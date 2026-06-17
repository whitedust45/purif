from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm
from torch_scatter import scatter_add, scatter_max, scatter_mean


@dataclass
class ClassifierConfig:
    in_node_dim: int = 4
    in_edge_dim: int = 72
    hidden_dim: int = 64
    num_classes: int = 2
    proto_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    threshold: float = 0.70
    tau_contrastive: float = 0.20
    tau_prototype: float = 1.00
    tau_mask: float = 0.50
    retention_threshold: float = 0.50
    alpha_contrastive: float = 0.03
    beta_sparsity: float = 0.02
    lambda_proto: float = 0.05


class PrototypeHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, proto_dim: int, tau_prototype: float):
        super().__init__()
        self.tau_prototype = tau_prototype
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, proto_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))

    def forward(self, graph_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(self.projector(graph_repr), dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        distances = torch.cdist(z, prototypes, p=2)
        logits = -distances / self.tau_prototype
        min_distance = distances.min(dim=1).values
        return logits, min_distance


class LearnableTopologyMask(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_learner = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        node_repr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_repr: torch.Tensor,
        tau_mask: float,
    ) -> torch.Tensor:
        src, dst = edge_index
        features = torch.cat([node_repr[src], node_repr[dst], edge_repr], dim=-1)
        logits = self.edge_learner(features).view(-1)
        return torch.sigmoid(logits / tau_mask).clamp(0.01, 0.99)


class LearnableFeatureMask(nn.Module):
    def __init__(self, in_edge_dim: int):
        super().__init__()
        hidden_dim = max(1, in_edge_dim // 2)
        self.feature_learner = nn.Sequential(
            nn.Linear(in_edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_attr: torch.Tensor, tau_mask: float) -> torch.Tensor:
        logits = self.feature_learner(edge_attr).view(-1, 1)
        return torch.sigmoid(logits / tau_mask).clamp(0.01, 0.99)


class OODAwareSubgraphClassifier(nn.Module):
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        heads = config.num_heads
        head_dim = max(1, hidden_dim // heads)
        hidden_dim = head_dim * heads
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Linear(config.in_node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(config.in_edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.edge_updates = nn.ModuleList()

        for _ in range(config.num_layers):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    head_dim,
                    heads=heads,
                    concat=True,
                    dropout=0.4,
                    edge_dim=hidden_dim,
                    add_self_loops=True,
                )
            )
            self.norms.append(GraphNorm(hidden_dim))
            self.edge_updates.append(
                nn.Sequential(
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

        graph_repr_dim = 12 * hidden_dim
        self.prototype_head = PrototypeHead(
            graph_repr_dim,
            config.num_classes,
            config.proto_dim,
            config.tau_prototype,
        )
        self.baseline_classifier = nn.Sequential(
            nn.Linear(graph_repr_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.num_classes),
        )
        self.topology_mask = LearnableTopologyMask(hidden_dim)
        self.feature_mask = LearnableFeatureMask(config.in_edge_dim)
        self.contrast_projector = nn.Sequential(
            nn.Linear(graph_repr_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.proto_dim),
        )

    def encode(
        self,
        data,
        topology_mask: torch.Tensor | None = None,
        feature_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = int(data.num_graphs)
        x = self.node_encoder(x)
        if feature_mask is not None:
            edge_attr = edge_attr * feature_mask.reshape(-1, 1)
        edge_attr = self.edge_encoder(edge_attr)

        if topology_mask is not None:
            keep = self._retain_topology(topology_mask, edge_index.device)
            if keep.numel() == 0:
                return torch.zeros(num_graphs, 12 * self.hidden_dim, device=x.device)
            edge_index = edge_index[:, keep]
            edge_attr = edge_attr[keep]

        src, dst = edge_index
        for conv, norm, edge_update in zip(self.convs, self.norms, self.edge_updates):
            x = (x + F.relu(norm(conv(x, edge_index, edge_attr)))) / 2.0
            edge_attr = edge_attr + edge_update(torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2.0

        return graph_readout(x, edge_index, edge_attr, batch, num_graphs, self.hidden_dim)

    def forward(self, data, use_prototype: bool = True):
        graph_repr = self.encode(data)
        if use_prototype:
            return self.prototype_head(graph_repr)
        return self.baseline_classifier(graph_repr)

    def augmented_views(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            node_repr = self.node_encoder(data.x)
            edge_repr = self.edge_encoder(data.edge_attr)

        topo_1 = self.topology_mask(node_repr, data.edge_index, edge_repr, self.config.tau_mask)
        feat_1 = self.feature_mask(data.edge_attr, self.config.tau_mask)
        view_1 = self.encode(data, topology_mask=topo_1, feature_mask=feat_1)

        topo_2 = self.topology_mask(node_repr, data.edge_index, edge_repr, self.config.tau_mask)
        feat_2 = self.feature_mask(data.edge_attr, self.config.tau_mask)
        view_2 = self.encode(data, topology_mask=topo_2, feature_mask=feat_2)
        return view_1, view_2, topo_1, topo_2

    def _retain_topology(self, topology_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.training:
            keep = topology_mask > self.config.retention_threshold
            return keep.nonzero(as_tuple=False).view(-1)
        return torch.arange(topology_mask.numel(), device=device)


def purif_classifier_loss(
    model: OODAwareSubgraphClassifier,
    data,
    criterion: nn.Module | None = None,
    use_prototype: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    criterion = criterion or nn.CrossEntropyLoss()
    cfg = model.config

    graph_repr = model.encode(data)
    if use_prototype:
        logits, min_distance = model.prototype_head(graph_repr)
        prototype_regularization = min_distance.mean()
    else:
        logits = model.baseline_classifier(graph_repr)
        prototype_regularization = torch.tensor(0.0, device=graph_repr.device)

    labels = data.y.view(-1).long()
    classification_loss = criterion(logits, labels)

    view_1, view_2, topo_1, topo_2 = model.augmented_views(data)
    contrastive_loss = contrastive_consistency_loss(
        model.contrast_projector(view_1),
        model.contrast_projector(view_2),
        tau=cfg.tau_contrastive,
    )
    sparsity_loss = 0.5 * (topo_1.mean() + topo_2.mean())

    loss = (
        classification_loss
        + cfg.alpha_contrastive * contrastive_loss
        + cfg.beta_sparsity * sparsity_loss
        + cfg.lambda_proto * prototype_regularization
    )
    metrics = {
        "classification_loss": float(classification_loss.detach().cpu()),
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
        "prototype_regularization": float(prototype_regularization.detach().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def predict_purified_candidates(
    model: OODAwareSubgraphClassifier,
    loader,
    device: torch.device,
    ood_distance_threshold: float | None = None,
) -> list[dict[str, float | int]]:
    model.eval()
    model.to(device)
    predictions = []

    for data in loader:
        data = data.to(device)
        logits, min_distance = model(data, use_prototype=True)
        probabilities = F.softmax(logits, dim=1)
        anomaly_prob = probabilities[:, 1]
        pred_label = (anomaly_prob >= model.config.threshold).long()

        if ood_distance_threshold is not None:
            is_ood = min_distance > ood_distance_threshold
            pred_label = torch.where(is_ood, torch.zeros_like(pred_label), pred_label)
        else:
            is_ood = torch.zeros_like(pred_label, dtype=torch.bool)

        graph_ids = getattr(data, "graph_id", torch.arange(pred_label.numel(), device=device)).view(-1)
        for idx in range(pred_label.numel()):
            predictions.append(
                {
                    "graph_id": int(graph_ids[idx].detach().cpu()),
                    "pred_label": int(pred_label[idx].detach().cpu()),
                    "anomaly_probability": float(anomaly_prob[idx].detach().cpu()),
                    "min_prototype_distance": float(min_distance[idx].detach().cpu()),
                    "is_ood": int(is_ood[idx].detach().cpu()),
                }
            )

    return predictions


def contrastive_consistency_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    if z1.size(0) <= 1:
        return torch.tensor(0.0, device=z1.device)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def graph_readout(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    hidden_dim: int,
) -> torch.Tensor:
    src = edge_index[0]
    edge_repr = torch.cat([x[edge_index.t()].reshape(-1, 2 * hidden_dim), edge_attr], dim=-1).relu()

    node_mean = scatter_mean(x, batch, dim=0, dim_size=num_graphs)
    node_max = scatter_max(x, batch, dim=0, dim_size=num_graphs)[0].clamp(min=0)
    node_sum = scatter_add(x, batch, dim=0, dim_size=num_graphs)
    node_graph_repr = torch.cat([node_mean, node_max, node_sum], dim=-1)

    edge_batch = batch[src]
    edge_mean = scatter_mean(edge_repr, edge_batch, dim=0, dim_size=num_graphs)
    edge_max = scatter_max(edge_repr, edge_batch, dim=0, dim_size=num_graphs)[0].clamp(min=0)
    edge_sum = scatter_add(edge_repr, edge_batch, dim=0, dim_size=num_graphs)
    edge_graph_repr = torch.cat([edge_mean, edge_max, edge_sum], dim=-1)

    return torch.cat([node_graph_repr, edge_graph_repr], dim=-1)
