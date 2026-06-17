from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


EDGE_FEATURE_COLUMNS = [
    "amount_sent",
    "sent_currency",
    "amount_received",
    "received_currency",
    "payment_format",
    "output_0",
    "output_1",
    "prob_0",
    "prob_1",
    "prediction",
]


def build_candidate_subgraphs(
    edge_csv: str | Path,
    community_csv: str | Path,
    output_pt: str | Path,
    min_edges: int = 1,
) -> list[Data]:
    edge_csv = Path(edge_csv)
    community_csv = Path(community_csv)
    output_pt = Path(output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(edge_csv)
    communities = pd.read_csv(community_csv)

    required_edges = {"from_id", "to_id", "prob_1"}
    required_communities = {"community_id", "node_id"}
    missing_edges = required_edges - set(edges.columns)
    missing_communities = required_communities - set(communities.columns)
    if missing_edges:
        raise ValueError(f"Missing edge columns: {missing_edges}")
    if missing_communities:
        raise ValueError(f"Missing community columns: {missing_communities}")

    candidates = []
    for graph_id, (community_id, group) in enumerate(communities.groupby("community_id")):
        node_set = set(group["node_id"].astype(int).tolist())
        sub_edges = edges[
            edges["from_id"].astype(int).isin(node_set)
            & edges["to_id"].astype(int).isin(node_set)
        ].copy()

        if len(sub_edges) < min_edges:
            continue

        data = _data_from_edge_frame(sub_edges, graph_id=graph_id, community_id=str(community_id))
        candidates.append(data)

    torch.save(candidates, output_pt)
    return candidates


def _data_from_edge_frame(sub_edges: pd.DataFrame, graph_id: int, community_id: str) -> Data:
    original_nodes = sorted(
        set(sub_edges["from_id"].astype(int).tolist())
        | set(sub_edges["to_id"].astype(int).tolist())
    )
    node_map = {node: idx for idx, node in enumerate(original_nodes)}

    src = [node_map[int(u)] for u in sub_edges["from_id"]]
    dst = [node_map[int(v)] for v in sub_edges["to_id"]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    feature_columns = [col for col in EDGE_FEATURE_COLUMNS if col in sub_edges.columns]
    feature_columns.extend([col for col in sub_edges.columns if col.startswith("edge_attr_")])
    feature_columns.extend([col for col in sub_edges.columns if col.startswith("edge_emb_")])
    edge_attr = torch.tensor(sub_edges[feature_columns].fillna(0.0).to_numpy(), dtype=torch.float)

    edge_y = None
    if "is_laundering" in sub_edges.columns:
        edge_y = torch.tensor(sub_edges["is_laundering"].to_numpy(), dtype=torch.long)

    graph_label = 0
    if edge_y is not None and int(edge_y.sum().item()) > 0:
        graph_label = 1

    x = _node_features(edge_index, num_nodes=len(original_nodes))

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_y=edge_y,
        y=torch.tensor([graph_label], dtype=torch.long),
        graph_id=torch.tensor([graph_id], dtype=torch.long),
        community_id=community_id,
        orig_node_ids=torch.tensor(original_nodes, dtype=torch.long),
    )


def _node_features(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if num_nodes == 0:
        return torch.zeros((0, 4), dtype=torch.float)

    src, dst = edge_index
    out_degree = torch.bincount(src, minlength=num_nodes).float()
    in_degree = torch.bincount(dst, minlength=num_nodes).float()
    total_degree = in_degree + out_degree
    bias = torch.ones(num_nodes, dtype=torch.float)
    return torch.stack([in_degree, out_degree, total_degree, bias], dim=1)
