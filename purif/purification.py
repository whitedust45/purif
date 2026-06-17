from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import math
import pandas as pd


PAYMENT_FORMAT_WEIGHTS = {
    0: 0.8,
    1: 1.0,
    2: 1.1,
    3: 1.1,
    4: 1.8,
    5: 1.5,
    6: 2.0,
}

CURRENCY_WEIGHTS = {
    0: 1.0,
    1: 2.0,
    2: 1.0,
    3: 1.0,
    4: 1.3,
    5: 1.4,
    6: 1.8,
    7: 1.3,
    8: 1.0,
    9: 1.0,
    10: 1.0,
    11: 0.9,
    12: 1.5,
    13: 1.2,
    14: 1.2,
}


@dataclass
class PurificationConfig:
    tau_h: float = 0.85
    tau_l: float = 0.15
    lambda_modularity: float = 0.30
    lambda_locality: float = 2.00
    epsilon: float = 1e-3
    max_iter: int = 30
    max_community_size: int = 500
    min_community_size: int = 3


@dataclass
class EdgeRecord:
    u: int
    v: int
    score: float
    weight: float
    phi: float
    count: int = 1


class Community:
    def __init__(self, cid: str, nodes: set[int], graph: "PurificationGraph", cfg: PurificationConfig):
        self.cid = cid
        self.nodes = set(nodes)
        self.graph = graph
        self.cfg = cfg
        self.refresh()

    def refresh(self) -> None:
        self.internal_edges = self.graph.internal_edges(self.nodes)
        self.sum_phi = sum(self.graph.edges[e].phi for e in self.internal_edges)
        self.internal_weight = sum(self.graph.edges[e].weight for e in self.internal_edges)
        self.volume = sum(self.graph.weighted_degree.get(node, 0.0) for node in self.nodes)
        self.locality = self._locality_penalty()
        self.modularity = self._modularity_score()
        self.objective = (
            self.sum_phi
            + self.cfg.lambda_modularity * self.modularity
            - self.cfg.lambda_locality * self.locality
        )

    def _locality_penalty(self) -> float:
        if not self.nodes:
            return 0.0
        return sum(self.graph.anchor_distance.get(node, 10_000.0) for node in self.nodes) / len(self.nodes)

    def _modularity_score(self) -> float:
        if self.graph.total_weight <= 0.0:
            return 0.0
        m = self.graph.total_weight
        return self.internal_weight / m - (self.volume / (2.0 * m)) ** 2

    def marginal_gain(self, node: int) -> float:
        if node in self.nodes:
            return -math.inf
        expanded = Community("candidate", self.nodes | {node}, self.graph, self.cfg)
        return expanded.objective - self.objective


class PurificationGraph:
    def __init__(self, edges: dict[tuple[int, int], EdgeRecord]):
        self.edges = edges
        self.adj = defaultdict(set)
        self.weighted_degree = defaultdict(float)

        for (u, v), record in edges.items():
            self.adj[u].add(v)
            self.adj[v].add(u)
            self.weighted_degree[u] += record.weight
            self.weighted_degree[v] += record.weight

        self.total_weight = sum(record.weight for record in edges.values())
        self.anchor_distance: dict[int, int] = {}

    def internal_edges(self, nodes: set[int]) -> set[tuple[int, int]]:
        internal = set()
        for u in nodes:
            for v in self.adj.get(u, ()):
                if v in nodes:
                    internal.add(_edge_key(u, v))
        return internal

    def compute_anchor_distance(self, anchor_edges: set[tuple[int, int]]) -> None:
        anchors = {node for edge in anchor_edges for node in edge}
        queue = deque()
        dist = {}

        for node in anchors:
            dist[node] = 0
            queue.append(node)

        while queue:
            u = queue.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)

        for node in self.adj:
            dist.setdefault(node, 10_000)

        self.anchor_distance = dist


def confidence_weighted_log_odds(score: float) -> float:
    eps = 1e-10
    score = max(eps, min(1.0 - eps, float(score)))
    confidence = 1.0 + score * math.log(score) + (1.0 - score) * math.log(1.0 - score)
    return confidence * math.log(score / (1.0 - score))


def risk_weight(
    score: float,
    amount: float = 1.0,
    currency: int | float = 0,
    payment_format: int | float = 1,
) -> float:
    try:
        currency_weight = CURRENCY_WEIGHTS.get(int(currency), 1.0)
        payment_weight = PAYMENT_FORMAT_WEIGHTS.get(int(payment_format), 1.0)
        return float(score) * abs(float(amount)) * currency_weight * payment_weight
    except (TypeError, ValueError):
        return float(score)


def load_purification_graph(csv_path: str | Path) -> tuple[PurificationGraph, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    required = {"from_id", "to_id", "prob_1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required purification columns: {missing}")

    edges: dict[tuple[int, int], EdgeRecord] = {}
    for _, row in df.iterrows():
        u = int(row["from_id"])
        v = int(row["to_id"])
        if u == v:
            continue
        key = _edge_key(u, v)
        score = float(row["prob_1"])
        amount = row.get("amount_sent", row.get("Amount Sent", 1.0))
        currency = row.get("sent_currency", row.get("Sent Currency", 0))
        payment_format = row.get("payment_format", row.get("Payment Format", 1))
        weight = risk_weight(score, amount=amount, currency=currency, payment_format=payment_format)
        phi = confidence_weighted_log_odds(score)

        if key in edges:
            previous = edges[key]
            new_count = previous.count + 1
            mean_score = (previous.score * previous.count + score) / new_count
            edges[key] = EdgeRecord(
                u=key[0],
                v=key[1],
                score=mean_score,
                weight=previous.weight + weight,
                phi=previous.phi + phi,
                count=new_count,
            )
        else:
            edges[key] = EdgeRecord(u=key[0], v=key[1], score=score, weight=weight, phi=phi)

    return PurificationGraph(edges), df


def partition_edges(
    graph: PurificationGraph,
    tau_h: float,
    tau_l: float,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    E_plus, E_minus, E_uncertain = set(), set(), set()
    for key, edge in graph.edges.items():
        if edge.score >= tau_h:
            E_plus.add(key)
        elif edge.score <= tau_l:
            E_minus.add(key)
        else:
            E_uncertain.add(key)
    return E_plus, E_minus, E_uncertain


def anchor_guided_purification(
    csv_path: str | Path,
    out_dir: str | Path,
    config: PurificationConfig | None = None,
) -> dict[str, Community]:
    cfg = config or PurificationConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph, _ = load_purification_graph(csv_path)
    E_plus, E_minus, E_uncertain = partition_edges(graph, cfg.tau_h, cfg.tau_l)

    graph.compute_anchor_distance(E_plus)

    communities = {
        f"C{idx}": Community(f"C{idx}", set(edge), graph, cfg)
        for idx, edge in enumerate(sorted(E_plus))
    }

    for _ in range(cfg.max_iter):
        old_objective = sum(comm.objective for comm in communities.values())
        _expand_communities(communities, graph, E_uncertain, cfg)
        communities = _merge_overlapping_communities(communities, graph, cfg)
        new_objective = sum(comm.objective for comm in communities.values())
        if abs(new_objective - old_objective) < cfg.epsilon:
            break

    communities = {
        cid: comm
        for cid, comm in communities.items()
        if len(comm.nodes) >= cfg.min_community_size
    }

    _save_communities(communities, out_dir)
    _save_edge_partitions(E_plus, E_minus, E_uncertain, out_dir)
    return communities


def _expand_communities(
    communities: dict[str, Community],
    graph: PurificationGraph,
    E_uncertain: set[tuple[int, int]],
    cfg: PurificationConfig,
) -> None:
    for comm in communities.values():
        if len(comm.nodes) >= cfg.max_community_size:
            continue

        frontier = set()
        for node in comm.nodes:
            for neighbor in graph.adj.get(node, ()):
                edge = _edge_key(node, neighbor)
                if neighbor not in comm.nodes and edge in E_uncertain:
                    frontier.add(neighbor)

        if not frontier:
            continue

        best_node = max(frontier, key=comm.marginal_gain)
        best_gain = comm.marginal_gain(best_node)
        if best_gain > 0.0:
            comm.nodes.add(best_node)
            comm.refresh()


def _merge_overlapping_communities(
    communities: dict[str, Community],
    graph: PurificationGraph,
    cfg: PurificationConfig,
) -> dict[str, Community]:
    updated = dict(communities)
    node_to_communities = defaultdict(set)
    for cid, comm in updated.items():
        for node in comm.nodes:
            node_to_communities[node].add(cid)

    candidate_pairs = set()
    for cids in node_to_communities.values():
        cids = sorted(cids)
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                candidate_pairs.add((cids[i], cids[j]))

    ranked_pairs = sorted(
        candidate_pairs,
        key=lambda pair: _leiden_screening_score(updated[pair[0]], updated[pair[1]], graph, cfg),
        reverse=True,
    )

    merge_idx = 0
    for c1, c2 in ranked_pairs:
        if c1 not in updated or c2 not in updated:
            continue
        comm1, comm2 = updated[c1], updated[c2]
        if not (comm1.nodes & comm2.nodes):
            continue

        merged_nodes = comm1.nodes | comm2.nodes
        if len(merged_nodes) > cfg.max_community_size:
            continue

        merged = Community(f"M{merge_idx}", merged_nodes, graph, cfg)
        if merged.objective > comm1.objective + comm2.objective:
            del updated[c1]
            del updated[c2]
            updated[merged.cid] = merged
            merge_idx += 1

    return updated


def _leiden_screening_score(
    comm1: Community,
    comm2: Community,
    graph: PurificationGraph,
    cfg: PurificationConfig,
) -> float:
    if graph.total_weight <= 0.0:
        return 0.0
    cross_weight = 0.0
    for u in comm1.nodes:
        for v in comm2.nodes:
            if u == v:
                continue
            edge = _edge_key(u, v)
            if edge in graph.edges:
                cross_weight += graph.edges[edge].weight
    return cfg.lambda_modularity * cross_weight / graph.total_weight


def _save_communities(communities: dict[str, Community], out_dir: Path) -> None:
    rows = []
    columns = ["community_id", "node_id", "objective", "modularity", "locality", "num_nodes"]
    for cid, comm in sorted(communities.items()):
        for node in sorted(comm.nodes):
            rows.append(
                {
                    "community_id": cid,
                    "node_id": node,
                    "objective": comm.objective,
                    "modularity": comm.modularity,
                    "locality": comm.locality,
                    "num_nodes": len(comm.nodes),
                }
            )
    pd.DataFrame(rows, columns=columns).to_csv(out_dir / "purified_communities.csv", index=False)


def _save_edge_partitions(
    E_plus: set[tuple[int, int]],
    E_minus: set[tuple[int, int]],
    E_uncertain: set[tuple[int, int]],
    out_dir: Path,
) -> None:
    rows = []
    columns = ["partition", "u", "v"]
    for name, edges in [("E_plus", E_plus), ("E_minus", E_minus), ("E_uncertain", E_uncertain)]:
        rows.extend({"partition": name, "u": u, "v": v} for u, v in sorted(edges))
    pd.DataFrame(rows, columns=columns).to_csv(out_dir / "edge_partitions.csv", index=False)


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)
