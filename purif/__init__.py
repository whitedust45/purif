from .edge_scoring import prepare_edge_score_csv
from .purification import PurificationConfig, anchor_guided_purification

__all__ = [
    "prepare_edge_score_csv",
    "PurificationConfig",
    "anchor_guided_purification",
    "build_candidate_subgraphs",
    "ClassifierConfig",
    "OODAwareSubgraphClassifier",
    "predict_purified_candidates",
]


def __getattr__(name):
    if name == "build_candidate_subgraphs":
        from .subgraph_extraction import build_candidate_subgraphs

        return build_candidate_subgraphs
    if name in {"ClassifierConfig", "OODAwareSubgraphClassifier", "predict_purified_candidates"}:
        from .ood_classifier import ClassifierConfig, OODAwareSubgraphClassifier, predict_purified_candidates

        return {
            "ClassifierConfig": ClassifierConfig,
            "OODAwareSubgraphClassifier": OODAwareSubgraphClassifier,
            "predict_purified_candidates": predict_purified_candidates,
        }[name]
    raise AttributeError(name)
