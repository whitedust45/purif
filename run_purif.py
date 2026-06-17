from __future__ import annotations

import argparse
import json
from pathlib import Path

from purif.edge_scoring import prepare_edge_score_csv
from purif.purification import PurificationConfig, anchor_guided_purification
from purif.subgraph_extraction import build_candidate_subgraphs


def run_pipeline(args: argparse.Namespace) -> dict[str, str]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_scores = prepare_edge_score_csv(
        args.train_csv,
        out_dir / "stage1_train_edge_scores.csv",
        filter_prob0_threshold=args.filter_prob0_threshold,
    )
    test_scores = prepare_edge_score_csv(
        args.test_csv,
        out_dir / "stage1_test_edge_scores.csv",
        filter_prob0_threshold=args.filter_prob0_threshold,
    )

    purification_cfg = PurificationConfig(
        tau_h=args.tau_h,
        tau_l=args.tau_l,
        lambda_modularity=args.lambda_modularity,
        lambda_locality=args.lambda_locality,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        max_community_size=args.max_community_size,
        min_community_size=args.min_community_size,
    )

    train_purif_dir = out_dir / "stage2_train_purification"
    test_purif_dir = out_dir / "stage2_test_purification"
    anchor_guided_purification(train_scores, train_purif_dir, purification_cfg)
    anchor_guided_purification(test_scores, test_purif_dir, purification_cfg)

    train_candidates = out_dir / "stage3_train_candidates.pt"
    test_candidates = out_dir / "stage3_test_candidates.pt"
    build_candidate_subgraphs(
        edge_csv=train_scores,
        community_csv=train_purif_dir / "purified_communities.csv",
        output_pt=train_candidates,
    )
    build_candidate_subgraphs(
        edge_csv=test_scores,
        community_csv=test_purif_dir / "purified_communities.csv",
        output_pt=test_candidates,
    )

    summary = {
        "train_edge_scores": str(train_scores),
        "test_edge_scores": str(test_scores),
        "train_purification_dir": str(train_purif_dir),
        "test_purification_dir": str(test_purif_dir),
        "train_candidates": str(train_candidates),
        "test_candidates": str(test_candidates),
        "classifier_module": "purif/ood_classifier.py",
        "classifier_inference": "purif.ood_classifier.predict_purified_candidates",
    }
    with open(out_dir / "purif_outputs.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PURIF release pipeline")
    parser.add_argument("--train_csv", required=True, help="Training edge-score CSV from the edge detector")
    parser.add_argument("--test_csv", required=True, help="Test edge-score CSV from the edge detector")
    parser.add_argument("--out_dir", required=True, help="Output directory")

    parser.add_argument(
        "--filter_prob0_threshold",
        type=float,
        default=None,
        help="Optional legacy pre-filter. Disabled by default to match the paper's full-edge partition.",
    )
    parser.add_argument("--tau_h", type=float, default=0.85)
    parser.add_argument("--tau_l", type=float, default=0.15)
    parser.add_argument("--lambda_modularity", type=float, default=0.30)
    parser.add_argument("--lambda_locality", type=float, default=2.00)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--max_community_size", type=int, default=500)
    parser.add_argument("--min_community_size", type=int, default=3)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
