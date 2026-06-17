from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RENAME_COLUMNS = {
    "edge_index_0": "from_id",
    "edge_index_1": "to_id",
    "Ground_Truth": "is_laundering",
    "Prediction": "prediction",
    "edge_attr_0": "timestamp",
    "edge_attr_1": "amount_received",
    "edge_attr_2": "received_currency",
    "edge_attr_3": "payment_format",
}

BASE_COLUMNS = [
    "from_id",
    "to_id",
    "timestamp",
    "amount_sent",
    "sent_currency",
    "amount_received",
    "received_currency",
    "payment_format",
    "type",
    "output_0",
    "output_1",
    "prob_0",
    "prob_1",
    "is_laundering",
    "prediction",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ordered_existing_columns(df: pd.DataFrame, column_groups: Iterable[str]) -> list[str]:
    columns = []
    for prefix in column_groups:
        columns.extend([col for col in df.columns if col.startswith(prefix)])
    return columns


def prepare_edge_score_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    filter_prob0_threshold: float | None = None,
) -> Path:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df.rename(columns=RENAME_COLUMNS)

    if "prob_0" not in df.columns or "prob_1" not in df.columns:
        missing_logits = {"output_0", "output_1"} - set(df.columns)
        if missing_logits:
            raise ValueError(f"Missing columns for probability derivation: {missing_logits}")
        df["prob_0"] = sigmoid(df["output_0"])
        df["prob_1"] = sigmoid(df["output_1"])

    if "amount_sent" not in df.columns and "amount_received" in df.columns:
        df["amount_sent"] = df["amount_received"]
    if "sent_currency" not in df.columns and "received_currency" in df.columns:
        df["sent_currency"] = df["received_currency"]
    if "type" not in df.columns:
        df["type"] = 0

    required = {"from_id", "to_id", "prob_0", "prob_1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required PURIF columns: {missing}")

    if filter_prob0_threshold is not None:
        df = df[df["prob_0"] <= filter_prob0_threshold].copy()

    edge_attr_cols = _ordered_existing_columns(df, ["edge_attr_"])
    edge_emb_cols = _ordered_existing_columns(df, ["edge_emb_"])
    ordered_cols = [col for col in BASE_COLUMNS if col in df.columns]
    ordered_cols.extend([col for col in edge_attr_cols + edge_emb_cols if col not in ordered_cols])

    df[ordered_cols].to_csv(output_csv, index=False)
    return output_csv
