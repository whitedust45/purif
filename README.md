# PURIF

Reference implementation for the PURIF method.

## Layout

```text
purif/
  edge_scoring.py
  purification.py
  subgraph_extraction.py
  ood_classifier.py
configs/
  default.yaml
run_purif.py
requirements.txt
```

## Entry

```bash
python run_purif.py \
  --train_csv path/to/train_predictions.csv \
  --test_csv path/to/test_predictions.csv \
  --out_dir outputs/purif_run
```

## Input

The pipeline expects edge-level prediction CSV files with node ids, logits or
probabilities, labels when available, and edge attributes.

Common columns:

```text
edge_index_0, edge_index_1
output_0, output_1
prob_0, prob_1
Prediction, Ground_Truth
edge_attr_*, edge_emb_*
```

`prob_1` is used as the anomaly score. If probabilities are absent, they are
computed from `output_0` and `output_1`.
