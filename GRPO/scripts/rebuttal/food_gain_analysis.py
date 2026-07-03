#!/usr/bin/env python3
"""Analyze Food gains using cached per-user predictions and routing weights."""

import argparse
import csv
import json
import math
import shlex
import socket
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from manual_eval import (
    DEFAULT_MODEL_NAMES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RECALLERS,
    dry_run_paths,
    load_or_generate_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHANNELS = ["pop", "itemknn", "lightgcn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Food gain analysis from manual RoutePO evaluation and best single-channel baseline."
    )
    parser.add_argument("--manifest", default=None, help="Optional legacy CSV/JSON manifest with prediction paths.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--routepo_method", default="top_k")
    parser.add_argument("--baseline_method", default="avg_score_weight")
    parser.add_argument("--recaller_order", nargs="*", default=None)
    parser.add_argument("--metric", default="ndcg@50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline_selector_metric", default="ndcg@50")
    parser.add_argument("--datasets", nargs="+", default=["ml-1m", "steam", "Food"])
    parser.add_argument("--model_names", nargs="+", default=DEFAULT_MODEL_NAMES)
    parser.add_argument("--recbole_models", nargs="+", default=DEFAULT_RECALLERS)
    parser.add_argument("--model_root", default="GRPO/data/pure_models")
    parser.add_argument("--data_path", default="dataset")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--profile_cutoff", type=int, default=500000)
    parser.add_argument("--prompt_top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--padding_side", default="left", choices=["left", "right"])
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, otherwise cpu.")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--force_eval", action="store_true", help="Regenerate manual evaluation cache.")
    parser.add_argument("--dry_run_paths", action="store_true", help="Print manual evaluation paths and exit.")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def read_manifest(path: Path) -> List[dict]:
    if path.suffix.lower() == ".json":
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("experiments", [])
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def list_like_records(value) -> bool:
    return isinstance(value, list) and (not value or isinstance(value[0], dict))


def records_from_prediction_file(path: Path, method: str) -> List[dict]:
    data = load_json(path)
    if list_like_records(data):
        return data
    if isinstance(data, dict):
        if method in data and list_like_records(data[method]):
            return data[method]
        if "predictions" in data and list_like_records(data["predictions"]):
            return data["predictions"]
        list_keys = [k for k, v in data.items() if list_like_records(v)]
        if len(list_keys) == 1:
            return data[list_keys[0]]
    raise ValueError(f"Could not find prediction records in {path} with method={method}")


def metric_at_k(predicted: Sequence[int], gt_items: Sequence[int], metric: str) -> float:
    name, k_raw = metric.split("@", 1)
    k = int(k_raw)
    gt = set(gt_items if isinstance(gt_items, list) else [gt_items])
    if not gt:
        return 0.0
    hits = [1 if item in gt else 0 for item in predicted[:k]]
    if name == "recall":
        return float(sum(hits) / min(len(gt), k))
    if name == "ndcg":
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
        return float(dcg / idcg) if idcg > 0 else 0.0
    raise ValueError(f"Unsupported metric: {metric}")


def normalize_weights(weights: Sequence[float], n: int) -> List[float]:
    if not weights or len(weights) != n:
        return [1.0 / n] * n if n else []
    weights = [max(0.0, float(w)) for w in weights]
    total = sum(weights)
    if total <= 0:
        return [1.0 / n] * n if n else []
    return [w / total for w in weights]


def infer_recaller_order(record: dict, explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        return [x.lower() for x in explicit]
    return [x.lower() for x in (record.get("recaller_predictions") or {}).keys()]


def merge_weighted_score(record: dict, recaller_order: Sequence[str]) -> List[int]:
    recaller_predictions = record.get("recaller_predictions") or {}
    weights = normalize_weights(record.get("merge_weights") or [], len(recaller_order))
    scores: Dict[int, float] = defaultdict(float)
    for name, weight in zip(recaller_order, weights):
        for item_id, score in recaller_predictions.get(name, []):
            scores[int(item_id)] += float(score) * weight
    return [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def merge_top_k(record: dict, recaller_order: Sequence[str], total_k: int) -> List[int]:
    recaller_predictions = record.get("recaller_predictions") or {}
    weights = normalize_weights(record.get("merge_weights") or [], len(recaller_order))
    names = [x.lower() for x in recaller_order]
    quota_weights = dict(zip(names, weights))
    ptr = {name: 0 for name in names}
    selected = {name: 0 for name in names}
    seen = set()
    merged: List[int] = []
    hard_cap = total_k * max(50, len(names) * 10)

    while len(merged) < total_k and hard_cap > 0:
        hard_cap -= 1
        t_next = len(merged) + 1
        best = None
        best_deficit = None
        for name in names:
            deficit = int(round(quota_weights[name] * t_next)) - selected[name]
            if best is None or deficit > best_deficit:
                best, best_deficit = name, deficit
            elif deficit == best_deficit:
                old_items = recaller_predictions.get(best, [])
                new_items = recaller_predictions.get(name, [])
                old_score = old_items[ptr[best]][1] if ptr[best] < len(old_items) else float("-inf")
                new_score = new_items[ptr[name]][1] if ptr[name] < len(new_items) else float("-inf")
                if new_score > old_score:
                    best, best_deficit = name, deficit
        if best is None:
            break
        items = recaller_predictions.get(best, [])
        while ptr[best] < len(items):
            item_id = int(items[ptr[best]][0])
            ptr[best] += 1
            if item_id in seen:
                continue
            seen.add(item_id)
            selected[best] += 1
            merged.append(item_id)
            break
        else:
            break
    return merged


def predicted_items(record: dict, method: str, recaller_order: Optional[Sequence[str]], max_k: int) -> List[int]:
    if "predicted_items" in record:
        return [int(x) for x in record["predicted_items"]]
    if method.startswith("single:"):
        name = method.split(":", 1)[1].lower()
        return [int(x[0]) for x in (record.get("recaller_predictions") or {}).get(name, [])]
    order = infer_recaller_order(record, recaller_order)
    if method in {"top_k", "topk", "avg_top_k", "snake"}:
        return merge_top_k(record, order, total_k=max_k)
    if method in {"weighted_score", "score", "average", "avg_score_weight", "multi_channel"}:
        return merge_weighted_score(record, order)
    raise ValueError(f"Unsupported prediction method: {method}")


def entropy(probs: Sequence[float]) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 0))


def spec_value(spec: dict, key: str, default: str) -> str:
    value = spec.get(key)
    return default if value is None or str(value).strip() == "" else str(value).strip()


def metric_cutoff(metric: str) -> int:
    return int(metric.split("@", 1)[1])


def required_eval_k(args: argparse.Namespace) -> int:
    return max(metric_cutoff(args.metric), metric_cutoff(args.baseline_selector_metric))


def analyze_prediction_records(
    dataset: str,
    routepo_records: Sequence[dict],
    baseline_records: Sequence[dict],
    routepo_method: str,
    baseline_method: str,
    routepo_label: str,
    baseline_label: str,
    recaller_order: Optional[Sequence[str]],
    channels: Sequence[str],
    metric: str,
    provenance: Optional[dict] = None,
) -> dict:
    max_k = int(metric.split("@", 1)[1])
    routepo_by_user = {int(r["user_id"]): r for r in routepo_records}
    baseline_by_user = {int(r["user_id"]): r for r in baseline_records}
    paired_users = sorted(set(routepo_by_user) & set(baseline_by_user))
    if not paired_users:
        raise ValueError(f"No paired users for dataset={dataset}")

    oracle_counts = {channel: 0.0 for channel in channels}
    tie_users = 0
    routepo_scores = []
    baseline_scores = []
    weight_entropies = []
    weight_variances = []
    channel_weight_values = {channel: [] for channel in channels}

    for user_id in paired_users:
        record = routepo_by_user[user_id]
        baseline_record = baseline_by_user[user_id]
        gt_items = record.get("gt_items") or record.get("ground_truth") or []
        recaller_predictions = record.get("recaller_predictions") or {}
        if not recaller_predictions:
            raise ValueError(f"RoutePO record for user {user_id} has no recaller_predictions")

        channel_scores = {}
        for channel in channels:
            items = [int(x[0]) for x in recaller_predictions.get(channel, [])]
            channel_scores[channel] = metric_at_k(items, gt_items, metric)
        best_score = max(channel_scores.values())
        winners = [channel for channel, score in channel_scores.items() if score == best_score]
        if len(winners) > 1:
            tie_users += 1
        for channel in winners:
            oracle_counts[channel] += 1.0 / len(winners)

        order = infer_recaller_order(record, recaller_order)
        weights = normalize_weights(record.get("merge_weights") or [], len(order))
        by_name = dict(zip(order, weights))
        ordered_channel_weights = [by_name.get(channel, 0.0) for channel in channels]
        s = sum(ordered_channel_weights)
        if s > 0:
            ordered_channel_weights = [w / s for w in ordered_channel_weights]
        weight_entropies.append(entropy(ordered_channel_weights))
        weight_variances.append(float(np.var(np.asarray(ordered_channel_weights, dtype=float))))
        for channel, weight in zip(channels, ordered_channel_weights):
            channel_weight_values[channel].append(weight)

        routepo_items = predicted_items(record, routepo_method, recaller_order, max_k)
        baseline_gt = baseline_record.get("gt_items") or baseline_record.get("ground_truth") or gt_items
        baseline_items = predicted_items(baseline_record, baseline_method, recaller_order, max_k)
        routepo_scores.append(metric_at_k(routepo_items, gt_items, metric))
        baseline_scores.append(metric_at_k(baseline_items, baseline_gt, metric))

    n_users = len(paired_users)
    oracle_probs = [oracle_counts[channel] / n_users for channel in channels]
    oracle_entropy = entropy(oracle_probs)
    max_entropy = math.log(len(channels)) if len(channels) > 1 else 1.0
    routepo_mean = float(np.mean(routepo_scores))
    baseline_mean = float(np.mean(baseline_scores))

    row = {
        "dataset": dataset,
        "routepo_variant": routepo_label,
        "baseline": baseline_label,
        "n_users": n_users,
        "metric": metric,
        "oracle_entropy": oracle_entropy,
        "oracle_entropy_norm": oracle_entropy / max_entropy if max_entropy > 0 else 0.0,
        "routepo_weight_entropy_mean": float(np.mean(weight_entropies)),
        "routepo_weight_entropy_norm_mean": float(np.mean(weight_entropies)) / max_entropy if max_entropy > 0 else 0.0,
        "routepo_weight_variance_mean": float(np.mean(weight_variances)),
        "routepo_metric": routepo_mean,
        "baseline_metric": baseline_mean,
        "routepo_gain": routepo_mean - baseline_mean,
        "oracle_tie_users": tie_users,
        "routepo_method": routepo_method,
        "baseline_method": baseline_method,
    }
    for channel, prob in zip(channels, oracle_probs):
        row[f"oracle_{channel}_pct"] = prob * 100.0
        row[f"routepo_{channel}_weight_mean"] = float(np.mean(channel_weight_values[channel]))
        row[f"routepo_{channel}_weight_var"] = float(np.var(channel_weight_values[channel]))
    if provenance:
        row.update(provenance)
    return row


def analyze_spec(spec: dict, args: argparse.Namespace) -> dict:
    dataset = spec.get("dataset", "")
    routepo_path = resolve_path(spec["routepo_predictions"])
    baseline_path = resolve_path(spec["baseline_predictions"])
    routepo_method = spec_value(spec, "routepo_method", args.routepo_method)
    baseline_method = spec_value(spec, "baseline_method", args.baseline_method)
    routepo_label = spec.get("routepo_label") or spec.get("routepo_variant") or routepo_method
    baseline_label = spec.get("baseline_label") or baseline_method
    recaller_order_raw = spec.get("recaller_order")
    recaller_order = (
        [x.lower() for x in recaller_order_raw.replace(",", " ").split()]
        if recaller_order_raw
        else ([x.lower() for x in args.recaller_order] if args.recaller_order else None)
    )
    routepo_records = records_from_prediction_file(routepo_path, routepo_method)
    baseline_records = records_from_prediction_file(baseline_path, baseline_method)
    return analyze_prediction_records(
        dataset,
        routepo_records,
        baseline_records,
        routepo_method,
        baseline_method,
        routepo_label,
        baseline_label,
        recaller_order,
        [x.lower() for x in args.channels],
        args.metric,
        {
            "routepo_predictions": str(routepo_path),
            "baseline_predictions": str(baseline_path),
        },
    )


def write_summary(rows: Sequence[dict], path: Path, channels: Sequence[str], command: str) -> None:
    lines = [
        "# Food Gain Analysis Summary",
        "",
        f"- Date: {datetime.now().isoformat(timespec='seconds')}",
        f"- Host: {socket.gethostname()}",
        f"- Command: `{command}`",
        f"- Output CSV: `{path.with_name('food_gain_analysis.csv')}`",
        "- Method: manual RoutePO evaluation with cached per-user predictions; no new RoutePO training.",
        "- Oracle distribution uses fractional credit for ties.",
        "",
        "## LaTeX Table",
        "",
        "| Dataset | Oracle channel entropy | Routing weight variance | RoutePO N@50 gain |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['oracle_entropy']:.4f} | "
            f"{row['routepo_weight_variance_mean']:.6f} | {row['routepo_gain']:.6f} |"
        )
    lines.extend(["", "## Details", ""])
    for row in rows:
        channel_bits = ", ".join(
            f"{channel}={row.get(f'oracle_{channel}_pct', 0.0):.1f}%"
            for channel in channels
        )
        lines.extend(
            [
                f"### {row['dataset']}",
                "",
                f"- Users: {row['n_users']}",
                f"- Oracle distribution: {channel_bits}",
                f"- Oracle entropy normalized: {row['oracle_entropy_norm']:.4f}",
                f"- RoutePO weight entropy normalized mean: {row['routepo_weight_entropy_norm_mean']:.4f}",
                f"- RoutePO {row['metric']}: {row['routepo_metric']:.6f}",
                f"- Baseline {row['metric']}: {row['baseline_metric']:.6f}",
                f"- Gain: {row['routepo_gain']:.6f}",
                "",
            ]
        )
        if row.get("prediction_cache"):
            lines.extend(
                [
                    f"- Prediction cache: `{row['prediction_cache']}`",
                    f"- Checkpoint: `{row['checkpoint_path']}`",
                    f"- Test dataset: `{row['test_dataset_path']}`",
                    f"- Best single channel: `{row['best_single_channel']}`",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"- RoutePO predictions: `{row['routepo_predictions']}`",
                    f"- Baseline predictions: `{row['baseline_predictions']}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        manifest_path = resolve_path(args.manifest)
        manifest = read_manifest(manifest_path)
        if not manifest:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        rows = [analyze_spec(spec, args) for spec in manifest]
    else:
        eval_k = required_eval_k(args)
        if args.dry_run_paths:
            dry_run_paths(args, eval_k)
            return
        rows = []
        channels = [x.lower() for x in args.channels]
        for dataset in args.datasets:
            for model_name in args.model_names:
                bundle = load_or_generate_predictions(args, dataset, model_name, eval_k)
                baseline_method = f"single:{bundle.selected_baseline_channel}"
                rows.append(
                    analyze_prediction_records(
                        dataset,
                        bundle.records,
                        bundle.records,
                        args.routepo_method,
                        baseline_method,
                        bundle.model_label,
                        f"best_single:{bundle.selected_baseline_channel}",
                        bundle.recaller_order,
                        channels,
                        args.metric,
                        {
                            "model_name": model_name,
                            "prediction_cache": str(bundle.prediction_cache),
                            "checkpoint_path": str(bundle.checkpoint_path),
                            "test_dataset_path": str(bundle.test_dataset_path),
                            "best_single_channel": bundle.selected_baseline_channel,
                            "baseline_selector_metric": args.baseline_selector_metric,
                            "single_channel_metrics": json.dumps(bundle.single_channel_metrics, sort_keys=True),
                            "max_length": bundle.max_length,
                        },
                    )
                )

    if not rows:
        raise RuntimeError("No food gain rows were produced.")

    csv_path = output_dir / "food_gain_analysis.csv"
    all_keys = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "food_gain_analysis_summary.md"
    command = " ".join(shlex.quote(x) for x in sys.argv)
    write_summary(rows, summary_path, [x.lower() for x in args.channels], command)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

# Default usage: python GRPO/scripts/rebuttal/food_gain_analysis.py --datasets ml-1m steam Food --model_names meta-llama/Llama-3.2-1B-Instruct --recbole_models ItemKNN LightGCN Pop --output_dir emnlp_recycle_outputs --channels pop itemknn lightgcn --metric ndcg@50
