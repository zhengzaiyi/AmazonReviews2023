#!/usr/bin/env python3
"""Compute paired user-level bootstrap confidence intervals from prediction files."""

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
DEFAULT_METRICS = ["ndcg@20", "ndcg@50", "recall@20", "recall@50"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap CI for RoutePO vs best single-channel baseline."
    )
    parser.add_argument("--manifest", default=None, help="Optional legacy CSV/JSON manifest with prediction paths.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--routepo_method", default="top_k")
    parser.add_argument("--baseline_method", default="avg_score_weight")
    parser.add_argument("--recaller_order", nargs="*", default=None)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument("--datasets", nargs="+", default=["ml-1m", "steam", "Food"])
    parser.add_argument("--model_names", nargs="+", default=DEFAULT_MODEL_NAMES)
    parser.add_argument("--recbole_models", nargs="+", default=DEFAULT_RECALLERS)
    parser.add_argument("--model_root", default="GRPO/data/pure_models")
    parser.add_argument("--data_path", default="dataset")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--profile_cutoff", type=int, default=500000)
    parser.add_argument("--prompt_top_k", type=int, default=3)
    parser.add_argument("--baseline_selector_metric", default="ndcg@50")
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


def normalize_method(method: Optional[str], default: str) -> str:
    if method is None or str(method).strip() == "":
        return default
    return str(method).strip()


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


def infer_recaller_order(record: dict, explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        return [x.lower() for x in explicit]
    keys = list((record.get("recaller_predictions") or {}).keys())
    return [x.lower() for x in keys]


def normalize_weights(weights: Sequence[float], n: int) -> List[float]:
    if not weights or len(weights) != n:
        return [1.0 / n] * n if n else []
    weights = [max(0.0, float(w)) for w in weights]
    total = sum(weights)
    if total <= 0:
        return [1.0 / n] * n if n else []
    return [w / total for w in weights]


def merge_weighted_score(record: dict, recaller_order: Sequence[str]) -> List[int]:
    recaller_predictions = record.get("recaller_predictions") or {}
    weights = normalize_weights(record.get("merge_weights") or [], len(recaller_order))
    scores: Dict[int, float] = defaultdict(float)
    for name, weight in zip(recaller_order, weights):
        for item_id, score in recaller_predictions.get(name, []):
            scores[int(item_id)] += float(score) * weight
    return [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def merge_top_k(record: dict, recaller_order: Sequence[str], total_k: int = 50) -> List[int]:
    recaller_predictions = record.get("recaller_predictions") or {}
    weights = normalize_weights(record.get("merge_weights") or [], len(recaller_order))
    names = [x.lower() for x in recaller_order]
    quota_weights = dict(zip(names, weights))
    ptr = {name: 0 for name in names}
    selected = {name: 0 for name in names}
    seen = set()
    merged: List[int] = []
    hard_cap = total_k * max(50, len(names) * 10)

    def quota(weight: float, t: int) -> int:
        return int(round(weight * t))

    while len(merged) < total_k and hard_cap > 0:
        hard_cap -= 1
        t_next = len(merged) + 1
        best = None
        best_deficit = None
        for name in names:
            deficit = quota(quota_weights[name], t_next) - selected[name]
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


def predicted_items(record: dict, method: str, recaller_order: Optional[Sequence[str]], max_k: int = 50) -> List[int]:
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


def per_user_metric_map(
    records: Sequence[dict],
    method: str,
    recaller_order: Optional[Sequence[str]],
    metrics: Sequence[str],
) -> Dict[int, Dict[str, float]]:
    max_k = max(int(metric.split("@", 1)[1]) for metric in metrics)
    out: Dict[int, Dict[str, float]] = {}
    for record in records:
        user_id = int(record["user_id"])
        gt_items = record.get("gt_items") or record.get("ground_truth") or []
        items = predicted_items(record, method, recaller_order, max_k=max_k)
        out[user_id] = {metric: metric_at_k(items, gt_items, metric) for metric in metrics}
    return out


def bootstrap(diff: np.ndarray, samples: int, seed: int, ci: float) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(samples, dtype=float)
    chunk = 1000
    offset = 0
    while offset < samples:
        take = min(chunk, samples - offset)
        indices = rng.integers(0, n, size=(take, n))
        means[offset : offset + take] = diff[indices].mean(axis=1)
        offset += take
    alpha = (100.0 - ci) / 2.0
    low, high = np.percentile(means, [alpha, 100.0 - alpha])
    p_value = 2.0 * min(float(np.mean(means <= 0.0)), float(np.mean(means >= 0.0)))
    return float(diff.mean()), float(low), float(high), min(1.0, p_value)


def metric_cutoff(metric: str) -> int:
    return int(metric.split("@", 1)[1])


def required_eval_k(args: argparse.Namespace) -> int:
    return max([metric_cutoff(metric) for metric in args.metrics] + [metric_cutoff(args.baseline_selector_metric)])


def rows_from_records(
    args: argparse.Namespace,
    dataset: str,
    routepo_records: Sequence[dict],
    baseline_records: Sequence[dict],
    routepo_method: str,
    baseline_method: str,
    routepo_label: str,
    baseline_label: str,
    recaller_order: Optional[Sequence[str]],
    provenance: Optional[dict] = None,
) -> List[dict]:
    routepo_metrics = per_user_metric_map(routepo_records, routepo_method, recaller_order, args.metrics)
    baseline_metrics = per_user_metric_map(baseline_records, baseline_method, recaller_order, args.metrics)
    paired_users = sorted(set(routepo_metrics) & set(baseline_metrics))
    if not paired_users:
        raise ValueError(f"No paired users for dataset={dataset}")

    rows = []
    for metric in args.metrics:
        diff = np.asarray(
            [routepo_metrics[u][metric] - baseline_metrics[u][metric] for u in paired_users],
            dtype=float,
        )
        mean_delta, low, high, p_value = bootstrap(diff, args.bootstrap_samples, args.seed, args.ci)
        row = {
            "dataset": dataset,
            "metric": metric,
            "routepo_variant": routepo_label,
            "baseline": baseline_label,
            "n_users": len(paired_users),
            "mean_delta": mean_delta,
            "ci95_low": low,
            "ci95_high": high,
            "p_value": p_value,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "routepo_method": routepo_method,
            "baseline_method": baseline_method,
        }
        if provenance:
            row.update(provenance)
        rows.append(row)
    return rows


def write_summary(rows: Sequence[dict], path: Path, command: str) -> None:
    lines = [
        "# Paired Bootstrap CI Summary",
        "",
        f"- Date: {datetime.now().isoformat(timespec='seconds')}",
        f"- Host: {socket.gethostname()}",
        f"- Command: `{command}`",
        f"- Output CSV: `{path.with_name('bootstrap_ci.csv')}`",
        "- Method: paired user-level bootstrap over per-user metric deltas.",
        "- Caveat: this measures robustness across users, not multi-seed training variance.",
        "",
        "## LaTeX Table",
        "",
        "| Dataset | Metric | Compared to | Delta RoutePO | 95% CI | p-value |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['baseline']} | "
            f"{row['mean_delta']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | "
            f"{row['p_value']:.4f} |"
        )
    lines.extend(["", "## Inputs", ""])
    for row in rows:
        if row.get("prediction_cache"):
            lines.append(
                f"- {row['dataset']} {row['metric']}: cache `{row['prediction_cache']}`; "
                f"checkpoint `{row['checkpoint_path']}`; test data `{row['test_dataset_path']}`; "
                f"best single `{row['best_single_channel']}`; n={row['n_users']}."
            )
        else:
            lines.append(
                f"- {row['dataset']} {row['metric']}: RoutePO `{row['routepo_predictions']}`, "
                f"baseline `{row['baseline_predictions']}`; n={row['n_users']}."
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.manifest:
        manifest_path = resolve_path(args.manifest)
        manifest = read_manifest(manifest_path)
        if not manifest:
            raise ValueError(f"Manifest is empty: {manifest_path}")

        for spec in manifest:
            dataset = spec.get("dataset", "")
            routepo_path = resolve_path(spec["routepo_predictions"])
            baseline_path = resolve_path(spec["baseline_predictions"])
            routepo_method = normalize_method(spec.get("routepo_method") or spec.get("routepo_merge_method"), args.routepo_method)
            baseline_method = normalize_method(spec.get("baseline_method"), args.baseline_method)
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
            rows.extend(
                rows_from_records(
                    args,
                    dataset,
                    routepo_records,
                    baseline_records,
                    routepo_method,
                    baseline_method,
                    routepo_label,
                    baseline_label,
                    recaller_order,
                    {
                        "routepo_predictions": str(routepo_path),
                        "baseline_predictions": str(baseline_path),
                    },
                )
            )
    else:
        eval_k = required_eval_k(args)
        if args.dry_run_paths:
            dry_run_paths(args, eval_k)
            return
        for dataset in args.datasets:
            for model_name in args.model_names:
                bundle = load_or_generate_predictions(args, dataset, model_name, eval_k)
                baseline_method = f"single:{bundle.selected_baseline_channel}"
                rows.extend(
                    rows_from_records(
                        args,
                        dataset,
                        bundle.records,
                        bundle.records,
                        args.routepo_method,
                        baseline_method,
                        bundle.model_label,
                        f"best_single:{bundle.selected_baseline_channel}",
                        bundle.recaller_order,
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
        raise RuntimeError("No bootstrap rows were produced.")

    csv_path = output_dir / "bootstrap_ci.csv"
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "bootstrap_ci_summary.md"
    command = " ".join(shlex.quote(x) for x in sys.argv)
    write_summary(rows, summary_path, command)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

# Default usage: python GRPO/scripts/rebuttal/paired_bootstrap_ci.py --datasets ml-1m steam Food --model_names meta-llama/Llama-3.2-1B-Instruct --recbole_models ItemKNN LightGCN Pop --output_dir emnlp_recycle_outputs --bootstrap_samples 10000 --seed 42
