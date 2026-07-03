#!/usr/bin/env python3
"""Shared manual evaluation helpers for rebuttal analysis scripts."""

import json
import re
import shlex
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_DIR = "emnlp_recycle_outputs"
DEFAULT_MODEL_NAMES = ["meta-llama/Llama-3.2-1B-Instruct"]
DEFAULT_RECALLERS = ["ItemKNN", "LightGCN", "Pop"]
DEFAULT_REPORT_METRICS = ["ndcg@10", "ndcg@20", "ndcg@50", "recall@10", "recall@20", "recall@50"]
SCHEMA_VERSION = 1


@dataclass
class PredictionBundle:
    dataset: str
    model_name: str
    model_label: str
    records: List[dict]
    recaller_order: List[str]
    selected_baseline_channel: str
    single_channel_metrics: Dict[str, float]
    prediction_cache: Path
    checkpoint_path: Path
    test_dataset_path: Path
    max_length: int
    metadata: dict


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def combo_name(recbole_models: Sequence[str]) -> str:
    return "_".join(sorted(recbole_models))


def model_short_name(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def ptk_suffix(prompt_top_k: int) -> str:
    return f"_ptk{prompt_top_k}" if prompt_top_k != 3 else ""


def checkpoint_suffix(profile_cutoff: int, prompt_top_k: int) -> str:
    pc_suffix = f"_pc{profile_cutoff}" if profile_cutoff != 20 else ""
    return f"{pc_suffix}{ptk_suffix(prompt_top_k)}"


def expected_paths(args, dataset: str, model_name: str) -> Tuple[Path, Path]:
    root = resolve_path(args.model_root)
    short = model_short_name(model_name)
    combo = combo_name(args.recbole_models)
    model_base = root / dataset / short
    checkpoint = model_base.parent / f"{short}_pure_sft_{combo}{checkpoint_suffix(args.profile_cutoff, args.prompt_top_k)}"
    test_data = model_base.parent / f"{short}_pure_sft_data_{combo}_{args.profile_cutoff}{ptk_suffix(args.prompt_top_k)}" / "test"
    return checkpoint, test_data


def resolve_model_path(path: Path) -> Path:
    if (path / "config.json").exists():
        return path
    try:
        from transformers.trainer_utils import get_last_checkpoint
    except Exception:
        return path
    last = get_last_checkpoint(str(path)) if path.exists() else None
    return Path(last) if last else path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def cache_path(args, dataset: str, model_name: str, eval_k: int) -> Path:
    output_dir = resolve_path(args.output_dir)
    cache_dir = output_dir / "predictions"
    combo = "_".join(x.lower() for x in sorted(args.recbole_models))
    limit = f"u{args.max_users}" if args.max_users else "all"
    name = (
        f"manual_predictions_{safe_name(dataset)}_{safe_name(model_short_name(model_name))}_"
        f"{safe_name(combo)}_pc{args.profile_cutoff}_ptk{args.prompt_top_k}_k{eval_k}_{limit}.json"
    )
    return cache_dir / name


def effective_max_length(args, dataset: str, data_dir: Path) -> int:
    if args.max_length is not None:
        return int(args.max_length)
    token_stats_path = data_dir.parent / "token_stats.json"
    if token_stats_path.exists():
        with token_stats_path.open() as f:
            return int(json.load(f).get("max_tokens_cls", 1536))
    return 11024 if dataset == "Food" else 1536


def metric_at_k(predicted: Sequence[int], gt_items: Sequence[int], metric: str) -> float:
    import math

    name, k_raw = metric.split("@", 1)
    k = int(k_raw)
    gt = set(gt_items if isinstance(gt_items, list) else [gt_items])
    if not gt:
        return 0.0
    hits = [1 if int(item) in gt else 0 for item in predicted[:k]]
    if name == "recall":
        return float(sum(hits) / min(len(gt), k))
    if name == "ndcg":
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
        return float(dcg / idcg) if idcg > 0 else 0.0
    raise ValueError(f"Unsupported metric: {metric}")


def metric_cutoff(metric: str) -> int:
    return int(metric.split("@", 1)[1])


def metrics_for_report(args, eval_k: int) -> List[str]:
    metrics: List[str] = []
    candidates: List[str] = list(DEFAULT_REPORT_METRICS)
    if getattr(args, "baseline_selector_metric", None):
        candidates.append(args.baseline_selector_metric)
    if getattr(args, "metric", None):
        candidates.append(args.metric)
    if getattr(args, "metrics", None):
        candidates.extend(args.metrics)
    for metric in candidates:
        try:
            if metric_cutoff(metric) > eval_k:
                continue
        except Exception:
            continue
        if metric not in metrics:
            metrics.append(metric)
    return metrics


def normalize_recaller_names(names: Sequence[str]) -> List[str]:
    return [str(name).lower() for name in names]


def metadata_template(
    args,
    dataset: str,
    model_name: str,
    checkpoint_path: Path,
    test_dataset_path: Path,
    eval_k: int,
    max_length: int,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "model_name": model_name,
        "model_label": model_short_name(model_name),
        "checkpoint_kind": "pure_sft",
        "checkpoint_path": str(checkpoint_path),
        "test_dataset_path": str(test_dataset_path),
        "recbole_models": normalize_recaller_names(sorted(args.recbole_models)),
        "profile_cutoff": int(args.profile_cutoff),
        "prompt_top_k": int(args.prompt_top_k),
        "eval_k": int(eval_k),
        "max_users": int(args.max_users) if args.max_users else None,
        "max_length": int(max_length),
        "baseline_selector_metric": args.baseline_selector_metric,
        "seed": int(getattr(args, "seed", 42)),
    }


def metadata_matches(cached: dict, expected: dict) -> bool:
    meta = cached.get("metadata") or {}
    return all(meta.get(key) == value for key, value in expected.items())


def single_channel_metrics(records: Sequence[dict], channels: Sequence[str], metric: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for channel in channels:
        values = []
        for record in records:
            predictions = (record.get("recaller_predictions") or {}).get(channel, [])
            items = [int(item[0]) for item in predictions]
            gt_items = record.get("gt_items") or record.get("ground_truth") or []
            values.append(metric_at_k(items, gt_items, metric))
        out[channel] = float(np.mean(values)) if values else 0.0
    return out


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
    scores: Dict[int, float] = {}
    for name, weight in zip(recaller_order, weights):
        for item_id, score in recaller_predictions.get(name, []):
            item_id = int(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + float(score) * weight
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


def predicted_items(record: dict, method: str, recaller_order: Sequence[str], max_k: int) -> List[int]:
    if method.startswith("single:"):
        name = method.split(":", 1)[1].lower()
        return [int(x[0]) for x in (record.get("recaller_predictions") or {}).get(name, [])]
    if method == "single_select":
        weights = normalize_weights(record.get("merge_weights") or [], len(recaller_order))
        name = recaller_order[int(np.argmax(np.asarray(weights, dtype=float)))] if recaller_order else ""
        return [int(x[0]) for x in (record.get("recaller_predictions") or {}).get(name, [])]
    if method in {"top_k", "topk", "avg_top_k", "snake"}:
        return merge_top_k(record, recaller_order, total_k=max_k)
    if method in {"weighted_score", "score", "average", "avg_score_weight", "multi_channel"}:
        return merge_weighted_score(record, recaller_order)
    raise ValueError(f"Unsupported report method: {method}")


def evaluate_method_metrics(
    records: Sequence[dict],
    method: str,
    recaller_order: Sequence[str],
    metrics: Sequence[str],
) -> Dict[str, float]:
    max_k = max(metric_cutoff(metric) for metric in metrics) if metrics else 0
    out: Dict[str, float] = {}
    for metric in metrics:
        values = []
        for record in records:
            gt_items = record.get("gt_items") or record.get("ground_truth") or []
            items = predicted_items(record, method, recaller_order, max_k=max_k)
            values.append(metric_at_k(items, gt_items, metric))
        out[metric] = float(np.mean(values)) if values else 0.0
    return out


def evaluation_metrics_report(
    records: Sequence[dict],
    recaller_order: Sequence[str],
    selected_baseline_channel: str,
    metrics: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for channel in recaller_order:
        report[f"single:{channel}"] = evaluate_method_metrics(records, f"single:{channel}", recaller_order, metrics)
    report["best_single"] = dict(report.get(f"single:{selected_baseline_channel}", {}))
    report["routepo_top_k"] = evaluate_method_metrics(records, "top_k", recaller_order, metrics)
    report["routepo_weighted_score"] = evaluate_method_metrics(records, "weighted_score", recaller_order, metrics)
    report["routepo_single_select"] = evaluate_method_metrics(records, "single_select", recaller_order, metrics)
    return report


def print_evaluation_metrics(
    dataset: str,
    model_name: str,
    records: Sequence[dict],
    recaller_order: Sequence[str],
    selected_baseline_channel: str,
    selector_metric: str,
    metrics: Sequence[str],
    report: Dict[str, Dict[str, float]],
    source: str,
) -> None:
    print("\n" + "=" * 80)
    print(f"Manual evaluation metrics ({source}): dataset={dataset}, model={model_short_name(model_name)}")
    print(f"Users: {len(records)}")
    print(f"Recaller order: {' '.join(recaller_order)}")
    print(f"Best single by {selector_metric}: {selected_baseline_channel}")
    if not metrics:
        print("No reportable metrics.")
        print("=" * 80)
        return
    name_width = max([len(name) for name in report.keys()] + [8])
    header = f"{'method':<{name_width}} " + " ".join(f"{metric:>12}" for metric in metrics)
    print(header)
    print("-" * len(header))
    for name, values in report.items():
        cells = " ".join(f"{values.get(metric, 0.0):12.6f}" for metric in metrics)
        print(f"{name:<{name_width}} {cells}")
    print("=" * 80)


def select_best_single_channel(records: Sequence[dict], channels: Sequence[str], metric: str) -> Tuple[str, Dict[str, float]]:
    metrics = single_channel_metrics(records, channels, metric)
    if not metrics:
        raise ValueError("No channel metrics were produced for best-single baseline selection.")
    selected = max(channels, key=lambda name: metrics.get(name, float("-inf")))
    return selected, metrics


def batch_iter(items: Sequence[dict], batch_size: int) -> Iterable[List[dict]]:
    step = max(1, int(batch_size))
    for start in range(0, len(items), step):
        yield list(items[start : start + step])


def apply_chat_format(texts: Sequence[str], tokenizer, model_name: str) -> List[str]:
    if "instruct" not in model_name.lower() or not hasattr(tokenizer, "apply_chat_template"):
        return list(texts)
    return [
        tokenizer.apply_chat_template(
            [{"role": "assistant", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for text in texts
    ]


def dtype_from_args(args):
    import torch

    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def resolve_torch_device(args):
    import os

    if not hasattr(args, "_routing_device"):
        args._routing_device = args.device
        args._recbole_device = args.device
        args._device_note = ""
        if args.device and str(args.device).startswith("cuda:") and "CUDA_VISIBLE_DEVICES" not in os.environ:
            physical_id = str(args.device).split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_id
            args._routing_device = "cuda"
            args._recbole_device = args.device
            args._device_note = (
                f"Mapped requested physical {args.device} to local cuda via CUDA_VISIBLE_DEVICES={physical_id}."
            )

    import torch

    device = torch.device(args._routing_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {device}, but CUDA is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device {device}, but only {torch.cuda.device_count()} CUDA device(s) are available."
            )
    return device


def load_label_mapping(data_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    mapping_path = data_dir.parent / "label_mapping.json"
    with mapping_path.open() as f:
        data = json.load(f)
    label2id = {str(k): int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): str(v).lower() for k, v in data["id2label"].items()}
    return label2id, id2label


def example_eval_fields(example: dict, idx: int) -> Tuple[int, List[int], List[int], List[int]]:
    user_id = int(example.get("user_id", idx))
    history = example.get("history") or example.get("eval_hist") or []
    gt_items = example.get("target_items") or example.get("gt_items") or example.get("ground_truth") or []
    full_hist = example.get("full_hist") or list(history) + list(gt_items)
    return user_id, [int(x) for x in history], [int(x) for x in gt_items], [int(x) for x in full_hist]


def infer_num_items(records: Iterable[dict]) -> int:
    max_item_id = 0
    for example in records:
        for field in ("history", "eval_hist", "target_items", "gt_items", "ground_truth", "full_hist"):
            value = example.get(field)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for item_id in values:
                try:
                    max_item_id = max(max_item_id, int(item_id))
                except (TypeError, ValueError):
                    continue
    return max_item_id + 1


def recaller_order_from_mapping(id2label: Dict[int, str], recallers: Dict[str, object]) -> List[str]:
    ordered = [id2label[i].lower() for i in sorted(id2label)]
    if set(ordered) == set(recallers):
        return ordered
    return sorted(recallers)


def generate_predictions(args, dataset: str, model_name: str, eval_k: int) -> PredictionBundle:
    import torch
    from datasets import Dataset
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from GRPO.models.main import initialize_recallers

    checkpoint_dir, test_data_dir = expected_paths(args, dataset, model_name)
    resolved_checkpoint = resolve_model_path(checkpoint_dir)
    max_length = effective_max_length(args, dataset, test_data_dir)
    expected_meta = metadata_template(args, dataset, model_name, resolved_checkpoint, test_data_dir, eval_k, max_length)

    if not test_data_dir.exists() or not (resolved_checkpoint / "config.json").exists():
        raise FileNotFoundError(
            f"Missing artifacts for dataset={dataset}, model={model_name}: "
            f"checkpoint={resolved_checkpoint}, test_data={test_data_dir}"
        )
    device = resolve_torch_device(args)

    label2id, id2label = load_label_mapping(test_data_dir)
    raw_dataset = Dataset.load_from_disk(str(test_data_dir))
    examples = [raw_dataset[i] for i in range(len(raw_dataset))]
    if args.max_users and args.max_users > 0:
        examples = examples[: args.max_users]

    seed = int(getattr(args, "seed", 42))
    num_items = infer_num_items(raw_dataset)
    recallers = initialize_recallers(
        model_names=args.recbole_models,
        dataset_name=dataset,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        seed=seed,
        use_latest_checkpoint=True,
        num_items=num_items,
        device=getattr(args, "_recbole_device", None) or str(device),
    )
    recaller_order = recaller_order_from_mapping(id2label, recallers)

    tokenizer_path = resolved_checkpoint if (resolved_checkpoint / "tokenizer_config.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = args.padding_side

    model = AutoModelForSequenceClassification.from_pretrained(
        str(resolved_checkpoint),
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=dtype_from_args(args),
    )
    model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.to(device)
    model.eval()

    records: List[dict] = []
    with torch.no_grad():
        indexed_examples = list(enumerate(examples))
        for batch in tqdm(list(batch_iter(indexed_examples, args.batch_size)), desc=f"Evaluating {dataset}/{model_short_name(model_name)}"):
            batch_indices = [idx for idx, _ in batch]
            batch_examples = [example for _, example in batch]
            texts = apply_chat_format([example["text"] for example in batch_examples], tokenizer, model_name)
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            weights_batch = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

            for idx, example, weights in zip(batch_indices, batch_examples, weights_batch):
                user_id, eval_hist, gt_items, full_hist = example_eval_fields(example, idx)
                if len(eval_hist) < 5:
                    continue
                recaller_predictions = {}
                for recaller_name in recaller_order:
                    items = recallers[recaller_name].recall(
                        user_id,
                        eval_k,
                        eval_hist,
                        full_hist=full_hist,
                        gt_items=gt_items,
                    )
                    recaller_predictions[recaller_name] = (
                        [[int(item_id), float(score)] for item_id, score in items] if items else []
                    )
                records.append(
                    {
                        "user_id": user_id,
                        "recaller_predictions": recaller_predictions,
                        "gt_items": gt_items,
                        "merge_weights": [float(x) for x in weights.tolist()],
                    }
                )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not records:
        raise RuntimeError(f"No prediction records were generated for dataset={dataset}, model={model_name}.")

    selected, channel_metrics = select_best_single_channel(records, recaller_order, args.baseline_selector_metric)
    report_metrics = metrics_for_report(args, eval_k)
    full_metrics = evaluation_metrics_report(records, recaller_order, selected, report_metrics)
    print_evaluation_metrics(
        dataset,
        model_name,
        records,
        recaller_order,
        selected,
        args.baseline_selector_metric,
        report_metrics,
        full_metrics,
        source="generated",
    )
    cache_file = cache_path(args, dataset, model_name, eval_k)
    metadata = {
        **expected_meta,
        "recaller_order": recaller_order,
        "selected_baseline_channel": selected,
        "single_channel_metrics": channel_metrics,
        "evaluation_metrics": full_metrics,
        "evaluation_metric_names": report_metrics,
        "device": str(device),
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "n_users": len(records),
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w") as f:
        json.dump({"metadata": metadata, "predictions": records}, f)

    return PredictionBundle(
        dataset=dataset,
        model_name=model_name,
        model_label=model_short_name(model_name),
        records=records,
        recaller_order=recaller_order,
        selected_baseline_channel=selected,
        single_channel_metrics=channel_metrics,
        prediction_cache=cache_file,
        checkpoint_path=resolved_checkpoint,
        test_dataset_path=test_data_dir,
        max_length=max_length,
        metadata=metadata,
    )


def load_or_generate_predictions(args, dataset: str, model_name: str, eval_k: int) -> PredictionBundle:
    checkpoint_dir, test_data_dir = expected_paths(args, dataset, model_name)
    resolved_checkpoint = resolve_model_path(checkpoint_dir)
    max_length = effective_max_length(args, dataset, test_data_dir)
    expected_meta = metadata_template(args, dataset, model_name, resolved_checkpoint, test_data_dir, eval_k, max_length)
    cache_file = cache_path(args, dataset, model_name, eval_k)

    if not args.force_eval and cache_file.exists():
        with cache_file.open() as f:
            cached = json.load(f)
        if metadata_matches(cached, expected_meta):
            records = cached.get("predictions") or []
            meta = cached.get("metadata") or {}
            recaller_order = meta.get("recaller_order") or normalize_recaller_names(sorted(args.recbole_models))
            selected = meta.get("selected_baseline_channel")
            channel_metrics = meta.get("single_channel_metrics")
            if not selected or not channel_metrics:
                selected, channel_metrics = select_best_single_channel(records, recaller_order, args.baseline_selector_metric)
            report_metrics = metrics_for_report(args, eval_k)
            full_metrics = meta.get("evaluation_metrics") or evaluation_metrics_report(
                records,
                recaller_order,
                selected,
                report_metrics,
            )
            print_evaluation_metrics(
                dataset,
                model_name,
                records,
                recaller_order,
                selected,
                args.baseline_selector_metric,
                report_metrics,
                full_metrics,
                source="cache",
            )
            return PredictionBundle(
                dataset=dataset,
                model_name=model_name,
                model_label=meta.get("model_label") or model_short_name(model_name),
                records=records,
                recaller_order=recaller_order,
                selected_baseline_channel=selected,
                single_channel_metrics={str(k): float(v) for k, v in channel_metrics.items()},
                prediction_cache=cache_file,
                checkpoint_path=resolved_checkpoint,
                test_dataset_path=test_data_dir,
                max_length=max_length,
                metadata=meta,
            )
        print(f"[cache miss] Metadata mismatch, regenerating {cache_file}")

    return generate_predictions(args, dataset, model_name, eval_k)


def dry_run_paths(args, eval_k: int) -> None:
    print("Dry-run path resolution:")
    for dataset in args.datasets:
        for model_name in args.model_names:
            checkpoint_dir, test_data_dir = expected_paths(args, dataset, model_name)
            resolved_checkpoint = resolve_model_path(checkpoint_dir)
            cache_file = cache_path(args, dataset, model_name, eval_k)
            print(f"dataset={dataset} model={model_name}")
            print(f"  checkpoint: {resolved_checkpoint}")
            print(f"  test data:  {test_data_dir}")
            print(f"  cache:      {cache_file}")
