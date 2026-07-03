#!/usr/bin/env python3
"""Profile RoutePO routing latency from downloaded pure-GRPO checkpoints."""

import argparse
import csv
import json
import os
import platform
import random
import re
import shlex
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_DIR = "emnlp_recycle_outputs"
DEFAULT_MODEL_NAMES = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]
DEFAULT_RECALLERS = ["ItemKNN", "LightGCN", "Pop"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile RoutePO routing latency for pure classification checkpoints."
    )
    parser.add_argument("--datasets", nargs="+", default=["ml-1m", "steam", "Food"])
    parser.add_argument("--model_names", nargs="+", default=DEFAULT_MODEL_NAMES)
    parser.add_argument("--recbole_models", nargs="+", default=DEFAULT_RECALLERS)
    parser.add_argument("--model_root", default="GRPO/data/pure_models")
    parser.add_argument("--data_path", default="dataset")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile_cutoff", type=int, default=500000)
    parser.add_argument(
        "--manual_history_cutoff",
        type=int,
        default=20,
        help="Temporarily keep only the most recent N history items for latency profiling; <=0 disables.",
    )
    parser.add_argument("--prompt_top_k", type=int, default=3)
    parser.add_argument("--recall_top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_users", type=int, default=500)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--padding_side", default="left", choices=["left", "right"])
    parser.add_argument("--bf16", action="store_true", help="Load model in bfloat16.")
    parser.add_argument("--fp16", action="store_true", help="Load model in float16.")
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, otherwise cpu.")
    parser.add_argument("--skip_recall_latency", action="store_true", help="Only profile routing forward latency.")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument(
        "--dry_run_paths",
        action="store_true",
        help="Only print the checkpoint/data paths that would be used.",
    )
    return parser.parse_args()


def combo_name(recbole_models: Sequence[str]) -> str:
    return "_".join(sorted(recbole_models))


def model_short_name(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def ptk_suffix(prompt_top_k: int) -> str:
    return f"_ptk{prompt_top_k}" if prompt_top_k != 3 else ""


def checkpoint_suffix(profile_cutoff: int, prompt_top_k: int) -> str:
    pc_suffix = f"_pc{profile_cutoff}" if profile_cutoff != 20 else ""
    return f"{pc_suffix}{ptk_suffix(prompt_top_k)}"


def expected_paths(args: argparse.Namespace, dataset: str, model_name: str) -> Tuple[Path, Path]:
    root = (REPO_ROOT / args.model_root).resolve()
    short = model_short_name(model_name)
    combo = combo_name(args.recbole_models)
    model_base = root / dataset / short
    ckpt = model_base.parent / f"{short}_pure_sft_{combo}{checkpoint_suffix(args.profile_cutoff, args.prompt_top_k)}"
    data = model_base.parent / f"{short}_pure_sft_data_{combo}_{args.profile_cutoff}{ptk_suffix(args.prompt_top_k)}" / "test"
    return ckpt, data


def resolve_model_path(path: Path) -> Path:
    if (path / "config.json").exists():
        return path
    last = get_last_checkpoint(str(path)) if path.exists() else None
    if last:
        return Path(last)
    return path


def load_label_mapping(data_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    mapping_path = data_dir.parent / "label_mapping.json"
    with mapping_path.open() as f:
        data = json.load(f)
    label2id = {str(k): int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): str(v) for k, v in data["id2label"].items()}
    return label2id, id2label


def select_examples(dataset, sample_users: int, sample_seed: int) -> List[dict]:
    records = [dataset[i] for i in range(len(dataset))]
    if sample_users <= 0 or sample_users >= len(records):
        return records
    rng = random.Random(sample_seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    chosen = sorted(indices[:sample_users])
    return [records[i] for i in chosen]


def truncate_purchase_history_text(text: str, cutoff: int) -> str:
    if cutoff <= 0 or "purchase history:" not in text:
        return text

    lines = text.splitlines(keepends=True)
    marker_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "purchase history:":
            marker_idx = idx
            break
    if marker_idx is None:
        return text

    block_start = marker_idx + 1
    block_end = block_start
    while block_end < len(lines):
        stripped = lines[block_end].strip()
        if stripped and not lines[block_end].startswith((" ", "\t")):
            break
        block_end += 1

    history_lines = lines[block_start:block_end]
    entries: List[List[str]] = []
    current: List[str] = []
    entry_pattern = re.compile(r"^(\s*)'?\d+'?:\s*$")
    for line in history_lines:
        if entry_pattern.match(line):
            if current:
                entries.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append(current)
    if not entries or len(entries) <= cutoff:
        return text

    kept_entries = entries[-cutoff:]
    rebuilt: List[str] = []
    for new_idx, entry in enumerate(kept_entries, start=1):
        first = entry[0]
        match = entry_pattern.match(first)
        if match:
            newline = "\n" if first.endswith("\n") else ""
            rebuilt.append(f"{match.group(1)}'{new_idx}':{newline}")
            rebuilt.extend(entry[1:])
        else:
            rebuilt.extend(entry)

    prefix = lines[:block_start]
    suffix = lines[block_end:]
    for idx in range(marker_idx - 1, -1, -1):
        if lines[idx].lstrip().startswith("purchased item numbers:"):
            prefix_line_idx = idx
            prefix[prefix_line_idx] = re.sub(
                r"(purchased item numbers:\s*)\d+",
                rf"\g<1>{len(kept_entries)}",
                prefix[prefix_line_idx],
                count=1,
            )
            break
    return "".join(prefix + rebuilt + suffix)


def apply_manual_history_cutoff(example: dict, cutoff: int) -> dict:
    if cutoff <= 0:
        return dict(example)

    out = dict(example)
    history = out.get("history") or out.get("eval_hist") or []
    history = [int(x) for x in history]
    if len(history) > cutoff:
        out["history"] = history[-cutoff:]
        if "eval_hist" in out:
            out["eval_hist"] = history[-cutoff:]
        for field in ("text", "prompt"):
            if isinstance(out.get(field), str):
                out[field] = truncate_purchase_history_text(out[field], cutoff)
        out["history_len_used"] = len(out["history"])
    return out


def batch_iter(items: Sequence[dict], batch_size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


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


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def hardware_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or "cpu"


def dtype_from_args(args: argparse.Namespace) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def setup_cuda_visibility(args: argparse.Namespace) -> None:
    """Treat --device cuda:N as a physical GPU request before CUDA is initialized."""
    args._routing_device = args.device
    args._recbole_device = args.device
    args._device_note = ""
    if not args.device or not args.device.startswith("cuda:"):
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        args._device_note = f"CUDA_VISIBLE_DEVICES already set to {visible}; using requested device as-is."
        return

    physical_id = args.device.split(":", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_id
    args._routing_device = "cuda"
    args._recbole_device = args.device
    args._device_note = f"Mapped requested physical {args.device} to local cuda via CUDA_VISIBLE_DEVICES={physical_id}."


def resolve_device(args: argparse.Namespace) -> torch.device:
    requested = getattr(args, "_routing_device", args.device)
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {device}, but CUDA is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device {device}, but only {torch.cuda.device_count()} CUDA device(s) are available."
            )
    return device


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(mean(values)) if values else 0.0,
        f"{prefix}_p50": percentile(values, 50),
        f"{prefix}_p95": percentile(values, 95),
    }


def example_eval_fields(example: dict, idx: int) -> Tuple[int, List[int], List[int], List[int]]:
    user_id = int(example.get("user_id", idx))
    history = example.get("history") or example.get("eval_hist") or []
    gt_items = example.get("target_items") or example.get("gt_items") or example.get("ground_truth") or []
    full_hist = example.get("full_hist") or list(history) + list(gt_items)
    return user_id, [int(x) for x in history], [int(x) for x in gt_items], [int(x) for x in full_hist]


def sampled_history_lengths(examples: Sequence[dict]) -> List[int]:
    lengths = []
    for idx, example in enumerate(examples):
        _, history, _, _ = example_eval_fields(example, idx)
        lengths.append(len(history))
    return lengths


def recaller_order_from_mapping(id2label: Dict[int, str], recallers: Dict[str, object]) -> List[str]:
    ordered = [id2label[i].lower() for i in sorted(id2label)]
    if set(ordered) == set(recallers):
        return ordered
    return sorted(recallers)


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


def profile_recall_latency(
    args: argparse.Namespace,
    dataset_name: str,
    examples: Sequence[dict],
    id2label: Dict[int, str],
    device: torch.device,
    num_items: int,
) -> Tuple[dict, Optional[dict]]:
    if args.skip_recall_latency:
        return {
            "recall_included": False,
            "recall_users": 0,
            "recall_top_k": args.recall_top_k,
            "recall_channels": "",
            "recall_latency_ms_mean": 0.0,
            "recall_latency_ms_p50": 0.0,
            "recall_latency_ms_p95": 0.0,
            "recall_users_per_sec": 0.0,
        }, None

    from GRPO.models.main import initialize_recallers

    recbole_device = getattr(args, "_recbole_device", args.device) or str(device)
    recallers = initialize_recallers(
        model_names=args.recbole_models,
        dataset_name=dataset_name,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        seed=args.seed,
        use_latest_checkpoint=True,
        num_items=num_items,
        device=recbole_device,
    )
    recaller_order = recaller_order_from_mapping(id2label, recallers)

    warmup_count = min(len(examples), max(0, args.warmup_batches * max(1, args.batch_size)))
    for idx, example in enumerate(examples[:warmup_count]):
        user_id, eval_hist, gt_items, full_hist = example_eval_fields(example, idx)
        if len(eval_hist) < 5:
            continue
        for recaller_name in recaller_order:
            _ = recallers[recaller_name].recall(
                user_id,
                args.recall_top_k,
                eval_hist,
                full_hist=full_hist,
                gt_items=gt_items,
            )
    sync_device(device)

    per_user_total_ms: List[float] = []
    per_channel_ms: Dict[str, List[float]] = {name: [] for name in recaller_order}
    measured_users = 0
    start_total = time.perf_counter()
    for idx, example in enumerate(examples):
        user_id, eval_hist, gt_items, full_hist = example_eval_fields(example, idx)
        if len(eval_hist) < 5:
            continue
        user_elapsed = 0.0
        for recaller_name in recaller_order:
            sync_device(device)
            start = time.perf_counter()
            _ = recallers[recaller_name].recall(
                user_id,
                args.recall_top_k,
                eval_hist,
                full_hist=full_hist,
                gt_items=gt_items,
            )
            sync_device(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            per_channel_ms[recaller_name].append(elapsed_ms)
            user_elapsed += elapsed_ms
        per_user_total_ms.append(user_elapsed)
        measured_users += 1
    total_time = time.perf_counter() - start_total

    stats = {
        "recall_included": True,
        "recall_users": measured_users,
        "recall_top_k": args.recall_top_k,
        "recall_channels": " ".join(recaller_order),
        "recall_users_per_sec": float(measured_users / total_time) if total_time > 0 else 0.0,
        "_recall_total_seconds": total_time,
        "_recall_per_user_ms": per_user_total_ms,
    }
    stats.update(latency_stats(per_user_total_ms, "recall_latency_ms"))
    for recaller_name, values in per_channel_ms.items():
        safe = recaller_name.replace("-", "_")
        stats.update(latency_stats(values, f"recall_{safe}_ms"))

    return stats, recallers


def profile_one(args: argparse.Namespace, dataset_name: str, model_name: str) -> Optional[dict]:
    from datasets import Dataset

    checkpoint_dir, data_dir = expected_paths(args, dataset_name, model_name)
    resolved_model_path = resolve_model_path(checkpoint_dir)
    if not data_dir.exists() or not (resolved_model_path / "config.json").exists():
        message = (
            f"Missing artifacts for dataset={dataset_name}, model={model_name}: "
            f"checkpoint={resolved_model_path}, data={data_dir}"
        )
        if args.skip_missing:
            print(f"[skip] {message}")
            return None
        raise FileNotFoundError(message)

    label2id, id2label = load_label_mapping(data_dir)
    test_dataset = Dataset.load_from_disk(str(data_dir))
    num_items = infer_num_items(test_dataset)
    examples = select_examples(test_dataset, args.sample_users, args.sample_seed)
    source_history_lengths = sampled_history_lengths(examples)
    examples = [apply_manual_history_cutoff(example, args.manual_history_cutoff) for example in examples]
    history_lengths = sampled_history_lengths(examples)
    max_length = args.max_length
    if max_length is None:
        token_stats_path = data_dir.parent / "token_stats.json"
        if token_stats_path.exists():
            with token_stats_path.open() as f:
                max_length = int(json.load(f).get("max_tokens_cls", 1536))
        else:
            max_length = 11024 if dataset_name == "Food" else 1536

    device = resolve_device(args)
    recall_stats, recallers = profile_recall_latency(args, dataset_name, examples, id2label, device, num_items)

    tokenizer_path = resolved_model_path if (resolved_model_path / "tokenizer_config.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = args.padding_side

    model = AutoModelForSequenceClassification.from_pretrained(
        str(resolved_model_path),
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=dtype_from_args(args),
    )
    model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.to(device)
    model.eval()

    batches = list(batch_iter(examples, max(1, args.batch_size)))
    warmup_batches = batches[: max(0, min(args.warmup_batches, len(batches)))]
    measured_batches = batches

    with torch.no_grad():
        for batch in warmup_batches:
            texts = apply_chat_format([ex["text"] for ex in batch], tokenizer, model_name)
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        per_user_ms: List[float] = []
        prompt_tokens: List[int] = []
        total_users = 0
        start_total = time.perf_counter()
        for batch in measured_batches:
            texts = apply_chat_format([ex["text"] for ex in batch], tokenizer, model_name)
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            attention = inputs.get("attention_mask")
            if attention is not None:
                prompt_tokens.extend([int(x) for x in attention.sum(dim=1).tolist()])
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            batch_users = len(batch)
            per_user_ms.extend([elapsed_ms / batch_users] * batch_users)
            total_users += batch_users
        total_time = time.perf_counter() - start_total

    peak_mem_gb = 0.0
    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    del model
    del recallers
    if device.type == "cuda":
        torch.cuda.empty_cache()

    recall_per_user_ms = recall_stats.pop("_recall_per_user_ms", [])
    recall_total_seconds = float(recall_stats.pop("_recall_total_seconds", 0.0))
    if recall_per_user_ms and len(recall_per_user_ms) == len(per_user_ms):
        end_to_end_ms = [r + fwd for r, fwd in zip(recall_per_user_ms, per_user_ms)]
    elif recall_per_user_ms:
        end_to_end_ms = [float(mean(recall_per_user_ms)) + fwd for fwd in per_user_ms]
    else:
        end_to_end_ms = list(per_user_ms)
    end_to_end_total_time = total_time + recall_total_seconds

    row = {
        "model": model_short_name(model_name),
        "model_name": model_name,
        "dataset": dataset_name,
        "sample_users": total_users,
        "batch_size": args.batch_size,
        "profile_cutoff": args.profile_cutoff,
        "manual_history_cutoff": args.manual_history_cutoff,
        "source_history_len_mean": float(mean(source_history_lengths)) if source_history_lengths else 0.0,
        "source_history_len_p50": percentile(source_history_lengths, 50),
        "source_history_len_p95": percentile(source_history_lengths, 95),
        "source_history_len_max": int(max(source_history_lengths)) if source_history_lengths else 0,
        "history_len_mean": float(mean(history_lengths)) if history_lengths else 0.0,
        "history_len_p50": percentile(history_lengths, 50),
        "history_len_p95": percentile(history_lengths, 95),
        "history_len_max": int(max(history_lengths)) if history_lengths else 0,
        "prompt_tokens_avg": float(mean(prompt_tokens)) if prompt_tokens else 0.0,
        "output_tokens_avg": 0.0,
        "latency_ms_mean": float(mean(per_user_ms)) if per_user_ms else 0.0,
        "latency_ms_p50": percentile(per_user_ms, 50),
        "latency_ms_p95": percentile(per_user_ms, 95),
        "users_per_sec": float(total_users / total_time) if total_time > 0 else 0.0,
        "end_to_end_users_per_sec": float(total_users / end_to_end_total_time) if end_to_end_total_time > 0 else 0.0,
        "peak_mem_gb": peak_mem_gb,
        "hardware": hardware_name(device),
        "requested_device": args.device or "",
        "resolved_device": str(device),
        "device_note": getattr(args, "_device_note", ""),
        "decoding": "classification_forward_no_generation",
        "candidate_lists": (
            f"online_recall_top{args.recall_top_k}_profiled; prompts_from_saved_test_dataset_manual_history_cutoff_{args.manual_history_cutoff}"
            if not args.skip_recall_latency
            else f"precomputed_in_test_dataset_manual_history_cutoff_{args.manual_history_cutoff}"
        ),
        "checkpoint_path": str(resolved_model_path),
        "test_dataset_path": str(data_dir),
        "data_path": args.data_path,
        "checkpoint_dir": args.checkpoint_dir,
        "sample_seed": args.sample_seed,
        "seed": args.seed,
        "max_length": max_length,
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
    }
    row.update(recall_stats)
    row.update(latency_stats(end_to_end_ms, "end_to_end_latency_ms"))
    return {
        key: value for key, value in row.items() if not key.startswith("_")
    }


def write_csv(rows: Sequence[dict], path: Path) -> None:
    base_fieldnames = [
        "model",
        "model_name",
        "dataset",
        "sample_users",
        "batch_size",
        "profile_cutoff",
        "manual_history_cutoff",
        "source_history_len_mean",
        "source_history_len_p50",
        "source_history_len_p95",
        "source_history_len_max",
        "history_len_mean",
        "history_len_p50",
        "history_len_p95",
        "history_len_max",
        "prompt_tokens_avg",
        "output_tokens_avg",
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p95",
        "users_per_sec",
        "recall_included",
        "recall_users",
        "recall_top_k",
        "recall_channels",
        "recall_latency_ms_mean",
        "recall_latency_ms_p50",
        "recall_latency_ms_p95",
        "recall_users_per_sec",
        "end_to_end_latency_ms_mean",
        "end_to_end_latency_ms_p50",
        "end_to_end_latency_ms_p95",
        "end_to_end_users_per_sec",
        "peak_mem_gb",
        "hardware",
        "requested_device",
        "resolved_device",
        "device_note",
        "decoding",
        "candidate_lists",
        "checkpoint_path",
        "test_dataset_path",
        "data_path",
        "checkpoint_dir",
        "sample_seed",
        "seed",
        "max_length",
        "date",
        "host",
    ]
    extras = sorted({key for row in rows for key in row if key not in base_fieldnames})
    fieldnames = base_fieldnames + extras
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: Sequence[dict], path: Path, command: str) -> None:
    lines = [
        "# Latency Profile Summary",
        "",
        f"- Date: {datetime.now().isoformat(timespec='seconds')}",
        f"- Host: {socket.gethostname()}",
        f"- Command: `{command}`",
        f"- Output CSV: `{path.with_name('latency_profile.csv')}`",
        "- Routing: classification forward pass only; no token generation.",
        "- Recall: online recaller calls are timed separately; recaller/model initialization is excluded.",
        "- Prompts: loaded from the saved test dataset; purchase history is manually truncated before tokenization.",
        "- Prompt construction time is not included.",
        "",
        "## LaTeX Table",
        "",
        "| Model | Dataset | Hist max | Prompt tok. | Recall p50 | Recall p95 | Route p50 | Route p95 | E2E p50 | E2E p95 | E2E Users/s | Peak Mem. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['history_len_max']} | {row['prompt_tokens_avg']:.1f} | "
            f"{row['recall_latency_ms_p50']:.1f} | {row['recall_latency_ms_p95']:.1f} | "
            f"{row['latency_ms_p50']:.1f} | {row['latency_ms_p95']:.1f} | "
            f"{row['end_to_end_latency_ms_p50']:.1f} | {row['end_to_end_latency_ms_p95']:.1f} | "
            f"{row['end_to_end_users_per_sec']:.2f} | "
            f"{row['peak_mem_gb']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['model']} / {row['dataset']}",
                "",
                f"- Checkpoint: `{row['checkpoint_path']}`",
                f"- Test dataset: `{row['test_dataset_path']}`",
                f"- Hardware: {row['hardware']}",
                f"- Requested device: {row.get('requested_device') or 'auto'}",
                f"- Resolved device: {row.get('resolved_device') or 'auto'}",
                f"- Batch size: {row['batch_size']}",
                f"- Source artifact profile cutoff: {row['profile_cutoff']}",
                f"- Manual history cutoff: {row['manual_history_cutoff']}",
                f"- Source history length mean/p50/p95/max: "
                f"{row['source_history_len_mean']:.1f}/{row['source_history_len_p50']:.1f}/"
                f"{row['source_history_len_p95']:.1f}/{row['source_history_len_max']}",
                f"- Profiled history length mean/p50/p95/max: "
                f"{row['history_len_mean']:.1f}/{row['history_len_p50']:.1f}/"
                f"{row['history_len_p95']:.1f}/{row['history_len_max']}",
                f"- Sample users: {row['sample_users']}",
                f"- Sample seed: {row['sample_seed']}",
                f"- Recall included: {row['recall_included']}",
                f"- Recall users: {row['recall_users']}",
                f"- Recall top-k: {row['recall_top_k']}",
                f"- Recall channels: {row['recall_channels']}",
                f"- Max length: {row['max_length']}",
                "",
            ]
        )
        if row.get("device_note"):
            lines.insert(-1, f"- Device note: {row['device_note']}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    setup_cuda_visibility(args)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run_paths:
        print("Dry-run path resolution:")
        for dataset in args.datasets:
            for model_name in args.model_names:
                ckpt, data = expected_paths(args, dataset, model_name)
                print(f"dataset={dataset} model={model_name}")
                print(f"  checkpoint: {resolve_model_path(ckpt)}")
                print(f"  test data:  {data}")
                print(f"  recbole checkpoints: {args.checkpoint_dir}/{dataset}")
                print(f"  recbole data:        {args.data_path}")
                print(f"  manual history cutoff: {args.manual_history_cutoff}")
        return

    rows = []
    for dataset in args.datasets:
        for model_name in args.model_names:
            row = profile_one(args, dataset, model_name)
            if row is not None:
                rows.append(row)

    if not rows:
        raise RuntimeError("No latency rows were produced.")

    csv_path = output_dir / "latency_profile.csv"
    summary_path = output_dir / "latency_profile_summary.md"
    command = " ".join(shlex.quote(x) for x in sys.argv)
    write_csv(rows, csv_path)
    write_summary(rows, summary_path, command)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

# Default usage: python GRPO/scripts/rebuttal/latency_profile.py --datasets ml-1m steam Food --model_names meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen3-4B-Instruct-2507 --recbole_models ItemKNN LightGCN Pop --profile_cutoff 500000 --manual_history_cutoff 20 --batch_size 1 --sample_users 500 --recall_top_k 50
