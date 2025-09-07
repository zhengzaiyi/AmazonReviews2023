GRPO package
=============

This folder contains a refactored version of the original `GRPO.py` script split
into smaller modules for readability and reuse.

Files
-----
- `data.py`: dataset loading and `InteractionData` wrapper. Supports RecBole when
  installed; otherwise falls back to a synthetic dataset for testing.
- `recallers.py`: RecBole-based recaller implementations. All recommendation 
  algorithms (including Pop, ItemKNN/ItemCF, BPR, SASRec, etc.) are unified 
  through the `RecBoleRecaller` class for consistency.
- `agents.py`: profile agent and LLM router helpers (HF local fallback).
- `selector.py`: selector network and GRPO trainer, plus router-only inference.
- `utils.py`: small utilities (seed, recall_at_k, merge_candidates).
- `main.py`: program entrypoint that mirrors the original script CLI.
- `__init__.py`: package init.

Usage
-----
Run the refactored entrypoint (same CLI as original):

    python GRPO.py --dataset ml-100k --data_path ./data --epochs 3

Or run the package main directly:

    python -m GRPO.main --help

Notes
-----
- The top-level `GRPO.py` remains a thin wrapper to keep backward compatibility.
- If you use RecBole models, ensure RecBole is installed in the environment.
- The HF local generator requires `transformers` and a local model; otherwise the
  router uses deterministic fallbacks.
