# Lucid DDoS Domain

This directory contains the DDoS detection slice of the multi‑domain explanation framework (state/sample description, embedding generation, surrogate trustee training).

## Model / Controller Provenance

The classification model analyzed here is derived from the original Lucid DDoS project:

https://github.com/doriguzzi/lucid-ddos

The original TensorFlow 2 implementation was ported to PyTorch for consistency with the other domains (ABR, congestion control). We only rely on exported feature tensors + labels; training or pipeline scripts from the original repository are not re‑implemented here.

A copy of the upstream README for reference exists locally as `lucid_readme.md`. It is informational only and NOT required to follow for experiments in this framework.

## Expected Data Artifacts

Each sample artifact (e.g., NPZ / parquet converted upstream) should provide at minimum:
- features: numeric vector (post any preprocessing consistent with the ported PyTorch model)
- label: integer or categorical label (attack / benign classes as mapped in code)
- optional metadata: flow / capture identifiers (ignored by core scripts unless explicitly used)

Locations and splits are resolved via this domain’s `global_constants.py`. Do not hard‑code paths elsewhere.

## Core Scripts (Parallel Pattern)

- `state_to_text.py`: Produces natural language descriptions for individual feature vectors (re-uses shared prompt/LLM constants).
- `embed_data.py`: Generates concept + sample embeddings (`--save_concept_embeddings`, `--save_sample_embeddings`, `--filter_saved_concepts` flags).
- `train_trustee.py`: Trains a surrogate (decision tree via TrustReport) to approximate the PyTorch Lucid classifier for interpretability.
- `train_model.py`: Trains / fine‑tunes the ported PyTorch Lucid model (not required for using pre‑generated data).

All configurable names, paths, model identifiers: `lucid_ddos/global_constants.py`.

## Workflow (Concise)

1. Prepare or obtain feature/label artifacts compatible with the ported PyTorch Lucid model (follow original repo only for semantics, not for tooling).
2. Place (or symlink) artifacts where `global_constants.py` expects them.
3. Run:
   - `state_to_text.py` (skips already described samples)
   - `embed_data.py`
   - `train_trustee.py`
4. Repeat step 1–2 to expand dataset; rerun downstream steps as needed.

## Trustee Notes

`train_trustee.py` flattens / directly consumes feature vectors; class names and feature ordering must remain stable. Changes to preprocessing require regenerating all prior descriptions and embeddings for consistency.

## Environment & Credentials

External LLM / embedding services controlled via environment variables (see constants). No secrets are stored in this repository.

## Troubleshooting (Brief)

Issue: No samples loaded  
Fix: Confirm data files discovered by split helpers in `global_constants.py`.

Issue: Feature mismatch / shape error  
Fix: Ensure preprocessing matches the PyTorch port (feature order identical to original Lucid pipeline).

Issue: LLM rate limits  
Fix: Lower batch size or concurrency constant (`N_QUERY_TOGETHER`) before retry.