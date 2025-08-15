# Adaptive Bitrate (ABR) Domain

This directory contains the ABR (adaptive bitrate streaming) slice of the multi‑domain explanation framework. It converts controller rollouts (states + actions) into natural language descriptions, embeddings, and surrogate (trustee) model explanations.

## Data / Controller Provenance

All raw ABR states and actions used here were generated with the `gelato_random` controller in the plume repository (module `puffer_abr`):

https://github.com/sagar-pa/plume/tree/main/puffer_abr

Follow the plume repository instructions (do not replicate them here) to:
1. Evaluate `gelato_random` on one or more network traces.
2. Produce per‑timestep serialized state + chosen action artifacts.

We only consume the exported artifacts; no plume training code is duplicated.

## Expected Artifact Format

Each saved sample (NPZ recommended) must provide at least:
- state: numeric tensor representation used by the controller
- action: selected bitrate index
- trace_idx (or similar identifier)

Name files consistently (e.g., `state_XXXXXXXX.npz`) and place them in the directory defined by `STATE_SAVE_PATH` in `abr/global_constants.py`.

## Core Scripts

- `state_to_text.py`: Converts numeric states into natural language descriptions using the LLM specified by `LLM_MODEL`.
- `embed_data.py`: Generates concept and/or sample embeddings (flags: `--save_concept_embeddings`, `--save_sample_embeddings`, `--filter_saved_concepts`).
- `train_trustee.py`: Trains a lightweight surrogate to approximate `gelato_random` behavior for explanation and robustness studies.
- `train_model.py`: (Domain model training; not required for state ingestion.)

All paths, model names, and tuning constants come from `abr/global_constants.py`. Do not hard‑code duplicates elsewhere.

## Typical Workflow (High Level)

1. Use plume (`gelato_random`) to generate additional rollout artifacts (follow plume docs).
2. Drop or symlink resulting NPZ files into `STATE_SAVE_PATH`.
3. Run `state_to_text.py` to create textual descriptions (skips existing).
4. Run `embed_data.py` for embeddings.
5. Run `train_trustee.py` to fit / refresh the surrogate.

Repeat steps 1–3 to expand the dataset; downstream steps can be re‑run as needed.

## Adding More States

Simply regenerate rollouts in plume with `gelato_random` over new traces (again, follow plume’s own instructions for flags / invocation) and place the outputs as above. No additional installation or conversion guidance is duplicated here.

## Environment & Credentials

External LLM / embedding backends are configured via environment variables (see references in code). Set required API keys before invoking scripts; none are stored in this repository.

## Notes

- Shared reusable logic (embedding, visualization scaffolding) lives in `abr/agua/`.
- Keep any refactoring of duplicated domain logic pointed toward shared utilities instead of copying code.
- Avoid modifying plume‑sourced semantics (state ordering, normalization) unless you regenerate all artifacts consistently.