# Congestion Control Domain

This directory contains the congestion control slice of the multi‑domain explanation framework (state description, embedding generation, trustee / surrogate training).

## Controller Provenance

Controller decisions analyzed here come from a crystalbox-based congestion control policy (reference repository):

https://github.com/sagar-pa/crystalbox

The underlying controller was trained / exported with `is_action_continuous = False` (discrete action setting). Action bins match those used in `train_trustee.py` (`ACTION_NAMES`):
1. 1/2x
2. -33%
3. -10%
4. No change
5. +10%
6. +33%
7. 2x

Follow the crystalbox repository instructions for training / rollouts. This directory only consumes serialized state/action artifacts—no training logic duplicated.

## Expected State Artifacts

Each sample (e.g., NPZ) must include:
- state: shape (HISTORY_LEN=10, F) numeric window (most recent last)
- action: integer index (0–6) aligned with the discrete bins above
- optional identifiers (trace, timestamp) if desired

Files should be placed according to paths resolved in this domain’s `global_constants.py` (e.g., train/test split helpers).

## Core Scripts

- `state_to_text.py`: Produces natural language descriptions from numeric windowed states (uses `FEATURES` list).
- `embed_data.py`: Generates concept / sample embeddings (flags: `--save_concept_embeddings`, `--save_sample_embeddings`, `--filter_saved_concepts`).
- `train_trustee.py`: Trains a decision‑tree-based surrogate (TrustReport) over flattened states (10 × F -> vector).
- `train_model.py`: Domain model training (not required for explanation pipeline ingestion).

All constants: `congestion_control/global_constants.py`. Do not hard‑code duplicates.

## Workflow (Concise)

1. Generate additional rollouts with the discrete action controller (`is_action_continuous=False`) per crystalbox docs.
2. Place (or symlink) resulting state/action NPZ files in the expected data directory (`split_state_files()` governs discovery).
3. Run:
   - `state_to_text.py`
   - `embed_data.py`
   - `train_trustee.py`
4. Repeat step 1–2 to expand dataset; rerun downstream steps as needed.

## Trustee Specifics

`train_trustee.py`:
- Flattens (N, 10, F) -> (N, 10*F).
- Wraps the teacher PyTorch model (`CCModel`) for action prediction.
- Produces full and pruned decision trees (see saved report directory).
- Uses `ACTION_NAMES` for class labels; modifying bin semantics requires regenerating all artifacts.

## Adding More States

Regenerate rollouts with the same discrete action configuration. Ensure feature ordering and normalization remain consistent; otherwise rebuild all prior artifacts for coherence.

## Environment & Credentials

Any external LLM / embedding providers must be configured via environment variables referenced by constants. No secrets are stored here.

## Troubleshooting

Issue: No training files found  
Cause: `split_state_files()` returned empty.  
Fix: Verify NPZ placement and naming.

Issue: Shape error flattening states  
Fix: Confirm each `state` array has leading dimension 10 (HISTORY_LEN).

Issue: Class mismatch  
Fix: Ensure exported actions already mapped to 0–6 indices consistent with bins above.
