# Agua: Concept-Based Explanations for Learning-Enabled Systems (SIGCOMM'25)

This is public repository of Agua, a framework for generating human-understandable, concept-based explanations of learned-enabled systems controllers. It supports three application domains:

1. Adaptive Bitrate (ABR) video streaming (`abr/`)
2. Congestion control (`congestion_control/`)
3. DDoS detection (Lucid dataset) (`lucid_ddos/`)


---
## Repository Structure

Top-level layout with purpose of key files (mirrored across domains where noted):

```
requirements.txt        # Python dependencies (install before use)
README.md               # This document
abr/                    # Adaptive Bitrate (video streaming) domain
congestion_control/     # Congestion control domain
lucid_ddos/             # DDoS detection domain
agua/                   # Shared reusable utilities (embedding, viz, robustness)
```

### Shared Utilities (`agua/`)
```
agua/__init__.py
agua/embed_data.py          # Generic embedding helpers (concept + sample)
agua/embedding_to_embedding.py  # Learns concept mapping function: controller features -> (concept,bin) distribution
agua/linear_policy_model.py # Learns output mapping function: (concept,bin) features -> action logits
agua/concept_viz.py         # Compute per-concept weights for explanations & format plots
agua/robustness.py          # Core robustness evaluation functions
```

### Domain Folder Pattern (illustrated with `abr/`)
```
abr/global_constants.py     # Domain-specific constants & paths (env-var load)
abr/state_to_text.py        # (abr & congestion_control) LLM state description generation
abr/input_to_text.py        # (lucid_ddos) Equivalent for packet/flow inputs
abr/embed_data.py           # CLI: save_concept_embeddings / save_sample_embeddings / filter
abr/train_model.py          # Train projection layers (--embedding_to_embedding / --linear_policy)
abr/train_trustee.py        # Train and report surrogate decision tree
abr/plot.py                 # Generate explanation plot for specific sample indices
abr/robustness.py           # Run robustness experiments (explainer & description variability)
abr/data/                   # Artifacts + initial zipped data (model, concepts, states, etc.)
	abr_model.pt              # Pretrained controller / domain model
	auto_concepts.txt|concepts.txt    # Concept list (auto or manual)
	concept_names.txt (if present)    # Optional human-readable names
	states/                   # Saved training states (*.npz)
	test_states/              # Saved test states (*.npz)
	state_descriptions/       # LLM text descriptions (delete to regenerate)
	state_embeddings/         # Per-sample description embeddings
	auto_concept_embeddings/|concept_embeddings/  # Per-concept embeddings
	embed_projection.pt       # Learned policy->text embedding projection
	final_projection.pt       # Final linear policy over concept bins
	train_log.txt             # Training log for linear policy stage
	trustee_report/           # Trustee artifacts (trees, metrics)
	robustness.txt            # Robustness experiment outputs
```

`congestion_control/` mirrors the ABR structure (uses `state_to_text.py`).

`lucid_ddos/` differences:
```
lucid_ddos/input_to_text.py     # Replaces state_to_text.py for network inputs
lucid_ddos/util_functions.py    # Domain-specific parsing helpers
lucid_ddos/lucid_cnn.py         # Domain model architecture
lucid_ddos/sample-dataset/      # Example raw data (if provided)
lucid_ddos/output/              # Domain-specific outputs (plots, etc.)
```

Environment variables (API keys) are read via `python-dotenv` from a root `.env` file (you create this—see setup below).

---
## Quick Start (Using Prepackaged Data)

1. Clone repo & create a Python 3.11 Conda environment (recommended):

Install Anaconda (https://www.anaconda.com/download/success) or Miniconda if you have not already. Then run:
```bash
conda create -n agua python=3.11 -y
conda activate agua
pip install -r requirements.txt
```
Optional (alternative without Conda): create a virtualenv with Python 3.11 manually: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

2. Unzip all bundled `data.zip` archives (each domain directory contains one). Run at repo root:
```bash
find . -maxdepth 2 -type f -name data.zip -execdir unzip -o data.zip \;
```
3. Generate an explanation plot (example for ABR, to generate the motivating scenario, use state indices 1617):
```bash
cd abr
python plot.py --idx 1617
```
An image `explanation_1617_None.png` will appear in the working directory (the trailing class specifier is `None` if no `--class` provided).

You can repeat in other domains (`congestion_control/`, `lucid_ddos/`) after unzipping their data.

---
## Training a New Explainer From Scratch

Follow these steps if you want to regenerate descriptions, embeddings, and projection layers (e.g., after changing concepts, model, or using your own data). Steps shown for `abr/`; adjust filenames for other domains.


### 1. Obtain an OpenAI API Key
Create or log into your OpenAI account, generate a key here:
https://platform.openai.com/api-keys

Create a `.env` file at the repository root:
```
OPENAI_API_KEY=sk-...yourkey...
```
The scripts load it automatically (see `load_dotenv` call in each domain's `global_constants.py`). Never commit your key.

### 2. Delete Included Text & Embeddings
#### **Deleting this data to regenerate it requires ~US$100-150 in API credits. Skip this step to test code with included data**

Remove the following directories to force regeneration:
* ABR / Congestion: `data/state_descriptions`, `data/state_embeddings`
* Lucid DDoS: `data/input_descriptions`, `data/input_embeddings`
* `data/embed_projection.pt` `data/final_projection.pt`

Also remove `auto_concept_embeddings/` (or `concept_embeddings/`) if you want to recompute concept embeddings.

### 3. Regenerate Natural Language Descriptions
Run the appropriate script (uses OpenAI Chat Completions):
```bash
cd abr
python state_to_text.py
```
For Lucid DDoS:
```bash
cd lucid_ddos
python input_to_text.py
```
This populates the description directory with one text file per state/input sample.

### 4. Embed Concepts & Sample Descriptions
From the domain directory:
```bash
python embed_data.py --save_concept_embeddings
python embed_data.py --save_sample_embeddings
```
Optionally filter near-duplicate concepts (threshold set in `global_constants.py`):
```bash
python embed_data.py --filter_saved_concepts
```

### 5. Train Mapping Functions (Explainer Stages)
Two sequential stages:
1. Concept mapping function ("embedding_to_embedding"): aligns controller feature embeddings with concept similarity bins.
2. Output mapping function ("linear_policy"): maps aggregated (concept, bin) probabilities to action logits.
```bash
python train_model.py --embedding_to_embedding
python train_model.py --linear_policy
```
Artifacts produced:
* `embed_projection.pt` – concept mapping function weights (controller features -> (concept,bin) probs)
* `final_projection.pt` – output mapping function weights (concept representation -> action logits)
* `train_log.txt` – training log

### 6. Visualize Concept-Based Explanations
`plot.py` loads both mapping functions, recomputes concept activations for the chosen samples, aggregates per-concept contribution weights, and renders a horizontal bar chart.

Generate a plot for one or more state indices (indices correspond to zero-based numbering in the `states/` directory filenames):
```bash
python plot.py --idx 0 1 2
```
Optionally focus on a specific action class predicted or of interest:
```bash
python plot.py --idx 42 --class 3
```

---
## Trustee (Surrogate) Model & Robustness Experiments

### Train Trustee
The trustee is a decision tree surrogate approximating the original controller while providing a simpler, rule-based explanation.
```bash
python train_trustee.py
```
Outputs go to `data/trustee_report/` and a log file (e.g. `trustee_train_log.txt`). The report includes pruned and full-depth trees plus performance metrics (classification report).

Trustee is provided through the [TrusteeML library](https://trusteeml.github.io/index.html), introduced in the paper linked [in their documentation](https://trusteeml.github.io/index.html#citing-us). Follow that page for installation directions; they may include installing Graphviz (e.g. `sudo apt-get install graphviz`) to render trust reports. If you encounter issues with Trustee, please open an issue on their GitHub project: https://github.com/TrusteeML/trustee.

### Run Robustness Experiments
Robustness evaluates stability of the LLM and Agua's data:
#### **This experiment uses ~US$10-20 in API credits**
```bash
python robustness.py
```
Results are written to `data/robustness.txt` summarizing:
* Explainer robustness to input noise
* Variability across repeated identical LLM queries
* Description robustness under noisy inputs

---
## Domain-Specific Notes

| Aspect | ABR / Congestion Control | Lucid DDoS |
|--------|--------------------------|------------|
| Description script | `state_to_text.py` | `input_to_text.py` |
| Description dirs | `state_descriptions/` | `input_descriptions/` |
| Embedding dirs | `state_embeddings/` | `input_embeddings/` |
| Concept embedding dir | `auto_concept_embeddings/` or `concept_embeddings/` | `auto_concept_embeddings/` |

All other training & plotting commands are parallel.

---
## Common Troubleshooting
* Missing states: ensure `states/` and `test_states/` were unzipped. If empty, you need to generate raw state collection (not provided here) before description generation.
* API errors / rate limits: scripts retry lightly; you may need to lower `N_QUERY_TOGETHER` in `global_constants.py` if you exceed rate limits.
* Zero concepts detected: confirm the concept embedding directory exists and contains per-concept embedding files after running `embed_data.py --save_concept_embeddings`.
* Plot shows unexpected class counts: verify `N_ACTIONS` in `global_constants.py` matches the controller model.

---
## Extending the Framework
1. Add a new domain folder following existing pattern (copy minimal scripts, adjust constants).
2. Implement domain-specific `*_to_text.py` serializer to turn raw state tensors into structured textual prompts.
3. Update `global_constants.py` with paths & model specifics.
4. Reuse shared utilities from `agua/` for embedding, training, plotting, robustness.

---
## License

See domain-specific licenses where provided (e.g. `lucid_ddos/LICENSE`) and the root `LICENSE` file. External datasets and models are subject to their original licenses.

---
## Citation
If you use Agua in academic work, please add an appropriate citation (placeholder):


---
## Acknowledgments
This framework builds on open-source tooling for LLM embeddings, PyTorch modeling, and decision tree surrogate analysis. It does not provide any gaurantees on the explanations or code generated.
