from typing import Dict

import numpy as np
import torch as th
from torch import nn
from openai import OpenAI

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # enable agua import

from agua.linear_policy_model import ConceptPredictor
from agua.concept_viz import get_concept_weights
from agua.robustness import (
    evaluate_explainer_robustness,
    evaluate_description_robustness,
)
from agua.embed_data import get_embedding

import global_constants as GC
from input_to_text import load_dataset, get_llm_description

N_QUERIES_PER_SAMPLE = 5
MAX_NUM_STATES = GC.MAX_NUM_INPUTS


class StateNoise:
    """Add bounded uniform noise to inputs using per-dimension std scaling.

    The noise range for each feature dimension is ``[-MAX_NOISE*std, +MAX_NOISE*std]``
    where ``std`` is computed over the provided training states.

    Attributes
    ----------
    MAX_NOISE : float
        Global multiplicative factor controlling maximum relative noise.
    std_state : np.ndarray
        Per-dimension standard deviation of the reference (training) states.
    min_noise : np.ndarray
        Per-dimension minimum noise value.
    max_noise : np.ndarray
        Per-dimension maximum noise value.
    rand : np.random.Generator
        Random number generator used for sampling noise.
    """

    MAX_NOISE = 0.07

    def __init__(self, train_states: np.ndarray, seed: int = 16) -> None:
        """Construct the noise helper.

        Parameters
        ----------
        train_states : np.ndarray
            Array of reference training states with shape (N, *feature_shape).
        seed : int, default=16
            Seed for deterministic noise sampling.
        """
        self.std_state = np.std(train_states, axis=0)
        self.min_noise = -1 * self.MAX_NOISE * self.std_state
        self.max_noise = self.MAX_NOISE * self.std_state
        self.rand = np.random.default_rng(seed=seed)

    def add_noise(self, state: np.ndarray) -> np.ndarray:
        """Return a noisy variant of ``state``.

        Parameters
        ----------
        state : np.ndarray
            Single input/state to perturb (shape matches training samples).

        Returns
        -------
        np.ndarray
            Noisy state with element-wise uniform perturbations applied.
        """
        noise = self.rand.uniform(low=self.min_noise, high=self.max_noise)
        return state + noise


def run_explainer_robustness() -> Dict[str, float]:
    """Evaluate robustness of concept weights to input noise.

    For each test input we:
    1. Compute concept embeddings & predicted action.
    2. Sample ``n_queries`` noisy variants, re-evaluating concept weights.
    3. Aggregate similarity statistics via ``evaluate_explainer_robustness``.

    Returns
    -------
    Dict[str, float]
        Mapping of metric name to aggregated statistic (e.g., cosine
        similarities, standard deviations).
    """
    train_inputs, train_outputs, test_inputs, test_outputs = load_dataset(
        n_samples=MAX_NUM_STATES
    )
    noise = StateNoise(train_inputs)

    model = GC.LucidModel()
    model.load_state_dict(th.load(GC.CONTROLLER_PATH, map_location="cpu", weights_only=True))
    model = model.eval()
    learned_projector = ConceptPredictor(GC.POLICY_EMBEDDING_SIZE, 
                                         GC.EMBEDDING_SIZE, GC.N_CONCEPTS, GC.BINS)
    learned_projector.load_state_dict(
    th.load(GC.EMBED_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    )

    final_projector = nn.Linear(GC.N_CONCEPTS * len(GC.BINS), GC.N_ACTIONS)
    loading_data = th.load(GC.OUTPUT_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    final_projector.load_state_dict(
        {"weight": loading_data["weight"], "bias": loading_data["bias"]}
    )

    def predict_fn(raw_state: np.ndarray):
        """Project a raw state to (concept_embedding, label).

        Parameters
        ----------
        raw_state : np.ndarray
            Raw test input sample of shape (H, W, C) or similar.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Concept embedding (1, n_concepts*len(bins)) and predicted label
            tensor shaped (1,).
        """
        input_sample = th.as_tensor(raw_state, dtype=th.float32)
        if input_sample.ndim == 3:
            input_sample = input_sample.unsqueeze(0)
        inverted = input_sample.permute(0, 3, 1, 2)
        embedding = model.feature_extractor(inverted)
        embedding = embedding.detach().cpu()
        embedding = learned_projector(embedding)
        output = (model(input_sample) > 0.5).long().reshape((1,))
        return embedding, output

    def concept_weight_fn(embedding: np.ndarray, _action: int) -> np.ndarray:
        """Compute concept weights for a given action.

        Parameters
        ----------
        embedding : np.ndarray or torch.Tensor
            Concept embedding tensor produced by ``predict_fn``.
        _action : int
            Action index whose weight profile we extract.

        Returns
        -------
        np.ndarray
            Flattened array of concept weights (length = n_concepts * len(bins)).
        """
        weights = get_concept_weights(
            projector=final_projector, embedding_samples=embedding,
            class_idx=_action
        )
        return weights.detach().flatten().numpy()

    return evaluate_explainer_robustness(
        states=test_inputs,
        predict_fn=predict_fn,
        concept_weight_fn=concept_weight_fn,
        noise_fn=noise.add_noise,
        n_queries=N_QUERIES_PER_SAMPLE,
    )


def run_multiple_requests() -> Dict[str, float]:
    """Evaluate variability across repeated LLM description & embedding calls.

    For each state we query the LLM ``n_queries`` times, embed each text, and
    compute dispersion metrics on the resulting embedding set.

    Returns
    -------
    Dict[str, float]
        Robustness metrics summarizing intra-state embedding variability.
    """
    train_inputs, _, test_inputs, _ = load_dataset(n_samples=MAX_NUM_STATES)
    states = test_inputs

    client = OpenAI()

    def describe_fn(state: np.ndarray) -> str:
        """Return a natural language description for ``state`` via LLM."""
        return get_llm_description(state=state, client=client)

    def embed_fn(text: str) -> np.ndarray:
        """Return embedding vector for provided description text."""
        return get_embedding(text=text, client=client, model=GC.DOC_EMBEDDING_MODEL)

    return evaluate_description_robustness(
        states=states,
        concept_embedding_dir=GC.CONCEPT_EMBEDDING_SAVE_PATH,
        describe_fn=describe_fn,
        embed_fn=embed_fn,
        n_queries=N_QUERIES_PER_SAMPLE,
    )


def run_input_robustness() -> Dict[str, float]:
    """Evaluate robustness of descriptions/embeddings to input perturbations.

    Noise is applied before generating each description. Differences capture
    how sensitive the LLM-generated explanations (and embeddings) are to small
    input changes.

    Returns
    -------
    Dict[str, float]
        Aggregated robustness statistics (e.g., mean cosine similarity).
    """
    train_inputs, _, test_inputs, _ = load_dataset(n_samples=MAX_NUM_STATES)
    noise = StateNoise(train_inputs)
    states = test_inputs

    client = OpenAI()

    def describe_fn(state: np.ndarray) -> str:
        """Return a natural language description for ``state`` via LLM."""
        return get_llm_description(state=state, client=client)

    def embed_fn(text: str) -> np.ndarray:
        """Return embedding vector for provided description text."""
        return get_embedding(text=text, client=client, model=GC.DOC_EMBEDDING_MODEL)

    return evaluate_description_robustness(
        states=states,
        concept_embedding_dir=GC.CONCEPT_EMBEDDING_SAVE_PATH,
        describe_fn=describe_fn,
        embed_fn=embed_fn,
        n_queries=N_QUERIES_PER_SAMPLE,
        noise_fn=noise.add_noise,
    )


def main() -> None:
    """Execute all robustness experiments and write a report.

    Creates (or overwrites) the robustness report specified by
    ``GC.ROBUSTNESS_FILE`` containing metrics for:
    - explainer_robustness
    - multiple_requests
    - input_robustness
    """
    results = {
        "explainer_robustness": run_explainer_robustness(),
        "multiple_requests": run_multiple_requests(),
        "input_robustness": run_input_robustness(),
    }
    with open(GC.ROBUSTNESS_FILE, "w", encoding="utf-8") as f:
        for exp_name, res in results.items():
            f.write(f"{exp_name}\n")
            for key, val in res.items():
                f.write(f"{key}: {val}\n")
            f.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main()
