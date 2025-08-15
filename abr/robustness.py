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
from state_to_text import load_dataset, get_llm_description

N_QUERIES_PER_SAMPLE = 5

class StateNoise:
    """Add bounded uniform noise to states."""

    MAX_NOISE = 0.07

    def __init__(self, train_states: np.ndarray, seed: int = 16) -> None:
        """Pre-compute noise bounds from training state statistics.

        Parameters
        ----------
        train_states : np.ndarray
            Array of training states used to derive per-feature std.
        seed : int, optional
            RNG seed for reproducible noise sampling (default 16).
        """
        self.std_state = np.std(train_states, axis=0)
        self.min_noise = -1 * self.MAX_NOISE * self.std_state
        self.max_noise = self.MAX_NOISE * self.std_state
        self.rand = np.random.default_rng(seed=seed)

    def add_noise(self, state: np.ndarray) -> np.ndarray:
        """Add uniform noise within pre-computed per-feature bounds.

        Parameters
        ----------
        state : np.ndarray
            Single state array to perturb (same shape as train samples).

        Returns
        -------
        np.ndarray
            Noisy state array (same shape as input).
        """
        noise = self.rand.uniform(low=self.min_noise, high=self.max_noise)
        return state + noise


def run_explainer_robustness() -> Dict[str, float]:
    """Evaluate robustness of concept attributions to input noise.

    Returns
    -------
    dict
        Mapping of robustness metric names to float scores aggregated
        over ``N_QUERIES_PER_SAMPLE`` repeated noisy perturbations.
    """
    train_data, _ = load_dataset(n_samples=GC.MAX_NUM_STATES)
    states, _, _ = train_data
    noise = StateNoise(states)

    # Load lightweight controller
    model = GC.ABRModel()
    model.load_state_dict(th.load(GC.CONTROLLER_PATH, map_location="cpu", weights_only=True))
    model.eval()
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
        """Return (concept_embedding, greedy_action) for a raw state."""
        # raw_state expected shape matches saved state
        state_tensor = th.as_tensor(raw_state, dtype=th.float32).unsqueeze(0)
        with th.no_grad():
            embedding = model.features_extractor(state_tensor)
            embedding = learned_projector(embedding)
            action = model(state_tensor)
        return embedding, int(action.item())

    def concept_weight_fn(embedding: np.ndarray, action: int) -> np.ndarray:
        """Compute per-concept weight vector for a given action.

        Parameters
        ----------
        embedding : np.ndarray
            Concept-bin embedding for one or more samples.
        action : int
            Target action index for which weights are extracted.
        """
        weights = get_concept_weights(
            projector=final_projector, embedding_samples=embedding, class_idx=action
        )
        return weights.detach().flatten().numpy()

    return evaluate_explainer_robustness(
        states=states,
        predict_fn=predict_fn,
        concept_weight_fn=concept_weight_fn,
        noise_fn=noise.add_noise,
        n_queries=N_QUERIES_PER_SAMPLE,
    )


def run_multiple_requests() -> Dict[str, float]:
    """Evaluate robustness of descriptions across repeated LLM queries.

    For each state, sends multiple identical prompts to the LLM and
    measures variability in downstream embeddings / similarity scores.

    Returns
    -------
    dict
        Aggregated variability metrics for description generation.
    """
    train_data, _ = load_dataset(n_samples=GC.MAX_NUM_STATES)
    states, _, _ = train_data

    client = OpenAI()

    def describe_fn(state: np.ndarray) -> str:
        """Generate an LLM description for a single state."""
        return get_llm_description(state=state, client=client)

    def embed_fn(text: str) -> np.ndarray:
        """Embed textual description using configured document model."""
        return get_embedding(text=text, client=client, model=GC.DOC_EMBEDDING_MODEL)

    return evaluate_description_robustness(
        states=states,
        concept_embedding_dir=GC.CONCEPT_EMBEDDING_SAVE_PATH,
        describe_fn=describe_fn,
        embed_fn=embed_fn,
        n_queries=N_QUERIES_PER_SAMPLE,
    )


def run_input_robustness() -> Dict[str, float]:
    """Evaluate description robustness under noisy input perturbations.

    Adds bounded noise to states before requesting descriptions and
    compares embedding / concept similarity statistics.

    Returns
    -------
    dict
        Mapping of robustness metric names to aggregated float values.
    """
    train_data, _ = load_dataset(n_samples=GC.MAX_NUM_STATES)
    states, _, _ = train_data
    noise = StateNoise(states)

    client = OpenAI()

    def describe_fn(state: np.ndarray) -> str:
        """Generate an LLM description for a single (possibly noisy) state."""
        return get_llm_description(state=state, client=client)

    def embed_fn(text: str) -> np.ndarray:
        """Embed textual description using configured document model."""
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
    """Run all robustness experiments and persist results to disk."""
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
