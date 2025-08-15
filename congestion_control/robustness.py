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
    """Add bounded uniform noise to states for robustness experiments."""
    MAX_NOISE = 0.07

    def __init__(self, train_states: np.ndarray, seed: int = 16) -> None:
        """Initialize noise bounds from training state statistics."""
        self.std_state = np.std(train_states, axis=0)
        self.min_noise = -1 * self.MAX_NOISE * self.std_state
        self.max_noise = self.MAX_NOISE * self.std_state
        self.rand = np.random.default_rng(seed=seed)

    def add_noise(self, state: np.ndarray) -> np.ndarray:
        """Return a noisy copy of ``state`` with per-feature uniform noise."""
        noise = self.rand.uniform(low=self.min_noise, high=self.max_noise)
        return state + noise


def run_explainer_robustness() -> Dict[str, float]:
    """Measure concept weight stability under input perturbations.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Mapping of robustness metric names to aggregated float scores.
    """
    states, actions, _, _ = load_dataset(n_samples=GC.MAX_NUM_STATES)
    noise = StateNoise(states)

    model = GC.CCModel()
    model.load_state_dict(th.load(GC.CONTROLLER_PATH, map_location="cpu", weights_only=True))
    model.eval()
    learned_projector = ConceptPredictor(GC.POLICY_EMBEDDING_SIZE, GC.EMBEDDING_SIZE, GC.N_CONCEPTS, GC.BINS)
    learned_projector.load_state_dict(
    th.load(GC.EMBED_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    )

    final_projector = nn.Linear(GC.N_CONCEPTS * len(GC.BINS), GC.N_ACTIONS)
    loading_data = th.load(GC.OUTPUT_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    final_projector.load_state_dict(
        {"weight": loading_data["weight"], "bias": loading_data["bias"]}
    )

    def predict_fn(raw_state: np.ndarray):
        """Return (embedding, greedy_action) for a raw state."""
        state_tensor = th.as_tensor(raw_state, dtype=th.float32).unsqueeze(0)
        with th.no_grad():
            embedding = model.features_extractor(state_tensor)
            embedding = learned_projector(embedding)
            action = model(state_tensor)
        return embedding, int(action.item())

    def concept_weight_fn(embedding: np.ndarray, action: int) -> np.ndarray:
        """Return flattened concept weight vector for given action."""
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
    """Assess description variability across repeated identical prompts.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Metrics quantifying variability across repeated descriptions.
    """
    states, *_ = load_dataset(n_samples=GC.MAX_NUM_STATES)

    client = OpenAI()

    def describe_fn(state: np.ndarray) -> str:
        """Generate an LLM description for a state."""
        return get_llm_description(state=state, client=client)

    def embed_fn(text: str) -> np.ndarray:
        """Embed textual description using the configured model."""
        return get_embedding(text=text, client=client, model=GC.DOC_EMBEDDING_MODEL)

    return evaluate_description_robustness(
        states=states,
        concept_embedding_dir=GC.CONCEPT_EMBEDDING_SAVE_PATH,
        describe_fn=describe_fn,
        embed_fn=embed_fn,
        n_queries=N_QUERIES_PER_SAMPLE,
    )


def run_input_robustness() -> Dict[str, float]:
    """Assess description robustness to noisy input perturbations.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Mapping of robustness metric names to float scores under noise.
    """
    states, *_ = load_dataset(n_samples=GC.MAX_NUM_STATES)
    noise = StateNoise(states)

    client = OpenAI()

    def describe_fn(state: np.ndarray) -> str:
        """Generate a description for (possibly noisy) state."""
        return get_llm_description(state=state, client=client)

    def embed_fn(text: str) -> np.ndarray:
        """Embed textual description using the configured model."""
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
    """Run all robustness experiments and write aggregated metrics to file.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Side-effects: writes metrics to ROBUSTNESS_FILE.
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
