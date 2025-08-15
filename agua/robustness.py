from pathlib import Path
from typing import Callable, Sequence, Dict, List

import numpy as np


def compute_robustness(
    reference_top: np.ndarray, query_tops: Sequence[np.ndarray], top_k: int = 5
) -> Dict[str, float]:
    """Return recall-style robustness statistics for ranked concepts.

    Parameters
    ----------
    reference_top : np.ndarray
        Array shape (n_samples, n_concepts) of reference ranked indices
        (descending importance). Only the first ``top_k`` positions are used.
    query_tops : Sequence[np.ndarray]
        Each element shape (n_samples, n_concepts) giving ranked indices for
        one noisy / repeated query run.
    top_k : int, default=5
        Number of leading concepts considered relevant.

    Returns
    -------
    Dict[str, float]
        Keys:
        - "Top voted concepts recalled": fraction of (sample, concept, query)
          pairs where a reference top-k concept appears in the query top-k.
        - "Concepts missed": complementary fraction.
    """
    votes = {"Top voted concepts recalled": 0, "Concepts missed": 0}
    ref = reference_top[:, :top_k]
    for state_idx in range(ref.shape[0]):
        ref_concepts = ref[state_idx]
        for query_top in query_tops:
            query = query_top[state_idx, :top_k]
            for concept in ref_concepts:
                if concept in query:
                    votes["Top voted concepts recalled"] += 1
                else:
                    votes["Concepts missed"] += 1
    total = votes["Top voted concepts recalled"] + votes["Concepts missed"]
    if total == 0:
        return {key: 0.0 for key in votes}
    return {key: val / total for key, val in votes.items()}


def evaluate_explainer_robustness(
    states: np.ndarray,
    predict_fn: Callable[[np.ndarray], tuple[np.ndarray, int]],
    concept_weight_fn: Callable[[np.ndarray, int], np.ndarray],
    noise_fn: Callable[[np.ndarray], np.ndarray],
    n_queries: int,
    top_k: int = 5,
) -> Dict[str, float]:
    """Evaluate stability of concept rankings under input perturbations.

    For each state a reference ranking is computed, then ``n_queries`` noisy
    variants are sampled via ``noise_fn`` and re-ranked. Rankings are
    summarized with :func:`compute_robustness`.

    Parameters
    ----------
    states : np.ndarray
        Array of input states (N, ...).
    predict_fn : Callable[[np.ndarray], Tuple[np.ndarray, int]]
        Maps a raw state to (concept_embedding, predicted_action).
    concept_weight_fn : Callable[[np.ndarray, int], np.ndarray]
        Produces raw concept weight vector from (embedding, action).
    noise_fn : Callable[[np.ndarray], np.ndarray]
        Perturbs a single state producing a noisy variant.
    n_queries : int
        Number of noisy evaluations per state.
    top_k : int, default=5
        Number of leading concepts considered in robustness stats.

    Returns
    -------
    Dict[str, float]
        Robustness statistics from :func:`compute_robustness`.
    """
    target_explanations: List[np.ndarray] = []
    for state in states:
        embedding, action = predict_fn(state)
        weights = concept_weight_fn(embedding, action)
        target_explanations.append(weights)

    noisy_explanations: List[List[np.ndarray]] = [list() for _ in range(n_queries)]
    for query_idx in range(n_queries):
        for state in states:
            noisy_state = noise_fn(state)
            embedding, action = predict_fn(noisy_state)
            weights = concept_weight_fn(embedding, action)
            noisy_explanations[query_idx].append(weights)

    reference = np.argsort(-np.array(target_explanations), axis=1)
    queries = [np.argsort(-np.array(exp), axis=1) for exp in noisy_explanations]
    return compute_robustness(reference, queries, top_k=top_k)


def evaluate_embedding_robustness(
    concept_embedding_dir: Path,
    text_embedding_dir: Path,
    n_samples: int,
    n_queries_per_sample: int,
    top_k: int = 5,
) -> Dict[str, float]:
    """Evaluate robustness using pre-computed text & concept embeddings.

    Parameters
    ----------
    concept_embedding_dir : Path
        Directory containing concept embedding ``*.npz`` files.
    text_embedding_dir : Path
        Directory containing sequential text embedding ``*.npz`` files
        (multiple per state, ordered/grouped by query count).
    n_samples : int
        Maximum number of states to include.
    n_queries_per_sample : int
        Number of repeated text embeddings (queries) per state.
    top_k : int, default=5
        Number of leading concepts considered.

    Returns
    -------
    Dict[str, float]
        Robustness statistics from :func:`compute_robustness`.
    """
    concept_files = sorted(concept_embedding_dir.glob("*.npz"))
    concept_embeddings = [np.load(f)["embedding"] for f in concept_files]
    concept_embeddings = np.array(concept_embeddings)

    embedding_files = sorted(text_embedding_dir.glob("*.npz"))
    if not embedding_files:
        raise ValueError(f"No embeddings found in {text_embedding_dir}")
    text_embeddings = [np.load(f)["embedding"] for f in embedding_files]
    text_embeddings = np.array(text_embeddings)

    # Determine number of states from available embeddings
    n_states = min(n_samples, text_embeddings.shape[0] // n_queries_per_sample)
    text_embeddings = text_embeddings[: n_states * n_queries_per_sample]

    concept_norm = np.linalg.norm(concept_embeddings, axis=1, keepdims=True)
    text_norm = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    sim = (text_embeddings @ concept_embeddings.T) / (text_norm * concept_norm.T)

    sim_per_state = sim.reshape(
        (n_states, n_queries_per_sample, concept_embeddings.shape[0])
    )
    sim_across_queries = [sim_per_state[:, i, :] for i in range(n_queries_per_sample)]
    top_n_concepts = [np.argsort(-sim_data, axis=1) for sim_data in sim_across_queries]

    vote_matrix = np.zeros((n_states, concept_embeddings.shape[0]), dtype=np.float32)
    for state_idx in range(n_states):
        for concept_idx in range(top_k):
            for query_top in top_n_concepts:
                vote_matrix[state_idx, query_top[state_idx, concept_idx]] += 1
    voted_top = np.argsort(-vote_matrix, axis=1)

    return compute_robustness(voted_top, top_n_concepts, top_k=top_k)


def evaluate_description_robustness(
    states: np.ndarray,
    concept_embedding_dir: Path,
    describe_fn: Callable[[np.ndarray], str],
    embed_fn: Callable[[str], np.ndarray],
    n_queries: int,
    top_k: int = 5,
    noise_fn: Callable[[np.ndarray], np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate robustness of LLM-generated descriptions & embeddings.

    For each input state the description is generated ``n_queries`` times
    (optionally after perturbation by ``noise_fn``). Each description is
    embedded and compared (cosine) to concept embeddings; top-k indices per
    query are aggregated by majority vote and the distribution summarized via
    :func:`compute_robustness`.

    Parameters
    ----------
    states : np.ndarray
        Array of input states (N, ...).
    concept_embedding_dir : Path
        Directory containing concept embedding ``*.npz`` files.
    describe_fn : Callable[[np.ndarray], str]
        Produces a textual description for one state.
    embed_fn : Callable[[str], np.ndarray]
        Maps description text to embedding vector.
    n_queries : int
        Number of description generations per state.
    top_k : int, default=5
        Number of leading concepts considered.
    noise_fn : Callable[[np.ndarray], np.ndarray], optional
        Perturbation function applied before description generation.

    Returns
    -------
    Dict[str, float]
        Robustness statistics from :func:`compute_robustness`.
    """

    concept_files = sorted(concept_embedding_dir.glob("*.npz"))
    concept_embeddings = [np.load(f)["embedding"] for f in concept_files]
    concept_embeddings = np.array(concept_embeddings)

    query_embeddings: List[np.ndarray] = []
    for _ in range(n_queries):
        embeds = []
        for state in states:
            perturbed = noise_fn(state) if noise_fn is not None else state
            desc = describe_fn(perturbed)
            embeds.append(embed_fn(desc))
        query_embeddings.append(np.vstack(embeds))

    concept_norm = np.linalg.norm(concept_embeddings, axis=1, keepdims=True)
    top_n_concepts: List[np.ndarray] = []
    for embeds in query_embeddings:
        text_norm = np.linalg.norm(embeds, axis=1, keepdims=True)
        sim = (embeds @ concept_embeddings.T) / (text_norm * concept_norm.T)
        top_n_concepts.append(np.argsort(-sim, axis=1))

    vote_matrix = np.zeros(
        (states.shape[0], concept_embeddings.shape[0]), dtype=np.float32
    )
    for state_idx in range(states.shape[0]):
        for concept_idx in range(top_k):
            for query_top in top_n_concepts:
                vote_matrix[state_idx, query_top[state_idx, concept_idx]] += 1
    voted_top = np.argsort(-vote_matrix, axis=1)

    return compute_robustness(voted_top, top_n_concepts, top_k=top_k)


def save_results(results: Dict[str, float], file_path: Path) -> None:
    """Append key/value robustness metrics to a text file.

    Parameters
    ----------
    results : Dict[str, float]
        Mapping of metric name to scalar value.
    file_path : Path
        Destination text file (created if absent, appended otherwise).
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        for key, val in results.items():
            f.write(f"{key}: {val}\n")
