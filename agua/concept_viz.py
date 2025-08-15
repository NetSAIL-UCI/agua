import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
from typing import Sequence, List, Tuple, Iterable

BINS_NAMES = {
    0: " is absent",
    1: "",
    2: "",
}


def load_concepts(concepts_file) -> List[Tuple[int, str]]:
    """Load concept ids and descriptions from a text file.

    The expected format per line is ``<id>.<free form description>``. Lines
    whose prefix cannot be parsed as an integer raise a ``ValueError``.

    Parameters
    ----------
    concepts_file : PathLike or str
        Path to concepts list file.

    Returns
    -------
    List[Tuple[int, str]]
        Sequence of (concept_id, description) pairs in file order.
    """
    concepts = []
    with open(concepts_file, "r") as f:
        for line in f:
            concept_id, *rest = line.strip().split(".")
            try:
                concept_id = int(concept_id)
            except ValueError as exc:
                raise ValueError(
                    "Concept id is not valid. Is the format of the concepts correct?"
                ) from exc
            feature = ".".join(rest)
            concepts.append([concept_id, feature])
    return concepts

def get_concept_weights(projector: nn.Linear, embedding_samples: th.Tensor,
                        class_idx: int = None, normalize: bool = True) -> th.Tensor:
    """Compute (concept, bin) weight scores for supplied embeddings.

    The linear layer ``projector`` maps flattened concept/bin activations to
    action logits. For an (optional) selected ``class_idx`` we recover a
    per-(concept, bin) contribution by element-wise multiplying the class
    weights with the input embedding (and optionally normalizing via softmax
    and scaling by the class probability).

    Parameters
    ----------
    projector : nn.Linear
        Trained output projection layer of shape (n_actions, n_concepts * n_bins).
    embedding_samples : torch.Tensor
        Tensor of shape (B, n_concepts * n_bins) or (n_concepts * n_bins,).
    class_idx : int, optional
        Target action index. If omitted, the argmax per-sample is used.
    normalize : bool, default=True
        Whether to apply softmax to weights and scale by class probability.

    Returns
    -------
    torch.Tensor
        Tensor shaped (B, n_concepts * n_bins) of (possibly normalized) weights.
    """
    if embedding_samples.ndim < 2:
        embedding_samples = embedding_samples.unsqueeze(0)
    scores = F.softmax(projector(embedding_samples), dim=1)
    if class_idx is None:
        class_idx = th.argmax(scores, dim=1)
    bias_compensation = projector.bias[class_idx] / projector.weight.size(1)
    bias_compensation = bias_compensation.unsqueeze(-1)
    concept_weights = projector.weight[class_idx] * embedding_samples
    concept_weights += bias_compensation
    if normalize:
        concept_weights = F.softmax(concept_weights, dim=-1)
        concept_weights *= scores[th.arange(scores.shape[0]), class_idx].unsqueeze(1)
    return concept_weights


def create_plotting_data(concept_weights: th.Tensor, concepts,
                         bins: Sequence[float], top_concepts_to_consider: int = 3) -> pd.DataFrame:
    """Convert weight tensor into a tidy DataFrame for plotting.

    Parameters
    ----------
    concept_weights : torch.Tensor
        Concept weight tensor of shape (B, n_concepts * n_bins) or flattened.
    concepts : Sequence[Tuple[int, str]]
        Output of :func:`load_concepts` providing (id, description).
    bins : Sequence[float]
        Bin thresholds used (length defines n_bins).
    top_concepts_to_consider : int, default=3
        Number of highest-magnitude concept/bin entries to include.

    Returns
    -------
    pandas.DataFrame
        Columns: ``Concept`` (str) and ``Weight`` (float) for top selected entries.
    """
    data = []
    if concept_weights.ndim == 1:
        concept_weights = concept_weights.unsqueeze(0)
    concept_weights = th.mean(concept_weights, dim=0)
    sorted_concepts = th.abs(concept_weights).argsort(descending=True)
    for i in range(top_concepts_to_consider):
        flat_idx = sorted_concepts[i].item()
        bin_idx, concept_idx = np.unravel_index(flat_idx, shape=(len(bins), len(concepts)))
        concept = concepts[concept_idx]
        bin_name = BINS_NAMES.get(bin_idx, "")
        concept_id, concept_desc = concept
        concept_name, *desc = concept_desc.split(":")
        data.append([f"{concept_name}{bin_name}", concept_weights[flat_idx].item()])
    data = pd.DataFrame(data, columns=["Concept", "Weight"])
    return data
