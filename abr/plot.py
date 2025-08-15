import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import global_constants as GC
from typing import List
from agua.concept_viz import get_concept_weights, create_plotting_data, load_concepts
from global_constants import ABRModel
from torch import nn
from agua.linear_policy_model import ConceptPredictor


def load_embeddings(indices: List[int]) -> th.Tensor:
    """Load concept-projected controller embeddings for given indices.

    Parameters
    ----------
    indices : list of int
        State indices (matching filename numbering) whose embeddings
        should be loaded and projected.

    Returns
    -------
    torch.Tensor
        Projected concept-bin embedding tensor of shape
        ``(N, N_CONCEPTS * len(BINS))`` suitable for weighting.

    Notes
    -----
    Loads raw states from disk, extracts controller features, then
    applies the learned embedding projector.
    """
    controller = ABRModel()
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, weights_only=True))  # load trained weights
    controller.eval()
    embeddings = []
    embedding_projector = ConceptPredictor(
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
        embedding_size=GC.EMBEDDING_SIZE,
        n_concepts=GC.N_CONCEPTS,
        bins=GC.BINS
    )
    embedding_projector.load_state_dict(th.load(GC.EMBED_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True))
    embedding_projector.eval()
    for idx in indices:
        file = GC.TEST_STATE_SAVE_PATH / f"state_{idx:07d}.npz"
        data = np.load(file)
        with th.no_grad():
            state = th.as_tensor(data["state"], dtype=th.float32).unsqueeze(0)
            embedding = controller.features_extractor(state).detach().cpu()
        embeddings.append(embedding)
    embeddings = th.cat(embeddings, dim=0)
    return embedding_projector(embeddings)


def main() -> None:
    """Generate concept-based explanation (bar plot) for given samples.

    Leverages the learned concept mapping function (`embed_projection.pt`)
    and output mapping function (`final_projection.pt`) to compute per-concept
    contribution weights toward the (optionally specified) action class.
    Saves a horizontal bar chart:
    `explanation_<idxs>_<class>.png`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, nargs="+", required=True, help="One or more state indices")
    parser.add_argument("--class", dest="class_idx", type=int, required=False, default= None)
    args = parser.parse_args()
    indices = args.idx
    concept_embedding = load_embeddings(indices)
    final_projector = nn.Linear(GC.N_CONCEPTS * len(GC.BINS), GC.N_ACTIONS)
    final_projector_state_dict = th.load(GC.OUTPUT_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    final_projector_state_dict = {"weight": final_projector_state_dict["weight"], "bias": final_projector_state_dict["bias"]}
    final_projector.load_state_dict(final_projector_state_dict)
    concepts = load_concepts(GC.CONCEPT_NAME_FILE)
    weights = get_concept_weights(final_projector, concept_embedding, class_idx=args.class_idx)
    df = create_plotting_data(weights, concepts, GC.BINS)
    ax = df.plot(kind="barh", x="Concept", y="Weight")
    plt.tight_layout()
    idx_tag = "-".join(f"{i}" for i in indices)
    plt.savefig(f"explanation_{idx_tag}_{args.class_idx}.png")
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
