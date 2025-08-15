import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # enable agua import
import argparse
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import global_constants as GC
from typing import List
from agua.concept_viz import get_concept_weights, create_plotting_data, load_concepts
from torch import nn
from agua.linear_policy_model import ConceptPredictor
from global_constants import CCModel  # domain-specific model


def load_embeddings(indices: List[int]) -> th.Tensor:
    """Load concept-projected embeddings for given state indices.

    Parameters
    ----------
    indices : list of int
        State indices whose embeddings should be loaded and projected.

    Returns
    -------
    torch.Tensor
        Concept-bin embedding tensor of shape ``(N, N_CONCEPTS * len(BINS))``.
    """
    controller = CCModel()
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, map_location="cpu", weights_only=True))
    controller.eval()
    embeddings = []
    embedding_projector = ConceptPredictor(
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
        embedding_size=GC.EMBEDDING_SIZE,
        n_concepts=GC.N_CONCEPTS,
        bins=GC.BINS,
    )
    embedding_projector.load_state_dict(
        th.load(GC.EMBED_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    )
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
    """CLI entry for generating concept weight bar plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, nargs="+", required=True, help="One or more state indices")
    parser.add_argument("--class", dest="class_idx", type=int, required=False, default=None)
    args = parser.parse_args()
    indices = args.idx
    concept_embedding = load_embeddings(indices)
    final_projector = nn.Linear(GC.N_CONCEPTS * len(GC.BINS), GC.N_ACTIONS)
    fp_state = th.load(GC.OUTPUT_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True)
    fp_state = {"weight": fp_state["weight"], "bias": fp_state["bias"]}
    final_projector.load_state_dict(fp_state)
    concepts = load_concepts(GC.CONCEPTS_FILE)
    weights = get_concept_weights(
        final_projector, concept_embedding, class_idx=args.class_idx
    )
    df = create_plotting_data(weights, concepts, GC.BINS)
    ax = df.plot(kind="barh", x="Concept", y="Weight")
    plt.tight_layout()
    idx_tag = "-".join(f"{i}" for i in indices)
    plt.savefig(f"explanation_{idx_tag}_{args.class_idx}.png")
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
