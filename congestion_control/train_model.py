import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from argparse import ArgumentParser
import numpy as np
import torch as th
import global_constants as GC
from agua.embedding_to_embedding import train_embed_layer
from agua.linear_policy_model import train_linear_policy_model, ConceptPredictor
from global_constants import CCModel


def _embed_paths_and_extractor():
    """Return resources for training the concept mapping function.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        (embed_path: Path, split_state_files_fn: Callable[[], (train, val)],
        extractor_fn: Callable[[Path], Tuple[torch.Tensor, np.ndarray]]) where
        extractor maps a state file path to ``(policy_embedding, text_embedding)``.
    """
    embed_path = GC.STATE_EMBEDDING_SAVE_PATH
    controller = CCModel()
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, weights_only=True))

    def extractor(file):
        """Load npz file and return policy + text embeddings.

        Parameters
        ----------
        file : Path
            Path to a saved ``state_XXXXXX.npz`` file containing a
            ``state`` array.

        Returns
        -------
        tuple
            (policy_state_embedding: torch.Tensor, text_embedding: np.ndarray).
        """
        data = np.load(file)
        with th.no_grad():
            state = th.as_tensor(data["state"], dtype=th.float32).unsqueeze(0)
            state_embedding = controller.features_extractor(state).detach().cpu()
        text_embedding = np.load(embed_path / file.name)["embedding"]
        return state_embedding, text_embedding

    return embed_path, GC.split_state_files, extractor


def _controller_and_files():
    """Return helpers for training the output mapping function.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        (split_state_files_fn, load_test_states_fn, extractor_fn) where the
        extractor maps loaded npz data dict to
        ``(concept_embedding, action_tensor, state_tensor)``.
    """
    controller = CCModel()
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, weights_only=True))
    embed_projection = ConceptPredictor(
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
        embedding_size=GC.EMBEDDING_SIZE,
        n_concepts=GC.N_CONCEPTS,
        bins=GC.BINS,
    )
    embed_projection.load_state_dict(th.load(GC.EMBED_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True))
    embed_projection.eval()

    def extractor(data):
        """Map loaded state dict to (concept_embedding, action, state_tensor).

        Parameters
        ----------
        data : Mapping
            Mapping (e.g., np.load result) with key ``state``.

        Returns
        -------
        tuple
            (concept_embedding: torch.Tensor, action: torch.Tensor, state_tensor: torch.Tensor).
        """
        state = th.as_tensor(data["state"], dtype=th.float32).unsqueeze(0)
        policy_embedding = controller.features_extractor(state).detach().cpu()
        concept_embedding = embed_projection(policy_embedding)
        action = controller(state)
        action = th.as_tensor(action, dtype=th.long)
        return concept_embedding, action, state

    return GC.split_state_files, GC.load_test_states, extractor


def main() -> None:
    """CLI for training mapping functions powering explanations.

    Flags
    -----
    --embedding_to_embedding : concept mapping function (controller features -> (concept,bin)).
    --linear_policy : output mapping function (concept representation -> action logits).
    """
    parser = ArgumentParser()
    parser.add_argument("--embedding_to_embedding", action="store_true")
    parser.add_argument("--linear_policy", action="store_true")
    args = parser.parse_args()

    if args.embedding_to_embedding:
        embed_path, split_files, extractor = _embed_paths_and_extractor()
        train_embed_layer(
            GC.CONCEPT_EMBEDDING_SAVE_PATH,
            embed_path,
            split_files,
            extractor,
            GC.EMBED_PROJECTION_SAVE_PATH,
            GC.POLICY_EMBEDDING_SIZE,
            GC.EMBEDDING_SIZE,
            GC.N_CONCEPTS,
            GC.BINS,
        )
    if args.linear_policy:
        split_files, load_test, extractor = _controller_and_files()
        train_linear_policy_model(
            split_files,
            load_test,
            extractor,
            GC.EMBED_PROJECTION_SAVE_PATH,
            GC.OUTPUT_PROJECTION_SAVE_PATH,
            GC.N_ACTIONS,
            GC.POLICY_EMBEDDING_SIZE,
            GC.EMBEDDING_SIZE,
            GC.N_CONCEPTS,
            GC.BINS,
            GC.TRAINING_LOG_FILE,
        )


if __name__ == "__main__":
    main()
