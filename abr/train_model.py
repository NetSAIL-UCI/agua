import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from argparse import ArgumentParser
import numpy as np
import torch as th
import global_constants as GC
from agua.embedding_to_embedding import train_embed_layer
from agua.linear_policy_model import train_linear_policy_model, ConceptPredictor
from global_constants import ABRModel


def _embed_paths_and_extractor():
    """Return paths and extractor for embedding-to-embedding training.

    Returns
    -------
    tuple
        ``(embed_path, split_state_files_fn, extractor_fn)`` where:
        * embed_path : Path to saved text embeddings for states.
        * split_state_files_fn : callable returning (train_files, val_files).
        * extractor_fn : function mapping a state file path ->
          ``(policy_state_embedding, text_embedding)``.

    Notes
    -----
    The policy embedding is produced by the lightweight controller's
    feature extractor; the text embedding is loaded from disk.
    """
    embed_path = GC.STATE_EMBEDDING_SAVE_PATH
    controller = ABRModel()
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, weights_only=True))

    def extractor(file):
        """Load state npz file and return (policy_embedding, text_embedding).

        Parameters
        ----------
        file : Path
            Path to ``state_XXXXXXX.npz`` containing a ``state`` array.

        Returns
        -------
        tuple
            ``(policy_state_embedding, text_embedding)`` as torch / numpy tensors.
        """
        data = np.load(file)
        with th.no_grad():
            state = th.as_tensor(data["state"], dtype=th.float32, device="cpu").unsqueeze(0)
            state_embedding = controller.features_extractor(state).detach().cpu()
        text_embedding = np.load(embed_path / file.name)["embedding"]
        return state_embedding, text_embedding

    return embed_path, GC.split_state_files, extractor


def _controller_and_files():
    """Return helpers required for linear policy training.

    Returns
    -------
    tuple
        ``(split_state_files_fn, load_test_states_fn, extractor_fn)`` where
        the extractor maps a loaded npz dict to ``(concept_embedding, action, state)``.

    Notes
    -----
    The concept embedding is produced by applying the learned embedding
    projector to the controller features.
    """
    controller = ABRModel()
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, weights_only=True))
    embed_projection = ConceptPredictor(
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
        embedding_size=GC.EMBEDDING_SIZE,
        n_concepts=GC.N_CONCEPTS,
        bins=GC.BINS
    )
    embed_projection.load_state_dict(th.load(GC.EMBED_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True))
    embed_projection.eval()

    def extractor(data):
        """Map raw loaded state dict -> (concept_embedding, action, state_tensor).

        Parameters
        ----------
        data : dict-like
            Loaded npz mapping with at least key ``state``.

        Returns
        -------
        tuple
            ``(concept_embedding, action_tensor, state_tensor)`` all torch tensors.
        """
        state = th.as_tensor(data["state"], dtype=th.float32).unsqueeze(0)
        embedding = controller.features_extractor(state).detach().cpu()
        embedding = embed_projection(embedding)
        action  = controller(state)
        action = th.as_tensor(action, dtype=th.long)
        return embedding, action, state

    return GC.split_state_files, GC.load_test_states, extractor


def main() -> None:
    """CLI for training mapping functions that power explanations.

    Flags
    -----
    --embedding_to_embedding : Train concept mapping function (controller features -> (concept,bin) distribution).
    --linear_policy : Train output mapping function (concept representation -> action logits).

    Returns
    -------
    None
        Saves learned weights under paths defined in ``global_constants``.
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
