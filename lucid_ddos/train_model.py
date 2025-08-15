import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from argparse import ArgumentParser
import numpy as np
import torch as th
import global_constants as GC
from agua.embedding_to_embedding import train_embed_layer
from agua.linear_policy_model import train_linear_policy_model, ConceptPredictor


def _embed_paths_and_extractor():
    """Return embedding path, split function, and raw->(policy,text) extractor.

    Returns
    -------
    Tuple[Path, Callable[[], Tuple[List[Path], List[Path]]], Callable[[Path], Tuple[torch.Tensor, np.ndarray]]]
        (embedding_dir, file_split_fn, extractor) where extractor loads an
        input file and returns (policy_embedding_tensor, text_embedding_array).
    """
    embed_path = GC.INPUT_EMBEDDING_SAVE_PATH
    controller = GC.LucidModel()
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, weights_only=True))
    controller = controller.eval()

    def extractor(file):
        """Load one saved input file -> (policy_embedding, text_embedding)."""
        data = np.load(file)
        with th.no_grad():
            inp = th.as_tensor(data["input_sample"], dtype=th.float32).unsqueeze(0)
            inp_chw = inp.permute(0, 3, 1, 2)
            input_embedding = controller.feature_extractor(inp_chw).detach().cpu()
        text_embedding = np.load(embed_path / file.name)["embedding"]
        return input_embedding, text_embedding

    return embed_path, GC.split_input_files, extractor


def _controller_and_files():
    """Prepare controller + projection for training linear policy layer.

    Returns
    -------
    Tuple[Callable, Callable, Callable]
        (split_files_fn, load_test_fn, extractor_fn) where ``extractor_fn`` maps
        a loaded npz dict to (concept_embedding, discrete_label, raw_input_tensor).
    """
    controller = GC.LucidModel()
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, weights_only=True))
    controller.eval()
    embed_projection = ConceptPredictor(
        policy_embedding_size=GC.POLICY_EMBEDDING_SIZE,
        embedding_size=GC.EMBEDDING_SIZE,
        n_concepts=GC.N_CONCEPTS,
        bins=GC.BINS,
    )
    embed_projection.load_state_dict(th.load(GC.EMBED_PROJECTION_SAVE_PATH, map_location="cpu", weights_only=True))
    embed_projection.eval()

    def extractor(data):
        """Convert one loaded sample dict to (concept_embed, label, raw_input)."""
        inp = th.as_tensor(
            data["input_sample"], dtype=th.float32, device="cpu"
        ).unsqueeze(0)
        chw = inp.permute(0, 3, 1, 2)
        policy_embedding = controller.feature_extractor(chw).detach().cpu()
        concept_embedding = embed_projection(policy_embedding)
        output = (controller(inp) > 0.5).long().reshape((1,))
        return concept_embedding, output, inp

    return GC.split_input_files, GC.load_test_inputs, extractor


def main() -> None:
    """CLI for training Lucid mapping functions.

    Flags
    -----
    --embedding_to_embedding : train concept mapping function (controller features -> (concept,bin)).
    --linear_policy : train output mapping function (concept representation -> action logits).
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
