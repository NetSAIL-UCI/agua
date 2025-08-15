import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from argparse import ArgumentParser
import global_constants as GC
from agua import embed_data as agua_embed


def _description_paths():
    """Return (description_dir, embedding_dir) for sample descriptions.

    Lucid uses input-centric naming (``INPUT_DESCRIPTION_SAVE_PATH``), but we
    fall back to any state-based paths if present to stay consistent with
    other domains.

    Returns
    -------
    Tuple[pathlib.Path, pathlib.Path]
        Directory for raw text descriptions and directory for their
        corresponding saved embeddings.
    """
    if hasattr(GC, "STATE_DESCRIPTION_SAVE_PATH"):
        return GC.STATE_DESCRIPTION_SAVE_PATH, GC.STATE_EMBEDDING_SAVE_PATH
    return GC.INPUT_DESCRIPTION_SAVE_PATH, GC.INPUT_EMBEDDING_SAVE_PATH


def main() -> None:
    """CLI entry point for generating / filtering Lucid embeddings.

    Flags
    -----
    --save_concept_embeddings : bool
        Generate embeddings for each concept in ``GC.CONCEPTS_FILE``.
    --save_sample_embeddings : bool
        Generate embeddings for each saved input description file.
    --filter_saved_concepts : bool
        Apply similarity-based filtering to prune near-duplicate concepts.
    """
    parser = ArgumentParser()
    parser.add_argument("--save_concept_embeddings", action="store_true", default=False,
                        help="Generate embeddings for each concept in GC.CONCEPTS_FILE.")
    parser.add_argument("--save_sample_embeddings", action="store_true", default=False,
                        help="Generate embeddings for each saved input description file.")
    parser.add_argument("--filter_saved_concepts", action="store_true", default=False,
                        help="Apply similarity-based filtering to prune near-duplicate concepts.")
    args = parser.parse_args()
    if args.save_concept_embeddings:
        agua_embed.save_concept_embeddings(GC.CONCEPTS_FILE, GC.CONCEPT_EMBEDDING_SAVE_PATH,
                                           GC.QUERY_EMBEDDING_MODEL)
    if args.save_sample_embeddings:
        desc_path, embed_path = _description_paths()
        agua_embed.save_sample_embeddings(desc_path, embed_path, GC.DOC_EMBEDDING_MODEL)
    if args.filter_saved_concepts:
        agua_embed.filter_concepts(GC.CONCEPT_EMBEDDING_SAVE_PATH, GC.MAX_INTRA_CONCEPT_SIMILARITY)


if __name__ == "__main__":
    main()
