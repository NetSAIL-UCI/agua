import sys
from pathlib import Path
from typing import Tuple
sys.path.append(str(Path(__file__).resolve().parents[1]))
from argparse import ArgumentParser
import global_constants as GC
from agua import embed_data as agua_embed


def _description_paths() -> Tuple[Path, Path]:
    """Resolve description and embedding output directories.

    Returns
    -------
    tuple of Path
        ``(description_path, embedding_path)`` where the first contains
        per-sample natural language descriptions and the second is the
        directory for their vector embeddings.
    """
    if hasattr(GC, "STATE_DESCRIPTION_SAVE_PATH"):
        return GC.STATE_DESCRIPTION_SAVE_PATH, GC.STATE_EMBEDDING_SAVE_PATH
    return GC.INPUT_DESCRIPTION_SAVE_PATH, GC.INPUT_EMBEDDING_SAVE_PATH


def main() -> None:
    """CLI entry for concept & sample embedding workflows.

    Actions (gated by flags):
    * ``--save_concept_embeddings``: embed each concept string once.
    * ``--save_sample_embeddings``: embed each saved state/input description.
    * ``--filter_saved_concepts``: prune concepts above intra-similarity threshold.

    Returns
    -------
    None
        Artifacts are written under paths declared in ``global_constants``.
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
        agua_embed.save_concept_embeddings(
            GC.CONCEPTS_FILE,
            GC.CONCEPT_EMBEDDING_SAVE_PATH,
            GC.QUERY_EMBEDDING_MODEL
        )
    if args.save_sample_embeddings:
        desc_path, embed_path = _description_paths()
        agua_embed.save_sample_embeddings(
            desc_path,
            embed_path,
            GC.DOC_EMBEDDING_MODEL
        )
    if args.filter_saved_concepts:
        agua_embed.filter_concepts(
            GC.CONCEPT_EMBEDDING_SAVE_PATH,
            GC.MAX_INTRA_CONCEPT_SIMILARITY
        )


if __name__ == "__main__":
    main()
