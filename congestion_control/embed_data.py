import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from argparse import ArgumentParser
import global_constants as GC
from agua import embed_data as agua_embed


def _description_paths():
    """Resolve description and embedding output directories.

    Returns
    -------
    tuple of Path
        ``(description_path, embedding_path)`` where the first contains
        per-state natural language descriptions and the second holds
        their vector embeddings.
    """
    if hasattr(GC, "STATE_DESCRIPTION_SAVE_PATH"):
        return GC.STATE_DESCRIPTION_SAVE_PATH, GC.STATE_EMBEDDING_SAVE_PATH
    return GC.INPUT_DESCRIPTION_SAVE_PATH, GC.INPUT_EMBEDDING_SAVE_PATH


def main() -> None:
    """CLI entry for concept & sample embedding workflows.

    Actions (controlled by flags):
    * ``--save_concept_embeddings``: embed each concept string.
    * ``--save_sample_embeddings``: embed each saved state description.
    * ``--filter_saved_concepts``: remove overly similar concepts.
    """
    parser = ArgumentParser()
    parser.add_argument("--save_concept_embeddings", action="store_true", default=False,
                        help="Generate and persist embeddings for all concepts")
    parser.add_argument("--save_sample_embeddings", action="store_true", default=False,
                        help="Generate embeddings for each state description")
    parser.add_argument("--filter_saved_concepts", action="store_true", default=False,
                        help="Filter concept set using MAX_INTRA_CONCEPT_SIMILARITY")
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
