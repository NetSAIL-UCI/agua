from pathlib import Path
from typing import Tuple, List
from torch import nn
import torch as th
import numpy as np

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

class LucidModel(nn.Module):
    """CNN-based classifier used as the underlying Lucid DDoS controller.

    Architecture
    ------------
    feature_extractor : Conv2d -> Dropout -> ReLU -> MaxPool2d -> Flatten
        Produces a fixed-size embedding (POLICY_EMBEDDING_SIZE) for an input
        history window.
    final_layer : Linear -> Sigmoid
        Maps extracted features to a scalar probability of the positive class.

    Input Format
    ------------
    Accepts either (B, H, W, C) or (H, W, C) arrays / tensors which are
    internally permuted into channel-first (B, C, H, W) for convolution.
    """

    N_KERNELS = 64
    DROPOUT = 0.5

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.N_KERNELS, kernel_size=(3, 11), stride=1),
            nn.Dropout(p=self.DROPOUT),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 1)),
            nn.Flatten(),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: th.Tensor) -> th.Tensor:
        """Return per-sample probability for the positive class.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor with shape (B, H, W, C) or (H, W, C). H/W correspond
            to temporal and feature axes respectively.

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, 1) containing probabilities after sigmoid.
        """
        if data.ndim == 4:
            data = data.permute(0, 3, 1, 2)
        elif data.ndim == 3:
            data = data.permute(2, 0, 1)
        else:
            raise ValueError("Expected input rank 3 or 4 for LucidModel")
        features = self.feature_extractor(data)
        pred = self.final_layer(features)
        return pred


MAX_INTRA_CONCEPT_SIMILARITY = 0.875

DATASET_PATH = Path(__file__).parent / "sample-dataset"
CONTROLLER_PATH = Path(__file__).parent / "output" / "10t-10n-DOS2019-LUCID.pt"
N_ACTIONS = 2
IS_ACTION_DISCRETE = True
SAVE_PATH = Path(__file__).parent / "data"
PROMPT_FILE = SAVE_PATH / "prompt_skeleton.txt"
CONCEPTS_FILE = SAVE_PATH / "auto_concepts.txt"

MAX_NUM_INPUTS = 1000
NUM_TEST_INPUTS = 450 
LLM_MODEL = "gpt-4o-2024-08-06"
QUERY_EMBEDDING_MODEL = "text-embedding-3-large"
DOC_EMBEDDING_MODEL = "text-embedding-3-large"

INPUT_SAVE_PATH = SAVE_PATH / "inputs"
TEST_INPUT_SAVE_PATH = SAVE_PATH / "test_inputs"
N_QUERY_TOGETHER = 250


INPUT_DESCRIPTION_SAVE_PATH = SAVE_PATH / "input_descriptions"
CONCEPT_EMBEDDING_SAVE_PATH = SAVE_PATH / "auto_concept_embeddings"
INPUT_EMBEDDING_SAVE_PATH = SAVE_PATH / "input_embeddings"


OUTPUT_PROJECTION_SAVE_PATH = SAVE_PATH / "final_projection.pt"
EMBED_PROJECTION_SAVE_PATH = SAVE_PATH / "embed_projection.pt"
TRAINING_LOG_FILE = SAVE_PATH / "train_log.txt"
ROBUSTNESS_FILE = SAVE_PATH / "robustness.txt"

TEST_FRACTION = 0.1
EXPLAIN_SEED = 14

POLICY_EMBEDDING_SIZE = 64
EMBEDDING_SIZE = 50
if CONCEPT_EMBEDDING_SAVE_PATH.exists():
    N_CONCEPTS = len(list(CONCEPT_EMBEDDING_SAVE_PATH.iterdir()))
else:
    N_CONCEPTS = 0
BINS = [20, 60, 100]


def split_input_files() -> Tuple[List[Path], List[Path]]:
    """Split saved input files into train / validation subsets.

    Files are randomly shuffled with a deterministic seed (``EXPLAIN_SEED``)
    to produce reproducible splits.

    Returns
    -------
    Tuple[List[Path], List[Path]]
        Two lists: (train_input_paths, validation_input_paths).
    """
    input_files = sorted(list(INPUT_SAVE_PATH.iterdir()))
    rand = np.random.RandomState(seed=EXPLAIN_SEED)
    rand.shuffle(input_files)
    n_train_samples = int((1 - TEST_FRACTION) * len(input_files))
    train_inputs = input_files[:n_train_samples]
    val_inputs = input_files[n_train_samples:]
    return (train_inputs, val_inputs)

def load_test_inputs() -> List[Path]:
    """Return sorted list of held-out test input file paths."""
    input_files = sorted(list(TEST_INPUT_SAVE_PATH.iterdir()))
    return input_files