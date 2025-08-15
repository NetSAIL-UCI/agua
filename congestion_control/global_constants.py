from pathlib import Path
from typing import Tuple, List
from torch import nn
import torch as th
import numpy as np
from torch.distributions import Categorical

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


class CCFeatureExtractor(nn.Module):
    """Simple MLP feature extractor for congestion control states.

    Parameters
    ----------
    features_dim : int, optional
        Output embedding dimensionality (default 128).
    """

    def __init__(self, features_dim: int = 128):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40, features_dim),
            nn.GELU()
        )

    def forward(self, observation: th.Tensor) -> th.Tensor:
        """Compute latent feature vector for a batch of observations."""
        return self.extractor(observation)
    
class CCModel(nn.Module):
    """Lightweight policy network over extracted features.

    Parameters
    ----------
    features_dim : int, optional
        Dimensionality of feature extractor output (default 128).
    action_dim : int, optional
        Number of discrete congestion control actions (default 7).
    """

    def __init__(self, features_dim: int = 128, action_dim: int = 7):
        super().__init__()
        self.features_extractor = CCFeatureExtractor(features_dim)
        self.policy_net = nn.Sequential(nn.Linear(features_dim, 128), nn.GELU())
        self.action_net = nn.Linear(128, action_dim)

    def forward(self, observation: th.Tensor) -> th.Tensor:
        """Return greedy action indices for a batch of observations."""
        features = self.features_extractor(observation)
        action_logits = self.action_net(self.policy_net(features))
        dist = Categorical(logits=action_logits)
        action = th.argmax(dist.probs, dim=1)
        return action

MAX_INTRA_CONCEPT_SIMILARITY = 0.875

SAVE_PATH = Path(__file__).parent / "data"
N_ACTIONS = 7
IS_ACTION_DISCRETE = True

PROMPT_FILE = SAVE_PATH / "prompt_skeleton.txt"
CONCEPTS_FILE = SAVE_PATH / "concepts.txt"
CONTROLLER_PATH = SAVE_PATH / "cc_model.pt"

MAX_NUM_STATES = 2750
NUM_TEST_STATES = 2000
LLM_MODEL = "gpt-4o-2024-08-06"
QUERY_EMBEDDING_MODEL = "text-embedding-3-large"
DOC_EMBEDDING_MODEL = "text-embedding-3-large"

STATE_SAVE_PATH = SAVE_PATH / "states"
TEST_STATE_SAVE_PATH = SAVE_PATH / "test_states"
N_QUERY_TOGETHER = 50


STATE_DESCRIPTION_SAVE_PATH = SAVE_PATH / "state_descriptions"
CONCEPT_EMBEDDING_SAVE_PATH = SAVE_PATH / "concept_embeddings"
STATE_EMBEDDING_SAVE_PATH = SAVE_PATH / "state_embeddings"

OUTPUT_PROJECTION_SAVE_PATH = SAVE_PATH / "final_projection.pt"
EMBED_PROJECTION_SAVE_PATH = SAVE_PATH / "embed_projection.pt"
TRAINING_LOG_FILE = SAVE_PATH / "train_log.txt"
ROBUSTNESS_FILE = SAVE_PATH / "robustness.txt"

POLICY_EMBEDDING_SIZE = 128
EMBEDDING_SIZE = 100
if CONCEPT_EMBEDDING_SAVE_PATH.exists():
    N_CONCEPTS = len(list(CONCEPT_EMBEDDING_SAVE_PATH.iterdir()))
else:
    N_CONCEPTS = 0
BINS = [20, 60, 100]


TEST_FRACTION = 0.25
SEED = 14

def split_state_files() -> Tuple[List[Path], List[Path]]:
    """Return shuffled (train, validation) state paths using TEST_FRACTION."""
    state_files = sorted(list(STATE_SAVE_PATH.iterdir()))
    rand = np.random.RandomState(seed=SEED)
    rand.shuffle(state_files)
    n_train_samples = int((1 - TEST_FRACTION) * len(state_files))
    train_states = state_files[:n_train_samples]
    val_states = state_files[n_train_samples:]
    return (train_states, val_states)

def load_test_states() -> List[Path]:
    """Return sorted list of test state file paths."""
    state_files = sorted(list(TEST_STATE_SAVE_PATH.iterdir()))
    return state_files