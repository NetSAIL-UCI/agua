from pathlib import Path
from typing import Tuple, List
import torch as th
from torch import nn
import numpy as np
from torch.distributions import Categorical

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

class ABRFeaturesExtractor(nn.Module):
    """1D convolutional feature extractor for ABR states.

    Parameters
    ----------
    history_len : int, optional
        Number of past timesteps encoded (default 10).
    network_features : int, optional
        Count of network-level scalar features per timestep (default 6).
    """

    def __init__(self, history_len: int = 10, network_features: int = 6):
        super().__init__()
        self.network_features = network_features
        self.history_len = history_len
        self.network_cnn = nn.Sequential(
            nn.Conv1d(self.history_len, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),   
            nn.Flatten())
        self.video_cnn = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten())
        self.quality_cnn = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Compute concatenated spatial features.

        Parameters
        ----------
        observations : torch.Tensor
            State tensor of shape ``(B, H, F)`` where the channel layout
            matches the slicing logic inside the method.

        Returns
        -------
        torch.Tensor
            Concatenated latent feature vector per batch element.
        """
        network_features = self.network_cnn(
            observations[..., 0:self.history_len, 0:self.network_features])
        video_features = self.video_cnn(
            observations[..., 0:5, self.network_features:self.network_features+10])
        quality_features = self.quality_cnn(
            observations[..., 0:5, self.network_features+10:self.network_features+20])
        features = th.cat((network_features, video_features, quality_features), dim=1)
        return features


class ABRModel(nn.Module):
    """Lightweight ABR policy network with CNN feature extractor.

    Parameters
    ----------
    history_len : int, optional
        Number of historical timesteps (default 10).
    network_features : int, optional
        Number of scalar network features (default 6).
    n_actions : int, optional
        Number of discrete actions (default 10).
    feature_dim : int, optional
        Dimensionality of concatenated feature extractor output
        (default 256).
    """

    def __init__(self, history_len: int = 10, network_features: int = 6,
                 n_actions: int = 10, feature_dim: int = 256):
        super().__init__()
        self.features_extractor = ABRFeaturesExtractor(history_len=history_len, 
                                          network_features=network_features)
        self.policy_net = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU())
        self.action_net = nn.Linear(256, n_actions)
               
    def forward(self, observation: dict) -> int:
        """Compute greedy (argmax) action for a batch of observations.

        Parameters
        ----------
        observation : dict or torch.Tensor
            Batched observation tensor compatible with feature extractor.

        Returns
        -------
        torch.Tensor
            Long tensor of chosen action indices with shape ``(B,)``.
        """
        features = self.features_extractor(observation)
        action_logits = self.action_net(self.policy_net(features))
        dist = Categorical(logits=action_logits)
        action = th.argmax(dist.probs, dim=1)
        return action


MAX_INTRA_CONCEPT_SIMILARITY = 0.875

N_ACTIONS = 10
IS_ACTION_DISCRETE = True

SAVE_PATH = Path(__file__).parent / "data"
PROMPT_FILE =  SAVE_PATH / "prompt_skeleton.txt"
CONCEPTS_FILE = SAVE_PATH / "auto_concepts.txt"
CONCEPT_NAME_FILE = SAVE_PATH / "concept_names.txt"
CONTROLLER_PATH = SAVE_PATH / "abr_model.pt"

MAX_NUM_STATES = 4000
NUM_TEST_STATES = 4000
LLM_MODEL = "gpt-4o-2024-08-06"
QUERY_EMBEDDING_MODEL = "text-embedding-3-large"
DOC_EMBEDDING_MODEL = "text-embedding-3-large"

STATE_SAVE_PATH = SAVE_PATH / "states"
TEST_STATE_SAVE_PATH = SAVE_PATH / "test_states"
N_QUERY_TOGETHER = 250


STATE_DESCRIPTION_SAVE_PATH = SAVE_PATH / "state_descriptions"
CONCEPT_EMBEDDING_SAVE_PATH = SAVE_PATH / "auto_concept_embeddings"
STATE_EMBEDDING_SAVE_PATH = SAVE_PATH / "state_embeddings"


OUTPUT_PROJECTION_SAVE_PATH = SAVE_PATH / "final_projection.pt"
EMBED_PROJECTION_SAVE_PATH = SAVE_PATH / "embed_projection.pt"
TRAINING_LOG_FILE = SAVE_PATH / "train_log.txt"
ROBUSTNESS_FILE = SAVE_PATH / "robustness.txt"

TEST_FRACTION = 0.1
SEED = 14

POLICY_EMBEDDING_SIZE = 256
EMBEDDING_SIZE = 100
if CONCEPT_EMBEDDING_SAVE_PATH.exists():
    N_CONCEPTS = len(list(CONCEPT_EMBEDDING_SAVE_PATH.iterdir()))
else:
    N_CONCEPTS = 0
BINS = [20, 60, 100]


def split_state_files() -> Tuple[List[Path], List[Path]]:
    """Return shuffled (train, validation) state file paths.

    Returns
    -------
    tuple of list[Path]
        ``(train_files, val_files)`` after random shuffle with seed ``SEED``.
    """
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