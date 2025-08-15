import numpy as np
from sklearn.metrics import classification_report
import torch as th
from trustee.report.trust import TrustReport
import global_constants as GC
from typing import Tuple
from state_to_text import HISTORY_FEATURES

# ABR state structure parameters (fixed by data generation scripts)
HISTORY_LEN = 10  # number of past timesteps stored
FUTURE_CHUNK_LEN = 5  # number of future chunk predictions

FEATURE_NAMES = []
for i in range(HISTORY_LEN):
    for feature in HISTORY_FEATURES:
        FEATURE_NAMES.append(f"{feature}__t-{HISTORY_LEN-i}")
for feature_name in ["sizes", "ssims"]:
    for time_idx in range(1, FUTURE_CHUNK_LEN + 1):
        for chunk_idx in range(GC.N_ACTIONS):
            FEATURE_NAMES.append(f"{feature_name}_t+{time_idx}_{chunk_idx}")
ACTION_NAMES = [f"video quality {action_idx+1}" for action_idx in range(GC.N_ACTIONS)]

def flatten_states(states: np.ndarray) -> np.ndarray:
    """Flatten padded ABR states into 2D feature matrix using fixed layout.

    Parameters
    ----------
    states : np.ndarray
        Shape (N, HISTORY_LEN, base_features + 2 * N_ACTIONS).

    Returns
    -------
    np.ndarray
        Shape (N, HISTORY_LEN * base_features + 2 * FUTURE_CHUNK_LEN * N_ACTIONS).
    """
    if states.ndim != 3:
        raise ValueError("Expected states with shape (N, T, F)")
    base_feature_count = len(HISTORY_FEATURES)
    future_chunk_len = FUTURE_CHUNK_LEN
    n_actions = GC.N_ACTIONS
    batch = states.shape[0]
    history_features = states[:, :, :base_feature_count].reshape(batch, -1)
    video_sizes = states[:, :future_chunk_len, base_feature_count: base_feature_count + n_actions].reshape(batch, -1)
    ssims = states[:, :future_chunk_len, base_feature_count + n_actions: base_feature_count + 2 * n_actions].reshape(batch, -1)
    return np.concatenate([history_features, video_sizes, ssims], axis=1).astype(np.float32)


def unflatten_states(flattened: np.ndarray) -> np.ndarray:
    """Reconstruct original 3D ABR state tensor from flattened representation.

    Parameters
    ----------
    flattened : np.ndarray
        Flattened state array produced by `flatten_states`.

    Returns
    -------
    np.ndarray
        Reconstructed states with shape (N, HISTORY_LEN, base_features + 2 * N_ACTIONS).
    """
    flattened = np.array(flattened)
    if flattened.ndim != 2:
        raise ValueError("Expected flattened states with shape (N, D)")
    # Reconstruct sizes based on known constants
    base_feature_count = len(HISTORY_FEATURES)
    n_actions = GC.N_ACTIONS
    future_chunk_len = FUTURE_CHUNK_LEN
    history_feat_dim = HISTORY_LEN * base_feature_count
    future_block = future_chunk_len * n_actions
    expected_dim = history_feat_dim + 2 * future_block
    if flattened.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected flattened dim {flattened.shape[1]} != expected {expected_dim}."
        )
    batch = flattened.shape[0]
    history = flattened[:, :history_feat_dim].reshape(batch, HISTORY_LEN, base_feature_count)
    video_sizes = flattened[:, history_feat_dim: history_feat_dim + future_block].reshape(batch, future_chunk_len, n_actions)
    ssims = flattened[:, history_feat_dim + future_block: history_feat_dim + 2 * future_block].reshape(batch, future_chunk_len, n_actions)
    full = np.zeros((batch, HISTORY_LEN, base_feature_count + 2 * n_actions), dtype=flattened.dtype)
    full[:, :, :base_feature_count] = history
    full[:, :future_chunk_len, base_feature_count: base_feature_count + n_actions] = video_sizes
    full[:, :future_chunk_len, base_feature_count + n_actions: base_feature_count + 2 * n_actions] = ssims
    return full


class PredictWrapper:
    """Adapter exposing a `predict(flat_states)` method for the A2C controller."""

    def __init__(self, model: th.nn.Module) -> None:
        """Initialize wrapper.

        Parameters
        ----------
        model : th.nn.Module
            Trained ABR controller model exposing ``forward`` and
            ``features_extractor`` compatible with ABR states.
        """
        self.model = model

    def predict(self, flat_states: np.ndarray) -> np.ndarray:
        """Predict greedy (argmax) actions for flattened states.

        Parameters
        ----------
        flat_states : np.ndarray
            2D flattened state matrix produced by ``flatten_states``.

        Returns
        -------
        np.ndarray
            Integer action indices of shape ``(N,)``.
        """
        if flat_states.ndim != 2:
            raise ValueError("Expected 2D flattened states")
        raw_states = unflatten_states(flat_states)
        raw_states = th.as_tensor(raw_states, dtype=th.float32)
        with th.no_grad():
            actions = self.model(raw_states).detach().cpu().numpy()
        return actions.reshape(-1)


def _load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and flatten ABR state/action datasets for trustee training.

    Returns
    -------
    tuple
        ``(train_states, train_actions, test_states, test_actions, val_states, val_actions)``
        where each state array is flattened via ``flatten_states``.
    """
    train_files, val_files = GC.split_state_files()
    test_files = GC.load_test_states()
    if not train_files:
        raise FileNotFoundError("No training state files found; run state collection first.")
    if not test_files:
        raise FileNotFoundError("No test state files found; generate test set.")

    # Load train (ignore val for now) and test arrays
    train_states_raw = [np.load(f)["state"] for f in train_files]
    train_actions = np.array([np.load(f)["action"] for f in train_files])
    test_states_raw = [np.load(f)["state"] for f in test_files]
    test_actions = np.array([np.load(f)["action"] for f in test_files])

    train_states = flatten_states(np.asarray(train_states_raw, dtype=np.float32))
    test_states = flatten_states(np.asarray(test_states_raw, dtype=np.float32))
    
    val_states_raw = [np.load(f)["state"] for f in val_files]
    val_actions = np.array([np.load(f)["action"] for f in val_files])
    val_states = flatten_states(np.asarray(val_states_raw, dtype=np.float32))
    return train_states, train_actions, test_states, test_actions, val_states, val_actions


def main() -> None:
    """Train decision tree trustee and produce trust report artifacts.

    Loads train/val/test states, fits pruned and full-depth trustees via
    ``TrustReport``, writes evaluation metrics, and saves serialized tree
    structures + associated report under ``trustee_report`` directory.
    """
    train_states, train_actions, test_states, test_actions, val_states, val_actions = _load_data()
    controller = GC.ABRModel()
    controller.load_state_dict(th.load(GC.CONTROLLER_PATH, map_location="cpu", weights_only=True))
    teacher = PredictWrapper(controller)

    report_dir = GC.SAVE_PATH / "trustee_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = GC.SAVE_PATH / "trustee_train_log.txt"

    trust_report = TrustReport(
        teacher,
        X_train=train_states,
        y_train=train_actions,
        X_test=val_states,
        y_test=val_actions,
        max_iter=0,
        trustee_num_iter=30,
        num_pruning_iter=10,
        trustee_sample_size=0.3,
        skip_retrain=True,
        verbose=False,
        is_classify=True,
        feature_names=FEATURE_NAMES,
        class_names=ACTION_NAMES
    )
    pruned_trustee_actions = trust_report.min_dt.predict(test_states)
    full_trustee_actions = trust_report.max_dt.predict(test_states)
    with open(log_file, "w") as f:
        print("-------------------------- Full Tree ------------------------------", file=f)
        print(f"Node Count: {trust_report.max_dt.tree_.node_count}", file=f)
        print(classification_report(test_actions, full_trustee_actions, digits=5), file=f)
        print("-------------------------- Pruned Tree ------------------------------", file=f)
        print(f"Node Count: {trust_report.min_dt.tree_.node_count}", file=f)
        print(classification_report(test_actions, pruned_trustee_actions, digits=5), file=f)
    trust_report.save(report_dir)


if __name__ == "__main__":
    main()
