import numpy as np
from sklearn.metrics import classification_report
from trustee.report.trust import TrustReport
import global_constants as GC
from typing import Tuple
from state_to_text import FEATURES
import torch as th

HISTORY_LEN = 10  # fixed window length in dataset generation
FEATURE_NAMES = []
for i in range(HISTORY_LEN):
    for feature in FEATURES:
        FEATURE_NAMES.append(f"{feature}__t-{HISTORY_LEN-i}")
ACTION_NAMES = ["1/2x", "-33%", "-10%", "No change", "+10%", "+33%", "2x"]


def flatten_states(states: np.ndarray) -> np.ndarray:
    """Flatten 3D state tensor to 2D feature matrix.

    Parameters
    ----------
    states : np.ndarray
        Array shaped (N, HISTORY_LEN, F).

    Returns
    -------
    np.ndarray
        Flattened array shaped (N, HISTORY_LEN * F).
    """
    if states.ndim != 3 or states.shape[1] != HISTORY_LEN:
        raise ValueError("Unexpected state shape; expected (N, 10, F)")
    n, t, f = states.shape
    return states.reshape(n, t * f).astype(np.float32)


def unflatten_states(flattened: np.ndarray) -> np.ndarray:
    """Inverse reshape of ``flatten_states``.

    Parameters
    ----------
    flattened : np.ndarray
        Array shaped (N, D) where D is divisible by HISTORY_LEN.

    Returns
    -------
    np.ndarray
        Reconstructed states shaped (N, HISTORY_LEN, F).
    """
    flattened = np.array(flattened)
    if flattened.ndim != 2:
        raise ValueError("Expected 2D flattened array")
    n, d = flattened.shape
    if d % HISTORY_LEN != 0:
        raise ValueError("Flattened dim not divisible by HISTORY_LEN")
    f = d // HISTORY_LEN
    return flattened.reshape(n, HISTORY_LEN, f)


class PredictWrapper:
    """Adapter exposing a ``predict(flattened_states)`` API."""
    def __init__(self, model: th.nn.Module) -> None:
        """Store underlying trained policy model.

        Parameters
        ----------
        model : torch.nn.Module
            Trained congestion control policy network used to produce
            greedy action predictions.

        Returns
        -------
        None
        """
        self.model = model

    def predict(self, flat_states: np.ndarray) -> np.ndarray:
        """Return greedy action indices for flattened states.

        Parameters
        ----------
        flat_states : np.ndarray
            2D array of shape ``(N, HISTORY_LEN * F)`` produced by
            ``flatten_states``.

        Returns
        -------
        np.ndarray
            1D array of integer action indices (shape ``(N,)``).
        """
        raw_states = unflatten_states(flat_states)
        raw_states = th.as_tensor(raw_states, dtype=th.float32)
        with th.no_grad():
            actions = self.model(raw_states).detach().cpu().numpy()
        return actions.reshape(-1)


def _load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and flatten state/action data for trustee training.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        (train_states, train_actions, test_states, test_actions, val_states, val_actions)
        with flattened representations.
    """
    train_files, val_files = GC.split_state_files()
    test_files = GC.load_test_states()
    if not train_files:
        raise FileNotFoundError("No training state files found; generate states first.")
    if not test_files:
        raise FileNotFoundError("No test state files found; generate test set.")
    train_states_raw = [np.load(f)["state"] for f in train_files]
    train_actions = np.array([np.load(f)["action"] for f in train_files])
    test_states_raw = [np.load(f)["state"] for f in test_files]
    test_actions = np.array([np.load(f)["action"] for f in test_files])
    val_states_raw = [np.load(f)["state"] for f in val_files]
    val_actions = np.array([np.load(f)["action"] for f in val_files])
    train_states = flatten_states(np.asarray(train_states_raw, dtype=np.float32))
    test_states = flatten_states(np.asarray(test_states_raw, dtype=np.float32))
    val_states = flatten_states(np.asarray(val_states_raw, dtype=np.float32))
    return train_states, train_actions, test_states, test_actions, val_states, val_actions


def main() -> None:
    """Train decision tree trustees and save trust report artifacts.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Side-effects: writes metrics and serialized trustees to disk.
    """
    train_states, train_actions, test_states, test_actions, val_states, val_actions = _load_data()
    controller = GC.CCModel()
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
        class_names=ACTION_NAMES,
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
