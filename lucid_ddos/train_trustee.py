import numpy as np
import torch as th
from sklearn.metrics import classification_report
from trustee.report.trust import TrustReport
import global_constants as GC
from typing import Tuple
from input_to_text import HISTORY_FEATURES

HISTORY_LEN = 10  # fixed sequence length
CHANNELS = 1      # final dim always 1 in stored input_sample tensors
STATE_DIM = (10, len(HISTORY_FEATURES), 1)
FEATURE_NAMES = []
for i in range(HISTORY_LEN):
    for feature in HISTORY_FEATURES:
        FEATURE_NAMES.append(f"{feature}__t-{HISTORY_LEN-i}")


def flatten_inputs(inputs: np.ndarray) -> np.ndarray:
    """Flatten a batch of temporal-feature tensors into 2D.

    Parameters
    ----------
    inputs : np.ndarray
        Array of shape (N, HISTORY_LEN, F, 1) where F = len(HISTORY_FEATURES).

    Returns
    -------
    np.ndarray
        Flattened array shaped (N, HISTORY_LEN * F).
    """
    inputs = np.array(inputs)
    batch_size = inputs.shape[0]
    return inputs.reshape(batch_size, -1)


def unflatten_inputs(flattened: np.ndarray) -> np.ndarray:
    """Inverse reshape of :func:`flatten_inputs`.

    Parameters
    ----------
    flattened : np.ndarray
        Array shaped (N, HISTORY_LEN * F).

    Returns
    -------
    np.ndarray
        Restored array shaped (N, HISTORY_LEN, F, 1).
    """
    flattened = np.array(flattened)
    batch_size = flattened.shape[0]
    return flattened.reshape((batch_size, ) + STATE_DIM)


class PredictWrapper:
    """Adapter exposing a scikit-learn style ``predict`` for LucidModel.

    Parameters
    ----------
    model : LucidModel
        Trained Lucid DDoS classifier.
    """

    def __init__(self, model: GC.LucidModel) -> None:
        self.model = model
        self.model.eval()

    def predict(self, flat_inputs: np.ndarray) -> np.ndarray:
        """Return boolean label predictions for flattened inputs.

        Parameters
        ----------
        flat_inputs : np.ndarray
            Array shaped (N, HISTORY_LEN * F) produced by ``flatten_inputs``.

        Returns
        -------
        np.ndarray
            Boolean predictions of shape (N,).
        """
        raw_inputs = th.as_tensor(unflatten_inputs(flat_inputs), dtype=th.float32)
        logits = self.model(raw_inputs)
        preds = (logits > 0.5).detach().cpu().numpy().reshape(-1)
        return preds.astype(bool)


def _load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and flatten training/validation/test input and output arrays.

    Returns
    -------
    Tuple[np.ndarray, ...]
        (train_inputs, train_outputs, test_inputs, test_outputs, val_inputs, val_outputs)
    """
    train_files, val_files = GC.split_input_files()
    test_files = GC.load_test_inputs()
    if not train_files:
        raise FileNotFoundError("No training input files found; generate inputs first.")
    if not test_files:
        raise FileNotFoundError("No test input files found; generate test set.")
    train_raw = [np.load(f)["input_sample"] for f in train_files]
    train_outputs = np.array([np.load(f)["output"] for f in train_files])
    test_raw = [np.load(f)["input_sample"] for f in test_files]
    test_outputs = np.array([np.load(f)["output"] for f in test_files])
    train_inputs = flatten_inputs(np.array(train_raw, dtype=np.float32))
    test_inputs = flatten_inputs(np.array(test_raw, dtype=np.float32))
    val_raw = [np.load(f)["input_sample"] for f in val_files]
    val_outputs = np.array([np.load(f)["output"] for f in val_files])
    val_inputs = flatten_inputs(np.array(val_raw, dtype=np.float32))
    return train_inputs, train_outputs, test_inputs, test_outputs, val_inputs, val_outputs


def main() -> None:
    """Train trustee decision trees and write evaluation report/logs."""
    train_inputs, train_outputs, test_inputs, test_outputs, val_inputs, val_outputs = _load_data()
    teacher = GC.LucidModel()
    teacher.load_state_dict(th.load(GC.CONTROLLER_PATH, map_location="cpu", weights_only=True))
    predictor = PredictWrapper(teacher)

    report_dir = GC.SAVE_PATH / "trustee_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = GC.SAVE_PATH / "trustee_train_log.txt"

    trust_report = TrustReport(
        predictor,
        X_train=train_inputs,
        y_train=train_outputs,
        X_test=val_inputs,
        y_test=val_outputs,
        max_iter=0,
        trustee_num_iter=30,
        num_pruning_iter=10,
        trustee_sample_size=0.3,
        skip_retrain=True,
        verbose=False,
        is_classify=True,
        feature_names=FEATURE_NAMES,
        class_names=["benign", "DDOS"]
    )
    pruned_trustee_outputs = trust_report.min_dt.predict(test_inputs)
    full_trustee_outputs = trust_report.max_dt.predict(test_inputs)
    with open(log_file, "w") as f:
        print("-------------------------- Full Tree ------------------------------", file=f)
        print(f"Node Count: {trust_report.max_dt.tree_.node_count}", file=f)
        print(classification_report(test_outputs, full_trustee_outputs, digits=5), file=f)
        print("-------------------------- Pruned Tree ------------------------------", file=f)
        print(f"Node Count: {trust_report.min_dt.tree_.node_count}", file=f)
        print(classification_report(test_outputs, pruned_trustee_outputs, digits=5), file=f)
    trust_report.save(report_dir)


if __name__ == "__main__":
    main()
