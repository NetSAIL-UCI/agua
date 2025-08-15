import torch as th
from torch import nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from typing import Callable, Tuple, Sequence
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from torch.nn import functional as F

BATCH_SIZE = 200
N_EPOCHS = 500
LR = 0.075
REG_WEIGHT = 1e-5
REG_ALPHA = 0.95


class ConceptPredictor(nn.Module):
    """Concept mapping function: policy embeddings -> (concept, bin) probabilities.

    Parameters
    ----------
    policy_embedding_size : int
        Input embedding dimension from controller.
    embedding_size : int
        Hidden projection layer width.
    n_concepts : int
        Number of concept prototypes.
    bins : Sequence[float]
        Percentile thresholds (defines number of bins).
    """

    def __init__(self, policy_embedding_size: int, embedding_size: int,
                 n_concepts: int, bins: Sequence[float]) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.n_bins = len(bins)
        self.embedding_projection = nn.Sequential(
            nn.Linear(policy_embedding_size, embedding_size),
            nn.ReLU(),
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, n_concepts * self.n_bins)
        )

    def forward(self, embeddded_states: th.Tensor) -> th.Tensor:
        """Return flattened softmax probabilities over (bin, concept).

        Parameters
        ----------
        embeddded_states : torch.Tensor
            Policy embeddings shaped (B, policy_embedding_size).

        Returns
        -------
        torch.Tensor
            Flattened tensor (B, n_concepts * n_bins) of probabilities.
        """
        pred_scores = self.embedding_projection(embeddded_states)
        pred_scores = pred_scores.view(-1, self.n_bins, self.n_concepts)
        pred_scores = F.softmax(pred_scores, dim=1)
        flat_preds = pred_scores.view(-1, self.n_concepts * self.n_bins)
        return flat_preds


def load_policy_dataset(split_files: Callable[[], Tuple[Sequence, Sequence]],
                       load_test: Callable[[], Sequence],
                       extractor: Callable[[object], Tuple[th.Tensor, th.Tensor, object]],
                       batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train/val/test loaders for policy head training.

    Parameters
    ----------
    split_files : Callable
        Returns (train_files, val_files) sample path sequences.
    load_test : Callable
        Returns list of test sample paths.
    extractor : Callable
        Function mapping loaded npz dict -> (concept_embedding, action_labels, raw_input_tensor).
    batch_size : int, default=BATCH_SIZE
        Batch size for dataloaders.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """
    train_files, val_files = split_files()
    test_files = load_test()
    datasets = [[[], [], []] for _ in range(3)]
    for target_dataset, dataset_files in [[datasets[0], train_files], [datasets[1], val_files], [datasets[2], test_files]]:
        for file in dataset_files:
            data = np.load(file)
            with th.no_grad():
                embedding, action, raw = extractor(data)
            for idx, sample in enumerate([embedding, action, raw]):
                target_dataset[idx].append(sample)
    for ds in datasets:
        for idx in range(len(ds)):
            ds[idx] = th.cat(ds[idx], dim=0)
    train_dataset = TensorDataset(*datasets[0])
    val_dataset = TensorDataset(*datasets[1])
    test_dataset = TensorDataset(*datasets[2])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader


def train_linear_policy_model(split_files: Callable[[], Tuple[Sequence, Sequence]],
                              load_test: Callable[[], Sequence],
                              extractor: Callable[[object], Tuple[th.Tensor, th.Tensor, object]],
                              embed_projection_path, output_projection_save_path,
                              n_actions: int, policy_embedding_size: int,
                              embedding_size: int, n_concepts: int, bins: Sequence[float],
                              training_log_file,
                              batch_size: int = BATCH_SIZE) -> None:
    """Train output mapping function (concept representation -> action logits).

    Uses a frozen concept mapping function (loaded from ``embed_projection_path``)
    to produce (concept,bin) probabilities, then fits a sparse-regularized
    linear classifier translating concept evidence into action logits. Logs
    validation metrics per epoch and saves final weights.

    Parameters
    ----------
    split_files : Callable
        Returns (train_files, val_files) sample paths.
    load_test : Callable
        Returns sequence of test file paths.
    extractor : Callable
        Maps loaded npz -> (concept_embedding, action_tensor, raw_input_tensor).
    embed_projection_path : PathLike
        Saved state dict for ``ConceptPredictor``.
    output_projection_save_path : PathLike
        Destination for trained linear layer state dict.
    n_actions : int
        Number of discrete action classes.
    policy_embedding_size : int
        Dimensionality of controller embedding input.
    embedding_size : int
        Hidden size used in the concept predictor (for documentation only).
    n_concepts : int
        Number of concepts (used for constructing linear layer shape).
    bins : Sequence[float]
        Percentile thresholds defining number of bins.
    training_log_file : PathLike
        File to append epoch-wise evaluation metrics.
    batch_size : int, default=BATCH_SIZE
        Mini-batch size.
    """
    learned_projector = ConceptPredictor(policy_embedding_size, embedding_size,
                                        n_concepts, bins)
    learned_projector.load_state_dict(th.load(embed_projection_path, weights_only=True))
    train_dataloader, val_dataloader, test_dataloader = load_policy_dataset(
        split_files, load_test, extractor, batch_size)

    embedding_projection = nn.Linear(n_concepts * len(bins), n_actions)
    optimizer = th.optim.Adam(params=embedding_projection.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    if output_projection_save_path.exists():
        return

    val_reports = []
    with open(training_log_file, "w") as log_file:
        for epoch_idx in tqdm(range(N_EPOCHS), position=0):
            val_loss = None
            all_true_actions = []
            all_pred_actions = []
            for (embedding_samples, actions, _) in val_dataloader:
                with th.no_grad():
                    output_scores = embedding_projection(embedding_samples)
                    pred_actions = th.argmax(output_scores, dim=1)
                    if val_loss is None:
                        val_loss = loss_fn(output_scores, actions)
                    else:
                        val_loss += loss_fn(output_scores, actions)
                    all_true_actions.extend(actions.tolist())
                    all_pred_actions.extend(pred_actions.tolist())
            val_reports.append(classification_report(
                all_true_actions, all_pred_actions, output_dict=True, zero_division=0.))
            print(f"Validation Loss: {val_loss / len(val_dataloader)}", file=log_file)
            print(f"{classification_report(all_true_actions, all_pred_actions, zero_division=0., digits=4)}", file=log_file)


            for (embedding_samples, actions, _) in tqdm(train_dataloader, leave=False, desc="Batch", position=1):
                output_scores = embedding_projection(embedding_samples)
                l1_penalty = REG_WEIGHT * ((1-REG_ALPHA) * embedding_projection.weight.pow(2).sum() +
                                           REG_ALPHA * embedding_projection.weight.abs().sum() +
                                           REG_ALPHA * embedding_projection.bias.abs().sum())
                loss = loss_fn(output_scores, actions) + l1_penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        test_loss = None
        all_true_actions = []
        all_pred_actions = []
        for (embedding_samples, actions, _) in test_dataloader:
            with th.no_grad():
                output_scores = embedding_projection(embedding_samples)
                pred_actions = th.argmax(output_scores, dim=1)
                if test_loss is None:
                    test_loss = loss_fn(output_scores, actions)
                else:
                    test_loss += loss_fn(output_scores, actions)
                all_true_actions.extend(actions.tolist())
                all_pred_actions.extend(pred_actions.tolist())
        val_reports.append(classification_report(
            all_true_actions, all_pred_actions, output_dict=True, zero_division=0., digits=4))
        tqdm.write(f"Final Test Loss: {test_loss / len(test_dataloader)}", end="\n")
        tqdm.write(f"{classification_report(all_true_actions, all_pred_actions, zero_division=0., digits=4)}", end="\n")
        print(f"Final Test Loss: {test_loss / len(test_dataloader)}", file=log_file)
        print(f"{classification_report(all_true_actions, all_pred_actions, zero_division=0., digits=4)}", file=log_file)

    embedding_projection = embedding_projection.cpu()
    final_params = embedding_projection.state_dict()
    th.save(final_params, output_projection_save_path)


