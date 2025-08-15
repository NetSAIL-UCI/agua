import torch as th
from torch import nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from typing import Callable, Tuple, Sequence
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from scipy import stats

BATCH_SIZE = 100
N_EPOCHS = 200
LR = 0.005


class ConceptPredictor(nn.Module):
    """Concept mapping function: controller features -> (concept, bin) logits.

    Parameters
    ----------
    policy_embedding_size : int
        Dimensionality of the controller (policy) embedding input.
    embedding_size : int
        Hidden layer size for intermediate projection.
    n_concepts : int
        Number of concept prototypes.
    bins : Sequence[float]
        Percentile thresholds defining similarity bins.
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
        """Return raw (bin, concept) logits for each input state.

        Parameters
        ----------
        embeddded_states : torch.Tensor
            Policy embeddings shaped (B, policy_embedding_size).

        Returns
        -------
        torch.Tensor
            Logits shaped (B, n_bins, n_concepts).
        """
        pred_scores = self.embedding_projection(embeddded_states)
        pred_scores = pred_scores.view(-1, self.n_bins, self.n_concepts)
        return pred_scores


class QuantileBinner:
    """Discretize similarity matrix into percentile-based classes.

    Performs per-feature (concept) binning based on percentile thresholds defined
    at initialization. ``fit`` stores the reference distribution; ``transform``
    maps new rows to integer bin indices.
    """
    def __init__(self, bins: Sequence[float]) -> None:
        self.values = None
        self.n_features = None
        self.bins = np.array(bins)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Return per-feature bin indices for input similarity rows.

        Parameters
        ----------
        data : np.ndarray
            Similarity scores shaped (N, n_features).

        Returns
        -------
        np.ndarray
            Integer bin indices shaped (N, n_features).
        """
        output = np.zeros_like(data, dtype=np.int64)
        if self.n_features is None or self.values is None:
            raise ValueError("Scaler Must be fit before using")
        for feature_idx in range(self.n_features):
            original_values = self.values[:, feature_idx]
            feature_values = data[:, feature_idx]
            feature_quantiles = stats.percentileofscore(a=original_values, score=feature_values)
            feature_classes = np.digitize(x=feature_quantiles, bins=self.bins)
            feature_classes = np.clip(feature_classes, 0, len(self.bins)-1)
            output[:, feature_idx] = feature_classes
        return output

    def fit(self, data: np.ndarray) -> None:
        """Store reference similarity values used for percentile binning."""
        self.values = data
        self.n_features = data.shape[1]


def load_embedding_dataset(concept_embedding_path, embed_path,
                           split_files_func: Callable[[], Tuple[Sequence, Sequence]],
                           extractor: Callable[[object], Tuple[th.Tensor, np.ndarray]],
                           bins: Sequence[float],
                           batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    """Build dataloaders for training the concept mapping function.

    Parameters
    ----------
    concept_embedding_path : PathLike
        Directory of concept ``.npz`` embedding files.
    embed_path : PathLike
        Directory of sample text embedding files.
    split_files_func : Callable
        Returns (train_files, val_files) sequences of sample file paths.
    extractor : Callable
        Function mapping a sample file path to (policy_embedding_tensor, text_embedding_array).
    bins : Sequence[float]
        Percentile thresholds used for similarity binning.
    batch_size : int, default=BATCH_SIZE
        Batch size for returned dataloaders.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        (train_loader, validation_loader)
    """
    concept_files = sorted(list(concept_embedding_path.iterdir()))
    concept_embeddings = []
    for concept_file in concept_files:
        data = np.load(concept_file)
        concept_embeddings.append(data["embedding"])
    concept_embeddings = np.array(concept_embeddings)

    train_files, val_files = split_files_func()
    train_dataset, val_dataset = [[], []], [[], []]

    for target_dataset, dataset_files in [[train_dataset, train_files], [val_dataset, val_files]]:
        for file in dataset_files:
            embedding, text_embedding = extractor(file)
            target_dataset[0].append(embedding)
            target_dataset[1].append(text_embedding)

    normalizer = None
    for target_dataset in [train_dataset, val_dataset]:
        text_embeddings = np.array(target_dataset[1])
        similarity_scores = cosine_similarity(text_embeddings, concept_embeddings)
        if normalizer is None:
            normalizer = QuantileBinner(bins)
            normalizer.fit(similarity_scores)
        similarity_classes = normalizer.transform(similarity_scores)
        target_dataset[0] = th.cat(target_dataset[0], dim=0)
        target_dataset[1] = th.as_tensor(similarity_classes)

    train_dataset = TensorDataset(*train_dataset)
    val_dataset = TensorDataset(*val_dataset)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader


def train_embed_layer(concept_embedding_path, embed_path,
                      split_files_func: Callable[[], Tuple[Sequence, Sequence]],
                      extractor: Callable[[object], Tuple[th.Tensor, np.ndarray]],
                      embed_projection_save_path,
                      policy_embedding_size: int, embedding_size: int,
                      n_concepts: int, bins: Sequence[float]) -> None:
    """Train the concept mapping function (embedding_to_embedding stage).

    Learns parameters that map controller (policy) feature embeddings
    into a binned similarity distribution over concepts. Saves the
    trained ``ConceptPredictor`` state dict to ``embed_projection_save_path``.

    Parameters
    ----------
    concept_embedding_path : PathLike
        Directory containing concept embeddings.
    embed_path : PathLike
        Directory containing sample text embedding files.
    split_files_func : Callable
        Returns (train_files, val_files) of sample file paths.
    extractor : Callable
        Maps file path -> (policy_embedding_tensor, text_embedding_array).
    embed_projection_save_path : PathLike
        Destination file for trained state dict.
    policy_embedding_size : int
        Dimensionality of controller embedding input.
    embedding_size : int
        Hidden layer width.
    n_concepts : int
        Number of concept prototypes.
    bins : Sequence[float]
        Percentile thresholds for binning similarity.
    """
    train_dataloader, val_dataloader = load_embedding_dataset(
        concept_embedding_path, embed_path, split_files_func, extractor, bins)
    embedding_projection = ConceptPredictor(policy_embedding_size, embedding_size,
                                           n_concepts, bins)

    optimizer = th.optim.SGD(params=embedding_projection.parameters(), lr=LR, momentum=0.25)
    loss_fn = nn.CrossEntropyLoss()
    if embed_projection_save_path.exists():
        return

    for epoch_idx in tqdm(range(N_EPOCHS), position=0):
        val_loss = None
        for (embedding_samples, sim_scores) in val_dataloader:
            with th.no_grad():
                output_scores = embedding_projection(embedding_samples)
                if val_loss is None:
                    val_loss = loss_fn(output_scores, sim_scores)
                else:
                    val_loss += loss_fn(output_scores, sim_scores)
        tqdm.write(f"Validation Loss: {val_loss / len(val_dataloader)}", end="\n")
            

        for (embedding_samples, sim_scores) in tqdm(train_dataloader, leave=False, desc="Batch", position=1):
            output_scores = embedding_projection(embedding_samples)
            loss = loss_fn(output_scores, sim_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    th.save(embedding_projection.state_dict(), embed_projection_save_path)
