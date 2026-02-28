import numpy as np
import torch
from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module, val_loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate model with Rank-1 accuracy and mAP.

    Returns (rank1_accuracy, mean_average_precision).
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            embeddings = model(imgs)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    if not all_embeddings:
        return 0.0, 0.0

    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)

    if len(embeddings) < 2:
        return 0.0, 0.0

    # Compute pairwise cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-8)
    similarity_matrix = embeddings_norm @ embeddings_norm.T

    # Rank-1 accuracy
    rank1_correct = 0
    average_precisions = []

    for i in range(len(labels)):
        # Exclude self
        sims = similarity_matrix[i].copy()
        sims[i] = -1.0

        # Sort by descending similarity
        ranked_indices = np.argsort(-sims)
        ranked_labels = labels[ranked_indices]
        query_label = labels[i]

        # Rank-1
        if ranked_labels[0] == query_label:
            rank1_correct += 1

        # AP for this query
        relevant = (ranked_labels == query_label)
        if relevant.sum() == 0:
            continue
        cumsum = np.cumsum(relevant)
        precision_at_k = cumsum / (np.arange(len(relevant)) + 1)
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        average_precisions.append(ap)

    rank1 = rank1_correct / len(labels) if len(labels) > 0 else 0.0
    mAP = np.mean(average_precisions) if average_precisions else 0.0

    return float(rank1), float(mAP)
