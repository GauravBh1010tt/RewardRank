import torch
import torch.nn.functional as F

def binary_accuracy(logits, labels, threshold=0.5):
    """
    Compute the accuracy for binary classification.

    Args:
        logits (torch.Tensor): Logits predicted by the model (before applying sigmoid).
        labels (torch.Tensor): Ground truth binary labels (0 or 1).
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        float: Accuracy as a percentage.
    """
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)
    
    # Apply threshold to get predictions (0 or 1)
    preds = (probs >= threshold).float()
    
    # Compare predictions with the true labels
    correct = (preds == labels).float()
    
    # Compute accuracy
    accuracy = correct.sum() / len(labels)
    
    return accuracy.item() * 100


import torch.distributed as dist

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def min_max_normalize(features):
    # Find the min and max across the second and third dimensions (document and feature dimensions)
    min_vals = features.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]  # Shape: (batch_size, 1, 1)
    max_vals = features.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]  # Shape: (batch_size, 1, 1)
    
    normalized_features = (features - min_vals) / (max_vals - min_vals + 1e-8)  # Add small value to avoid division by zero
    return normalized_features

# Method 2: Z-Score Normalization for a 3D tensor
def z_score_normalize(features):
    # Compute the mean and standard deviation across the document and feature dimensions
    mean_vals = features.mean(dim=(1, 2), keepdim=True)  # Shape: (batch_size, 1, 1)
    std_vals = features.std(dim=(1, 2), keepdim=True)    # Shape: (batch_size, 1, 1)
    
    standardized_features = (features - mean_vals) / (std_vals + 1e-8)  # Add small value to avoid division by zero
    return standardized_features