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