import torch
import argparse
import glob
import os
import pdb
import numpy as np

import torch.distributed as dist
from PIL import Image
import torch.nn.functional as F

from numbers import Number
from typing import Any, Callable
from torch import Tensor

from typing import Dict
import pandas as pd

def infer_ultr(pos_idx, device, model='ips'):
    df = pd.read_csv("/ubc/cs/home/g/gbhatt/borg/ranking/CF_ranking/bbm/propensities/global_all_pairs.csv")
    model = torch.zeros(500, dtype=torch.float64).to(device)
    positions = df["position"].values
    propensities = torch.tensor(df.iloc[:, 1].values).to(device)

    model[positions] = propensities
    examination = model[pos_idx]

    return examination

def binary_accuracy(logits, labels, threshold=0.5, soft=False):
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)
    
    # Apply threshold to get predictions (0 or 1)
    preds = (probs >= threshold).float()
    
    # Compare predictions with the true labels
    if soft:
        correct = torch.abs(preds - labels) < 0.5
        accuracy = correct.float().mean()
    else:
        correct = (preds == labels).float()
        accuracy = correct.sum() / len(labels)
    
    return accuracy.item() * 100

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def soft_sort_group_parallel(
    s: Tensor, temperature: Number = 1.0, power: Number = 1.0, dummy_indices: Tensor = None
):
    LARGE_NEG_SOFTMAX_INPUT = -(6.5 * 10**4)

    if dummy_indices is None:
        dummy_indices = torch.zeros_like(s, dtype=bool)

    # Set dummy_indices in clone of s smaller than min
    s_clone = torch.clone(s)
    s_clone[dummy_indices] = s.min() - 1.0

    # Extract sorted non-dummy indices from s_clone
    # Sort currently does not support bool dtype on CUDA
    nondummy_perm_rows = ~(dummy_indices.float().sort(dim=-1).values.bool())
    s_sorted = s_clone.sort(descending=True, dim=-1).values

    # Expanding dummy_indices shape to match that of permutation matrix
    s_shape = dummy_indices.shape
    dummy_indices_unsqueezed = dummy_indices.unsqueeze(-2).expand(*s_shape, s_shape[-1])

    pairwise_distances = (s.unsqueeze(-2) - s_sorted.unsqueeze(-1)).abs().pow(
        power
    ).neg() / temperature
    # Set pairwise distance to dummy_indices as LARGE_NEG_SOFTMAX_INPUT
    pairwise_distances[dummy_indices_unsqueezed] = LARGE_NEG_SOFTMAX_INPUT

    perm_mat = torch.zeros_like(pairwise_distances)

    # Apply softmax only to non-dummy rows of pairwise distance matrix
    perm_mat[nondummy_perm_rows] = torch.nn.functional.softmax(
        pairwise_distances[nondummy_perm_rows], dim=-1
    )

    return perm_mat


def custom_dcg(relevance, k, mask=None, gain_fn='exp'):
    """
    Calculates the Discounted Cumulative Gain (DCG) at k for a batch of examples with masking support.
    Args:
        relevance (torch.Tensor): Tensor of shape (batch_size, num_items) representing relevance scores.
        k (int): Rank up to which DCG is calculated.
        mask (torch.Tensor): Optional mask of shape (batch_size, num_items) with 1 for valid items and 0 for invalid/padded items.
        gain_fn (str): The gain function to use ('exp' for exponential or 'lin' for linear).
    Returns:
        torch.Tensor: DCG values for each example in the batch.
    """

    k = min(k, relevance.shape[1]) # should be min(top-k, n_items)

    if mask!=None:
        relevance = relevance * mask # padded values are replaced with 0 and hence gain=0 for those

    def gain_function(rel):
        if gain_fn == 'exp':
            return 2 ** rel - 1
        return rel

    # Take top-k relevance scores
    relevance = relevance[:, :k]

    # Compute log denominator
    log_factors = torch.log2(torch.arange(3, k + 2, device=relevance.device, dtype=relevance.dtype))

    # Compute DCG
    gains = gain_function(relevance)

    #pdb.set_trace()
    
    dcg = gains[:, 0] + torch.sum(gains[:, 1:] / log_factors, dim=1)
    return dcg

def custom_ndcg(preds, targets, k, mask=None, gain_fn='lin'):
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) at k for a batch of examples with masking support.
    Args:
        preds (torch.Tensor): Predicted scores of shape (batch_size, num_items).
        target (torch.Tensor): Ground truth relevance scores of shape (batch_size, num_items).
        k (int): Rank up to which NDCG is calculated.
        mask (torch.Tensor): Optional mask of shape (batch_size, num_items) with 1 for valid items and 0 for invalid/padded items.
        gain_fn (str): The gain function to use ('exp' for exponential or 'lin' for linear).
    Returns:
        torch.Tensor: NDCG values for each example in the batch.
    """

    if mask is not None:
        # mask = torch.where(mask == 0, -8e+8, mask)
        # preds = preds*mask
        # targets = targets*mask

        mask_new = torch.where(mask == 0, 8e+8, mask)
        #pdb.set_trace()

        preds = preds + (1-mask_new) # addition masking preserves negative values
        targets = targets + (1-mask_new)

    # Sort relevance scores based on predicted order
    sorted_indices = torch.argsort(preds, descending=True, dim=1)
    relevance = torch.gather(targets, dim=1, index=sorted_indices)

    # Compute ideal DCG by sorting target relevance in descending order
    ideal_relevance = torch.sort(targets, descending=True, dim=1)[0]

    # Calculate DCG and ideal DCG
    actual_dcg = custom_dcg(relevance, k, mask=mask, gain_fn=gain_fn)
    ideal_dcg = custom_dcg(ideal_relevance, k, mask=mask, gain_fn=gain_fn)

    #pdb.set_trace()

    # Handle division by zero (when ideal_dcg is 0)
    #print (actual_dcg, ideal_dcg)
    ndcg_values = actual_dcg / (ideal_dcg + 1e-8)
    ndcg_values[ideal_dcg == 0] = 0.0

    return ndcg_values, actual_dcg

def get_ndcg(preds, targets, k=10, mask=None, gain_fn='exp', return_mean=True, use_dcg=False):

    ndcg, dcg = custom_ndcg(preds=preds, targets=targets, k=k, mask=mask, gain_fn=gain_fn)
    #dcg = get_ndcg_old(preds=preds, targets=targets, k=k, mask=mask, use_rax=True)
    #score = dcg
    #if args:
    if use_dcg:
        score = dcg
    else:
        score = ndcg
    #return score
    #return score.item()
    #pdb.set_trace()

    if return_mean:
        return score.mean().item()
    
    return score

def eval_po(batch, pred_scores, ips_model=None, device=None, args=None):

    relevance_org = torch.tensor(batch['relevance_ips']).to(device)
    #examination_org = torch.tensor(batch['examination_ips']).to(device)

    # mask = 1.0*torch.tensor(batch['mask']).to(device)
    # pred_scores_mask = pred_scores * mask
    # pred_scores_padded = torch.where(pred_scores_mask==0, -8e+8, pred_scores_mask)

    mask = torch.tensor(batch['mask']).to(device)

    if args.ips_ideal:
        pred_scores_padded = torch.where(~mask, -8e+8, relevance_org)
    else:
        pred_scores_padded = torch.where(~mask, -8e+8, pred_scores)
    
    sorted_indices = torch.argsort(pred_scores_padded, dim=1, descending=True)
    new_positions = torch.empty_like(sorted_indices).to(device)

    new_positions.scatter_(1, sorted_indices, torch.arange(pred_scores.size(1)).unsqueeze(0).expand_as(pred_scores).to(device))
    
    if args.ips_production:
        examination = infer_ultr(pos_idx=batch['position'], device=device)
    else:
        examination = infer_ultr(pos_idx=1+new_positions, device=device) # positions should start from 1 as exam[0] = 0

    relevance = relevance_org * mask
    examination = examination * mask

    prob_click = examination * torch.sigmoid(relevance)
    prob_noclick = torch.prod(1-prob_click, dim=1)
    prob_atleast_1click = 1 - prob_noclick

    if args.ips_rel:
        relevance_org = torch.tensor(batch['relevance_ips']).to(device)
        score = get_ndcg(preds=torch.sigmoid(pred_scores), targets=torch.sigmoid(relevance_org), 
                        mask=1.0*mask, use_dcg=False)
    else:
        clicks = torch.tensor(batch['click'], dtype=torch.float).to(device)

        score = get_ndcg(preds=torch.sigmoid(pred_scores), targets=clicks, 
                        mask=1.0*mask, use_dcg=False)

    return prob_atleast_1click.mean(), score