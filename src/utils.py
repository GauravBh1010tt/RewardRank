import torch
import argparse
import glob
import os
import pdb
import numpy as np

from PIL import Image
import torch.nn.functional as F

from numbers import Number
from typing import Any, Callable
from torch import Tensor

from src.ultr_models import infer_ultr

# import jax.numpy as jnp
# import rax

from torchmetrics.retrieval import RetrievalNormalizedDCG

LARGE_NEG_SOFTMAX_INPUT = -(6.5 * 10**4)

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--task', default='merge_imgs', type=str)
    parser.add_argument('--output_path', default='/home/ec2-user/workspace/output/', type=str)
    parser.add_argument('--output_folder', default='ultr_ips', type=str)

    return parser

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

def z_score_normalize(features):
    # Compute the mean and standard deviation across the document and feature dimensions
    mean_vals = features.mean(dim=(1, 2), keepdim=True)  # Shape: (batch_size, 1, 1)
    std_vals = features.std(dim=(1, 2), keepdim=True)    # Shape: (batch_size, 1, 1)
    
    standardized_features = (features - mean_vals) / (std_vals + 1e-8)  # Add small value to avoid division by zero
    return standardized_features

def one_hot_binary_batch(batch):
    return torch.stack([1 - batch, batch], dim=1).float()

def distance_prob(prob1, prob2, distance_type='kl'):
    
    if distance_type == 'kl':
        # Ensure non-zero values to avoid log(0) in KL divergence
        prob1 = prob1 + 1e-8
        prob2 = prob2 + 1e-8
        kl_div = F.kl_div(prob1.log(), prob2, reduction='batchmean')  # KL Divergence
        return kl_div
    
    elif distance_type == 'js':
        # Jensen-Shannon Divergence (symmetrized version of KL)
        prob1 = prob1 + 1e-8
        prob2 = prob2 + 1e-8
        m = 0.5 * (prob1 + prob2)
        js_div = 0.5 * (F.kl_div(prob1.log(), m, reduction='batchmean') + F.kl_div(prob2.log(), m, reduction='batchmean'))
        return js_div
    
    # elif distance_type == 'l2':
    #     # L2 Distance (Euclidean distance)
    #     l2_dist = torch.norm(prob1 - prob2, p=2)
    #     return l2_dist
    
    elif distance_type == 'l1':
        # L1 Distance (Manhattan distance)
        l1_dist = torch.abs(prob1 - prob2).mean()
        return l1_dist
    
    elif distance_type == 'tv':
        # prob1 [batch x 2]
        # L1 Distance (Manhattan distance)
        tv = 0.5*torch.abs(prob1 - prob2).sum(dim=1).mean()
        return tv*100

    elif distance_type == 'mse':
        # L1 Distance (Manhattan distance)
        mse = F.mse_loss(prob1, prob2)
        return mse

def merge_images(image_paths, output_path, direction="horizontal"):

    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(i.size for i in images))

    if direction == "horizontal":
        total_width = sum(widths)
        max_height = max(heights)
        new_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for image in images:
            new_image.paste(image, (x_offset, 0))
            x_offset += image.size[0]

    elif direction == "vertical":
        max_width = max(widths)
        total_height = sum(heights)
        new_image = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for image in images:
            new_image.paste(image, (0, y_offset))
            y_offset += image.size[1]

    else:
        raise ValueError("Invalid direction. Choose 'horizontal' or 'vertical'")

    new_image.save(output_path)

def sample_without_replacement_with_prob(delta, pos, click=torch.ones(1)):
    weights = torch.ones_like(pos)
    remaining_idx = []

    if click.sum() == 0:
        #print('here')
        return pos

    if delta*len(pos)<1:
        idx = torch.multinomial(pos, len(pos), replacement=False)
        return pos[idx]
    
    delta_sample_idx = torch.multinomial(pos,int(delta*len(pos)), replacement=False)
    #print ('retaining', pos[delta_sample_idx])
    weights[delta_sample_idx] = 0
    #remaining_idx = len(pos) - int(delta*len(pos))

    for i,j in enumerate(pos):
        if j not in pos[delta_sample_idx]:
            remaining_idx.append(i)
    #print ('perturbing', remaining_idx)
    d_pos = pos.clone()

    for i in remaining_idx:
        # Normalize weights to ensure they sum to 1
        normalized_weights = weights / weights.sum()
        # Sample one index based on normalized weights
        if weights.sum() == 0:
            return d_pos
        
        sampled_index = torch.multinomial(normalized_weights, 1).item()
        #print (i, sampled_index, normalized_weights)
        d_pos[sampled_index] = pos[i]
        # Set the weight of the sampled index to 0 for the next iteration
        weights[sampled_index] = 0

    #print ('there', pos, d_pos)

    return d_pos

def sample_swap(pos, click=torch.ones(1), fn='swap_rand', ultr_mod=None):
    
    def swap(pos, idx, swap_idx=-1):    
        temp = pos[idx].clone()
        pos[idx] = pos[swap_idx]
        pos[swap_idx] = temp
        return pos
    
    if ultr_mod:
        examination, relevance = click[0], click[1]
        prob_click = examination * torch.sigmoid(relevance)
        idx = prob_click.argmax()
    else:
        if click.sum()<1:
            return pos
        indices = (click == 1).nonzero()
        idx = indices[0].item()
    
    if fn == 'swap_rand':
        return swap(pos, idx=0)
    
    if fn=='swap_first_click_bot':
        swap_idx = np.random.randint(len(pos)/2,len(pos))
    elif fn=='swap_first_click_top':
        swap_idx = np.random.randint(0,len(pos)/2)
    elif fn=='swap_first_click_rand':
        swap_idx = torch.multinomial(pos, 1)
    return swap(pos, idx, swap_idx)


def soft_sort_group_parallel(
    s: Tensor, temperature: Number = 1.0, power: Number = 1.0, dummy_indices: Tensor = None
):

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

def dcg(relevance, k):
    """
    Calculates the Discounted Cumulative Gain (DCG) at k.
    """
    relevance = relevance[:k]
    dcg = relevance[0] + torch.sum(relevance[1:] / torch.log2(torch.arange(2, k+1)))
    return dcg

def ndcg(relevance, k, ideal_relevance=None):
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) at k.
    """
    if ideal_relevance is None:
        ideal_relevance = torch.sort(relevance, descending=True)[0]
    
    actual_dcg = dcg(relevance, k)
    ideal_dcg = dcg(ideal_relevance, k)
    
    if ideal_dcg == 0:
        return 0.0
    else:
        return actual_dcg / ideal_dcg
    

def ndcg_all(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=-1,
         filler_value=1.0):
    """
    Normalized Discounted Cumulative Gain at k.

    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param filler_value: a filler NDCG value to use when there are no relevant items in listing
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = filler_value  # if idcg == 0 , set ndcg to filler_value

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=-1):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def dcg_all(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=-1):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def mrr_all(y_pred, y_true, ats=None, padding_indicator=-1):
    """
    Mean Reciprocal Rank at k.

    Compute MRR at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: MRR values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    values, indices = torch.max(true_sorted_by_preds, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t().expand(len(y_true), len(ats))

    ats_rep = torch.tensor(data=ats, device=indices.device, dtype=torch.float32).expand(len(y_true), len(ats))

    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0

    result = result * within_at_mask

    return result

def get_ndcg(preds, targets, k=10, return_dcg=False, use_rax=False, mask=None):

    if use_rax:
        import jax.numpy as jnp
        import rax
        preds = jnp.array(preds)
        targets = jnp.array(targets)

        score = rax.dcg_metric(preds, targets, topn=k, where=mask)
        return score.item()
    
    ndcg = RetrievalNormalizedDCG(top_k=k, return_dcg=return_dcg, ignore_index=0) # ignore padded index for targets: 0 is used for padding
    idx = torch.repeat_interleave(torch.arange(preds.shape[0]), preds.shape[1])

    score = ndcg(preds.contiguous().view(-1), targets.contiguous().view(-1), idx)

    return score.item()


def eval_ultr(batch, pred_scores, ips_model=None, device=None):
    row,col = batch['tokens'].shape[0], batch['tokens'].shape[1]

    pred_scores = pred_scores * torch.tensor(batch['mask']).to(device)

    pred_scores_padded = torch.where(pred_scores==0, -8e+8, pred_scores)
    
    sorted_indices = torch.argsort(pred_scores_padded, dim=1, descending=True)
    new_positions = torch.empty_like(sorted_indices).to(device)
    new_positions.scatter_(1, sorted_indices, torch.arange(pred_scores.size(1)).unsqueeze(0).expand_as(pred_scores).to(device))

    #new_positions = np.array(new_positions.detach().cpu())

    #pdb.set_trace()
    
    # new_batch = {'tokens':batch['tokens'].reshape(row*col,128),
    #         'attention_mask':batch['attention_mask'].reshape(row*col,128),
    #         'token_types':batch['token_types'].reshape(row*col,128),
    #         'positions':new_positions.reshape(row*col)}
    
    # out_c = ips_model(new_batch) #TODO:use propensities for inference
    # exam = np.array(out_c.examination).reshape(row,col)
    # rel = np.array(out_c.relevance).reshape(row,col)

    # relevance = torch.tensor(rel).to(device)
    # examination = torch.tensor(exam).to(device)

    relevance = torch.tensor(batch['relevance_ips']).to(device)
    examination = infer_ultr(pos_idx=new_positions, device=device)
    #examination = torch.tensor(batch['examination_ips']).to(device)

    relevance = relevance * torch.tensor(batch['mask']).to(device)
    examination = examination * torch.tensor(batch['mask']).to(device)

    prob_click = examination * torch.sigmoid(relevance)
    prob_noclick = torch.prod(1-prob_click, dim=1)
    prob_atleast_1click = 1 - prob_noclick

    #pdb.set_trace()

    return prob_atleast_1click.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rank BERT', parents=[get_args_parser()])
    args = parser.parse_args()

    save_path = os.path.join(args.output_path, args.output_folder, 'figs', '*')
    all_files = glob.glob(save_path)
    all_files.sort()
    
    merge_images(all_files, output_path=os.path.join(args.output_path, args.output_folder,'eval_viz.jpg'),direction='vertical')