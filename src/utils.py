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


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=-1):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


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


def get_ndcg_old(preds, targets, k=10, return_dcg=False, use_rax=False, mask=None, args=None):

    if use_rax:
        import jax.numpy as jnp
        import rax
        preds = jnp.array(preds)
        targets = jnp.array(targets)

        score = rax.dcg_metric(preds, targets, topn=k, where=mask)
        return score.item()
    
    if args!=None:
        if args.ultr_models:
            ndcg = RetrievalNormalizedDCG(top_k=k, ignore_index=0) # ignore padded index for targets: 0 is used for padding
        else:
            ndcg = RetrievalNormalizedDCG(top_k=k)
    idx = torch.repeat_interleave(torch.arange(preds.shape[0]), preds.shape[1])

    score = ndcg(preds.contiguous().view(-1), targets.contiguous().view(-1), idx)

    return score.item()

def get_ndcg(preds, targets, k=10, mask=None, args=None, return_mean=True):

    ndcg, dcg = custom_ndcg(preds=preds, targets=targets, k=k, mask=mask, gain_fn=args.gain_fn)
    
    score = ndcg
    if args:
        if args.use_dcg:
            score = dcg
    #pdb.set_trace()

    if return_mean:
        return score.mean().item()
    
    return score

def gumbel_softmax_sample(logits, tau=0.1, mask=None):
    """Sample a ranking from Gumbel-Softmax distribution with masking."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    logits = logits + gumbel_noise
    if mask is not None:
        logits = logits * mask + (1 - mask) * -1e9  # Mask invalid positions
    probs = torch.softmax(logits / tau, dim=-1)
    return probs

def eval_ultr(batch, pred_scores, ips_model=None, device=None, args=None):

    relevance_org = torch.tensor(batch['relevance_ips']).to(device)
    #examination_org = torch.tensor(batch['examination_ips']).to(device)

    # mask = 1.0*torch.tensor(batch['mask']).to(device)
    # pred_scores_mask = pred_scores * mask
    # pred_scores_padded = torch.where(pred_scores_mask==0, -8e+8, pred_scores_mask)

    mask = torch.tensor(batch['mask']).to(device)
    pred_scores_padded = torch.where(~mask, -8e+8, pred_scores)

    #pdb.set_trace()
    
    sorted_indices = torch.argsort(pred_scores_padded, dim=1, descending=True)
    new_positions = torch.empty_like(sorted_indices).to(device)

    new_positions.scatter_(1, sorted_indices, torch.arange(pred_scores.size(1)).unsqueeze(0).expand_as(pred_scores).to(device))
    examination = infer_ultr(pos_idx=1+new_positions, device=device) # positions should start from 1 as exam[0] = 0

    relevance = relevance_org * mask
    examination = examination * mask

    prob_click = examination * torch.sigmoid(relevance)
    prob_noclick = torch.prod(1-prob_click, dim=1)
    prob_atleast_1click = 1 - prob_noclick

    score = get_ndcg(preds=torch.sigmoid(pred_scores), targets=torch.sigmoid(relevance_org), mask=1.0*mask, args=args)

    return prob_atleast_1click.mean(), score


def eval_ultr_choice2(batch, pred_scores, ips_model=None, device=None, args=None):

    relevance_org = torch.tensor(batch['relevance_ips']).to(device) # rel ABD
    examination_org = torch.tensor(batch['examination_ips']).to(device) # exam pos_(A), pos_(B), pos_(D) :: pos_(A)<pos_(B)<pos_(D)

    # mask = 1.0*torch.tensor(batch['mask']).to(device)
    # pred_scores_mask = pred_scores * mask
    # pred_scores_padded = torch.where(pred_scores_mask==0, -8e+8, pred_scores_mask)

    mask = torch.tensor(batch['mask']).to(device)
    pred_scores_padded = torch.where(~mask, -8e+8, pred_scores) # 0.5, 0.25, 0.75, &, &, &

    #pdb.set_trace()
    
    sorted_indices = torch.argsort(pred_scores_padded, dim=1, descending=True) # D, A, B, &, &, & ==> 2,0,1,3,4,5
    rel_ranked_items = relevance_org[torch.arange(pred_scores.size(1)).unsqueeze(0).to(device), sorted_indices] # relevance is independent of positions

    # new_positions = torch.empty_like(sorted_indices).to(device)

    # new_positions.scatter_(1, sorted_indices, torch.arange(pred_scores.size(1)).unsqueeze(0).expand_as(pred_scores).to(device))
    # examination_org = infer_ultr(pos_idx=new_positions, device=device) # choice 1: positions starts with 1...n, missing positions are not taken care of

    relevance = rel_ranked_items * mask
    examination = examination_org * mask

    prob_click = examination * torch.sigmoid(relevance)
    prob_noclick = torch.prod(1-prob_click, dim=1)
    prob_atleast_1click = 1 - prob_noclick

    score = get_ndcg(preds=torch.sigmoid(pred_scores), targets=torch.sigmoid(relevance_org), mask=1.0*mask, args=args)

    return prob_atleast_1click.mean(), score


def eval_ultr_ideal(batch, ips_model=None, device=None, args=None):

    relevance_org = torch.tensor(batch['relevance_ips']).to(device)
    pos_idx = 1 + torch.arange(batch['position'].shape[1]).repeat(batch['position'].shape[0],1).to(device)

    examination_ideal = infer_ultr(pos_idx=pos_idx, device=device)
    
    ############ Ideal positions ###################################
    mask = torch.tensor(batch['mask']).to(device)
    relevance_org_padded = torch.where(~mask, -8e+8, relevance_org) # 0.5, 0.25, 0.75, &, &, &

    #pdb.set_trace()
    rel_sorted, rel_sorted_idx = torch.sort(relevance_org_padded, dim=1, descending=True)

    if args.production:
        rel_sorted = relevance_org

    relevance_ideal = rel_sorted * mask
    examination_ideal = examination_ideal * mask    
    
    prob_click = examination_ideal * torch.sigmoid(relevance_ideal)
    prob_noclick = torch.prod(1-prob_click, dim=1)
    prob_atleast_1click = 1 - prob_noclick

    #score = get_ndcg(preds=torch.sigmoid(pred_scores), targets=torch.sigmoid(relevance_org), mask=1.0*mask, args=args)
    score = 0.0

    return prob_atleast_1click.mean(), score

def eval_ultr_ideal_prev(batch, ips_model=None, device=None, args=None):

    relevance_org = torch.tensor(batch['relevance_ips']).to(device)
    
    ############ Ideal positions ###################################
    mask = torch.tensor(batch['mask']).to(device)
    relevance_org_padded = torch.where(~mask, -8e+8, relevance_org) # 0.5, 0.25, 0.75, &, &, &
    
    examination = torch.tensor(batch['examination_ips']).to(device)
    #pdb.set_trace()
    exam_sorted_idx = torch.argsort(examination, dim=1, descending=True)
    rel_sorted_idx = torch.argsort(relevance_org_padded, dim=1, descending=True)
    optimal_pos = torch.zeros_like(relevance_org_padded, dtype=int)
    
    for i in range(examination.shape[0]):
        optimal_pos[i, exam_sorted_idx[i]] = rel_sorted_idx[i]
        
    pred_scores = relevance_org.gather(1,optimal_pos)

    prob_atleast_1click, score = eval_ultr_choice2(batch, pred_scores, ips_model=None, device=None, args=None)

    return prob_atleast_1click, score

def swap_columns(tensor, i, j):
    temp = tensor[:, i].clone()
    tensor[:, i] = tensor[:, j]
    tensor[:, j] = temp
    return tensor

def loss_urcc(logits, labels, device, mask, reward_mod, inputs_embeds, position_ids, doc_feats, 
						   		labels_click, attention_mask, avg_lables):
    #pdb.set_trace()

    def utility_metric(click, f_logit, cf_logit, mask):
        out = torch.div(torch.sigmoid(cf_logit[0]), torch.sigmoid(f_logit[0])) * click[:,0].unsqueeze(1) * (1-mask.int()[:,0].unsqueeze(1))
        for i in range(1,click.shape[1]):
            out += torch.div(torch.sigmoid(cf_logit[i]), torch.sigmoid(f_logit[i])) * click[:,i].unsqueeze(1) * (1-mask.int()[:,i].unsqueeze(1))

        return out

    u_pi = labels * (1-mask.int())
    u_pi = u_pi.sum(dim=1)

    with torch.no_grad():
        out_factual = reward_mod(inputs_embeds=inputs_embeds, position_ids=position_ids, 
	    					   labels_click = labels_click, attention_mask=attention_mask) # TODO: detach gradients
    
    loss = torch.tensor([0.0], requires_grad=True).to(device)

    count = 0
    aggr_pos_idx = []
    

    for i in range(labels.shape[1]-1):
        temp_pos_idx = position_ids.clone()
        count = 0
        aggr_pos_idx = []
        for j in range(i+1,labels.shape[1]):
            temp_pos_idx = swap_columns(temp_pos_idx, i,j)
            aggr_pos_idx.append(temp_pos_idx)
            count+=1
            
            # with torch.no_grad():
            #     out_counterfact = reward_mod(inputs_embeds=inputs_embeds, position_ids=temp_pos_idx, 
            #                 labels_click = labels_click, attention_mask=attention_mask)
            
            # u_pi_ = utility_metric(labels, out_factual, out_counterfact, mask)

            # temp_loss = (u_pi_ - u_pi.unsqueeze(1))*torch.log(1+torch.exp(-1*(torch.sigmoid(out_factual['logits']) - torch.sigmoid(out_counterfact['logits']))))
            # loss += temp_loss.sum()
            
        with torch.no_grad():
        
            out_counterfact = reward_mod(inputs_embeds=inputs_embeds.repeat(count,1,1), position_ids=torch.concat(aggr_pos_idx), 
                        labels_click = labels_click.repeat(count,1), attention_mask=attention_mask.repeat(count,1))

        u_pi_ = utility_metric(labels.repeat(count,1), torch.stack(out_factual['per_item_logits']).repeat(1,count,1),
                               torch.stack(out_counterfact['per_item_logits']), mask.repeat(count,1))

        
        temp_loss = (u_pi_ - u_pi.repeat(count).unsqueeze(1))*torch.log(1+torch.exp(-1*(torch.sigmoid(out_factual['logits'].repeat(count,1)) - torch.sigmoid(out_counterfact['logits']))))
        loss += temp_loss.sum()
        
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rank BERT', parents=[get_args_parser()])
    args = parser.parse_args()

    save_path = os.path.join(args.output_path, args.output_folder, 'figs', '*')
    all_files = glob.glob(save_path)
    all_files.sort()
    
    merge_images(all_files, output_path=os.path.join(args.output_path, args.output_folder,'eval_viz.jpg'),direction='vertical')