import torch
import torch.nn as nn
import numpy as np
import pdb

from src.utils import get_ndcg
import torch.nn.functional as F

def gumbel_softmax_sample(logits: torch.Tensor, mask, num_samples: int = 1, temp: float = 0.1):
    """
    Samples rankings using the Gumbel-Softmax trick with masking.

    Args:
        logits (torch.Tensor): Logits over items, shape (batch_size, n_docs).
        mask (torch.Tensor): Binary mask indicating valid documents.
        num_samples (int): Number of samples to draw. (Overwritten by logits.size(0))
        temp (float): Temperature for Gumbel noise.

    Returns:
        torch.Tensor: Sampled rankings (indices), shape (batch_size, n_docs).
    """
    def clamp_probs(probs, eps=1e-8):
        # Prevent log(0) during Gumbel noise generation
        return torch.clamp(probs, eps, 1 - eps)

    num_samples = logits.size()[0]  # Override with batch size

    with torch.no_grad():
        logits = logits.expand(num_samples, -1)
        mask = mask.expand(num_samples, -1)

        # Sample Gumbel noise
        u = clamp_probs(torch.rand_like(logits))
        z = logits - torch.log(-torch.log(u)) * temp

        # Apply mask (mask out padded positions)
        z = z.masked_fill(mask == 0, float('-inf'))

        # Sample ranking by sorting Gumbel-perturbed logits
        sample = torch.sort(z, descending=True, dim=1)[1].detach()

    return sample

def plackett_luce_log_prob(logits, ranking, mask):
    """
    Computes the log-likelihood under the Plackett-Luce model for a given ranking.

    Args:
        logits (torch.Tensor): Predicted scores, shape (batch_size, n_docs).
        ranking (torch.Tensor): Permutation indices, shape (batch_size, n_docs).
        mask (torch.Tensor): Binary mask, shape (batch_size, n_docs).

    Returns:
        torch.Tensor: Log-likelihood, shape (batch_size,)
    """
    ranked_scores = logits.gather(1, ranking)  # Get scores in ranked order
    log_cumsum = torch.logcumsumexp(ranked_scores.flip(dims=[1]), dim=1).flip(dims=[1])
    log_probs = ranked_scores - log_cumsum  # Compute log P(rank[i] | rest)
    log_probs = log_probs * mask  # Mask out invalid positions
    return log_probs.sum(dim=1)  # Sum over ranks

def compute_pg_rank_loss(scores, targets, args, mask, device=None, reward_mod=None, inputs_embeds=None, position_ids=None, doc_feats=None, 
                         labels_click=None, attention_mask=None):
    """
    Computes the policy gradient ranking loss using sampled rankings.

    Args:
        scores (torch.Tensor): Predicted relevance scores.
        targets (torch.Tensor): Ground truth relevance scores or binary labels.
        args: argparse.Namespace with training hyperparameters (e.g., MC_samples).
        mask (torch.Tensor): Binary mask over documents.
        device (torch.device): Unused, reserved for compatibility.
        reward_mod (callable): Optional reward model to compute sample-specific rewards.
        inputs_embeds, position_ids, doc_feats, labels_click, attention_mask: Optional inputs for reward model.

    Returns:
        torch.Tensor: Scalar loss.
    """
    mask = (1 - mask.int())  # Invert mask for internal usage

    log_probs_samples = []  # Stores log-likelihoods for each MC sample
    rewards = []  # Stores corresponding rewards

    for _ in range(args.MC_samples):
        with torch.no_grad():
            # Sample a ranking using Gumbel-Softmax
            sampled_ranking = gumbel_softmax_sample(logits=scores, mask=mask)

        # Compute log-likelihood of sampled ranking under Plackett-Luce
        log_probs_samples.append(plackett_luce_log_prob(
            logits=scores, ranking=sampled_ranking.detach(), mask=mask
        ))

        # Compute reward (either NDCG or learned reward model)
        if reward_mod is None:
            rewards.append(get_ndcg(sampled_ranking, targets, mask=mask, return_mean=False))
        else:
            with torch.no_grad():
                out = reward_mod(inputs_embeds=inputs_embeds.detach(),
                                 position_ids=sampled_ranking.detach(),
                                 labels_click=labels_click.detach(),
                                 attention_mask=attention_mask.detach())
                reward_sample = torch.sigmoid(out['logits']).squeeze()
                rewards.append(reward_sample.detach())

    rewards = torch.stack(rewards)  # Shape: (MC_samples, batch_size)

    # Compute variance-reduced REINFORCE loss using leave-one-out baseline
    batch_loss = 0
    for i, log_probs in enumerate(log_probs_samples):
        baseline = (torch.sum(rewards, dim=0) - rewards[i]) / max(1, args.MC_samples - 1)
        
        if args.pgrank_nobaseline:
            baseline = 0.0  # No baseline used if flag is set

        advantage = rewards[i] - baseline
        batch_loss += -(log_probs * advantage).sum()  # Accumulate loss

    batch_loss /= args.MC_samples  # Average over MC samples
    return batch_loss