import torch
import torch.nn as nn
import numpy as np
import pdb
import copy

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


def compute_grpo_loss(cur_scores, old_scores, ref_scores, targets, args, mask, device=None,
                      reward_mod=None, inputs_embeds=None, burnout_period=False,
                      labels_click=None, attention_mask=None, step=None):
    """
    Computes the GRPO loss using sampled rankings.

    Args:
        cur_scores (torch.Tensor): Predicted relevance scores from current policy.
        old_scores (torch.Tensor): Predicted relevance scores from old policy: policy before latest gradient update.
        ref_scores (torch.Tensor): Predicted relevance scores from ref policy: policy from last checkpoint.
        burnout_period (torch.Bool): Before 1st epoch the reference policy is discarded; KL is set after 1st epoch
        targets (torch.Tensor): Ground truth relevance scores or binary labels.
        args: argparse.Namespace with training hyperparameters (e.g., MC_samples).
        mask (torch.Tensor): Binary mask over documents.
        device (torch.device): Unused, reserved for compatibility.
        reward_mod (callable): Optional reward model to compute sample-specific rewards.
        inputs_embeds, position_ids, doc_feats, labels_click, attention_mask: Optional inputs for reward model.

    Returns:
        torch.Tensor: Scalar loss.
    """
    mask = (1 - mask.int())
    B, K = cur_scores.shape

    log_probs_curr = []   # [R, B]
    log_probs_old  = []   # [R, B]
    log_probs_ref  = []   # [R, B] (optional)
    rewards_list   = []   # [R, B]

    eps_std = 1e-8
    logratio_clip = 20.0     # exp(±20) ~ 4.85e8, prevents inf
    adv_clip = 5.0           # cap extreme advantages

    for _ in range(args.grpo_rollouts):
        # --- Sample actions from OLD policy (behavior) ---
        with torch.no_grad():
            sampled_ranking_old = gumbel_softmax_sample(old_scores, mask)             # actions ~ π_old
            old_log_pi = plackett_luce_log_prob(old_scores, sampled_ranking_old, mask)  # [B]
            if not burnout_period and ref_scores is not None:
                ref_log_pi = plackett_luce_log_prob(ref_scores, sampled_ranking_old, mask)  # [B]
            else:
                ref_log_pi = None

            # Rewards on OLD samples only
            if reward_mod is None:
                r = get_ndcg(sampled_ranking_old, targets, mask=mask, return_mean=False)  # [B]
            else:
                out = reward_mod(
                    inputs_embeds=(inputs_embeds.detach() if inputs_embeds is not None else None),
                    position_ids=sampled_ranking_old,    # rewards on old-policy actions
                    labels_click=(labels_click.detach() if labels_click is not None else None),
                    attention_mask=(attention_mask.detach() if attention_mask is not None else None),
                )
                r = torch.sigmoid(out['logits']).squeeze()  # [B]

        # --- Evaluate CURRENT policy on the SAME actions (WITH grad) ---
        cur_log_pi = plackett_luce_log_prob(cur_scores, sampled_ranking_old.detach(), mask)  # [B], requires_grad

        log_probs_old.append(old_log_pi)
        log_probs_curr.append(cur_log_pi)
        rewards_list.append(r)
        if ref_log_pi is not None:
            log_probs_ref.append(ref_log_pi)

    # Stack to [R, B] then transpose to [B, R]
    rewards    = torch.stack(rewards_list, dim=0).transpose(0, 1)         # [B, R]
    old_log_pi = torch.stack(log_probs_old,  dim=0).transpose(0, 1)       # [B, R]
    cur_log_pi = torch.stack(log_probs_curr, dim=0).transpose(0, 1)       # [B, R]
    if len(log_probs_ref) > 0:
        ref_log_pi = torch.stack(log_probs_ref, dim=0).transpose(0, 1)    # [B, R]
    else:
        ref_log_pi = None

    # Group-relative advantages per query (over R)
    #pdb.set_trace()
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r  = rewards.std(dim=1, keepdim=True, unbiased=False)
    adv = (rewards - mean_r) / (std_r + eps_std)                           # [B, R]
    adv = torch.clamp(adv, min=-adv_clip, max=adv_clip)                    # tame outliers

    # Importance ratios with log-ratio clamp
    log_ratio = (cur_log_pi - old_log_pi).clamp(min=-logratio_clip, max=logratio_clip)  # [B, R]
    ratio = torch.exp(log_ratio)
    ratio_clipped = torch.clamp(ratio, 1.0 - args.grpo_eps, 1.0 + args.grpo_eps)

    # PPO surrogate (stop grad through advantage baseline)
    # (adv is fine to keep as tensor; no gradient path to rewards anyway)
    surrogate = torch.minimum(ratio * adv, ratio_clipped * adv).mean()

    # Optional KL-to-reference on the SAME actions (simple surrogate)
    if ref_log_pi is not None:
        # Penalize moving away from ref: E[log πθ - log πref]
        kl_surr = (cur_log_pi - ref_log_pi).mean()
        kl_term = args.grpo_beta * kl_surr
    else:
        kl_term = cur_log_pi.new_zeros(())

    loss = -(surrogate - kl_term)
    return loss