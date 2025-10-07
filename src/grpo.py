import torch
import copy
from src.utils import get_ndcg

def gumbel_softmax_sample(logits: torch.Tensor, mask, temp: float = 0.1):

    def clamp_probs(probs, eps=1e-8):
        return torch.clamp(probs, eps, 1 - eps)

    B, K = logits.shape
    with torch.no_grad():
        u = clamp_probs(torch.rand_like(logits))
        z = logits - torch.log(-torch.log(u)) * temp
        z = z.masked_fill(mask == 0, float('-inf'))
        sample = torch.sort(z, descending=True, dim=1)[1].detach()  # [B, K]
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

def compute_grpo_loss(scores, targets, args, mask, device=None,
                      reward_mod=None, inputs_embeds=None,
                      position_ids=None, doc_feats=None,
                      labels_click=None, attention_mask=None,
                      model=None, ref_model=None, step=None):
    """
    Computes GRPO loss (Generalized Reinforce with PPO-style clipping).
    """
    mask = (1 - mask.int())  # invert for consistency with PG-Rank
    B, K = scores.shape

    # Initialize / update reference model
    if step is not None and step % args.ref_update == 1:
        ref_model = copy.deepcopy(model).eval()

    # Sample rankings
    with torch.no_grad():
        sampled_ranking = gumbel_softmax_sample(scores, mask, temp=args.gumbel_temp)
        old_log_pi = plackett_luce_log_prob(scores, sampled_ranking.detach(), mask)

    # Compute rewards
    rewards = []
    if reward_mod is None:
        rewards.append(get_ndcg(sampled_ranking, targets, mask=mask, return_mean=False))
    else:
        with torch.no_grad():
            out = reward_mod(inputs_embeds=inputs_embeds.detach(),
                             position_ids=sampled_ranking.detach(),
                             labels_click=(labels_click.detach() if labels_click is not None else None),
                             attention_mask=(attention_mask.detach() if attention_mask is not None else None))
            reward_sample = torch.sigmoid(out['logits']).squeeze()
            rewards.append(reward_sample.detach())

    rewards = torch.stack(rewards)  # [1, B] in our current sampling
    rewards = rewards.view(B, -1)

    # Normalize rewards to get advantage
    advantage = (rewards - rewards.mean(dim=1, keepdim=True)) / (
        rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-10
    )
    advantage = advantage.view(-1)

    # Reference log prob
    if ref_model is not None:
        with torch.no_grad():
            ref_scores = ref_model(inputs_embeds).squeeze(-1)
            ref_log_pi = plackett_luce_log_prob(ref_scores, sampled_ranking.detach(), mask)
    else:
        ref_log_pi = old_log_pi.detach()

    # Current log prob
    log_pi = plackett_luce_log_prob(scores, sampled_ranking.detach(), mask)

    # PPO-style clipped objective
    coef1 = torch.exp(log_pi - old_log_pi)
    coef2 = torch.clamp(coef1, 1.0 - args.eps, 1.0 + args.eps)
    kl = torch.exp(ref_log_pi - log_pi) - (ref_log_pi - log_pi) - 1.0

    loss = -torch.mean(torch.min(coef1 * advantage, coef2 * advantage) - args.beta * kl)

    avg_reward = rewards.mean()
    return loss, avg_reward.item(), ref_model