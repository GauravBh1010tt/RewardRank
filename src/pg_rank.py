
import torch
import torch.nn as nn
import numpy as np
import pdb

from src.utils import get_ndcg
import torch.nn.functional as F

def gumbel_softmax_sample_v1(logits, mask, tau=0.1, hard=True):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))  # Gumbel(0,1) noise
    gumbel_logits = logits + gumbel_noise  # Add noise

    gumbel_logits = gumbel_logits.masked_fill(mask == 0, float('-inf'))  
    probs = F.gumbel_softmax(gumbel_logits, tau=tau, hard=hard, dim=-1)  # Shape: (batch_size, n_docs)
    ranking = torch.argsort(probs, descending=True, dim=-1)

    return ranking

def plackett_luce_log_prob_v1(logits, ranking, mask):
    batch_size, n_docs = logits.shape
    log_probs = torch.zeros(batch_size, device=logits.device)  # Initialize log_probs
    
    for i in range(n_docs):
        valid_mask = mask.gather(1, ranking[:, i:i+1]).squeeze(-1)  # Get mask for ranked docs
        logit_i = logits.gather(1, ranking[:, i:i+1]).squeeze(-1)  # Get scores for ranked positions
        denominator = torch.logsumexp(logits.gather(1, ranking[:, i:]), dim=-1)  # Log sum exp denominator

        # Apply mask to valid values, skipping invalid ones
        log_probs += (logit_i - denominator) * valid_mask  # Multiply by mask to ignore invalid terms
    
    return log_probs

########################## v2: Based on Tesi's implementation ############################################

def gumbel_softmax_sample_v2(logits: torch.Tensor, mask, num_samples: int = 1, temp: float = 0.1):
    
    def clamp_probs(probs, eps=1e-8): # TODO: eps=1e-10, temp=0.1
        return torch.clamp(probs, eps, 1 - eps)

    num_samples = logits.size()[0]
    
    with torch.no_grad():
        logits = logits.expand(num_samples, -1)
        mask = mask.expand(num_samples, -1)
        u = clamp_probs(torch.rand_like(logits))
        z = logits - torch.log(-torch.log(u)) * temp
        z = z.masked_fill(mask == 0, float('-inf'))
        sample = torch.sort(z, descending=True, dim=1)[1].detach()

    return sample

def plackett_luce_log_prob_v2(logits, ranking, mask):
    ranked_scores = logits.gather(1, ranking)  # (batch_size, n_docs)
    log_cumsum = torch.logcumsumexp(ranked_scores.flip(dims=[1]), dim=1).flip(dims=[1])
    log_probs = ranked_scores - log_cumsum
    log_probs = log_probs * mask

    return log_probs.sum(dim=1)  # Shape: (batch_size,)

############################################################################################################

def compute_pg_rank_loss(scores, targets, args, mask, device=None, reward_mod=None, inputs_embeds=None, position_ids=None, doc_feats=None, 
						   		labels_click=None, attention_mask=None):
    
    mask = (1-mask.int())

    log_probs_samples = []
    rewards = []
    for _ in range(args.MC_samples):
        
        with torch.no_grad():
            sampled_ranking = gumbel_softmax_sample_v2(logits=scores, mask=mask)
        
        log_probs_samples.append(plackett_luce_log_prob_v2(logits=scores, ranking=sampled_ranking.detach(),
                                                        mask=mask))
        
        if reward_mod is None:
            rewards.append(get_ndcg(sampled_ranking, targets, mask=mask, return_mean=False, args=args))  # Compute reward
        else:
            with torch.no_grad():
                out = reward_mod(inputs_embeds=inputs_embeds.detach(), position_ids=sampled_ranking.detach(), 
                                labels_click = labels_click.detach(), attention_mask=attention_mask.detach())
                reward_sample = torch.sigmoid(out['logits']).squeeze()
                rewards.append(reward_sample.detach())

    rewards = torch.stack(rewards)

    # Compute variance-reduced loss for all MC samples
    # N: number of MC samples
    # B: batch size
    # log_probs_samples : N*B
    # rewards: N*B

    batch_loss = 0
    for i, log_probs in enumerate(log_probs_samples):
        baseline = (torch.sum(rewards, dim=0) - rewards[i]) / max(1, args.MC_samples - 1)  # Leave-one-out baseline, computed for each query across the MC samples
        
        if args.pgrank_nobaseline:
            baseline = 0.0

        advantage = rewards[i] - baseline
        batch_loss += -(log_probs * advantage).sum()
        
    batch_loss /= args.MC_samples  # Average over samples
    
    return batch_loss