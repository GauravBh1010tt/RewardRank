from numbers import Number
import warnings
import torch
import torch.nn.functional as F
from functools import partial
import numpy as np

from src.loss_utils import (
    DISCOUNT_FN,
    EXP2,
    GAIN_FN,
    LARGE_NEG_SOFTMAX_INPUT,
    LINEAR,
    LOG2,
    NEURAL_SORT,
    RELAXED_GROUP_PARALLEL_SORT_FN,
    RELAXED_SORT_FN,
    compute_max_dcg,
    hard_sort,
    hard_sort_group_parallel,
    stochastic_neural_sort,
    deterministic_neural_sort,
    sinkhorn_scaling,
)


PRED = "pred"
TARGET = "target"
DEFAULT_EPS=1e-10
PADDED_Y_VALUE=-1


class PiRank_Loss(torch.nn.Module):

    def __init__(
        self,
        sort_func: str = NEURAL_SORT,
        gain_fn: str = EXP2,
        rank_discount_fn: str = LOG2,
        k: int = 8,
        ste: bool = False,
        epsilon: float = 1.0e-10,
        **sort_kwargs,
    ):
        
        super().__init__()

        if "temperature" in sort_kwargs:
            assert isinstance(
                sort_kwargs["temperature"], Number
            ), "'temperature' must be numerical."
        if "power" in sort_kwargs:
            assert isinstance(sort_kwargs["power"], Number), "'power' must be numerical."
        assert (k is None) or isinstance(k, int), "'k' must be int or None."
        if not gain_fn in GAIN_FN:
            raise ValueError(f"Unknown gain function: `{gain_fn}` is not in {list(GAIN_FN.keys())}")
        if not rank_discount_fn in DISCOUNT_FN:
            raise ValueError(
                f"Unknown rank discount function: `{rank_discount_fn}` is not in {list(DISCOUNT_FN.keys())}"
            )
        if not sort_func in RELAXED_GROUP_PARALLEL_SORT_FN:
            raise ValueError(
                f"Unknown differentiable sort function: `{sort_func}` is not in {list(RELAXED_GROUP_PARALLEL_SORT_FN.keys())}"
            )

        self.sort_func = RELAXED_GROUP_PARALLEL_SORT_FN[sort_func]
        self.gain_fn = GAIN_FN[gain_fn]
        self.rank_discount_fn = DISCOUNT_FN[rank_discount_fn]
        self.sort_kwargs = sort_kwargs
        self.k = k
        self.ste = ste

        assert (
            type(epsilon) == float
        ), f"epsilon must be a float, but given epsilon={epsilon} is of type={type(epsilon)} given"
        assert epsilon > 0, f"epsilon must be positive, but epsilon={epsilon}"
        self.epsilon = epsilon

        self._pad_values = self._initialize_pad_values()

    def _initialize_pad_values(self):
        pad_values = {PRED: LARGE_NEG_SOFTMAX_INPUT}
        if self.gain_fn in [GAIN_FN[LINEAR], GAIN_FN[EXP2]]:
            pad_values[TARGET] = 0.0
        else:
            raise ValueError(f"No pad value defined for gain_fn={self.gain_fn}!")
        return pad_values

    @property
    def pad_values(self):
        return self._pad_values

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, dummy_indices: torch.Tensor = None, device=None,
    ) -> torch.Tensor:
        
        if dummy_indices is None:
            dummy_indices = torch.zeros_like(pred, dtype=bool)

        if (
            self.ste
        ):  # soft and hard permutation matrices for backward and forward passes, respectively
            perm_mat_backward = self.sort_func(
                pred, **self.sort_kwargs, dummy_indices=dummy_indices
            )
            # this step requires pred dummy_indexes to be padded with very small number
            perm_mat_forward = hard_sort_group_parallel(pred, dummy_indices=dummy_indices)
            perm_mat = perm_mat_backward + (perm_mat_forward - perm_mat_backward).detach()
        else:  # soft permutation matrix
            perm_mat = self.sort_func(pred, **self.sort_kwargs, dummy_indices=dummy_indices)

        # form the gain vector
        gains = self.gain_fn(target).unsqueeze(-1)
        gains_sorted = torch.matmul(perm_mat, gains).squeeze(-1)  # n

        ## compute the DCG score
        # Sort currently does not support bool dtype on CUDA
        nondummy_perm_rows = ~(dummy_indices.float().sort(dim=-1).values.bool())
        I = (nondummy_perm_rows).cumsum(dim=-1)
        rank_discount = self.rank_discount_fn(I)
        rank_discount[~nondummy_perm_rows] = 0.0
        dg = gains_sorted * rank_discount  # [n * n] = [n]

        dg_cumsum = dg.cumsum(dim=-1)
        dcg = dg_cumsum[..., -1]
        if self.k is not None:
            # Handle groups with less than k nondummy indices separately
            large_list = I[..., -1] > self.k
            dcg[large_list] = dg_cumsum[large_list][I[large_list] == self.k]

        # compute the max DCG score
        # requires target dummy_indexes to be padded with 0.0
        max_dcg = compute_max_dcg(
            target,
            gain_fn=self.gain_fn,
            rank_discount_fn=self.rank_discount_fn,
            k=self.k,
            dummy_indices=dummy_indices,
        )

        # avoiding division-by-zero without losing the gradient
        ndcg = dcg / (self.epsilon + max_dcg)

        return 1.0 - ndcg  # turning gain into loss

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
            
        with torch.no_grad():
        
            out_counterfact = reward_mod(inputs_embeds=inputs_embeds.repeat(count,1,1), position_ids=torch.concat(aggr_pos_idx), 
                        labels_click = labels_click.repeat(count,1), attention_mask=attention_mask.repeat(count,1))

        u_pi_ = utility_metric(labels.repeat(count,1), torch.stack(out_factual['per_item_logits']).repeat(1,count,1),
                               torch.stack(out_counterfact['per_item_logits']), mask.repeat(count,1))

        
        temp_loss = (u_pi_ - u_pi.repeat(count).unsqueeze(1))*torch.log(1+torch.exp(-1*(torch.sigmoid(out_factual['logits'].repeat(count,1)) - torch.sigmoid(out_counterfact['logits']))))
        loss += temp_loss.sum()
        
    return loss


def segment_softmax(scores: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    """
    Softmax over grouped segments (e.g., each query).
    scores: [N]
    segments: [N] — segment/group ID per score
    """
    out = torch.empty_like(scores)
    for seg_id in segments.unique():
        mask = segments == seg_id
        seg_scores = scores[mask]
        out[mask] = F.softmax(seg_scores, dim=0)
    return out


def first_item_segment_mask(segments: torch.Tensor) -> torch.Tensor:
    """
    Returns a boolean mask [N] indicating the first item in each segment.
    Assumes sorted segments.
    """
    return torch.cat([torch.tensor([True], device=segments.device),
                      segments[1:] != segments[:-1]])


def convert_query_ids_to_segment_ids(query_ids_np: np.ndarray, device=None) -> torch.Tensor:
    """
    Convert a numpy array of string query IDs to unique integer segment IDs as a PyTorch tensor.
    Returns a tensor of shape [N] with contiguous integer segment IDs.
    """
    # Convert string array to list of strings
    query_ids_str = query_ids_np.tolist()

    # Map each unique string to an integer
    unique_ids = list(sorted(set(query_ids_str)))
    id_to_segment = {qid: i for i, qid in enumerate(unique_ids)}

    # Map all query_ids to segment IDs
    segment_ids = [id_to_segment[qid] for qid in query_ids_str]
    return torch.tensor(segment_ids, dtype=torch.long, device=device)

def normalize_probabilities(scores: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    """
    Normalize scores within each segment to form probability distributions.
    """
    probs = torch.zeros_like(scores)
    for seg_id in segments.unique():
        mask = segments == seg_id
        seg_scores = scores[mask]
        total = seg_scores.sum()
        if total > 0:
            probs[mask] = seg_scores / total
    return probs

def listwise_softmax_ips(
    relevance_scores: torch.Tensor,     # shape [B, N]
    examination_probs: torch.Tensor,    # shape [B, N]
    clicks: torch.Tensor,               # shape [B, N]
    query_ids,            # shape [B, N]; each row shares same ID
    dummy_indices: torch.Tensor,        # shape [B, N]; 1 = pad, 0 = valid
    max_weight: float = 10.0,
    device=None,
) -> torch.Tensor:
    """
    PyTorch version of listwise_softmax_ips using segment-aware softmax and normalization.

    Flattens inputs to [B*N] and uses query_ids as segments.
    """
    B, N = relevance_scores.shape
    mask = (dummy_indices == 0)

    query_ids = convert_query_ids_to_segment_ids(query_ids, device=device)

    # Flatten all [B, N] → [B*N]
    flat_scores = relevance_scores.reshape(-1)
    flat_exam = examination_probs.reshape(-1)
    flat_clicks = clicks.reshape(-1)
    flat_mask = mask.reshape(-1)
    flat_qids = query_ids.reshape(-1)

    # Apply mask: zero out padded elements
    flat_exam = flat_exam * flat_mask
    flat_clicks = flat_clicks * flat_mask

    # Step 1: IPS weights = 1 / examination
    weights = 1.0 / (flat_exam + 1e-9)
    weights = torch.clamp(weights, max=max_weight)

    # Step 2: Label = weight * click
    labels = weights * flat_clicks

    # Step 3: Normalize labels per segment
    norm_labels = normalize_probabilities(labels, flat_qids)

    # Step 4: Softmax over scores (log-softmax used in cross-entropy)
    log_softmax_scores = torch.zeros_like(flat_scores)
    for qid in flat_qids.unique():
        mask = (flat_qids == qid)
        scores_q = flat_scores[mask]
        log_softmax_scores[mask] = F.log_softmax(scores_q, dim=0)

    # Step 5: Cross entropy: - sum p_i * log q_i
    loss = -norm_labels * log_softmax_scores
    loss = loss[flat_mask.bool()]  # apply final mask
    return loss.mean()

def pointwise_sigmoid_ips(
    relevance_scores: torch.Tensor,      # shape [B, N]
    examination_probs: torch.Tensor,     # shape [B, N]
    clicks: torch.Tensor,                # shape [B, N]
    max_weight: float = 10.0,
    eps: float = 1e-9,
    query_ids=None,
    device=None,
    dummy_indices: torch.Tensor = None,  # shape [B, N]; 1=pad, 0=item
) -> torch.Tensor:
    """
    Batched Pointwise IPS loss with masking based on dummy indices.

    Args:
        relevance_scores: Predicted logits, shape [B, N]
        examination_probs: Propensity scores, shape [B, N]
        clicks: Binary click labels, shape [B, N]
        max_weight: Max cap for inverse propensity weight
        eps: Small constant for numerical stability
        dummy_indices: Tensor of shape [B, N], where 1 = pad (ignored), 0 = valid item

    Returns:
        Scalar tensor: mean IPS loss over valid items.
    """

    # Compute inverse propensity weights
    weights = 1.0 / (examination_probs + eps)
    weights = torch.clamp(weights, min=0.0, max=max_weight)

    # Apply weights to clicks
    labels = weights * clicks.float()  # shape [B, N]

    # Predicted probabilities
    probs = torch.sigmoid(relevance_scores)  # shape [B, N]

    # BCE-style log terms
    log_p = torch.log(torch.clamp(probs, min=eps))
    log_not_p = torch.log(torch.clamp(1.0 - probs, min=eps))

    # Pointwise IPS loss per item
    loss = -labels * log_p - (1.0 - labels) * log_not_p  # shape [B, N]

    # Mask out padded items using dummy_indices
    if dummy_indices is not None:
        mask = (dummy_indices == 0)  # shape [B, N]; True for valid
        loss = loss * mask.float()
        total_valid = mask.float().sum()
        return loss.sum() / (total_valid + eps)
    else:
        return loss.mean()

def listwise_softmax_loss(pred_scores, true_relevance, device=None):
    """
    pred_scores: shape [batch_size, list_size]
    true_relevance: same shape, ideally using rank-based gains
    """
    pred_prob = F.log_softmax(pred_scores, dim=1)
    true_prob = F.softmax(true_relevance, dim=1)
    return F.kl_div(pred_prob, true_prob, reduction='batchmean')

def listMLE(y_pred, y_true, dummy_indices, eps=1e-9, device=None):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]
    mask = dummy_indices
    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))

def listNet(y_pred, y_true, dummy_indices, eps=1e-9, device=None):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    mask = dummy_indices
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


def lambdaLoss(y_pred, y_true, dummy_indices, eps=1e-9, device=None, weighing_scheme='lambdaRank_scheme', k=None, sigma=1., mu=10.,
               reduction="mean", reduction_log="binary"):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """
    #weighing_scheme = lambdaRank_scheme
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = dummy_indices
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.
    else:
        weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))