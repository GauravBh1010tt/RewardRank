from numbers import Number
from typing import Any, Callable

import torch
from torch import Tensor

# Gain and Discount functions
EXP2 = "exp2"
LINEAR = "linear"
LOG2 = "log2"

# Normalizations
L1 = "l1"
NONE = "none"
SOFTMAX = "softmax"

# PiRank loss constants

# For fp16 precision, we use a value close to the smallest representable number in fp16
# which is approximately -65504. Using a slightly smaller value to ensure safety.
# This value will support all precisions, including fp16, bf16 and fp32.
LARGE_NEG_SOFTMAX_INPUT = -(6.5 * 10**4)
NEURAL_SORT = "neural_sort"
SOFT_SORT = "soft_sort"

# Reductions
MEAN = "mean"
REDUCTION = "reduction"
SUM = "sum"

# Three-part Hinge loss defaults
NEGATIVE = 3
NEUTRAL = 2
POSITIVE = 1


def identity(x: Any) -> Any:
    return x


def noop_normalization(x: Tensor) -> Tensor:
    """
    Identity function with an additional check of non-negativity of the entries.
    """
    assert torch.all(x >= 0), "The values must be non-negative in no-op normalization."
    return x


def softmax_1d(x: Tensor) -> Tensor:
    """
    Softmax normalization in the last dimension
    """
    return torch.nn.functional.softmax(x, dim=-1)


def l1_normalization_1d(x: Tensor) -> Tensor:
    """
    L1 normalization in the last dimension
    """
    assert torch.all(x >= 0), "The values must be non-negative in L1 normalization."
    return torch.nn.functional.normalize(x, p=1, dim=-1)


def exp2(x: Tensor) -> Tensor:
    """
    Exponential gains used for a ranking metric e.g. NDCG
    """
    return 2 ** x - 1


def inverse_log2(x: Tensor) -> Tensor:
    """
    Logarithmic (base 2) discount used for a ranking metric e.g. NDCG
    """
    return 1.0 / torch.log2(1 + x)


def inverse(x: Tensor) -> Tensor:
    """
    Linear discount used for a ranking metric e.g. NDCG
    """
    return 1.0 / x


def apply_pairwise_op(t: Tensor, op: Callable[[Tensor, Tensor], Tensor]):
    """
    Apply the given operation 'op' on all the pairs in the last dimension of 't'.
    Arguments:
        t: input N-d tensor
        op: a pairwise torch operation
    """
    return op(t.unsqueeze(-2), t.unsqueeze(-1))


def compute_rank(scores: Tensor) -> Tensor:
    """
    Computes ranks induced by the given scores. For example, [30.1, 10.0, 200.0] -> [2.0, 3.0, 1.0].
    Ranks are float type, as some tensor operations it will be part of, don't support
    integer or long values.
    Args:
        scores: an N-d Tensor [batch dim1 x ... x batch dim(N-1) x #items] with scores list in last dimension, the higher the score the better the rank
    Returns:
        rank: an N-d tensor with ranks corresponding to the scores list in the last dimension.
    """
    # The order vector contains indices
    # that yield descending order of the scores vector
    order = torch.argsort(scores, dim=-1, descending=True)
    rank = torch.zeros_like(scores)

    indexes = torch.arange(scores.shape[-1], device=scores.device) + 1.0
    for dim in range(len(scores.shape) - 1):
        indexes = indexes.unsqueeze(0)
    indexes = indexes.expand(*tuple(scores.shape))

    # Note: scatter_ backward pass is implemented only for src.shape == index.shape.
    # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_
    rank.scatter_(dim=-1, index=order, src=indexes)
    return rank


def compute_max_dcg(
    target: Tensor, gain_fn=exp2, rank_discount_fn=inverse_log2, k: int = None, dummy_indices=None
):
    """
    Compute the maximum DCG score
    Arguments:
        target: the target  N-d Tensor [batch dim1 x ... x batch dim(N-1) x #items] based on which
        the max DCG score is computed in the last dimension. Negative taget values can be used, but
        it may lead to instability issues downstream.
        https://webis.de/downloads/publications/papers/gienapp_2020c.pdf
        gain_fn: the gain function used in DCG. Exponential gain is the default.
        rank_discount_fn: the discount function used in DCG. Logarithmic discount is the default.
        k: the K as in DCG @K.
    Returns:
        max_dcg: (N-1)-d tensor containing maximum DCG score computed along last dimension of target
    """

    _gain = gain_fn(target)
    if dummy_indices is not None:
        gain_for_rank = torch.clone(_gain)
        min_gain = _gain.min()
        gain_for_rank[dummy_indices] = min_gain - 1.0

        gain = torch.clone(_gain)
        gain[dummy_indices] = 0.0
    else:
        gain_for_rank = gain = _gain

    ideal_rank = compute_rank(gain_for_rank)
    discount = rank_discount_fn(ideal_rank)
    if k is not None:
        discount *= ideal_rank <= k

    gain_discounted = gain * discount
    max_dcg = gain_discounted.sum(dim=-1)
    return max_dcg


def neural_sort(s: Tensor, temperature=1.0):
    """
    NeuralSort -- the relaxed/softened sorting operator. Given the scores, it returns a
    smooth differentiable permutation matrix representing the order of the scores. Lower
    temperatures will cause the final matrix to be close to a hard permutation matrix with
    0/1 entries.
    References:
        * Paper: https://arxiv.org/pdf/1903.08850.pdf (Section 3)
        * GitHub (PyTorch):
          https://github.com/ermongroup/neuralsort/blob/master/pytorch/neuralsort.py
        * GitHub (TensorFlow): https://github.com/ermongroup/neuralsort/blob/master/tf/util.py#L20
    Arguments:
        s: a 1-D Tensor of ranking scores
        temperature: the temperature parameter
    Returns:
        soft version of the permutation matrix based on the given scores.
    """
    # A_s{ij} = |s_i - s_j|
    A_s = torch.abs(apply_pairwise_op(s, torch.sub))
    one = torch.ones_like(s)

    # B = A_s x one
    B = torch.matmul(A_s, one)

    n = s.shape[0]
    I = torch.arange(n, device=s.device) + 1

    # C_i = (n + 1 - 2i) s
    C = (n + 1 - 2 * I).unsqueeze(1) * s.unsqueeze(0)

    return torch.nn.functional.softmax((C - B) / temperature, dim=1)


def neural_sort_group_parallel(s: Tensor, temperature: Number = 1.0, dummy_indices: Tensor = None):
    """
    NeuralSort -- the relaxed/softened sorting operator. Given the scores, it returns a
    smooth differentiable permutation matrix representing the order of the scores. Lower
    temperatures will cause the final matrix to be close to a hard permutation matrix with
    0/1 entries.
    References:
        * Paper: https://arxiv.org/pdf/1903.08850.pdf (Section 3)
        * GitHub (PyTorch):
          https://github.com/ermongroup/neuralsort/blob/master/pytorch/neuralsort.py
        * GitHub (TensorFlow): https://github.com/ermongroup/neuralsort/blob/master/tf/util.py#L20
    Arguments:
        s: an N-d Tensor [batch dim1 x ... x batch dim(N-1) x #items] of list of ranking scores in
        the last dimension
        temperature: the temperature parameter
        dummy_indices: N-d bool tensor (same shape as s) identifying the padded dummy indices at the
        end of the groups. If dummy_indices=None, then all items are considered as non-dummy.
    Returns:
        soft version of the permutation matrix based on the given scores.
    """
    if dummy_indices is None:
        dummy_indices = torch.zeros_like(s, dtype=bool)

    s_selected = torch.clone(s)
    s_selected[dummy_indices] = 0.0  # Set dummy_indices as 0.0

    # A_s{ij} = |s_i - s_j|
    A_s = torch.abs(apply_pairwise_op(s_selected, torch.sub))
    s_shape = dummy_indices.shape
    dummy_indices_unsqueezed = dummy_indices.unsqueeze(-2).expand(*s_shape, s_shape[-1])
    A_s[dummy_indices_unsqueezed] = 0.0  # Set all (...,i,j) pairs with j=dummy_index[..., :] as 0.0

    # B = A_s x one
    B = A_s.sum(dim=-1)
    B = B.unsqueeze(-2)

    I = (~dummy_indices).cumsum(dim=-1)
    n = I[..., -1:]

    # C_i = (n + 1 - 2i) s
    C = (n + 1 - 2 * I).unsqueeze(-1) * s_selected.unsqueeze(-2)
    C_B_diff = C - B
    # Setting dummy_indices logits value as -10.**10
    # so that they don't contribute to softmax
    C_B_diff[dummy_indices_unsqueezed] = LARGE_NEG_SOFTMAX_INPUT

    perm_mat = torch.zeros_like(C_B_diff)
    # Sort currently does not support bool dtype on CUDA
    nondummy_perm_rows = ~(dummy_indices.float().sort(dim=-1).values.bool())
    perm_mat[nondummy_perm_rows] = torch.nn.functional.softmax(
        C_B_diff[~dummy_indices] / temperature, dim=-1
    )

    return perm_mat


def hard_sort(s: Tensor):
    """
    Given a Tensor of scores in the last the dim, return a Tensor of permutation matrices in the
    last two dimensions representing the order of the values of the scores in descending order.
    Arguments:
        s: N-d Tensor [batch dim1 x ... x batch dim(N-1) x #items] of scores in the last dimension
        of the groups. If dummy_indices=None, then all items are considered as non-dummy.
    Returns:
        the list of permutation matrices based on the given scores
    """
    order = torch.argsort(s, descending=True, dim=-1)

    batch_dims = s.shape[:-1]
    group_dim = s.shape[-1]
    perm = torch.zeros(*batch_dims, group_dim, group_dim, device=s.device)

    order = order.unsqueeze(-1).expand(perm.shape)
    perm.scatter_(dim=-1, index=order, src=torch.ones_like(order, dtype=perm.dtype))

    return perm


def hard_sort_group_parallel(s: Tensor, dummy_indices: Tensor = None):
    """
    Given a Tensor of scores in the last the dim, return a Tensor of permutation matrices in the
    last two dimensions representing the order of the values of the scores in descending order.
    Arguments:
        s: an N-d Tensor [batch dim1 x ... x batch dim(N-1) x #items] of list of ranking scores in
        the last dimension
        dummy_indices: N-d bool tensor (same shape as s) identifying the padded dummy indices at the
        end
    Returns:
        the list of permutation matrices based on the given scores
    """
    if dummy_indices is not None and torch.any(~dummy_indices):
        # Clone s and set dummy values to be smaller than min
        s_clone = torch.clone(s)
        s_clone[dummy_indices] = s_clone[~dummy_indices].min() - 1.0
    else:
        s_clone = s

    order = torch.argsort(s_clone, descending=True, dim=-1)

    batch_dims = s.shape[:-1]
    group_dim = s.shape[-1]
    perm = torch.zeros(*batch_dims, group_dim, group_dim, device=s.device)

    order = order.unsqueeze(-1).expand(perm.shape)
    perm.scatter_(dim=-1, index=order, src=torch.ones_like(order, dtype=perm.dtype))

    if dummy_indices is not None:
        # Rows corresponding to sorted positions of dummy_indices are set to all zeros
        perm_clone = torch.zeros_like(perm)
        # Sort currently does not support bool dtype on CUDA
        nondummy_perm_rows = ~(dummy_indices.float().sort(dim=-1).values.bool())
        perm_clone[nondummy_perm_rows] = perm[nondummy_perm_rows]
        perm = perm_clone

    return perm


def soft_sort(s: Tensor, temperature=1.0, power=1.0):
    """
    SoftSort -- another relaxed/softened sorting operator similar to NeuralSort. Given
    the scores, it returns a smooth differentiable permutation matrix representing the
    order of the scores. Lower temperatures will cause the final matrix to be close
    to a hard permutation matrix with 0/1 entries.
    References:
        * Paper: https://arxiv.org/pdf/2006.16038.pdf (Section 3)
        * GitHub (PyTorch): https://github.com/sprillo/softsort/blob/master/pytorch/softsort.py
        * GitHub (TensorFlow): https://github.com/sprillo/softsort/blob/master/tf/util.py#L56
    Arguments:
        s: a 1-D Tensor of ranking scores
        temperature: the temperature parameter
    Returns:
        soft version of the permutation matrix based on the given scores.
    """
    s_sorted = s.sort(descending=True, dim=0)[0]
    pairwise_distances = (s.unsqueeze(0) - s_sorted.unsqueeze(-1)).abs().pow(
        power
    ).neg() / temperature
    perm = pairwise_distances.softmax(dim=-1)

    return perm


def soft_sort_group_parallel(
    s: Tensor, temperature: Number = 1.0, power: Number = 1.0, dummy_indices: Tensor = None
):
    """
    SoftSort -- another relaxed/softened sorting operator similar to NeuralSort. Given
    the scores, it returns a smooth differentiable permutation matrix representing the
    order of the scores. Lower temperatures will cause the final matrix to be close
    to a hard permutation matrix with 0/1 entries.
    References:
        * Paper: https://arxiv.org/pdf/2006.16038.pdf (Section 3)
        * GitHub (PyTorch): https://github.com/sprillo/softsort/blob/master/pytorch/softsort.py
        * GitHub (TensorFlow): https://github.com/sprillo/softsort/blob/master/tf/util.py#L56
    Arguments:
        s: an N-d Tensor [batch dim1 x ... x batch dim(N-1) x #items] of list of ranking scores in
        the last dimension
        temperature: the temperature parameter
        dummy_indices: N-d bool tensor (same shape as s) identifying the padded dummy indices at the
        end of the groups. If dummy_indices=None, then all items are considered as non-dummy.
    Returns:
        soft version of the permutation matrix based on the given scores.
    """
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


REDUCTION_FN = {SUM: torch.sum, MEAN: torch.mean, NONE: identity}
NORMALIZATION_FN = {SOFTMAX: softmax_1d, L1: l1_normalization_1d, NONE: noop_normalization}
GAIN_FN = {LINEAR: identity, EXP2: exp2}
DISCOUNT_FN = {LOG2: inverse_log2, LINEAR: inverse}
RELAXED_SORT_FN = {NEURAL_SORT: neural_sort, SOFT_SORT: soft_sort}
RELAXED_GROUP_PARALLEL_SORT_FN = {
    NEURAL_SORT: neural_sort_group_parallel,
    SOFT_SORT: soft_sort_group_parallel,
}


import torch
import numpy as np

DEFAULT_EPS = 1e-10


def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    """
    Sinkhorn scaling procedure.
    :param mat: a tensor of square matrices of shape N x M x M, where N is batch size
    :param mask: a tensor of masks of shape N x M
    :param tol: Sinkhorn scaling tolerance
    :param max_iter: maximum number of iterations of the Sinkhorn scaling
    :return: a tensor of (approximately) doubly stochastic matrices
    """
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=DEFAULT_EPS)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=DEFAULT_EPS)

        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat


def deterministic_neural_sort(s, tau, mask, dev):
    """
    Deterministic neural sort.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :return: approximate permutation matrices of shape [batch_size, slate_length, slate_length]
    """

    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=dev)
    s = s.masked_fill(mask[:, :, None], -1e8)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    A_s = A_s.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)

    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))

    temp = [n - m + 1 - 2 * (torch.arange(n - m, device=dev) + 1) for m in mask.squeeze(-1).sum(dim=1)]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=dev))) for t in temp]
    scaling = torch.stack(temp).type(torch.float32).to(dev)  # type: ignore

    s = s.masked_fill(mask[:, :, None], 0.0)
    C = torch.matmul(s, scaling.unsqueeze(-2))

    P_max = (C - B).permute(0, 2, 1)
    P_max = P_max.masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    P_max = P_max.masked_fill(mask[:, :, None] & mask[:, None, :], 1.0)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def sample_gumbel(samples_shape, device, eps=1e-10) -> torch.Tensor:
    """
    Sampling from Gumbel distribution.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param samples_shape: shape of the output samples tensor
    :param device: device of the output samples tensor
    :param eps: epsilon for the logarithm function
    :return: Gumbel samples tensor of shape samples_shape
    """
    U = torch.rand(samples_shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def stochastic_neural_sort(s, n_samples, tau, mask, dev, beta=1.0, log_scores=True, eps=1e-10):
    """
    Stochastic neural sort. Please note that memory complexity grows by factor n_samples.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param n_samples: number of samples (approximations) for each permutation matrix
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :param beta: scale parameter for the Gumbel distribution
    :param log_scores: whether to apply the logarithm function to scores prior to Gumbel perturbation
    :param eps: epsilon for the logarithm function
    :return: approximate permutation matrices of shape [n_samples, batch_size, slate_length, slate_length]
    """

    batch_size = s.size()[0]
    n = s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=dev)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)

    P_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat