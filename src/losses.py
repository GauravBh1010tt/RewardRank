from numbers import Number
import warnings
import torch

#from .group_parallel_loss_fn import GroupParallelLossFn

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


class PiRankLoss(torch.nn.Module):
    def __init__(
        self,
        sort_func=NEURAL_SORT,
        gain_fn=EXP2,
        rank_discount_fn=LOG2,
        k=8,
        ste=False,
        **sort_kwargs,
    ):
        super().__init__()
        self._check_deprecation_warning()
        self._validate_inputs(sort_func, gain_fn, rank_discount_fn, k, **sort_kwargs)

        self.sort_func = RELAXED_SORT_FN[sort_func]
        self.gain_fn = GAIN_FN[gain_fn]
        self.rank_discount_fn = DISCOUNT_FN[rank_discount_fn]
        self.sort_kwargs = sort_kwargs
        self.k = k
        self.ste = ste

    @staticmethod
    def _check_deprecation_warning():
        """Warn users about deprecation."""
        warnings.warn(
            "Please switch to using PiRankGPLoss. We will be deprecating PiRankLoss soon.",
            DeprecationWarning,
        )

    @staticmethod
    def _validate_inputs(sort_func, gain_fn, rank_discount_fn, k, **sort_kwargs):
        """Validate the inputs to avoid runtime errors."""
        if "temperature" in sort_kwargs:
            assert isinstance(
                sort_kwargs["temperature"], Number
            ), "'temperature' must be numerical."

        if "power" in sort_kwargs:
            assert isinstance(sort_kwargs["power"], Number), "'power' must be numerical."

        assert (k is None) or isinstance(k, int), "'k' must be int or None."

        if gain_fn not in GAIN_FN:
            raise ValueError(f"Unknown gain function: `{gain_fn}` is not in {list(GAIN_FN.keys())}")

        if rank_discount_fn not in DISCOUNT_FN:
            raise ValueError(
                f"Unknown rank discount function: `{rank_discount_fn}` is not in {list(DISCOUNT_FN.keys())}"
            )

        if sort_func not in RELAXED_SORT_FN:
            raise ValueError(
                f"Unknown differentiable sort function: `{sort_func}` is not in {list(RELAXED_SORT_FN.keys())}"
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        perm_mat = self._get_permutation_matrix(pred)
        gains_sorted = self._compute_sorted_gains(perm_mat, target)
        dcg = self._compute_dcg(gains_sorted)
        max_dcg = compute_max_dcg(
            target, gain_fn=self.gain_fn, rank_discount_fn=self.rank_discount_fn, k=self.k
        )

        # Avoiding division-by-zero without losing the gradient
        ndcg = dcg.sum() / (1e-10 + max_dcg)
        return 1.0 - ndcg  # Turning gain into loss

    def _get_permutation_matrix(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute the permutation matrix based on the prediction and STE flag."""
        if self.ste:
            perm_mat_backward = self.sort_func(pred, **self.sort_kwargs)
            perm_mat_forward = hard_sort(pred)
            perm_mat = perm_mat_backward + (perm_mat_forward - perm_mat_backward).detach()
        else:
            perm_mat = self.sort_func(pred, **self.sort_kwargs)
        return perm_mat

    def _compute_sorted_gains(self, perm_mat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the sorted gains by multiplying the permutation matrix with gains."""
        gains = self.gain_fn(target)
        return torch.matmul(perm_mat, gains)

    def _compute_dcg(self, gains_sorted: torch.Tensor) -> torch.Tensor:
        """Compute the DCG (Discounted Cumulative Gain) score."""
        n = gains_sorted.shape[0]
        discount = self.rank_discount_fn(torch.arange(n, device=gains_sorted.device) + 1)
        dcg = gains_sorted * discount
        return dcg[: self.k]



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