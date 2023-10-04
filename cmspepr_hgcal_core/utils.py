from __future__ import annotations
import torch
from torch_scatter import scatter_add


def assert_no_nans(x):
    """
    Raises AssertionError if there is a nan in the tensor
    """
    assert not torch.isnan(x).any()


def huber(d: torch.FloatTensor, delta: float) -> torch.FloatTensor:
    """Huber function; see https://en.wikipedia.org/wiki/Huber_loss#Definition.
    Multiplied by 2 w.r.t Wikipedia version.

    Args:
        d (torch.FloatTensor): Input array
        delta (float): Point at which quadratic behavior should switch to linear.

    Returns:
        torch.FloatTensor: Huberized array
    """

    """
    See: https://en.wikipedia.org/wiki/Huber_loss#Definition
    Multiplied by 2 w.r.t Wikipedia version (aligning with Jan's definition)
    """
    return torch.where(
        torch.abs(d) <= delta, d**2, 2.0 * delta * (torch.abs(d) - delta)
    )


def scatter_count(input: torch.Tensor) -> torch.LongTensor:
    """Returns ordered counts over an index array.

    Args:
        input (torch.Tensor): An array of indices, assumed to start at 0. The array does
            not need to be sorted.

    Returns:
        torch.LongTensor: An array with counts per index.

    Example:
        >>> scatter_count(torch.Tensor([0, 0, 0, 1, 1, 2, 2]))
        tensor([3, 2, 2])

        Index assumptions work like in torch_scatter, so:
        >>> scatter_count(torch.Tensor([1, 1, 1, 2, 2, 4, 4]))
        tensor([0, 3, 2, 0, 2])
    """
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def scatter_counts_to_indices(input: torch.LongTensor) -> torch.LongTensor:
    """
    Converts counts to indices. This is the inverse operation of scatter_count
    Example:
    input:  [3, 2, 2]
    output: [0, 0, 0, 1, 1, 2, 2]
    """
    return torch.repeat_interleave(
        torch.arange(input.size(0), device=input.device), input
    ).long()


class LossResult:
    """
    Wrapper class for keeping track of several loss contributions that combined make up
    a single loss.
    Has a nice printout and basic operator overloading.
    """

    def __init__(self, **components):
        self.components = components
        self.offset = 1.0

    def __getitem__(self, *args, **kwargs):
        return self.components.__getitem__(*args, **kwargs)

    @property
    def loss(self) -> torch.float:
        """The final summed-up loss value of all passed components.

        Returns:
            torch.float: _description_
        """
        return self.offset + sum(self.components.values())

    def __repr__(self):
        loss = self.loss
        r = [f"final loss:     {loss:.10f}"]
        for c, v in self.components.items():
            perc = 100.0 * v / (loss - self.offset)
            r.append(f"  {c:20} {v:15.10f}   {perc:5.2f}%")
        return "\n".join(r)

    def __add__(self, o: LossResult) -> LossResult:
        lr = LossResult(**self.components)
        for c in o.components.keys():
            if c not in lr.components:
                lr.components[c] = 0.0
            lr.components[c] += o.components[c]
        return lr

    def __truediv__(self, num: float) -> LossResult:
        lr = LossResult(**self.components)
        for c in lr.components.keys():
            lr.components[c] /= num
        return lr

    def backward(self):
        return self.loss.backward()
