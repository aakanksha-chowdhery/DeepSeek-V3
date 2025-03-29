from typing import Any

import torch

def _reduce(ctx: Any, input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    # group = get_model_parallel_group()

    if ctx:
        ctx.mark_dirty(input_)

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_)

    return input_


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _reduce(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output
    
    

def reduce_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _reduce(None, grad_output)
    

def copy_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)