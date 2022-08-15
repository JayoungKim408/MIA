import torch


def _updated_create_or_extend_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int
) -> None:
    """
    Create a 'grad_sample' attribute in the given parameter, or append to it
    if the 'grad_sample' attribute already exists.
    """
    if hasattr(param, "grad_sample"):
        param.grad_sample += grad_sample
    else:
        param.grad_sample = grad_sample