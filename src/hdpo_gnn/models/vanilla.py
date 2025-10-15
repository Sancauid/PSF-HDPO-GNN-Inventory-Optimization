from typing import Tuple
import torch
from torch import nn


class VanillaPolicy(nn.Module):
  """
  A simple MLP-based policy network producing logits for action selection.

  The network flattens the input and applies two hidden layers with ReLU
  activations, outputting a vector of size `output_size`.
  """

  def __init__(self, input_size: int, output_size: int) -> None:
    """
    Initialize the VanillaPolicy.

    Args:
      input_size: Number of input features after flattening.
      output_size: Number of output logits (e.g., n_stores + 1).
    """
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_size, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, output_size),
    )
    self.output_size = int(output_size)

  def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Compute forward pass and split outputs for stores and warehouse.

    Args:
      x: Input tensor of shape [batch, ...] which will be flattened to [batch, input_size].

    Returns:
      A dictionary with:
        - 'stores': tensor of shape [batch, n_stores]
        - 'warehouses': tensor of shape [batch, 1]
    """
    if x.dim() > 2:
      x = x.view(x.size(0), -1)
    logits = self.net(x)
    n_stores = self.output_size - 1
    stores_logits = logits[:, :n_stores]
    warehouse_logits = logits[:, n_stores:]
    return {"stores": stores_logits, "warehouses": warehouse_logits}


