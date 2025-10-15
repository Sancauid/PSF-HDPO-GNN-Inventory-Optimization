from typing import Any, Dict, Iterable, Optional, Tuple
import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm


class Trainer:
  """
  Encapsulates the end-to-end training loop with a differentiable simulator.

  This class orchestrates epoch and batch loops, resets the simulator for each
  batch, rolls out multi-period episodes, computes losses, performs backprop,
  and steps the optimizer and scheduler.
  """

  def __init__(
    self,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[Any],
    simulator: Any,
    train_loader: Iterable[Dict[str, Any]],
    configs: Dict[str, Any],
  ) -> None:
    """
    Initialize the Trainer.

    Args:
      model: Policy network.
      optimizer: Optimizer instance.
      scheduler: Optional LR scheduler.
      simulator: Differentiable simulator with reset/step APIs.
      train_loader: Iterable yielding batch dictionaries.
      configs: Configuration mapping for problem, data, and training params.
    """
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.simulator = simulator
    self.train_loader = train_loader
    self.configs = configs

    self.problem_params = configs.get("problem_params", {})
    self.data_params = configs.get("data_params", {})
    self.training_params = configs.get("training_params", {})
    self.architecture = (
      configs.get("model_params", {}).get("architecture", "vanilla").lower()
    )

  def train(self) -> None:
    """
    Run the main training loop over epochs.
    """
    epochs: int = int(self.training_params.get("epochs", 1))
    for epoch in range(epochs):
      avg_loss = self._train_epoch(epoch)
      if self.scheduler is not None:
        self.scheduler.step()
      print(f"epoch={epoch} avg_loss={avg_loss:.6f}")

  def _train_epoch(self, epoch: int) -> float:
    """
    Train for a single epoch over all batches.

    Args:
      epoch: Current epoch index (0-based).

    Returns:
      Average reported loss across all batches.
    """
    losses: list[float] = []
    for batch in tqdm(self.train_loader, desc=f"train epoch {epoch}"):
        batch_loss = self._train_batch(batch)
        losses.append(batch_loss)
    return float(sum(losses) / max(len(losses), 1))

  def _train_batch(self, batch: Dict[str, Any]) -> float:
    """
    Train on a single batch by running a full simulated episode.

    Args:
      batch: Dictionary containing tensors for resetting the simulator and any
        additional inputs required by the model.

    Returns:
      A scalar float with the loss value reported for logging.
    """
    self.optimizer.zero_grad(set_to_none=True)

    inventories = batch.get("inventories")
    demands = batch.get("demands")
    cost_params = batch.get("cost_params")
    lead_times = batch.get("lead_times")

    if inventories is None or demands is None or cost_params is None or lead_times is None:
      raise ValueError("Batch missing required keys: inventories, demands, cost_params, lead_times")

    periods: int = int(self.problem_params.get("periods", demands.shape[0]))
    batch_size: int = int(self.data_params.get("n_samples", demands.shape[1]))
    num_stores: int = int(self.problem_params.get("n_stores", inventories["stores"].shape[1]))
    num_warehouses: int = int(self.problem_params.get("n_warehouses", inventories["warehouses"].shape[1]))

    store_mask = torch.ones(num_stores, dtype=torch.bool, device=demands.device)

    total_episode_cost = torch.zeros(batch_size, device=demands.device, dtype=demands.dtype)

    self.simulator.reset(
      inventories=inventories,
      demands=demands,
      cost_params=cost_params,
      lead_times=lead_times,
    )

    for t in range(periods):
      if self.architecture == "vanilla":
        obs = {
          "inventory_stores": self.simulator.inventory_stores,
          "inventory_warehouses": self.simulator.inventory_warehouses,
        }
        x = torch.cat([obs["inventory_stores"], obs["inventory_warehouses"]], dim=1)
        logits = self.model(x)
        store_logits = logits[:, :num_stores]
        wh_logits = logits[:, -num_warehouses:]
        actions_stores = torch.sigmoid(store_logits)
        actions_wh = torch.sigmoid(wh_logits)
      else:
        if not {"x", "edge_index"}.issubset(set(batch.keys())):
          actions_stores = torch.zeros(batch_size, num_stores, device=demands.device, dtype=demands.dtype)
          actions_wh = torch.zeros(batch_size, num_warehouses, device=demands.device, dtype=demands.dtype)
        else:
          node_out = self.model(batch["x"], batch["edge_index"])  # [num_nodes, 1]
          total_nodes_per_sample = num_stores + num_warehouses
          if node_out.dim() == 2 and node_out.size(1) == 1 and node_out.size(0) == batch_size * total_nodes_per_sample:
            node_out = node_out.view(batch_size, total_nodes_per_sample, 1)
            actions_stores = torch.nn.functional.softplus(node_out[:, :num_stores, 0])
            actions_wh = torch.nn.functional.softplus(node_out[:, num_stores:, 0])
          else:
            actions_stores = torch.zeros(batch_size, num_stores, device=demands.device, dtype=demands.dtype)
            actions_wh = torch.zeros(batch_size, num_warehouses, device=demands.device, dtype=demands.dtype)

      _, step_cost = self.simulator.step({
        "stores": actions_stores,
        "warehouses": actions_wh,
      })

      total_episode_cost = total_episode_cost + step_cost

    loss_for_backward = total_episode_cost.mean()
    loss_to_report = loss_for_backward.detach()

    loss_for_backward.backward()
    max_norm = float(self.training_params.get("grad_clip_norm", 1.0))
    clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
    self.optimizer.step()

    return float(loss_to_report.item())


