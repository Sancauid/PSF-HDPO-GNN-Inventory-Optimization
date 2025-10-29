from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from .engine import (
    calculate_losses,
    perform_gradient_step,
    prepare_batch_for_simulation,
    run_simulation_episode,
)


class Trainer:
    """
    Encapsulates the end-to-end training loop with a graph-based differentiable simulator.

    This class orchestrates epoch and batch loops, prepares PyG batches for simulation,
    rolls out multi-period episodes using the EdgeGNN model, computes losses, performs
    backprop, and steps the optimizer and scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Any],
        train_loader: Iterable[Batch],
        configs: Dict[str, Any],
    ) -> None:
        """
        Initialize the Trainer.

        Args:
          model: EdgeGNN policy network for predicting edge flows.
          optimizer: Optimizer instance.
          scheduler: Optional LR scheduler.
          train_loader: Iterable yielding PyG Batch objects.
          configs: Configuration mapping for problem, data, and training params.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.simulator = None
        self.train_loader = train_loader
        self.configs = configs

        self.problem_params = configs.get("problem_params", {})
        self.data_params = configs.get("data_params", {})
        self.training_params = configs.get("training_params", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)

    def train(self, epochs: int | None = None) -> None:
        """
        Run the main training loop over epochs.
        """
        num_epochs: int = (
            int(epochs)
            if epochs is not None
            else int(self.training_params.get("epochs", 1))
        )
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(epoch)
            if self.scheduler is not None:
                # Support ReduceLROnPlateau-style schedulers
                try:
                    self.scheduler.step(avg_loss)
                except TypeError:
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

    def _train_batch(self, batch: Batch) -> float:
        """
        Train on a single batch by running a full simulated episode using the graph-based engine.

        Args:
            batch: PyG Batch containing graphs with node features, edge indices, and attributes.

        Returns:
            A scalar float with the loss value reported for logging.
        """
        # Prepare batch for simulation using the new engine function
        data_for_reset, pyg_batch = prepare_batch_for_simulation(batch, self.device)

        # Get simulation parameters
        periods = int(self.problem_params.get("periods", data_for_reset["demands"].shape[0]))
        ignore_periods = int(self.problem_params.get("ignore_periods", 0))
        
        # Get number of nodes from the graph structure
        num_nodes = data_for_reset["node_features"].shape[0]

        # Run simulation episode using the new engine function
        total_episode_cost, cost_to_report = run_simulation_episode(
            model=self.model,
            pyg_batch=pyg_batch,
            data_for_reset=data_for_reset,
            periods=periods,
            ignore_periods=ignore_periods,
        )

        # Calculate losses using the new engine function
        loss_for_backward, loss_to_report_tensor = calculate_losses(
            total_episode_cost=total_episode_cost,
            cost_to_report=cost_to_report,
            num_nodes=num_nodes,
            periods=periods,
            ignore_periods=ignore_periods,
        )

        # Perform gradient step using the new engine function
        max_norm = float(self.training_params.get("grad_clip_norm", 1.0))
        perform_gradient_step(
            loss=loss_for_backward,
            model=self.model,
            optimizer=self.optimizer,
            grad_clip_norm=max_norm,
        )

        return float(loss_to_report_tensor.item())

