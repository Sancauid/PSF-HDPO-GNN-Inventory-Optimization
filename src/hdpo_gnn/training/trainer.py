from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from tqdm.auto import tqdm

from .engine import (
    calculate_losses,
    perform_gradient_step,
    prepare_batch_for_simulation,
    run_simulation_episode,
)


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
        self.simulator = None
        self.train_loader = train_loader
        self.configs = configs

        self.problem_params = configs.get("problem_params", {})
        self.data_params = configs.get("data_params", {})
        self.training_params = configs.get("training_params", {})
        self.architecture = (
            configs.get("model_params", {}).get("architecture", "vanilla").lower()
        )

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

    def _train_batch(self, batch: Dict[str, Any] | Batch) -> float:
        """
        Train on a single batch by running a full simulated episode.

        Args:
            batch: Dictionary containing tensors for resetting the simulator and any
                additional inputs required by the model.

        Returns:
            A scalar float with the loss value reported for logging.
        """
        self.optimizer.zero_grad(set_to_none=True)

        # Prepare reset data and store mask depending on architecture
        if self.architecture == "vanilla":
            assert isinstance(batch, dict)
            data_for_reset = {
                "inventories": batch["inventories"],
                "demands": batch["demands"],
                "cost_params": batch["cost_params"],
                "lead_times": batch["lead_times"],
            }
            num_stores = int(
                self.problem_params.get(
                    "n_stores", data_for_reset["inventories"]["stores"].shape[1]
                )
            )
            store_mask = torch.ones(
                num_stores,
                dtype=torch.bool,
                device=data_for_reset["demands"].device,
            )
        else:
            assert isinstance(batch, Batch)
            data_for_reset, store_mask = self._prepare_gnn_batch(batch)

        periods = int(
            self.problem_params.get("periods", data_for_reset["demands"].shape[0])
        )
        ignore_periods = int(self.problem_params.get("ignore_periods", 0))
        num_real_stores = int(self.problem_params.get("n_stores", store_mask.numel()))

        total_episode_cost, cost_to_report = run_simulation_episode(
            model=self.model,
            simulator=None,
            batch=(
                data_for_reset.get("pyg_batch") if self.architecture == "gnn" else None
            ),
            periods=periods,
            ignore_periods=ignore_periods,
            data_for_reset=data_for_reset,
            store_mask=store_mask,
            architecture=self.architecture,
        )

        loss_for_backward, loss_to_report_tensor = calculate_losses(
            total_episode_cost=total_episode_cost,
            cost_to_report=cost_to_report,
            total_real_stores=num_real_stores,
            periods=periods,
            ignore_periods=ignore_periods,
        )

        loss_for_backward.backward()
        max_norm = float(self.training_params.get("grad_clip_norm", 1.0))
        clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        self.optimizer.step()

        return float(loss_to_report_tensor.item())

    def _prepare_gnn_batch(self, batch: Batch) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Prepare PyG batch into dense tensors for simulator.reset and create store mask.

        Args:
            batch: A PyG Batch containing graphs with attributes x, edge_index, batch,
                and per-graph attributes like demands, costs, and lead times.

        Returns:
            A tuple (data_for_reset, store_mask) where data_for_reset is a mapping with
            keys 'inventories', 'demands', 'cost_params', 'lead_times', plus references
            to the original PyG batch and shape metadata to support the simulation loop.
        """
        dense_x, mask = to_dense_batch(batch.x, batch.batch)  # [B, N, F]
        B, N = dense_x.size(0), dense_x.size(1)
        device, dtype = dense_x.device, dense_x.dtype
        periods: int = int(self.problem_params.get("periods"))

        inventories = {
            "stores": dense_x[..., 0],
            "warehouses": torch.zeros(
                B,
                int(self.problem_params.get("n_warehouses", 1)),
                device=device,
                dtype=dtype,
            ),
        }

        if hasattr(batch, "demands"):
            d = batch.demands
            if d.dim() == 3 and d.size(0) == B and d.size(2) == N:
                demands = d.permute(1, 0, 2).to(device=device, dtype=dtype)  # [T,B,N]
            elif d.dim() == 2 and d.size(1) == N:
                T = periods
                demands = (
                    d.view(B, T, N).permute(1, 0, 2).to(device=device, dtype=dtype)
                )
            else:
                raise ValueError("Unexpected demands shape in PyG batch")
        else:
            raise ValueError("PyG batch missing 'demands' attribute")

        cost_params = {
            "holding_store": torch.as_tensor(
                getattr(batch, "holding_store", 1.0), device=device, dtype=dtype
            ).flatten()[0],
            "underage_store": torch.as_tensor(
                getattr(batch, "underage_store", 1.0), device=device, dtype=dtype
            ).flatten()[0],
            "holding_warehouse": torch.as_tensor(
                getattr(batch, "holding_warehouse", 0.5), device=device, dtype=dtype
            ).flatten()[0],
        }
        lead_times = {
            "stores": int(
                torch.as_tensor(getattr(batch, "lead_time_stores", 0))
                .flatten()[0]
                .item()
            ),
            "warehouses": int(
                torch.as_tensor(getattr(batch, "lead_time_warehouses", 0))
                .flatten()[0]
                .item()
            ),
        }

        store_mask = torch.ones(N, dtype=torch.bool, device=device)

        data_for_reset: Dict[str, Any] = {
            "inventories": inventories,
            "demands": demands,
            "cost_params": cost_params,
            "lead_times": lead_times,
            "pyg_batch": batch,
            "B": B,
            "N": N,
        }
        return data_for_reset, store_mask
