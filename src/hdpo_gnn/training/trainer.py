# src/hdpo_gnn/training/trainer.py
import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from . import simulation_engine

log = logging.getLogger(__name__)

class Trainer:
    """
    Orchestrates the HDPO training and evaluation loop.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: PyGDataLoader,
        dev_loader: PyGDataLoader,
        test_loader: PyGDataLoader,
        config: DictConfig,
        device: torch.device,
        env: 'Environment',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        log.info("--- Initializing Trainer ---")
        log.debug("RECEIVED model, optimizer, loaders, config, device, and scheduler.")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.env = env
        
        self.epochs = self.config.training.epochs
        self.early_stopping_patience = self.config.training.early_stopping_patience
        self.best_dev_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None
        
        # Create a unique name for this run for saving results
        lr = config.training.optimizer.lr
        if OmegaConf.is_config(lr):
            if OmegaConf.is_list(lr):
                lr = float(lr[0])
            else:
                lr = float(lr)
        elif isinstance(lr, (list, tuple)):
            lr = float(lr[0])
        else:
            lr = float(lr)
        self.run_name = f"{config.experiment.name}_lr_{lr}"
        
        log.info(f"Trainer configured for {self.epochs} epochs with early stopping patience of {self.early_stopping_patience}.")
        log.info(f"Initialized run with name: {self.run_name}")

    def train(self):
        """Main training loop over epochs."""
        log.debug("--> Calling train...")
        for epoch in range(1, self.epochs + 1):
            log.info(f"--- Starting Epoch {epoch}/{self.epochs} ---")
            
            # --- Training Phase ---
            self.model.train()
            train_loss = self._run_epoch(self.train_loader, is_training=True, epoch_num=epoch)
            log.info(f"Epoch {epoch} | Average Training Loss: {train_loss:.4f}")

            # --- Validation (Dev) Phase ---
            self.model.eval()
            with torch.no_grad():
                dev_loss = self._run_epoch(self.dev_loader, is_training=False, epoch_num=epoch)
            log.info(f"Epoch {epoch} | Average Validation Loss: {dev_loss:.4f}")

            # --- Learning Rate Scheduling & Early Stopping ---
            if self.scheduler:
                self.scheduler.step(dev_loss)
            
            if self._check_early_stopping(dev_loss):
                log.info("Early stopping criteria met. Halting training.")
                break
        
        log.info("--- Training Finished ---")
        
        # --- FINAL EVALUATION STEP ---
        self.evaluate_on_test_set()
        
        log.debug("<-- train COMPLETED.")

    def _run_epoch(self, data_loader: PyGDataLoader, is_training: bool, epoch_num: int) -> float:
        """Runs a single epoch of training or evaluation over the provided data loader."""
        mode = "Training" if is_training else "Validation"
        log.debug(f"...... Calling _run_epoch (mode={mode})...")
        
        total_reported_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num} {mode}", leave=False)
        
        for batch in progress_bar:
            batch = batch.to(self.device)
            
            if is_training:
                self.optimizer.zero_grad()

            # Delegate the entire simulation and loss calculation to the engine
            total_episode_cost, reported_costs = simulation_engine.run_simulation_episode(
                self.model,
                pyg_batch=batch,
                env=self.env, 
                config=self.config, 
                is_training=is_training
            )
            
            # The loss for backpropagation is the mean of the *total* episode cost
            loss_for_backward = total_episode_cost.mean()
            
            if is_training:
                loss_for_backward.backward()
                grad_clip_norm = self.config.training.get("gradient_clip_norm")
                if grad_clip_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()

            # The loss for reporting/logging is the mean of the *reported* costs (after warmup)
            reported_loss = reported_costs.mean().item()
            total_reported_loss += reported_loss
            progress_bar.set_postfix(loss=reported_loss)

        avg_loss = total_reported_loss / len(data_loader)
        log.debug(f"...... _run_epoch RETURNING average reported loss: {avg_loss:.4f}")
        return avg_loss

    def _check_early_stopping(self, dev_loss: float) -> bool:
        """Checks if early stopping criteria are met and saves the best model state."""
        log.debug("...... Calling _check_early_stopping...")
        if dev_loss < self.best_dev_loss:
            log.info(f"Validation loss improved from {self.best_dev_loss:.4f} to {dev_loss:.4f}. Saving model state.")
            self.best_dev_loss = dev_loss
            self.epochs_no_improve = 0
            # Save the model's state dictionary in memory
            self.best_model_state = self.model.state_dict()
        else:
            self.epochs_no_improve += 1
            log.info(f"Validation loss did not improve. Patience: {self.epochs_no_improve}/{self.early_stopping_patience}.")

        stop = self.epochs_no_improve >= self.early_stopping_patience
        log.debug(f"...... _check_early_stopping RETURNING {stop}")
        return stop
    
    def evaluate_on_test_set(self):
        """Loads the best model and evaluates it on the test set."""
        log.info("--- Starting Final Evaluation on Test Set ---")
        if self.best_model_state is None:
            log.warning("No best model state found (training may have been too short). Evaluating with the final model.")
        else:
            log.info("Loading best model state from training.")
            self.model.load_state_dict(self.best_model_state)
            
        self.model.eval()
        with torch.no_grad():
            test_loss = self._run_epoch(self.test_loader, is_training=False, epoch_num="Test")
            
        log.info(f"FINAL TEST LOSS: {test_loss:.4f}")
        
        # Save result to a file
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, f"{self.run_name}.json")
        
        result_data = {
            "run_name": self.run_name,
            "best_validation_loss": float(self.best_dev_loss),
            "final_test_loss": float(test_loss),
            "config": OmegaConf.to_container(self.config, resolve=True)
        }
        
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=4)
            
        log.info(f"Results saved to {result_path}")