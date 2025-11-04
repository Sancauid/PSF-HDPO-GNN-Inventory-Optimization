# train.py

import argparse
import logging
import pprint
import random
import sys
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from src.hdpo_gnn.utils.environment_builder import Environment
from src.hdpo_gnn.data.dataset_manager import DatasetManager
from src.hdpo_gnn.models.factory import ModelFactory
from src.hdpo_gnn.training.trainer import Trainer

log = logging.getLogger(__name__)

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    logging.basicConfig(level=level, format=log_format, handlers=[logging.StreamHandler(sys.stdout)])
    log.info(f"Logging level set to {'DEBUG' if verbose else 'INFO'}")

def setup_reproducibility(seed: int) -> None:
    log.info(f"Setting global random seed to {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main() -> None:
    parser = argparse.ArgumentParser(description="Worker script for a single training run.")
    parser.add_argument("config_file", help="Path to the unified experiment config.")

    parser.add_argument("--lr", type=float, required=True, help="Learning rate for this specific run.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log.info(f"--- STARTING SINGLE TRAINING RUN ---")
    log.info(f"Config: {args.config_file}, Learning Rate: {args.lr}")

    config = OmegaConf.load(args.config_file)
    OmegaConf.resolve(config)
    log.info(f"Experiment Name: {config.experiment.name}")

    setup_reproducibility(config.experiment.seed)
    device = torch.device(config.experiment.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"Target device: {device}")

    try:
        log.info("PHASE 1 & 2: Initializing Environment and DataLoaders...")
        env = Environment(config.environment)
        dataset_manager = DatasetManager(env, config.data, config.features)
        train_loader, dev_loader, test_loader = dataset_manager.get_data_loaders()

        log.info("PHASE 3: Creating policy network...")
        model_factory = ModelFactory(config.model, config.features, env)
        model = model_factory.create_model().to(device)

        log.info("PHASE 4: Setting up optimizer and trainer...")
        optimizer_cfg = config.training.optimizer

        learning_rate = args.lr

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=optimizer_cfg.get("betas", [0.9, 0.999])
        )
        log.info(f"Optimizer '{optimizer_cfg.name}' configured with learning rate: {learning_rate}")

        scheduler = None

        trainer = Trainer(
            model=model, optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, dev_loader=dev_loader, test_loader=test_loader,
            config=config, device=device, env=env
        )

        log.info("PHASE 5: Starting training process...")
        start_time = datetime.now()
        trainer.train()
        end_time = datetime.now()
        log.info(f"Training finished in {end_time - start_time}.")

    except Exception as e:
        log.exception("An unhandled error occurred during the training pipeline.")
        raise e

if __name__ == "__main__":
    main()