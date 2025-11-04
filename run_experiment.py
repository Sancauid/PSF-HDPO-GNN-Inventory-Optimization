# run_experiment.py

import subprocess
import logging
import sys
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [MANAGER] %(message)s")
log = logging.getLogger(__name__)

def run_single_training(config_path: str, lr: float, verbose: bool) -> None:
    """
    Executes a single training run as a subprocess.
    """
    log.info("="*80)
    log.info(f"Launching training worker for LR = {lr}")
    log.info("="*80)

    command = [
        sys.executable,
        "train.py",
        config_path,
        "--lr",
        str(lr),
    ]
    if verbose:
        command.append("-v")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    rc = process.poll()
    if rc != 0:
        log.error(f"Training worker for LR={lr} failed with return code {rc}.")
    else:
        log.info(f"Training worker for LR={lr} completed successfully.")

    return rc

def main():
    """
    Main manager function to run the hyperparameter sweep.
    """
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <path_to_config_file> [-v]")
        sys.exit(1)

    config_path = sys.argv[1]
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    log.info(f"Starting experiment sweep for config: {config_path}")

    config = OmegaConf.load(config_path)
    learning_rates_raw = config.training.optimizer.lr

    if OmegaConf.is_config(learning_rates_raw):
        learning_rates = OmegaConf.to_container(learning_rates_raw, resolve=True)
    else:
        learning_rates = learning_rates_raw

    if not isinstance(learning_rates, list):
        learning_rates = [learning_rates]
    else:
        while len(learning_rates) > 0 and isinstance(learning_rates[0], list):
            learning_rates = learning_rates[0]

    learning_rates = [float(lr) for lr in learning_rates]

    log.info(f"Found {len(learning_rates)} learning rates to test: {learning_rates}")

    results = {}
    for lr in learning_rates:
        return_code = run_single_training(config_path, lr, verbose)
        results[lr] = "Success" if return_code == 0 else "Failure"
        if return_code != 0:
            log.warning(f"Aborting sweep due to failure on LR={lr}")
            break

    log.info("--- Experiment Sweep Summary ---")
    for lr, status in results.items():
        log.info(f"  - LR: {float(lr):<10.6f} | Status: {status}")
    log.info("-----------------------------")
    log.info("Experiment sweep finished.")

if __name__ == "__main__":
    main()

