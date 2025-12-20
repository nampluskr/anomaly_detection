# src/anomaly_detection/utils.py

import os
import logging
from datetime import datetime


def setup_logging(log_dir, name="train"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)


def get_experiment_dir(output_root, model_name, dataset_name, category):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{dataset_name}_{category}_{timestamp}"
    exp_dir = os.path.join(output_root, "experiments", model_name, exp_name)
    
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    return exp_dir