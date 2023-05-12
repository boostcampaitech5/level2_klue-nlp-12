import logging
import os
import random

import numpy as np
import torch
import wandb
from wandb import AlertLevel

log = logging.getLogger(__name__)


def seed_everything(seed, workers: bool = False) -> int:
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed


def init_wandb(config, run_name):
    if not config.use_wandb:
        return

    wandb.init(
        entity=config.wandb['entity'],
        project=config.wandb['project_name'],
        name=run_name,
        config=config,
    )
    wandb.alert(title='start', level=AlertLevel.INFO, text=f'{run_name}')


def alert_wandb(config, run_name, title):
    if config.use_wandb:
        wandb.alert(title='finished', level=AlertLevel.INFO, text=f'{run_name}')
