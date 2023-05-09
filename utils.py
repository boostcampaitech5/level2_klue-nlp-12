import os
import torch
import random
import logging
import numpy as np

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