# PyTorch Test
# Started on 2025/08/04
# by Drew Wingfield

#region Imports
import logging

import torch # I used "python -m pip install -r .\requirements.txt --index-url https://download.pytorch.org/whl/cu128" - see https://pytorch.org/get-started/locally/ for details.
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

#endregion Imports

#region Setup
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)
logger.debug("Logging set up correctly.")

logger.info(f"CUDA: {torch.cuda.is_available()}")
logger.info(f"Found {torch.accelerator.device_count()} accelerator devices.")

#endregion Setup

def main() -> None:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logger.info(f"Using {device} device")


if __name__ == "__main__":
    logger.info("Running as main.")
    main()



# -- End of file --