# PyTorch Test
# Started on 2025/08/04
# by Drew Wingfield

#region Imports
print("Wait for imports...")
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
    logger.info("Loading data")
    training_data = pd.read_csv("training_data.csv", dtype={
        "scoreRedFinal": np.uint8,"scoreRedAuto": np.uint8, "scoreBlueFinal": np.uint8,"scoreBlueAuto": np.uint8, 
        "redOPR": np.float16,"redAutoOPR": np.float16,"redCCWM": np.float16,
        "blueOPR": np.float16,"blueAutoOPR": np.float16,"blueCCWM": np.float16,
        "recentredOPR": np.float16,"recentredAutoOPR": np.float16,"recentredCCWM": np.float16,
        "recentblueOPR": np.float16,"recentblueAutoOPR": np.float16,"recentblueCCWM": np.float16
        })
    logger.debug("Training data: ")
    logger.debug(training_data)

    x_data = training_data[["redOPR","redAutoOPR","redCCWM","blueOPR","blueAutoOPR","blueCCWM","recentredOPR","recentredAutoOPR","recentredCCWM","recentblueOPR","recentblueAutoOPR","recentblueCCWM"]].to_numpy()
    logger.debug(x_data)

    x_data = torch.from_numpy(x_data)
    logger.debug("X data torch tensor from numpy:")
    logger.debug(x_data)

    # Move the tensor to accelerator if available
    if torch.accelerator.is_available():
        x_data = x_data.to(torch.accelerator.current_accelerator())

    logger.debug(f"Shape of tensor: {x_data.shape}")
    logger.debug(f"Datatype of tensor: {x_data.dtype}")
    logger.debug(f"Device tensor is stored on: {x_data.device}")

    y_data = training_data[["scoreRedFinal","scoreRedAuto","scoreBlueFinal","scoreBlueAuto"]].to_numpy()
    logger.debug("Y data:")
    logger.debug(y_data)
    



if __name__ == "__main__":
    logger.info("Running as main.")
    main()



# -- End of file --