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


SCORE_TYPE = np.uint8   # The numpy type for team scores (final and auto)
STATS_TYPE = np.float16 # The numpy type for team statistics (OPR, AutoOPR, CCWM, etc)


#region Setup
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)
logger.debug("Logging set up correctly.")

logger.info(f"CUDA: {torch.cuda.is_available()}")
logger.info(f"Found {torch.accelerator.device_count()} accelerator devices.")

#endregion Setup


def who_won_to_bool(x) -> bool:
    """ Returns true if x is Red, false otherwise. """
    return x in ("Red", "red")



def main() -> None:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
    logger.info(f"Using {device} device")
    logger.info("Loading data")

    training_data = pd.read_csv("training_data.csv", dtype={
        "scoreRedFinal": SCORE_TYPE,"scoreRedAuto": SCORE_TYPE, "scoreBlueFinal": SCORE_TYPE,"scoreBlueAuto": SCORE_TYPE, 
        "redOPR" : np.float16, "redAutoOPR" : STATS_TYPE, "redCCWM" : STATS_TYPE,
        "blueOPR": STATS_TYPE, "blueAutoOPR": STATS_TYPE, "blueCCWM": STATS_TYPE,
        "recentredOPR" : STATS_TYPE, "recentredAutoOPR" : STATS_TYPE, "recentredCCWM" : STATS_TYPE,
        "recentblueOPR": STATS_TYPE, "recentblueAutoOPR": STATS_TYPE, "recentblueCCWM": STATS_TYPE
        })
    
    #TODO: Figure out memory efficiency stuff later
    
    logger.debug("Training data: ")
    logger.debug(training_data)

    x_data = training_data[["redOPR","redAutoOPR","redCCWM","blueOPR","blueAutoOPR","blueCCWM","recentredOPR","recentredAutoOPR","recentredCCWM","recentblueOPR","recentblueAutoOPR","recentblueCCWM"]].to_numpy()
    logger.debug("x_data (limited to input data): ")
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

    # Vectorization with help from https://stackoverflow.com/a/46470401/25598210
    y_bool_data = np.vectorize(who_won_to_bool)(training_data[["whoWon"]])
    logger.debug("Y bool data:")
    logger.debug(y_bool_data)
    



if __name__ == "__main__":
    logger.info("Running as main.")
    main()



# -- End of file --