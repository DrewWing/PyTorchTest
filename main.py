# PyTorch Test
# Started on 2025/08/04
# by Drew Wingfield

# Custom datasets with help from https://github.com/utkuozbulak/pytorch-custom-dataset-examples

#region Imports
print("Wait for imports...")
import logging
import os
from typing import Literal

import torch # I used "python -m pip install -r .\requirements.txt --index-url https://download.pytorch.org/whl/cu128" - see https://pytorch.org/get-started/locally/ for details.
from torch import nn
from torch.utils.data.dataset import Dataset

import pandas as pd
import numpy as np

#endregion Imports


SCORE_TYPE = np.uint8   # The numpy type for team scores (final and auto)
STATS_TYPE = np.float16 # The numpy type for team statistics (OPR, AutoOPR, CCWM, etc)
BATCH_SIZE = 64
SHUFFLE    = False

#region Setup
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)
logger.debug("Logging set up correctly.")

logger.info(f"CUDA: {torch.cuda.is_available()}")
logger.info(f"Found {torch.accelerator.device_count()} accelerator devices.")

#endregion Setup

#region Functions
def who_won_to_bool(x) -> bool:
    """ Returns true if x is Red, false otherwise. """
    return x in ("Red", "red")



class FtcDataset(Dataset):
    def __init__(self, path: str | os.PathLike, type: Literal["discrete","continuous"]):
        """ 
        First Tech Challenge Dataset.

        path is the path to a csv file.

        type is either discrete (boolean true/false on if red wins)
        or continuous (scores of Red, RedAuto, Blue, BlueAuto).
        """
        self.path = path
        self.type = type

        logger.debug(f'[FtcDataset][__init__] Loading FTC Dataset with type "{type} from path "{path}"')

        # Load the data
        self.data_full = pd.read_csv(path, dtype={
            "scoreRedFinal": SCORE_TYPE,"scoreRedAuto": SCORE_TYPE, "scoreBlueFinal": SCORE_TYPE,"scoreBlueAuto": SCORE_TYPE, 
            "redOPR" : np.float16, "redAutoOPR" : STATS_TYPE, "redCCWM" : STATS_TYPE,
            "blueOPR": STATS_TYPE, "blueAutoOPR": STATS_TYPE, "blueCCWM": STATS_TYPE,
            "recentredOPR" : STATS_TYPE, "recentredAutoOPR" : STATS_TYPE, "recentredCCWM" : STATS_TYPE,
            "recentblueOPR": STATS_TYPE, "recentblueAutoOPR": STATS_TYPE, "recentblueCCWM": STATS_TYPE
            })
        
        # Split the data
        self.input_arr = self.data_full[[
            "redOPR","redAutoOPR","redCCWM",
            "blueOPR","blueAutoOPR","blueCCWM",
            "recentredOPR","recentredAutoOPR","recentredCCWM",
            "recentblueOPR","recentblueAutoOPR","recentblueCCWM"
            ]].to_numpy()
        logger.debug("[FtcDataset][__init__] input_arr:")
        logger.debug(self.input_arr)

        #self.input_arr = torch.from_numpy(self.input_arr)
        # logger.debug("X data torch tensor from numpy:")
        # logger.debug(x_data)# Move the tensor to accelerator if available
        #if torch.accelerator.is_available():
        #    self.input_arr = self.input_arr.to(torch.accelerator.current_accelerator())

        # logger.debug(f"Shape of tensor: {x_data.shape}")
        # logger.debug(f"Datatype of tensor: {x_data.dtype}")
        # logger.debug(f"Device tensor is stored on: {x_data.device}")

        if type == "continuous":
            self.label_arr = self.data_full[["scoreRedFinal","scoreRedAuto","scoreBlueFinal","scoreBlueAuto"]].to_numpy()
            # logger.debug("Y data:")
            # logger.debug(y_data)
        
        else:
            # Vectorization with help from https://stackoverflow.com/a/46470401/25598210
            self.label_arr = np.vectorize(who_won_to_bool)(self.data_full[["whoWon"]])
            # logger.debug("Y bool data:")
            # logger.debug(y_bool_data)


        logger.debug("[FtcDataset][__init__] label_arr:")
        logger.debug(self.label_arr)

        logger.debug(f"[FtcDataset][__init__] input_arr.shape={self.input_arr.shape}  label_arr.shape={self.label_arr.shape}")        
        assert self.input_arr.shape[0] == self.label_arr.shape[0]
        logger.debug("[FtcDataset][__init__] Initializatin complete.")

        
    def __getitem__(self, index):
        """ Returns tensor, label. """
        # Get the data
        item = self.input_arr[index] # TODO: Add index range validation later
        label = self.label_arr[index] # TODO: Add index range validation later

        # Turn the item into a tensor
        item = torch.tensor(item)

        # Return the tensor and label
        return (item, label)

    def __len__(self):
        return self.input_arr.shape[0] # of how many examples(images?) you have


#endregion Functions
# Procedural

def main() -> None:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
    logger.info(f"Using {device} device")
    logger.info("Loading data")
    
    #TODO: Figure out memory efficiency stuff later
    csv_data = FtcDataset(path="training_data.csv", type="continuous")

    # Split into training and testing sets (with help from https://stackoverflow.com/a/51768651/25598210)
    train_dataset, test_dataset = torch.utils.data.random_split(csv_data, [0.8, 0.2])
    
    # Define data loader
    train_dataset_loaded = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE) 
    test_dataset_loaded  = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE, shuffle=SHUFFLE) 
    



if __name__ == "__main__":
    logger.info("Running as main.")
    main()



# -- End of file --