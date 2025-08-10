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
MODEL_TYPE_TORCH = torch.float16
BATCH_SIZE = 64
SHUFFLE    = False
TYPE       = "continuous"

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


class NeuralNetwork(nn.Module):
    def __init__(self, type: Literal["discrete", "continuous"]):
        """
        type is either discrete (boolean true/false on if red wins)
        or continuous (scores of Red, RedAuto, Blue, BlueAuto).
        """
        super().__init__()

        num_inputs = 12
        num_middle = 10
        if type == "discrete":
            num_outputs = 1
        else:
            num_outputs = 4

        self.flatten = nn.Flatten() # Not quite sure what this does

        # Create the stack of modules
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_inputs, num_middle),
            nn.ReLU(),
            nn.Linear(num_middle, num_outputs)
            # nn.Linear(28*28, 512),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 10),
        )

    def forward(self, x):
        # Unsure what any of this does, really. I'm assuming it evaluates the model?
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device): 
    """ This function was modified from the example at https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html """
    size = len(dataloader.dataset)
    logger.debug("[train] Training model")
    model.train() # Set the model in training mode
    logger.debug("[train] Model is in training mode")
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device=device, dtype=MODEL_TYPE_TORCH), y.to(device=device, dtype=MODEL_TYPE_TORCH)

        #logger.debug(f"X is type {type(X)}, y is type {type(y)}") # For heavy debug use only

        pred = model(X) # Make predictions for this batch

        # Compute prediction error
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step() # Adjust learning weights
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.debug(f"batch {batch:5}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    """ This function was modified from the example at https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device=device, dtype=MODEL_TYPE_TORCH), y.to(device=device, dtype=MODEL_TYPE_TORCH)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item() # Disabled because it's for classification models only
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



#endregion Functions
# Procedural

def main() -> None:
    torch.manual_seed(0) # TODO: Remove this later. it's for cacheing and speed
    # Main procedure with help from the PyTorch documentation: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
    logger.info(f"Using {device} device")
    logger.info("Loading data")
    
    #TODO: Figure out memory efficiency stuff later
    csv_data = FtcDataset(path="training_data.csv", type=TYPE)

    # Split into training and testing sets (with help from https://stackoverflow.com/a/51768651/25598210)
    train_dataset, test_dataset = torch.utils.data.random_split(csv_data, [0.8, 0.2])
    
    # Define data loader
    train_dataset_loaded = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test_dataset_loaded  = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    logger.debug("train dataset loaded:")
    logger.debug(train_dataset_loaded)
    logger.debug(train_dataset)
    logger.debug("test dataset loaded:")
    logger.debug(test_dataset_loaded)
    logger.debug(test_dataset)

    logger.info("Creating model...")
    model = NeuralNetwork(type=TYPE).to(device)

    logger.debug(f"Casting model to type {MODEL_TYPE_TORCH}") #TODO: Unsure if this is needed
    model = model.to(dtype=MODEL_TYPE_TORCH)
    
    logger.debug("Model: ")
    logger.debug(model)


    # Test INput
    # scoreRedFinal,scoreRedAuto,scoreBlueFinal,scoreBlueAuto,redOPR,               redAutoOPR,         redCCWM,            blueOPR,            blueAutoOPR,blueCCWM,recentredOPR,recentredAutoOPR,recentredCCWM,recentblueOPR,recentblueAutoOPR,recentblueCCWM,whoWon
    # 190,  75, 100,70,                                     295.35789473679995,     106.4017543859,     8.331578947399999,  282.0859259259,     97.3066666667,7.512592592599999,312.7883295194,113.0205949656,-2.042334096100001,307.2037037037,106.4444444445,7.7592592592999985,Red
    # 32,   8,  46, 11,                                     43.4265873016,          14.934523809600002, -6.4424603175,      34.2876984127,      6.351190476200001,6.585317460300001,43.4265873016,14.934523809600002,-6.4424603175,34.2876984127,6.351190476200001,6.585317460300001,Blue
    # 36,   11, 43, 11,                                     34.711864406800004,     6.1016949153,       0.32203389829999995,47.0677966102,      8.4858757062,-0.3502824857999993,34.711864406800004,6.1016949153,0.32203389829999995,47.0677966102,8.4858757062,-0.3502824857999993,Blue
    # 34,   5,  75, 11,                                     54.166666666699996,     9.3333333333,       2.1666666667000003, 51.6666666666,      12.666666666600001,8.3333333334,54.166666666699996,9.3333333333,2.1666666667000003,51.6666666666,12.666666666600001,8.3333333334,Blue
    # 36,   0,  26, 11,                                     61.1093951094,          10.2702702703,      4.378378378400001,  54.7927927928,      13.1351351351,12.1891891892,61.1093951094,10.2702702703,4.378378378400001,54.7927927928,13.1351351351,12.1891891892,Red
    # 29,   0,  68, 11,                                     57.0383333333,          10.094999999999999, 2.17,               52.3733333334,      13.36,9.96,57.0383333333,10.094999999999999,2.17,52.3733333334,13.36,9.96,Blue
    test_input = torch.tensor(data=[[295.35789473679995, 106.4017543859, 8.331578947399999, 282.0859259259, 97.3066666667, 7.512592592599999, 312.7883295194, 113.0205949656, -2.042334096100001, 307.2037037037, 106.4444444445, 7.7592592592999985]])

    #test_input = torch.rand(3, 12).to(device)
    test_input = test_input.to(device=device, dtype=MODEL_TYPE_TORCH)
    logger.debug("test input:")
    logger.debug(test_input)
    logits = model.linear_relu_stack(test_input)

    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)

    logger.debug("Prediction probabilities:")
    logger.debug(str(pred_probab))

    logger.info("Creating loss function...")
    loss_fn = nn.CrossEntropyLoss() # Using this for now bc it's the one used in the example, tune later
    logger.info("Creating optimizer...")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4) # Again used example, tune later.
    # lr is learning rate


    epochs = 5
    logger.info(f"Training with {epochs} epochs...")
    for t in range(epochs):
        logger.debug(f"Epoch {t+1}\n-------------------------------")
        train(dataloader=train_dataset_loaded, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
        test( dataloader=test_dataset_loaded,  model=model, loss_fn=loss_fn, device=device)
    logger.info("Training complete.")


    #print(f"Model structure: {model}\n\n")

    #for name, param in model.named_parameters():
    #    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")



if __name__ == "__main__":
    logger.info("Running as main.")
    main()



# -- End of file --