# PyTorch Test
# Started on 2025/08/04
# by Drew Wingfield

# Custom datasets with help from https://github.com/utkuozbulak/pytorch-custom-dataset-examples

#region Imports
print("Wait for imports...")

print("  - Builtins")
import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)
logger.debug("Logging set up correctly.")

logger.info("  - Torch")
import torch # I used "python -m pip install -r .\requirements.txt --index-url https://download.pytorch.org/whl/cu128" - see https://pytorch.org/get-started/locally/ for details.
from torch import nn
from torch.utils.data.dataset import Dataset

logger.info("  - Scikit preprocessing")
from sklearn.preprocessing import RobustScaler

logger.info("  - Pandas")
import pandas as pd
logger.info("  - Numpy")
import numpy as np

logger.info("  - Joblib")
from joblib import dump, load

logger.info("Imports complete.")
#endregion Imports


SCORE_TYPE = np.int16   # The numpy type for team scores (final and auto)
STATS_TYPE = np.float16 # The numpy type for team statistics (OPR, AutoOPR, CCWM, etc)
MODEL_TYPE_TORCH = torch.float16
BATCH_SIZE = 64
SHUFFLE    = False
TYPE       = "continuous"

#region Setup
logger.info(f"CUDA: {torch.cuda.is_available()}")
logger.info(f"Found {torch.accelerator.device_count() if torch.cuda.is_available() else 0} accelerator devices.")

#endregion Setup

#region Functions
def who_won_to_bool(x) -> bool:
    """ Returns true if x is Red, false otherwise. """
    return x in ("Red", "red")


class FtcDataset(Dataset):
    def __init__(
            self, path: str | os.PathLike, 
            type: Literal["discrete","continuous"], 
            path_data_scalar: str = "", 
            path_label_scalar: str= "",
            disable_scaling: bool = False
            ):
        """ 
        First Tech Challenge Dataset.

        path is the path to a csv file.

        type is either discrete (boolean true/false on if red wins)
        or continuous (scores of Red, RedAuto, Blue, BlueAuto).

        path_data_scalar is the path to load a saved scalar from, or blank for a new Scalar object.
        if disable_scaling is True, disables all scaling
        """
        self.path = path
        self.type = type
        if path_data_scalar == "" or disable_scaling: # TODO: make into a load_scalar func
            self.data_scalar = None
        else:
            try:
                self.data_scalar = load(path_data_scalar)
            except Exception as e:
                logger.error(f"[FtcDataset][__init__] Loading a data scalar from path ({path_data_scalar}) failed.")
                raise e
        
        if path_label_scalar == "" or disable_scaling:
            self.label_scalar = None
        else:
            try:
                self.label_scalar = load(path_label_scalar)
            except Exception as e:
                logger.error(f"[FtcDataset][__init__] Loading a label scalar from path ({path_label_scalar}) failed.")
                raise e

        logger.debug(f'[FtcDataset][__init__] Loading FTC Dataset with type "{type} from path "{path}"')

        #region loading
        # Load the data
        self.data_full = pd.read_csv(path, dtype={
            "scoreRedFinal": SCORE_TYPE,"scoreRedAuto": SCORE_TYPE, "scoreBlueFinal": SCORE_TYPE,"scoreBlueAuto": SCORE_TYPE, 
            "redOPR" : STATS_TYPE, "redAutoOPR" : STATS_TYPE, "redCCWM" : STATS_TYPE,
            "blueOPR": STATS_TYPE, "blueAutoOPR": STATS_TYPE, "blueCCWM": STATS_TYPE,
            "recentredOPR" : STATS_TYPE, "recentredAutoOPR" : STATS_TYPE, "recentredCCWM" : STATS_TYPE,
            "recentblueOPR": STATS_TYPE, "recentblueAutoOPR": STATS_TYPE, "recentblueCCWM": STATS_TYPE
            })
        

        self.data_full.fillna(0, inplace=True) # Replace all nan values with zero
        
        # Split the data
        self.data_arr = self.data_full[[
            "redOPR","redAutoOPR","redCCWM",
            "blueOPR","blueAutoOPR","blueCCWM",
            "recentredOPR","recentredAutoOPR","recentredCCWM",
            "recentblueOPR","recentblueAutoOPR","recentblueCCWM"
            ]]#.to_numpy()
        logger.debug("[FtcDataset][__init__] data_arr:")
        logger.debug(self.data_arr)

        #self.data_arr = torch.from_numpy(self.data_arr)
        # logger.debug("X data torch tensor from numpy:")
        # logger.debug(x_data)# Move the tensor to accelerator if available
        #if torch.accelerator.is_available():
        #    self.data_arr = self.data_arr.to(torch.accelerator.current_accelerator())

        # logger.debug(f"Shape of tensor: {x_data.shape}")
        # logger.debug(f"Datatype of tensor: {x_data.dtype}")
        # logger.debug(f"Device tensor is stored on: {x_data.device}")

        if type == "continuous":
            self.label_arr = self.data_full[["scoreRedFinal","scoreRedAuto","scoreBlueFinal","scoreBlueAuto"]]
            # logger.debug("Y data:")
            # logger.debug(y_data)
        
        else:
            # Vectorization with help from https://stackoverflow.com/a/46470401/25598210
            self.label_arr = np.vectorize(who_won_to_bool)(self.data_full[["whoWon"]])
            # logger.debug("Y bool data:")
            # logger.debug(y_bool_data)


        logger.debug("[FtcDataset][__init__] label_arr:")
        logger.debug(self.label_arr)

        logger.debug(f"[FtcDataset][__init__] data_arr.shape={self.data_arr.shape}  label_arr.shape={self.label_arr.shape}")        
        assert self.data_arr.shape[0] == self.label_arr.shape[0]
        #endregion loading


        if disable_scaling:
            logger.debug("[FtcDataset][__init__] Scaling disabled. Skipping scaling steps.")
        else:
            logger.debug("[FtcDataset][__init__] Scaling data...")
            self.data_arr = self.scale_data(data_to_scale=self.data_arr, scalar=self.data_scalar)

            if type == "continuous":
                logger.debug("[FtcDataset][__init__] Scaling labels...")
                self.label_arr = self.scale_data(data_to_scale=self.label_arr, scalar=self.label_scalar)

        try:
            self.label_arr = self.label_arr.to_numpy()
            self.data_arr  = self.data_arr.to_numpy()
        except AttributeError as e:
            pass # If scaling worked correctly, this throws errors. Not sure why but whatever.

        logger.debug("[FtcDataset][__init__] Initializatin complete.")

    # TODO: add __repr__ and other class funcs.

    def scale_data(self, data_to_scale, scalar: None | RobustScaler):
        """ Returns a scaled version of the data. MAY OR MAY NOT DEEP COPY. If no scalar present, creates one. """

        if scalar is None:
            # If not loading a scalar, create and fit one
            logger.debug("[FtcDataset][scale_data] No scalar loaded. Creating and fitting one...")
            scalar = RobustScaler()
            scalar.fit(data_to_scale)

        # Actually scale the array
        # With help from the scikit learn docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
        scaled_data = scalar.transform(data_to_scale) # pyright: ignore[reportOptionalMemberAccess]

        logger.debug("[FtcDataset][__init__] Scaling complete. Saled dataset:")
        logger.debug(scaled_data)

        return scaled_data
        
        # TODO: Implement saving scalarsd

    def __getitem__(self, index):
        """ Returns tensor, label. """
        # Get the data
        item = self.data_arr[index] # TODO: Add index range validation later
        label = self.label_arr[index] # TODO: Add index range validation later

        # Turn the item into a tensor
        item = torch.tensor(item)

        # Return the tensor and label
        return (item, label)

    def __len__(self):
        return self.data_arr.shape[0] # of how many examples(images?) you have


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
        #x = self.flatten(x) # Not necessary
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model: NeuralNetwork, loss_fn, optimizer, device): 
    """ This function was modified from the example at https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html """
    size = len(dataloader.dataset)
    logger.debug("[train]")
    logger.debug("[train]")

    logger.debug("[train] - - - - - - - - ")
    logger.debug("[train] Training model")
    logger.debug("[train]")
    logger.debug("[train]")


    model.train() # Set the model in training mode
    logger.debug("[train] Model is in training mode")
    train_losses = []
    matches_correct = [] # An item for every match set, 1 if all correct and 0 if all incorrect

    for batch, (X, y) in enumerate(dataloader):
        logger.debug(f"[train] batch {batch:5}")

        
        X, y = X.to(device=device, dtype=MODEL_TYPE_TORCH), y.to(device=device, dtype=MODEL_TYPE_TORCH)

        #logger.debug(f"X is type {type(X)}, y is type {type(y)}") # For heavy debug use only

        if torch.any(torch.isnan(X)):
            assert False # NaNs found in input X!
        
        if torch.any(torch.isnan(y)):
            assert False # NaNs found in input y!

        pred = model(X) # Make predictions for this batch

        # Compute prediction error
        loss = loss_fn(pred, y)

        # print("x")
        # print(X)
        # print("\n\ny")
        # print(y)
        # print("\n\n")
        # print("pred")
        # print(pred)

        #corr = (y[0] > y[2]) == (pred[0] > pred[2])
        corr = torch.where((y[:,0] > y[:,2]) == (pred[:,0] > pred[:,2]), 1, 0) # TODO: Add a variable to disable stats stuff later.
        #matches_correct.append(torch.mean(corr))
        # print("\n\n corr")
        # print(corr)

        # print(f"\n\nlen success: {len(corr)}")
        # print(f"\n\nlen total: {len(y)}")
        matches_correct.append( corr.to(device='cpu').numpy().mean() )
        # exit()

        try:
            train_losses.append(loss.item()) # Pushing it back to the CPU with help from https://stackoverflow.com/a/72742578/25598210
        except Exception as e:
            pass
            # If it's NaN or something, we'll catch it later

        logger.debug(f"[train]   Loss={loss} ({type(loss)})")

        
        optimizer.zero_grad()

        # Backpropagation
        try:
            loss.backward()
        except RuntimeError as e:
            logger.error("[train] Error occured!")
            logger.error("[train] X: ")
            logger.error(X)
            logger.error("[train] y: ")
            logger.error(y)
            logger.error("[train] model parameteres:")
            for par in model.parameters():
                logger.error("  - "+str(par))
            logger.error("[train] grad:")
            logger.error(X.grad)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip the gradient norm to prevent nan and infinite values
            try:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1) # Clip the gradient norm to prevent nan and infinite values
                logger.error("[train] Clipped parameters:")
                for par in model.parameters():
                    logger.error("  - "+str(par))
            except:
                logger.info("[train[ Unable to clip parameters.")
            raise e

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip the gradient norm to prevent nan and infinite values
        optimizer.step() # Adjust learning weights
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.debug(f"[train] batch {batch:5}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return np.mean(train_losses), np.mean(matches_correct)


def test(dataloader, model: NeuralNetwork, loss_fn, device):
    """ This function was modified from the example at https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device=device, dtype=MODEL_TYPE_TORCH), y.to(device=device, dtype=MODEL_TYPE_TORCH)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item() # Disabled because it's for classification models only
            corr = torch.where((y[:,0] > y[:,2]) == (pred[:,0] > pred[:,2]), 1, 0) # TODO: Add a variable to disable stats stuff later.
            correct.append(corr.to(device="cpu").numpy().mean())
    
    test_loss /= num_batches
    correct = np.mean(correct)
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    logger.debug(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss, correct



def graph_losses(validation_losses, train_losses, validation_correct, train_correct):
    """ Uses MatPlotLib to plot thhe loss and correct percentage graphs as a function of epochs. """
    from matplotlib import pyplot as plt
    plt.plot(validation_losses, label='Validation Loss')
    plt.plot(train_losses, label='Train Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    #plt.ylim(bottom=0)
    plt.show()
    plt.plot(validation_correct, label='Validation Matches Correct')
    plt.plot(train_correct, label='Train Matches Correct')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


#endregion Functions
# Procedural

def main() -> None:
    torch.manual_seed(0) # TODO: Remove this later. it's for cacheing and speed
    torch.autograd.set_detect_anomaly(True) # Halts training when something goes wrong
    
    # Add file handler to logging
    fileout = logging.FileHandler("log.log")
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fileout.setFormatter(formatter)
    logger.addHandler(fileout)

    
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

    logger.info("Creating loss function...")
    #loss_fn = nn.CrossEntropyLoss() # Using this for now bc it's the one used in the example, tune later
    loss_fn = nn.L1Loss()
    logger.info("Creating optimizer...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # lr is learning rate



    epochs = 50
    validation_loss_graph = []
    train_loss_graph = []
    train_m_corr_graph = [] # Train matches correct
    valid_m_corr_graph = []

    logger.info(f"Training with {epochs} epochs...")
    for t in range(epochs):
        logger.debug("Parameters:")
        for par in model.parameters():
            logger.debug("  - "+str(par))

        logger.debug("\n-------------------------------")
        logger.info(f"Epoch {t+1}")

        train_loss, train_m_corr = train(dataloader=train_dataset_loaded, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
        train_loss_graph.append(train_loss)
        train_m_corr_graph.append(train_m_corr)
        
        validation_loss, valid_m_corr = test( dataloader=test_dataset_loaded,  model=model, loss_fn=loss_fn, device=device)
        validation_loss_graph.append(validation_loss)
        valid_m_corr_graph.append(valid_m_corr)

        logger.info(f"    Training loss: {train_loss:.4f} Validation loss: {validation_loss:.4f}     Train Matches Correct: {train_m_corr:.2%}%  Validation Matches Correct: {valid_m_corr:.2%}")


    logger.info("Training complete.")
    logger.info("Loss graph:")
    logger.info(validation_loss_graph)
    a = [(i, validation_loss_graph[i]) for i in range(len(validation_loss_graph)) ]
    logger.info(str(a))


    graph_losses(
        train_losses        = train_loss_graph, 
        validation_losses   = validation_loss_graph, 
        train_correct       = train_m_corr_graph, 
        validation_correct  = valid_m_corr_graph
    )

    #print(f"Model structure: {model}\n\n")

    #for name, param in model.named_parameters():
    #    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")



if __name__ == "__main__":
    logger.info("Running as main.")
    main()



# -- End of file --