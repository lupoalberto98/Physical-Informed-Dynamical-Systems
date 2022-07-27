import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm_notebook
from torch.utils.data import Dataset, DataLoader
import warnings

### Train epoch function
def train_epoch(net, device, dataloader, optimizer):
    """
    Train an epoch of data
    Args:
        net : pytorch_lightning class object
        device : training device (cuda/cpu)
        dataloader : dataloader of data
    Returns:
        mean(train_epoch_loss) : average epoch loss
    """
    # Set the train mode
    net.train()
    # List to save batch losses
    train_epoch_loss = []
    # Iterate the dataloader
    for batch_idx, batch in enumerate(dataloader):

        # Move to device
        batch = batch.to(device)
    
        # Compute loss
        loss = net.training_step(batch, batch_idx)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute batch losses
        train_batch_loss = loss.detach().cpu().numpy()
        train_epoch_loss.append(train_batch_loss)
        
    return np.mean(train_epoch_loss)
    

### Test epoch function
def val_epoch(net,  device, dataloader):
    """
    Validate an epoch of data
    -----------
    Parameters:
    net = network
    device = training device (cuda/cpu)
    dataloader = dataloader of data
    loss_function = loss function
    optimzer = optimizer used
    --------
    Returns:
    mean(val_epoch_loss) = average validation loss
    """
    # Set evaluation mode
    net.eval()
    # List to save evaluation losses
    val_epoch_loss = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
                
            # Move to device
            batch = batch.to(device)
            
            # Compute loss
            loss = net.validation_step(batch, batch_idx)

            # Compute batch_loss
            val_batch_loss = loss.detach().cpu().numpy()
            val_epoch_loss.append(val_batch_loss)

    return np.mean(val_epoch_loss)



### Create k-folds splits
def KF_split(k_fold, batch_size, dataset):
    """
    Split dataset in k-folds for cross validation procedure
    ___________
    Parameters:
    k_folds: number of folds to divide train dataset
    dataset: dataset to be divided into k-folds during each epoch
    ________
    Returns:
    dataloaders: list containing all k_fold dataloaders after splitting
    """
    # Take the length of dataset (pytorch dataset)
    len_dataset = dataset.__len__()
    
    # Define the k-folds
    len_fold = len_dataset//k_fold
    folds = torch.utils.data.random_split(dataset[:k_fold*len_fold], k_fold*[len_fold])
    
    # Defin empty list dataloaders
    kf_dataloaders = []
    
    for f in range(k_fold):
        dataloader = DataLoader(folds[f], batch_size=batch_size, shuffle=True, num_workers=0)
        kf_dataloaders.append(dataloader)
        
    return kf_dataloaders

### simple early stoppinng costume class
class early_stopping():
    
    def __init__(self, patience, mode="min"):
        """
        Initialize
        Args:
            patience : number of steps without improvement before stopping
            mod : "min" or "max"
        """
        self.patience = patience
        self.mode = mode
    
    def stop_training(self, loss):
        """
        Args:
            loss : metric to be evaluated, iterable object
        """
        break_loop = False
        trained_epochs = len(loss)
        if trained_epochs >= self.patience:
            if self.mode=="min":
                min_value = min(loss)
                if min(loss[-self.patience:]) > min_value:
                    warnings.warn("Training stopped at epoch %d"%trained_epochs)
                    break_loop = True
            
            elif self.mode=="max":
                max_value = max(loss)
                if max(loss[-self.patience:]) < max_value:
                    warnings.warn("Training stopped at epoch %d"%trained_epochs)
                    break_loop = True
                    
        return break_loop
    
### Train epochs with k-fold cross validation
def kf_train_epochs(net, device, k_fold, batch_size, dataset, optimizer, max_num_epochs, early_stopping=None):
    """
    """
    # Progress bar
    pbar = tqdm_notebook(range(max_num_epochs))

    # Inizialize empty lists to save fold list losses
    mean_train = []
    std_train = []
    mean_val = []
    std_val = []
    
    # Define dataloaders
    kf_dataloaders = KF_split(k_fold, batch_size, dataset)
    
    for epoch_num in pbar:
        
        # Empty lìsts to save fold losses
        train_loss_folds = []
        val_loss_folds = []
        
        # Iterate over each fold
        for f in range(k_fold):
                       
            # Compute validation loss on f fold
            val_loss_fold = val_epoch(net, device, kf_dataloaders[f])
            
            # Compute train loss on the other folds
            train_loss_fold = []
            
            for j in range(k_fold):
                if j != f:
                    # Train fold
                    mean_train_loss = train_epoch(net, device, kf_dataloaders[j], optimizer)
                    train_loss_fold.append(mean_train_loss)
            
            train_loss_fold = np.mean(train_loss_fold)
            
            # Append in list for each epoch
            train_loss_folds.append(train_loss_fold)
            val_loss_folds.append(val_loss_fold)
            
        
        
        # Set pbar description
        pbar.set_description("Train loss: %s" %round(np.mean(train_loss_folds),2)+", "+"Val loss %s" %round(np.mean(val_loss_folds),2))
       
        
                
        # Append fold losses lists on logs
        mean_train.append(np.mean(train_loss_folds))
        std_train.append(np.std(train_loss_folds))
        mean_val.append(np.mean(val_loss_folds))
        std_val.append(np.std(val_loss_folds))
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping.stop_training(mean_val):
                break
        
    return mean_train, std_train, mean_val, std_val
