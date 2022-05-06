#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm


### Train epoch function
def pi_train_epoch(net, device, dataloader, loss_fn, optimizer, epoch_num):
    """
    Train an epoch of data (sweep through dataloader)
    -----------
    Parameters:
    net = network
    device = training device (cuda/cpu)
    dataloader = dataloader of data
    loss_function = loss function
    optimzer = optimizer used
    --------
    Returns:
    mean(train_epoch_loss) = average epoch loss
    """
    # Set train mode
    net.train()
    
    # List to append losses
    train_epoch_losses = []
      
    # Loop over other steps

    for batch_sample in dataloader:

        ### Move samples to the proper device
        batch_sample = batch_sample.to(device)

        ### Prepare network input and labels and first net_out for curriculum learning
        state  = batch_sample[:, :-1, :]
        labels = batch_sample[:, 1:, :]

        ### Forward pass
        # Clear previous recorded gradients
        optimizer.zero_grad()
        
        # Forward pass
        next_state, _ = net(state) # we do not need the rnn state at this point, we can ignore the output with "_"
    
        ### Update network
        # Evaluate loss 
        loss = loss_fn(state, next_state, labels, epoch_num)
    
        # Backward pass
        loss.backward()
        
        # Update
        optimizer.step()
        
        # Save batch loss
        train_epoch_losses.append(loss.detach().cpu().numpy())
    
        
        
    return np.mean(train_epoch_losses)

### Validation epoch function
def pi_val_epoch(net,  device, dataloader, loss_fn, epoch_num):
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
        for batch_sample in dataloader:
                
            # Move to device
            batch_sample = batch_sample.to(device)
            
            ### Prepare network input and labels
            state  = batch_sample[:, :-1, :]
            labels = batch_sample[:, 1:, :]
      
            # Forward pass
            next_state, _ = net(state) 

            # Compute loss
            loss = loss_fn(state, next_state, labels, epoch_num)
            
            # Compute batch_loss
            val_epoch_loss.append(loss.detach().cpu().numpy())
       
    return np.mean(val_epoch_loss)

### Training epochs
def pi_train(net, device, train_dataloader, val_dataloader, loss_fn, optimizer, max_num_epochs, early_stopping = False):
    """
    Train an epoch
    ___________
    Parameters:
    max_num_epochs: maximum number of epochs (sweeps tthrough the datasets) to train the model
    early_stopping: if true stop the training if the last validation loss is greater 
    than the average of last 100 epochs
    """
    
    # Progress bar
    pbar = tqdm(range(max_num_epochs))

    # Inizialize empty lists to save losses
    train_loss_log = []
    val_loss_log = []

    for epoch_num in pbar:

        # Train epoch
        mean_train_loss = pi_train_epoch(net, device, train_dataloader, loss_fn, optimizer, epoch_num)

        # Validate epoch
        mean_val_loss = pi_val_epoch(net, device, val_dataloader, loss_fn, epoch_num)

        # Append losses and accuracy
        train_loss_log.append(mean_train_loss)
        val_loss_log.append(mean_val_loss)

        # Set pbar description
        pbar.set_description("Train loss: %s" %round(mean_train_loss,2)+", "+"Val loss %s" %round(mean_val_loss,2))
        
        # Early stopping
        if early_stopping:
            if epoch_num>100 and np.mean(val_loss_log[-100:]) < np.mean(val_loss_log[-10:]):
                print("Training stopped at epoch "+str(epoch_num)+" to avoid overfitting.")
                break
    
    return train_loss_log, val_loss_log



### For transformer
### Train epoch function
def tf_train_epoch(net, tgt_mask, device, dataloader, loss_fn, optimizer, epoch_num):
    """
    Train an epoch of data (sweep through dataloader)
    -----------
    Parameters:
    net = network
    device = training device (cuda/cpu)
    dataloader = dataloader of data
    loss_function = loss function
    optimzer = optimizer used
    --------
    Returns:
    mean(train_epoch_loss) = average epoch loss
    """
    # Set train mode
    net.train()
    
    # List to append losses
    train_epoch_losses = []
    
    for batch_sample in dataloader:

        ### Move samples to the proper device
        batch_sample = batch_sample.to(device)

        ### Prepare network input and labels
        net_input  = batch_sample[:, :-1, :]
        labels = batch_sample[:, 1:, :]

        ### Forward pass
        # Clear previous recorded gradients
        optimizer.zero_grad()
        # Forward pass
        net_out[:,1:,:] = net(net_input, net_input, labels, labels) 
    
        ### Update network
        # Evaluate loss 
        loss = loss_fn(state, next_state, labels, epoch_num)
    
        # Backward pass
        loss.backward()
        
        # Update
        optimizer.step()
        
        # Save batch loss
        train_epoch_losses.append(loss.detach().cpu().numpy())
    
    return np.mean(train_epoch_losses)

### Validation epoch function
def tf_val_epoch(net, tgt_mask, device, dataloader, loss_fn, epoch_num):
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
        for batch_sample in dataloader:
                
            # Move to device
            batch_sample = batch_sample.to(device)
            
            ### Prepare network input and labels
            net_input  = batch_sample[:, :-1, :]
            labels = batch_sample[:, 1:, :]

            # Forward pass
            net_out[:,1:,:] = net(net_input, net_input, labels, labels) 

            # Compute loss
            loss = loss_fn(state, next_state, labels, epoch_num)
            
            # Compute batch_loss
            val_epoch_loss.append(loss.detach().cpu().numpy())

    return np.mean(val_epoch_loss)

### Training epochs
def tf_train(net, tgt_mask, device, train_dataloader, val_dataloader, loss_fn, optimizer, max_num_epochs, early_stopping = False):
    """
    Train an epoch
    ___________
    Parameters:
    max_num_epochs: maximum number of epochs (sweeps tthrough the datasets) to train the model
    early_stopping: if true stop the training if the last validation loss is greater 
    than the average of last 100 epochs
    """
    
    # Progress bar
    pbar = tqdm(range(max_num_epochs))

    # Inizialize empty lists to save losses
    train_loss_log = []
    val_loss_log = []

    for epoch_num in pbar:

        # Train epoch
        mean_train_loss = tf_train_epoch(net, tgt_mask, device, train_dataloader, loss_fn, optimizer, epoch_num)

        # Validate epoch
        mean_val_loss = tf_val_epoch(net, tgt_mask, device, val_dataloader, loss_fn, epoch_num)

        # Append losses and accuracy
        train_loss_log.append(mean_train_loss)
        val_loss_log.append(mean_val_loss)

        # Set pbar description
        pbar.set_description("Train loss: %s" %round(mean_train_loss,2)+", "+"Val loss %s" %round(mean_val_loss,2))
        
        # Early stopping
        if early_stopping:
            if epoch_num>100 and np.mean(val_loss_log[-100:]) < np.mean(val_loss_log[-10:]):
                print("Training stopped at epoch "+str(epoch_num)+" to avoid overfitting.")
                break
    
    return train_loss_log, val_loss_log