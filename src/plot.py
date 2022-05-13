import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def gen_trajectory(net, state0, prediction_steps = 1000):
    " Generate a trajectory of prediction_steps lenght starting from test_dataset[0]. Return np.array"
    state = torch.tensor(state0, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    h0 = torch.zeros(net.layers_num, 1,net.hidden_units)
    c0 = torch.zeros(net.layers_num, 1, net.hidden_units)
    rnn_state0 = (h0, c0)
    rnn_state = rnn_state0
    n_update = 2000# Reset the dynamics to a true data every n_update steps

    net_states = []
    net.eval()
   
    for i in range(prediction_steps):
        with torch.no_grad():
            net_states.append(state[-1].squeeze().numpy())
            """
            # Reset state 
            if i%n_update==0:
                state = torch.tensor(test_dataset[i], dtype=torch.float).unsqueeze(0).unsqueeze(0)
                rnn_state = rnn_state0
            """
            # Forward pass
            state, rnn_state = net(state, rnn_state)

    return np.array(net_states)
    



def plot_trajectory(t, time=None, filename = None, prediction_steps=100, labels=None, color=None):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=3)
    gs = axs[1].get_gridspec()
        
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        ax.set_xlabel("t")
        if time is None:
            ax.plot(t[:prediction_steps, -1], t[:prediction_steps, index], c=color)
        else:
            ax.plot(time[:prediction_steps], t[:prediction_steps, index], c=color)
        
        index += 1

   
    if labels is not None:
        axs[0].set_ylabel(labels[0])
        axs[1].set_ylabel(labels[1])
        axs[2].set_ylabel(labels[2])
    
    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig

def compare_trajectories(t1, t2, time=None, filename = None, prediction_steps=100, labels=None, color=None):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=3)
    gs = axs[1].get_gridspec()
    
        
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        ax.set_xlabel("t")
        if time is None:
            ax.plot(t1[:prediction_steps, -1], t1[:prediction_steps, index], label="Predicted", c=color)
            ax.plot(t2[:prediction_steps, -1], t2[:prediction_steps, index], label="Actual", c=color)
        else:
            ax.plot(time[:prediction_steps], t1[:prediction_steps, index], label="Predicted", c=color)
            ax.plot(time[:prediction_steps], t2[:prediction_steps, index], label="Actual", c=color)
        ax.legend(loc = "upper right", fontsize = "x-small")
        index += 1
    
    if labels is not None:
        axs[0].set_ylabel(labels[0])
        axs[1].set_ylabel(labels[1])
        axs[2].set_ylabel(labels[2])
    
    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig


def plot_3Dtrajectory(net_states, filename=None, color=None):
    ### Plot 3d trajectory
    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(net_states[:,0], net_states[:,1], net_states[:,2], s=0.1, c=color, cmap="Reds")
    ax.legend()
    if color is not None:
        fig.colorbar(cm.ScalarMappable(norm=Normalize(), cmap="Reds"), ax=ax)

    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
    
    
        
    return fig
    
