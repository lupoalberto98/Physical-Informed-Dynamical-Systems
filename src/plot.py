import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy import signal

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
    



def plot_trajectory(t, time=None, n_var=3, filename = None, prediction_steps=100, labels=None, color=None):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=n_var)
    gs = axs[1].get_gridspec()
        
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        ax.set_xlabel("t")
        if time is None:
            ax.plot(t[:prediction_steps, -1], t[:prediction_steps, index], c=color, lw=0.5)
        else:
            ax.plot(time[:prediction_steps], t[:prediction_steps, index], c=color, lw=0.5)
        
        index += 1

   
    # Set labels
    for i in range(n_var):
        axs[i].set_ylabel("x"+str(i+1))
    
    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig

def compare_trajectories(t1, t2, time=None, n_var=3, filename = None, prediction_steps=100, labels=None, color=None):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=n_var)
    
        
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        ax.set_xlabel("t")
        if time is None:
            ax.plot(t1[:prediction_steps, -1], t1[:prediction_steps, index], label="Predicted", c=color, lw=0.5)
            ax.plot(t2[:prediction_steps, -1], t2[:prediction_steps, index], label="Actual", c=color, lw=0.5)
        else:
            ax.plot(time[:prediction_steps], t1[:prediction_steps, index], label="Predicted", c=color, lw=0.5)
            ax.plot(time[:prediction_steps], t2[:prediction_steps, index], label="Actual", c=color, lw=0.5)
        ax.legend(loc = "upper right", fontsize = "x-small")
        index += 1
    
    # Set labels
    for i in range(n_var):
        axs[i].set_ylabel("x"+str(i+1))
        
    
    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig


def plot_3Dtrajectory(net_states, var=[0,1,2], filename=None, color=None):
    ### Plot 3d trajectory
    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(net_states[:,var[0]], net_states[:,var[1]], net_states[:,var[2]], cmap = "RdBu_r", s=0.1, c=color)
    ax.legend()
    if color is not None:
        fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=min(color), vmax=max(color)), cmap="RdBu_r"), ax=ax)

    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig
    

def poincare_plot(states, true_states=None, n_var=3, filename=None, prediction_steps=1000):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=n_var)
    gs = axs[1].get_gridspec()
        
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        ax.set_xlabel("$x$"+str(index+1))
        ax.scatter(states[:prediction_steps, index], states[1:prediction_steps+1,index], s=0.2, label="Predicted") 
        if true_states is  not None:
            ax.scatter(true_states[:prediction_steps, index], true_states[1:prediction_steps+1,index], s=0.2, label="Actual")
            ax.legend(loc = "upper right", fontsize = "x-small")
        index += 1
        

   
    # Set labels
    for i in range(n_var):
        axs[i].set_ylabel("y"+str(i+1))
    
    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig

def plot_powspec(states, true_states=None, n_var=3, filename=None):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=n_var)
    gs = axs[1].get_gridspec()
        
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        ax.set_xlabel("Frequency")
        f, P = signal.periodogram(states[:,index])
        ax.semilogy(f[1:], P[1:], label="Predicted") 
        if true_states is  not None:
            f, P = signal.periodogram(true_states[:,index])
            ax.semilogy(f[1:], P[1:], label="Actual") 
            ax.legend(loc = "upper right", fontsize = "x-small")
        index += 1
        

   
    # Set labels
    for i in range(n_var):
        axs[i].set_ylabel("PS of (x"+str(i+1)+")")
    
    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig

