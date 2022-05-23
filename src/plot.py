import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy import signal


def gen_trajectory(net, state0, dt=0.01, dd_mode=False, prediction_steps = 1000):
    " Generate a trajectory of prediction_steps lenght starting from test_dataset[0]. Return np.array"
    state = state0.unsqueeze(0).unsqueeze(0)
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
            # Forward past
            if dd_mode is False:
                state, rnn_state = net(i, state, rnn_state)
            else:
                k1, rnn_state = net(i, state, rnn_state)
                k2, rnn_state = net(i, state+dt*k1/2., rnn_state)
                k3, rnn_state = net(i, state+dt*k2/2., rnn_state)
                k4, rnn_state = net(i, state+dt*k3, rnn_state)
                df = dt*(k1+2*k2+2*k3+k4)/6.0
                state = state+df

    return torch.tensor(net_states)
    



def plot_trajectory(t, time=None, n_var=3, filename = None, prediction_steps=100, labels=None, color=None):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=n_var)
    gs = axs[1].get_gridspec()
        
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        ax.set_xlabel("t")
        if time is None:
            ax.plot(t.detach().cpu().numpy()[:prediction_steps, -1], t.detach().cpu().numpy()[:prediction_steps, index], c=color, lw=0.5)
        else:
            ax.plot(time.detach().cpu().numpy()[:prediction_steps], t.detach().cpu().numpy()[:prediction_steps, index], c=color, lw=0.5)
        
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
            ax.plot(t1.detach().cpu().numpy()[:prediction_steps, -1], t1.detach().cpu().numpy()[:prediction_steps, index], label="Predicted", c=color, lw=0.5)
            ax.plot(t2.detach().cpu().numpy()[:prediction_steps, -1], t2.detach().cpu().numpy()[:prediction_steps, index], label="Actual", c=color, lw=0.5)
        else:
            ax.plot(time.detach().cpu().numpy()[:prediction_steps], t1.detach().cpu().numpy()[:prediction_steps, index], label="Predicted", c=color, lw=0.5)
            ax.plot(time.detach().cpu().numpy()[:prediction_steps], t2.detach().cpu().numpy()[:prediction_steps, index], label="Actual", c=color, lw=0.5)
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
    ax.scatter(net_states.detach().cpu().numpy()[:,var[0]], net_states.detach().cpu().numpy()[:,var[1]], net_states.detach().cpu().numpy()[:,var[2]], cmap = "RdBu_r", s=0.1, c=color)
    ax.legend()
    if color is not None:
        fig.colorbar(cm.ScalarMappable( cmap="RdBu_r"), ax=ax)

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
        ax.scatter(states.detach().cpu().numpy()[:prediction_steps, index], states.detach().cpu().numpy()[1:prediction_steps+1,index], s=0.2, label="Predicted") 
        if true_states is  not None:
            ax.scatter(true_states.detach().cpu().numpy()[:prediction_steps, index], true_states.detach().cpu().numpy()[1:prediction_steps+1,index], s=0.2, label="Actual")
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
        f, P = signal.periodogram(states.detach().cpu().numpy()[:,index])
        ax.semilogy(f[1:], P[1:], label="Predicted") 
        if true_states is  not None:
            f, P = signal.periodogram(true_states.detach().cpu().numpy()[:,index])
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

### Plot autoencoder reconstructed trajectory
def plot_rec_trajectory(rec, filename=None):
    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection="3d")
    # Plot reconstructed trajectory
    for i in range(len(rec)):
        ax.scatter(rec.detach().cpu().numpy()[i,0,:,0], rec.detach().cpu().numpy()[i,0,:,1], rec.detach().cpu().numpy()[i,0,:,2], c="b", s=0.1)
        
    # Set labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig


# Plot learned parameters distribution
def plot_params_distr(enc, true_params, bins=100, filename=None):
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=len(true_params))
    gs = axs[1].get_gridspec()
    # Plot reconstructed trajectory
    index = 0
    statistics = []
    for ax in axs[0:]:
        
        ax.set_xlabel("p"+str(index+1))
        ax.set_ylabel("density")
        mean = np.mean(enc.detach().cpu().numpy()[:,index])
        std = np.std(enc.detach().cpu().numpy()[:,index])
        ax.hist(enc.detach().cpu().numpy()[:,index], bins=bins, density=True, label="$\mu$="+str(mean)+","+"$\sigma$="+str(std))            
        ax.axvline(x=true_params[index], c="red", lw=2)
        ax.axvline(x=mean, c="red", lw=2, ls="--")
        ax.axvspan(mean-std, mean+std, alpha=0.3, color='red')
        ax.grid()
        index = index+1
        statistics.append([mean, std])

    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig, np.array(statistics)


# Plot parameter distribution in 3d
def plot_3ddistr(enc, true_params, indeces=[0,1,2],filename=None):
    # Check indeces length
    if len(indeces) != 3:
        raise ValueError("Invalid indeces list.")
        
    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection="3d")
    
    # Plot 3d distribution
    ax.scatter(enc.detach().cpu().numpy()[:,indeces[0]], enc.detach().cpu().numpy()[:,indeces[1]], enc.detach().cpu().numpy()[:,indeces[2]], s=10)
    ax.scatter(true_params[...,indeces[0]],true_params[...,indeces[1]], true_params[...,indeces[2]], s=100)
    # Set labels
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("p3")
    
    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig