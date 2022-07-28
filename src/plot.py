"""
@author : Alberto Bassi
"""

#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy import signal
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
from DsTools import Box

    
def compare_R2scores(net, true_states, time=20):
    """
    Report R2 scores of pieces of length pred_steps
    Args:
    true_states is a torch tensor of shape (samples, feature)
    prediction_steps is a int specifing how many steps to take
    return np.array
    """
    # retrieve sequence length
    seq_length = true_states.shape[0]
    
    r2_scores = []
    prediction_steps = net.num_timesteps(time)
    for i in range(seq_length-prediction_steps):
        states = net.predict(time, true_states[i:i+prediction_steps], input_is_looped=True)
        r2_scores.append(r2_score(true_states[i:i+prediction_steps,:].detach().cpu().numpy(), states.detach().cpu().numpy(), multioutput="raw_values"))
    
    return np.array(r2_scores)


def plot_trajectory(t, time=None, n_var=3, filename = None, prediction_steps=100, labels=None, color=None):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=n_var, sharex=True)
    gs = axs[1].get_gridspec()
    fig.subplots_adjust(hspace=0)
    
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        if time is None:
            ax.plot(t.detach().cpu().numpy()[:prediction_steps, -1], t.detach().cpu().numpy()[:prediction_steps, index], c=color, lw=0.5)
        else:
            ax.plot(time.detach().cpu().numpy()[:prediction_steps], t.detach().cpu().numpy()[:prediction_steps, index], c=color, lw=0.5)
        
        index += 1

   
    # Set labels
    axs[-1].set_xlabel("t")
    for i in range(n_var):
        if labels is None:
            axs[i].set_ylabel("x"+str(i+1))
        else:
            axs[i].set_ylabel(labels[i])
    
    #fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig

def compare_trajectories(t1, t2, time=None, n_var=3, filename = None, prediction_steps=100, labels=None, color=None):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=n_var, sharex=True)
    fig.subplots_adjust(hspace=0)
    
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        if time is None:
            ax.plot(t1.detach().cpu().numpy()[:prediction_steps, -1], t1.detach().cpu().numpy()[:prediction_steps, index], label="Predicted", c=color, lw=0.5)
            ax.plot(t2.detach().cpu().numpy()[:prediction_steps, -1], t2.detach().cpu().numpy()[:prediction_steps, index], label="Actual", c=color, lw=0.5)
        else:
            ax.plot(time.detach().cpu().numpy()[:prediction_steps], t1.detach().cpu().numpy()[:prediction_steps, index], label="Predicted", c=color, lw=0.5)
            ax.plot(time.detach().cpu().numpy()[:prediction_steps], t2.detach().cpu().numpy()[:prediction_steps, index], label="Actual", c=color, lw=0.5)
        ax.legend(loc = "upper right", fontsize = "x-small")
        index += 1
    
    # Set labels
    axs[-1].set_xlabel("t")
    for i in range(n_var):
        if labels is None:
            axs[i].set_ylabel("x"+str(i+1))
        else:
            axs[i].set_ylabel(labels[i])
        
    
    #fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig


def plot_3Dtrajectory(net_states, var=[0,1,2], filename=None, color=None, labels=None):
    ### Plot 3d trajectory
    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection="3d")

    if labels is None:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    
    
    
    if color is not None:
        plot = ax.scatter(net_states.detach().cpu().numpy()[:,var[0]], net_states.detach().cpu().numpy()[:,var[1]], net_states.detach().cpu().numpy()[:,var[2]], cmap = "RdBu_r", s=0.1, c=color)
    else:
        ax.plot(net_states.detach().cpu().numpy()[:,var[0]], net_states.detach().cpu().numpy()[:,var[1]], net_states.detach().cpu().numpy()[:,var[2]], lw=0.1)
        
    ax.legend()
    
    if color is not None:
        ax.figure.colorbar(plot, ax=ax)
        

    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig
    

def poincare_plot(states, delay=1, true_states=None, n_var=3, filename=None, prediction_steps=1000, c1="blue", c2="orange"):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=n_var)
    gs = axs[1].get_gridspec()
        
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        ax.set_xlabel("$x^{("+str(index+1)+")}_{t-\\tau}$")
        ax.scatter(states.detach().cpu().numpy()[:prediction_steps-delay, index], states.detach().cpu().numpy()[delay:prediction_steps,index], s=0.2, label="Predicted", c=c1) 
        if true_states is  not None:
            ax.scatter(true_states.detach().cpu().numpy()[:prediction_steps-delay, index], true_states.detach().cpu().numpy()[delay:prediction_steps,index], s=0.2, label="Actual", c=c2)
            ax.legend(loc = "upper right", fontsize = "x-small")
        index += 1
        

   
    # Set labels
    for i in range(n_var):
        axs[i].set_ylabel("$x^{("+str(i+1)+")}_t$")
    
    fig.tight_layout()
    # Save and return
    if filename is not None:
        plt.savefig(filename)
        
    return fig

def plot_powspec(states, true_states=None, n_var=3, filename=None):
    ### Plot two trajectories to compare, varaibles against time
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=n_var, sharex=True)
    gs = axs[1].get_gridspec()
    fig.subplots_adjust(hspace=0)
    
    # Plot dynamic variables
    index = 0
    for ax in axs[0:]:
        f, P = signal.periodogram(states.detach().cpu().numpy()[:,index])
        ax.semilogy(f[1:], P[1:], label="Predicted") 
        if true_states is  not None:
            f, P = signal.periodogram(true_states.detach().cpu().numpy()[:,index])
            ax.semilogy(f[1:], P[1:], label="Actual") 
            ax.legend(loc = "upper right", fontsize = "x-small")
        index += 1
        

   
    # Set labels
    axs[-1].set_xlabel("Frequency")
    for i in range(n_var):
        axs[i].set_ylabel("PS of (x"+str(i+1)+")")
    
    #fig.tight_layout()
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
        ax.plot(rec.detach().cpu().numpy()[i,0,:,0], rec.detach().cpu().numpy()[i,0,:,1], rec.detach().cpu().numpy()[i,0,:,2], c="b", lw=0.1)
        
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
def plot_params_distr(enc, plot_stat=False, true_params=None, labels=None, bins=100, range=None, filename=None):
    fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=enc.shape[1])
    gs = axs[1].get_gridspec()
    # Plot reconstructed trajectory
    index = 0
    statistics = []
    for ax in axs[0:]:
        if labels is not None:
            ax.set_xlabel(labels[index])
        else: 
            ax.set_xlabel("x"+str(index+1))
            
        ax.set_ylabel("density")
        
        # Copmute statistics
        mean = np.mean(enc.detach().cpu().numpy()[:,index])
        std = np.std(enc.detach().cpu().numpy()[:,index])
        statistics.append([mean, std])
        
        # Plot histogram and show mean and std
        ax.hist(enc.detach().cpu().numpy()[:,index], bins=bins, density=True, range=range, log=True)
        if plot_stat:
            ax.axvspan(mean-std, mean+std, alpha=0.3, color='red')
            ax.axvline(x=mean, c="red", lw=2, ls="--")
        
        # Plot true parameters
        if true_params is not None:
            ax.axvline(x=true_params[index], c="red", lw=2)
            
            
        ax.grid()
        index = index+1
        

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


def plot_compute_pdf(points, epsilon=0.5, filename=None):
    """
    Plot pdf (normalized) for 1d,2d,3d distributions
    For 1d plot the histogram, while box for 2d, 3d
    Args:
        points : data to be copmuted the pdf
        epsilon : length of cubes 
        labels : labels to put
        filename : if not None file where to save plot
    """
    
    fig, ax = plt.subplots(nrows=1,ncols=points.shape[-1], figsize = (15,5))
    
    for i in range(points.shape[-1]):
        
       
        if i==0:
            points_reduced = points[:,:2]
            ax[i].set_xlabel("$x^{(1)}$")
            ax[i].set_ylabel("$x^{(2)}$")
        if i==1:
            points_reduced = points[:,0::2]
            ax[i].set_xlabel("$x^{(1)}$")
            ax[i].set_ylabel("$x^{(3)}$")
        if i==2:
            points_reduced = points[:,1:]
            ax[i].set_xlabel("$x^{(2)}$")
            ax[i].set_ylabel("$x^{(3)}$")
            
        # compute box
        center = (np.amax(points_reduced, axis=0) + np.amin(points_reduced, axis=0))/2.
        box_sizes = np.amax(points_reduced, axis=0) - np.amin(points_reduced, axis=0)
        box = Box(center, box_sizes, epsilon)
        
        # compute hist (pdf=h[0])
        h = ax[i].hist2d(points_reduced[:,0], points_reduced[:,1], bins = box.int_sizes,cmap="RdBu_r", density = True)
        
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h[3], cax=cax)
        
    fig.tight_layout()
    # save
    if filename is not None:
        fig.savefig(filename)

    return fig