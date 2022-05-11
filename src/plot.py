import numpy as np
import torch
import matplotlib.pyplot as plt

def gen_trajectory(net, test_dataset, prediction_steps = 1000):
    " Generate a trajectory of prediction_steps lenght starting from test_dataset[0]. Return np.array"
    state = torch.tensor(test_dataset[0], dtype=torch.float).unsqueeze(0).unsqueeze(0)
    h0 = torch.zeros(net.layers_num, 1,net.hidden_units)
    c0 = torch.zeros(net.layers_num, 1, net.hidden_units)
    rnn_state0 = (h0, c0)
    rnn_state = rnn_state0
    n_update = 2000# Reset the dynamics to a true data every n_update steps

    net_states = []
    net.eval()
    
    # Adjust length
    if prediction_steps > len(test_dataset):
        prediction_steps = len(test_dataset)

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
    



def plt_gen_trajectory(net_states, t_test, test_dataset, filename, prediction_steps=100, include_time=False):
    fig, axs = plt.subplots(figsize=(10,5), ncols=2, nrows=3)
    gs = axs[1, 1].get_gridspec()
    
    # Adjust length
    if prediction_steps > len(test_dataset):
        prediction_steps = len(test_dataset)
        
    index = 0
    for ax in axs[0:,0]:
        ax.set_xlabel("t")
        if include_time:
            ax.plot(net_states[:prediction_steps, -1], net_states[:prediction_steps, index], label="Predicted")
            ax.plot(test_dataset[:prediction_steps, -1], test_dataset[:prediction_steps, index], label="Actual")
        else:
            ax.plot(t_test[:prediction_steps], net_states[:prediction_steps, index], label="Predicted")
            ax.plot(t_test[:prediction_steps], test_dataset[:prediction_steps, index], label="Actual")
        ax.legend(loc = "upper right", fontsize = "x-small")
        index += 1

    axs[0,0].set_ylabel("x")
    axs[1,0].set_ylabel("y")
    axs[2,0].set_ylabel("z")

    # remove the underlying axes
    for ax in axs[0:, -1]:
        ax.remove()

    axbig = fig.add_subplot(gs[0:, -1], projection="3d")
    axbig.set_xlabel("x")
    axbig.set_ylabel("y")
    axbig.set_zlabel("z")
    axbig.set_title("Predicted Lorenz attractor")
    axbig.plot(net_states[:prediction_steps,0], net_states[:prediction_steps,1], net_states[:prediction_steps,2])
    axbig.legend()


    fig.tight_layout()
    plt.savefig(filename)
    plt.show()
