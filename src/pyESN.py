"""
@author : Varun Ojha
"""

#!/usr/bin/env python3
import numpy as np
import utils
import torch


def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.

    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s

    Returns:
        None if s is None, else numpy vector of targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x


class ESN:
    def __init__(
        self,
        n_inputs,
        n_outputs,
        system,
        n_reservoir=200,
        spectral_radius=0.95,
        sparsity=0,
        erdos_graph=False,
        noise=1e-6,
        timestep=0.01,
        input_shift=None,
        input_scaling=None,
        teacher_forcing=True,
        feedback_scaling=None,
        teacher_scaling=None,
        teacher_shift=None,
        extended_states=False,
        out_activation=identity,
        inverse_out_activation=identity,
        random_state=None,
        silent=True,
        use_pi_loss = False,
    ):
        """
        Args:
            n_inputs: nr of input dimensions (n_inputs = K): u(n) = <u_1(n),u_2(n),...,u_K(n)>; K nodes n-> time (or length of signal),
            n_outputs: nr of output dimensions (n_outputs = L): y(n) = = <y_1(n),y_2(n),...,y_L(n)> ; L nodes
            n_reservoir: nr of reservoir neurons (n_reservoir = N): x(n) = = <x_1(n),x_2(n),...,x_N(n)>; N nodes
            
            spectral_radius: spectral radius of the recurrent weight matrix: concerning weight iniailization
            sparsity: proportion of recurrent weights set to zero: 0 < sparsity < 1.
            For an average connectivity of d, set sparsity = 1 - d/N
            noise: noise added to each neuron (regularization)
            timestep: change of time between each data point
            
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
                        
            teacher_forcing: if True, feed the target back into output units
            
            teacher_scaling: factor applied to the target signal: teacher is target output signal
            teacher_shift: additive term applied to the target signal
            
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
            
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builtin RandomState.
            silent: suppress messages
        """
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.pi_method = utils.RK4(timestep, model=system)
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.erdos_graph = erdos_graph
        self.noise = noise
        self.timestep = timestep
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.extended_states = extended_states
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.silent = silent
        self.use_pi_loss = use_pi_loss
        self.initweights()

    def num_timesteps(self, time):
        """Returns the number of timesteps required to pass time time.
        Raises an error if timestep value does not divide length time.
        """
        num_timesteps = time / self.timestep
        if not num_timesteps.is_integer():
            raise Exception
        return int(num_timesteps)

    def initweights(self):
        # RESERVOIR WEIGHTS - initialize recurrent weights:
        # begin with a random matrix centered around zero:
        # W is a NxN, matrix where N is number of nodes in reservoir.
        # RandomState.rand(array_dimensions) creates an array of the given shape
        # and populates it with random samples from a uniform distribution over
        # [0, 1).
        if self.erdos_graph:
            self.W = np.ones((self.n_reservoir, self.n_reservoir))
        else:
            self.W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete the fraction of connections given by (self.sparsity):
        self.W[self.random_state_.rand(*self.W.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights:
        if self.erdos_graph:
            pass
        else:
            radius = np.max(np.abs(np.linalg.eigvals(self.W)))
            # rescale them to reach the requested spectral radius:
            self.W = self.W * self.spectral_radius / radius
            
        rho_W = np.max(np.abs(np.linalg.eigvals(self.W)))
        if not self.silent:
            print(f"Spectral radius of W is {rho_W}")
            print(f"Modulus of W is {np.sum(W**2)}")
        # INPUTS WEIGHTS - random input weights, each between -1 and 1:
        # W_in is NxK matrix where N is the number of reservoir nodes and K is the input nodes
        self.W_in = (
            self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        )

        # FEEDBACK WEIGHTS - random feedback (teacher forcing) weights:
        # W_feedb is NxL matrix where N is the number of reservoir nodes and L is the output nodes
        self.W_feedb = (
            self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1
        )

    def _update(self, state, input_pattern, output_pattern):
        """performs one update step.

        i.e., computes and returns the next network state by applying the
        recurrent weights to the last state & and feeding in the current input
        and output patterns.
        N.B. Does not scale inputs or outputs.
        """
        if self.teacher_forcing:
            # x(n+1) = f( W.x(n)) + W_in.u(n+1) + W_feedb.y(n) ), f is tanh here and feedback weights are considered
            preactivation = (
                np.dot(self.W, state)
                + np.dot(self.W_in, input_pattern)
                + np.dot(self.W_feedb, output_pattern)
            )
        else:
            # x(n+1) = f( W.x(n)) + W_in.u(n+1)), f is tanh here adn NO feedback weights are considered
            preactivation = np.dot(self.W, state) + np.dot(
                self.W_in, input_pattern
            )
        return np.tanh(preactivation) + self.noise * (
            self.random_state_.rand(self.n_reservoir) - 0.5
        )

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher *= self.teacher_scaling
        if self.teacher_shift is not None:
            teacher += self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, inputs, outputs, inspect=False):
        """
        Collect the network's reaction to training data, train readout weights.

        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs). This
            is the signal the esn attempts to replicate.
            inspect: show a visualisation of the collected reservoir states

        Returns:
            the network's output on the training data, using the trained weights
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        if not self.silent:
            print("harvesting states...")
        # step the reservoir through the given input,output pairs:
        # for all input 0 to length of inputs (sample size),
        # x(n) = <x_1(n),...x_N(n)> is the a state of N reservior nodes for n = 1 to length of input
        # State should be a input_length x N matrix.
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        # Only loop in RNN trainig is to update state for each input signal. DONE!
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(
                states[n - 1], inputs_scaled[n], teachers_scaled[n]
            )

        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        if not self.silent:
            print("fitting...")
        # we'll disregard the first few states:
        transient = min(
            int(inputs.shape[0] / 10), 100
        )  # input forgetting property
        if not self.silent:
            print(f"transient = {transient}")

        # include the raw inputs:
        # W_out should be L x (K + N + L) matrix, W_out = L x (K + N) matrix was chosen
        # In this code self feedback to output node is probably ignored,
        # hence, extended_states only include input_scaled and
        # it signifies activation of input and resevoir nodes
        if self.extended_states:
            extended_states = np.hstack((states, inputs_scaled))
        else:
            extended_states = states  # non-extended states do not matter
            # because we are using the pseudoinverse.
            
        print(extended_states[transient:].shape)
        # Solve for W_out:
        if self.use_pi_loss:
            df = self.pi_method(torch.tensor(inputs[transient:], dtype=torch.float32))
            self.W_out = np.dot(
                np.linalg.pinv(extended_states[transient:]), inputs[transient:]+df.detach().numpy(),
            ).T
        else:
            self.W_out = np.dot(
                np.linalg.pinv(extended_states[transient:]), outputs[transient:],
            ).T

        # remember the last state for later:
        self.laststate = states[-1]
        self.lastinput = inputs[-1]
        self.lastoutput = teachers_scaled[-1]

        # optionally visualize the collected states
        if inspect:
            from matplotlib import pyplot as plt

            # (^-- we depend on matplotlib only if this option is used)
            plt.figure(
                figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01)
            )
            plt.imshow(
                extended_states.T, aspect="auto", interpolation="nearest"
            )
            plt.colorbar()

        # apply learned weights to the collected states:
        # y(n+1) = f( W_out.(x(n),u(n)) );
        pred_train = self.out_activation(np.dot(extended_states, self.W_out.T))
        training_rmse = np.sqrt(np.mean((pred_train - outputs) ** 2))
        if not self.silent:
            print("training root mean squared error:")
            print(training_rmse)

        return pred_train, training_rmse, transient

    def predict(
        self, time, inputs=None, input_is_looped=True, continuation=True
    ):
        """
        Apply the learned weights to the network's reactions to new input.

        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training
            state
            input_is_looped: if True, the output is fed into the network as an
            input at the next step.

        Returns:
            Array of output activations
        """
        # Convert input
        inputs = inputs.detach().numpy()
        # Prepare arrays
        if not input_is_looped:
            if inputs.ndim < 2:
                inputs = np.reshape(inputs, (len(inputs), -1))
            n_samples = inputs.shape[0]
        else:
            n_samples = self.num_timesteps(time)

        if continuation:
            laststate = self.laststate
            lastinput = self._scale_inputs(self.lastinput)
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = inputs[0]
            lastoutput = inputs[1]

        if not input_is_looped:
            inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        else:
            input_subarray = np.empty((n_samples, self.n_inputs))
            inputs = np.vstack([lastinput, input_subarray])
            
        states = np.vstack([laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack([lastoutput, np.zeros((n_samples, self.n_outputs))])

        # Calculate prediction
        for n in range(n_samples):
            # If input_is_looped is True, then we set ESN in prediction phase
            # following fig 1 in Pathak et al 2017:
            # https://arxiv.org/abs/1710.07313
            if input_is_looped:
                inputs[n + 1] = self._scale_inputs(outputs[n])

            states[n + 1] = self._update(
                states[n], inputs[n + 1], self._scale_teacher(outputs[n]),
            )

            if self.extended_states:
                outputs[n + 1, :] = self.out_activation(
                    np.dot(
                        np.concatenate([states[n + 1], inputs[n + 1]]),
                        self.W_out.T,
                    )
                )
            else:
                outputs[n + 1, :] = self.out_activation(
                    np.dot(states[n + 1], self.W_out.T)
                )

        # Return till -1 to assure that it has same length as input
        return torch.tensor(np.array(outputs[:-1]))
