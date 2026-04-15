import math
import numpy as np
from viterbi_hmm import HMM


class ForwardBackward(HMM):

    def forward(self, observation_sequence):
        n = len(observation_sequence)
        if n == 0:
            return [], []
        # Initialize Forward Matrix
        forward_matrix = {s: [float("-inf")] * n for s in self.states}

        # INITIALIZATION (t = 0)
        for s in self.states:
            # Initial probability + Emission probability
            forward_matrix[s][0] = self.log_initial[s] + self.log_emission[s].get(observation_sequence[0], float("-inf"))

        # ITERATION (t = 1 to n-1)
        for t in range(1, n):
            obs = observation_sequence[t]
            for current_state in self.states:
                # Initialize an empty list to store the values that need to be added for the current observation
                values = []
                for prev_state in self.states:
                    # Probability in matrix at t-1 for previous state + Transition probability of previous state to current state + Emission probability of current observation at current state
                    values.append(forward_matrix[prev_state][t-1] + self.log_transition[prev_state][current_state] + self.log_emission[current_state].get(obs, float("-inf")))
                
                # log-sum-exp to add probabilities safely since the probabilities are in log-space
                forward_matrix[current_state][t] = np.logaddexp.reduce(values)

        # Add final probabilities (using log-sum-exp) to get total log probability
        final_values = [forward_matrix[s][n-1] for s in self.states]
        total_forward_prob = np.logaddexp.reduce(final_values)

        return forward_matrix, total_forward_prob
    

    def backward(self, observation_sequence):
        n = len(observation_sequence)
        if n == 0:
            return [], []
        
        # Initialize Backward Matrix
        backward_matrix = {s: [float("-inf")] * n for s in self.states}

        # INITIALIZATION (t = n-1)
        for s in self.states:
            # Initialize to 0.0 since log(1) = 0
            backward_matrix[s][n-1] = 0.0

        # ITERATION (t = n-2 to 0)
        for t in range(n-2, -1, -1):
            next_obs = observation_sequence[t+1]
            for current_state in self.states:
                # Initialize an empty list to store the values that need to be added for the current observation
                values = []
                for next_state in self.states:
                    # Probability in matrix at t+1 for next state + Transition probability from current state to next state + Emission probability of next observation at next state
                    values.append(backward_matrix[next_state][t+1] + self.log_transition[current_state][next_state] + self.log_emission[next_state].get(next_obs, float("-inf")))
                # log-sum-exp to add probabilities safely since the probabilities are in log-space
                backward_matrix[current_state][t] = np.logaddexp.reduce(values)
        
        # Add initial probability and emission probability for observation[0] to get final probabilities for each state
        final_values = [(backward_matrix[s][0] + self.log_emission[s].get(observation_sequence[0], float("-inf")) + self.log_initial[s]) for s in self.states]
        # Add final probabilities (using log-sum-exp) to get total log probability
        total_backward_prob = np.logaddexp.reduce(final_values)

        return backward_matrix, total_backward_prob
    

    def run(self, observation_sequence):
        
        # Run forward and backward algorithm
        forward_matrix, total_forward_prob = self.forward(observation_sequence)
        backward_matrix, total_backward_prob = self.backward(observation_sequence)

        n = len(observation_sequence)
        # Initialize a matrix to hold probabilities for each observation at each state
        final_prob_matrix = {s: [0.0] * n for s in self.states}

        # Calculate the probability for each observation at each state using the forward and backward matrices
        for t in range(n):
            for s in self.states:
                # Normalizing the final result
                final_prob_matrix[s][t] = math.exp(forward_matrix[s][t] + backward_matrix[s][t] - total_forward_prob)

        return forward_matrix, total_forward_prob, backward_matrix, total_backward_prob, final_prob_matrix
    