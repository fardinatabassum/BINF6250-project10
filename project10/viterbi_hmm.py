import math


class HMM:
    """
    Base class for Hidden Markov Model data management.
    Handles initialization and log-space conversion for Numerical Stability.
    """

    def __init__(self, states, emissions, initial_probs, transition_probs, emission_probs):
        self.states = states
        self.emissions = emissions
        self.initial_probs = initial_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs
        # Convert all probabilities to log-scale 
        self.log_initial = {s: self.convert_to_log_scale(initial_probs[s]) for s in states}

        # log_transition[previous][current]
        self.log_transition = {from_state: {to_state: self.convert_to_log_scale(transition_probs[from_state][to_state])
                                for to_state in states} for from_state in states}

        # log_emission[state][observation]
        self.log_emission = {s: {obs: self.convert_to_log_scale(prob)
                             for obs, prob in emits.items()}
                         for s, emits in emission_probs.items()}

    def convert_to_log_scale(self, prob):
        """Helper to calculate log(p) and handling p=0 as -infinity."""
        if prob <= 0:
            return -float('inf')
        return math.log(prob)


class Viterbi(HMM):
    """
    Subclass implementing the Viterbi Algorithm for Optimal Path Finding.
    Uses Iterative Tabulation.
    """

    def run(self, observation_sequence):
        """
        Executes the Viterbi algorithm on a given sequence of observations.
        """
        n = len(observation_sequence)
        if n == 0:
            return []

        # Initialize Score and Traceback Matrices: O(N x K) Space
        viterbi_matrix = {s: [0.0] * n for s in self.states}
        traceback_matrix = {s: [None] * n for s in self.states}

        # INITIALIZATION (t = 0)
        initial_observation = observation_sequence[0]
        for s in self.states:
            # Score(s,0) = log_initial(s) + log_emission(s, obs[0])
            current_emission= self.log_emission[s].get(initial_observation, -float('inf'))
            viterbi_matrix[s][0] = self.log_initial[s] + current_emission
            traceback_matrix[s][0] = None  # Boundary for traceback

        # ITERATION (t = 1 to N-1)
        for t in range(1, n):
            current_observation = observation_sequence[t]
            for current_state in self.states:

                # Initialize to negative infinity to find the maximum in log-space
                best_prob = -float('inf')
                best_previous_state = None

                # Find the best transition from the previous column
                for previous_state in self.states:
                    # Adds emission once within the maximization loop
                    path_score = viterbi_matrix[previous_state][t - 1] + self.log_transition[previous_state][current_state] + self.log_emission[current_state][observation_sequence[t]]

                    if path_score > best_prob:
                        best_prob = path_score
                        best_previous_state = previous_state

                # Update matrix with path score
                viterbi_matrix[current_state][t] = best_prob

                # Update Traceback Matrix with the ptr
                traceback_matrix[current_state][t] = best_previous_state

        return self.traceback(viterbi_matrix, traceback_matrix, n)

    def traceback(self, viterbi_matrix, traceback_matrix, n):
        """Reconstructing the Optimal Path from the stored pointers."""
        result_path = []

        # TERMINATION
        final_state = None
        max_final_score = -float('inf')
        for s in self.states:
            if viterbi_matrix[s][n - 1] > max_final_score:
                max_final_score = viterbi_matrix[s][n - 1]
                final_state = s

        # Walk backward from the end to the start
        if final_state is not None:
            result_path.append(final_state)

            # Move from t = N-1 down to t = 1
            for t in range(n - 1, 0, -1):
                final_state = traceback_matrix[final_state][t]
                result_path.append(final_state)

        # Reverse to get chronological order (Time 0 to Time N-1)
        result_path.reverse()
        return result_path
