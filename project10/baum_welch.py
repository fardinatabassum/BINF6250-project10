from collections import defaultdict
from scipy.special import logsumexp
from forward_backward import *
import copy


class BaumWelch(ForwardBackward):
    """
    Implementation of the Baum-Welch algorithm
    """

    def expectation(self, sequences):
        """
        Performs expectation step of the algorithm
        """

        # initialize the model parameters to update for each iteration
        next_transition_counts = [[float('-inf') for _ in self.states] for _ in self.states]
        next_emission_counts = [[float('-inf') for _ in self.emissions] for _ in self.states]
        next_initial_counts = [float('-inf') for _ in self.states]


        # pass over each sequence and update the parameters
        for sequence in sequences:
            # get the probabilities needed to update model parameters based on the sequence
            forward_matrix, forward_prob = self.forward(sequence)
            backward_matrix, backward_prob = self.backward(sequence)

            # update our initial state soft counts
            for i, state in enumerate(self.states):
                log_emission = forward_matrix[state][0] + backward_matrix[state][0] - forward_prob
                next_initial_counts[i] = np.logaddexp(next_initial_counts[i], log_emission)

            # update our emission and transition counts
            for i in range(len(sequence) - 1):
                nucleotide = sequence[i]

                for j, state in enumerate(self.states):
                    log_fk = forward_matrix[state][i]
                    log_bk = backward_matrix[state][i]

                    # get emission soft counts for each state
                    log_emission = log_fk + log_bk - forward_prob
                    nucleotide_index = self.emissions.index(nucleotide)
                    next_emission_counts[j][nucleotide_index] = np.logaddexp(next_emission_counts[j][nucleotide_index], log_emission)

                    # get transition soft counts for each state
                    for k,next_state in enumerate(self.states):
                        log_akl = self.log_transition[state][next_state]
                        log_el = self.log_emission[next_state][sequence[i + 1]]
                        log_bl = backward_matrix[next_state][i + 1]

                        # update soft counts for each states transition probs
                        log_transition = log_fk + log_akl + log_el + log_bl - forward_prob
                        next_transition_counts[j][k] = np.logaddexp(next_transition_counts[j][k], log_transition)

        return next_transition_counts, next_emission_counts, next_initial_counts


    def convert_counts(self, transition_counts, emission_counts, initial_counts):
        """
        Converts soft counts to log probabilities
        """

        log_transition_probs = []
        for state in transition_counts:
            log_transitions = []
            for log_count in state:
                log_prob = log_count - logsumexp(state)
                log_transitions.append(log_prob)
            log_transition_probs.append(log_transitions)

        log_emission_probs = []
        for state in emission_counts:
            log_emissions = []
            for log_count in state:
                log_prob = log_count - logsumexp(state)
                log_emissions.append(log_prob)
            log_emission_probs.append(log_emissions)

        log_initial_probs = []
        for log_count in initial_counts:
            log_prob = log_count - logsumexp(initial_counts)
            log_initial_probs.append(log_prob)


        return log_transition_probs, log_emission_probs, log_initial_probs


    def maximization(self, sequences, threshold, iterations):
        """
        Performs maximization step of the Baum-Welch algorithm
        """

        for iter_idx in range(iterations):
            # make a copy of our current model
            transition_copy = copy.deepcopy(self.log_transition)
            emission_copy = copy.deepcopy(self.log_emission)
            initial_copy = copy.deepcopy(self.log_initial)

            # perform expectation step
            trans_counts, emit_counts, init_counts = self.expectation(sequences)

            # convert to log probabilities
            log_trans_probs, log_emit_probs, log_init_probs = self.convert_counts(trans_counts, emit_counts, init_counts)

            # update the copy of the model with the new log probs
            for i,key in enumerate(transition_copy.keys()):
                for j,state in enumerate(self.states):
                    transition_copy[key][state] = log_trans_probs[i][j]

            for i,key in enumerate(emission_copy.keys()):
                for j,emission in enumerate(self.emissions):
                    emission_copy[key][emission] = log_emit_probs[i][j]

            for i, key in enumerate(initial_copy.keys()):
                initial_copy[key] = log_init_probs[i]


            # compare the log likelihood of our old model with our new model
            old_ll = 0.0
            for seq in sequences:
                forward_matrix, forward_prob = self.forward(seq)
                old_ll += forward_prob

            # reassignment to update model
            self.log_transition = transition_copy
            self.log_emission = emission_copy
            self.log_initial = initial_copy

            new_ll = 0.0
            for seq in sequences:
                forward_matrix, forward_prob = self.forward(seq)
                new_ll += forward_prob

            print(new_ll - old_ll)
            if abs(new_ll - old_ll) < threshold:
                print("Convergence reached! Your model has been trained")
                return self.log_transition, self.log_emission, self.log_initial

        print("Convergence criteria not met")
        return self.log_transition, self.log_emission, self.log_initial


def model_randomizer(states, emissions, seed=7):
    """
    Creates a random model
    """
    rng = np.random.default_rng(seed)

    init_probs = defaultdict()
    trans_probs = defaultdict(defaultdict)
    emit_probs = defaultdict(defaultdict)

    random_ints = rng.integers(1, 101, size=len(states))
    for i in range(len(states)):
        init_probs[states[i]] = (random_ints[i] / sum(random_ints))

    for i in range(len(states)):
        random_ints = rng.integers(1, 101, size=len(states))
        for j in range(len(states)):
            trans_probs[states[i]][states[j]] = random_ints[j] / sum(random_ints)

    for i in range(len(states)):
        random_ints = rng.integers(1, 101, size = len(emissions))
        for j in range(len(emissions)):
            emit_probs[states[i]][emissions[j]] = random_ints[j] / sum(random_ints)

    return init_probs, trans_probs, emit_probs
