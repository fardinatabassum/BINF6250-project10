# Introduction
In this project we implement the Baum-Welch algorithm to estimate the parameters of a Hidden Markov Model with two states: High GC content and Low GC Content. It estimates initial probabilities, transition probabilities, and emission probabilities. Our code divides this algorithm into two steps:

1. Expectation - where we use our previous forward-backward algorithm implementation to estimate soft counts for set of probabilities in each part of the model based on a set of sequences.
2. Maximization - where we iterate, normalize the counts after each iteration, and compare log-likelihood of the previous iterations model with the current iterations model.

The examples provides in this project are just small demonstrations of how our implementation works. To get realistic biological models, the algorithm should be trained on a larger set of sequences from an appropriate biological context.

# Pseudocode

```
Expectation:

  Initilaize lists to store transition_counts, emission_counts, initial_counts

  For each sequence in our data set:
    calculate forward_matrix, sequence_prob
    calculate backwad_matrix

    # Get our initial state soft counts
    for each state in our model:
      log_initial = forward_matrix[state][first position in sequence] + backward_matrix[state][first position in sequence] - sequence_prob
      initial_counts = np.logaddexp(initial_counts[state], log_initial)

    # Get our emission and transition counts
    for each nucleotide in our sequence except the last one:
      for each state in our model:
        # emission counts
        log_forward = forward_matrix[state][nucleotide]
        log_backward = backward_matrix[state][nucleotide]
        log_emission = log_forward + log_backward - sequence_prob
        emission_counts[state][nucleotide] = np.logaddexp(emission_counts[state][nucleotide], log_emission)

        # tranisiton counts
        for each next_state in our model:
          log_state2state = log(transition[state][next_state])
          log_emission = log(emission[next_state][next nucleotide in the sequence])
          log_finish = backward_matrix[next_state][next nucleotide in the sequence]
          log_transition = log_forward + log_state2state + log_emission + log_finish - sequence_prob
          transition_counts[state][next_state] = np.logaddexp(transition_counts[state][next_state], log_transition)

    return transition_counts, emission_counts, initial_counts

Maximization:

  for every iteration in our algorithm:
    make a copy of each part of our model
    perform expectation step

    convert log soft counts for each set of counts to log probabilites by subtracting the
    logsumexp of each set of counts associated with a state (a set of emissions for a state, a set of transitions for a state, and all initial counts for all states)
    from a specific count of a state (count of emission "G" for state "H", count of transiton from state "H" to state "L", initial count of state "L", etc.)

    update our copies with the new log_probabilities
    compute the log likelihood of our old model given our sequences
    update our model with the new log_probabilities
    compute the log likelihood of our new model given our sequences
    compare the ll of our new model to our old model, if the change is smaller than the given threshold we've reached convergence

    return the trained model

    

      
```

# Successes
Once we broke the algorithm into the expectation and maximization steps our group was able to think through this algorithm in a more productive manner. At first we were almost combining both steps into one, leading to a lot of confusion. Once we got past this, we were able to work through the logic and find a strategy for implementation. Keeping these parts seperate made the debugging and sanity checks much easier when we started writing code. Additionally within the expectation step we were able to break down the equations provided in the lecture for generating soft counts - once we internalized these and figured out how to use our forward-backward algorithm implementation to get the probabilities needed for them, the whole project to started to flow better.

# Struggles
As mentioned, breaking the algorithm into it's two main parts helped us implement the algorithm, but it was a bit of a struggle to get there. At first there was confusion around what an iteration looked like, and we originally thought each iteration was a move through the set of sequences. So analysis of seq1 = iteration 1, seq2 = iteration 2, so on and so forth. Within this framework we though about expectation and maximization happening as one (calculate new model after analyzing a sequence, compare it to the model before that sequence), which as expected, was not making sense from a counts -> normalization perspective. After meeting with Marcus and coming to the realization that you analyze all sequences in one iteration that was when we were able to discern between the two steps. In addition to this we had some difficulty figuring out the method to compare our current model with the previous one - we initially thought of it as comparing the actual probabilities of the model with one another rather than comparing the log-likelihood of the two models. Where our implementation uses dictionaries to represent our models, this was quite a pain to try and implement, and also not as good of a metric as log-likelihood.  

# Personal Reflections
## Group Leader
**Fardina Tabassum** -  I found this project the most challenging out of all the other hmm implementations. I had a hard time wrapping my head around understanding the Baum-Welch algorithms, especially breaking down the equations he had on the lecture slides. I think understanding each step of the algorithm was quite tricky as well, especially the expectation step. It was a little confusing to grasp what he meant by expected counts and also how to create snapshots of the original model before updating it but after a constructive meeting with Marcus, my team and I were able to better understand the algorithm. Marcus also introduced us to the copy package and the concept of deepcopy which we then used to make a full copy of our model to then update it. My teammates really helped me with the full pseudocode breakdown, which helped me understand all the iterations that were happening. Once we broke down the equation and the pseudocode, the implementation became easier. It was also a big plus that my team and I were able to set up our previous HMM projects in a modular and robust way so that it can be used with Baum Welch without making any major changes.

## Other member
**Meghana Ravi** - This project was more challenging for me to understand compared to the previous HMM projects, especially in terms of the underlying logic and understanding the mathematical equations. Aside from the mathematical aspect of this algorithm, what the iterations were supposed to do also confused me a lot. Discussing the process with my group and attending office hours helped clarify many of my confusions. Going through the extra material in the module also helped me visualize the process better. Once I understood the logic, it became easier to work on. 


**Connor Crawford** - This project, probably more than any other project, taught me how important it is to consider how you iterate through your data and what you're trying to accomplish when you do. In this project we needed multiple layers of iteration: The outer iteration to perform maximization and the inner iteration to calculate expected counts across each sequence in our observations - not to mention the iterations within that to access states and next_states. These all needed to be lined up in a proper manner to keep the system functioning in a way that at least appears to behave in a reasonable manner. If you lose track of what you're doing at each level it's very easy to get lost.

# Generative AI Appendix
ChatGPT was used to breakdown equations and the conceptual framework of the algorithm.
