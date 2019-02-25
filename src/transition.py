from preprocess import Preprocessor
import pandas as pd


class Transition(object):
    def __init__(self):
        self.start = {}  # probabilities of each states being a start word
        self.stop = {}  # probabilities of each states being a stop word
        self.matrix = {}  # probabilities of each states leading into the next states
        self.states = []  # list of states

    # Computes transition matrix and probabilities of start and stop words
    def compute_params(self, preprocessor):
        print("Building transition parameters...")
        # Setting up start/stop counters
        self.states = preprocessor.get_states()
        start_count = {}
        stop_count = {}
        for state in self.states:
            start_count[state] = 0
            stop_count[state] = 0

        # Setting up empty transition matrix, read as "Transition from row label to column label"
        transition_count = pd.DataFrame(0, index=self.states, columns=self.states)

        # Bringing in ordered tweet labels from dataset
        tweet_labels = preprocessor.get_ordered_states()

        for tweet in tweet_labels:
            # Counts start and stop words
            start_count[tweet[0]] += 1
            stop_count[tweet[len(tweet)-1]] += 1

            # Counting transitions from state x to state y
            for tag_index in range(len(tweet)):
                if tag_index == len(tweet) - 1:  # if reached last state of tweet
                    continue
                transition_count.at[tweet[tag_index], tweet[tag_index+1]] += 1

        # Normalises values across rows ino probabilities, assigns to self.matrix
        transition_count["sum"] = transition_count.sum(axis=1)
        self.matrix = transition_count.loc[:, transition_count.columns.values[0]:transition_count.columns.values[-2]].div(transition_count["sum"], axis=0)
        print("Calculated transition parameter matrix with {} rows and columns.".format(self.matrix.shape[0]))

        # Calculates probabilities of start and stop states
        self.start = self.edge_state_compute(start_count)
        self.stop = self.edge_state_compute(stop_count)

    def edge_state_compute(self, edge_words):
        total = sum(edge_words.values())
        for key in edge_words.keys():
            edge_words[key] = edge_words[key]/total
        return edge_words

    # Probability that sentence starts with given state
    def startwith(self, state):
        if state in self.start:
            return self.start[state]
        else:
            return 0

    # Probability that sentence ends with given state
    def stopwith(self, state):
        if state in self.stop:
            return self.stop[state]
        else:
            return 0

    # Probability that state0 is followed by state1
    def transit_prob(self, state0, state1):
        if state0 in self.matrix:
            if state1 in self.matrix[state0]:
                if state0 in self.states:
                    return self.matrix.at[state0, state1]
                else:
                    raise RuntimeError("current tag:" + state0 + "\n never occurred in train data")
            else:
                return 0
        else:
            return 0

    def get_start_words(self):
        return self.start

    def get_stop_words(self):
        return self.stop

    def get_transition_matrix(self):
        return self.matrix.values




