from preprocess import Preprocessor
import pandas as pd
import numpy as np

class Transition2(object):
    def __init__(self):
        self.start = {}  # probabilities of each states being a start word
        self.stop = {}  # probabilities of each states being a stop word
        self.start_u_matrix = {} # probabilities of (START,u)->v
        self.matrix = {}  # probabilities of each states leading into the next states
        self.states = []  # list of states
        self.states_order2 = [] # list of tuples of states

    # Computes transition matrix and probabilities of start and stop words
    # start_start is a vector of size = state*1 
    # start_u is a matrix of size state*state
    # matrix is the transition matrix of size state^2*state
    # stop is a vector of size state^2*1
    def compute_params(self, preprocessor):
        print("Building transition parameters...")
        # Setting up start/stop counters
        self.states = preprocessor.get_states()
        start_start_count = {}
        stop_count = {}
        for state in self.states:
            start_start_count[state] = 0  #first word
            # set up rows of tuples for different permutation of states
            for state2 in self.states:
                self.states_order2.append((state,state2))
                stop_count[(state,state2)] = 0  #last word
        
        # first transition (start,u) -> v
        start_u_count = pd.DataFrame(0, index=self.states, columns=self.states)
        
        # Setting up empty transition matrix, read as "Transition from row label to column label"
        transition_count = pd.DataFrame(0, index=self.states_order2, columns=self.states)

        # Bringing in ordered tweet labels from dataset
        tweet_labels = preprocessor.get_ordered_states()

        for tweet in tweet_labels:
            # Counts start words
            start_start_count[tweet[0]] += 1
            
            # Count transition from first to second state (START,u) -> v
            if(len(tweet)>1):
                start_u_count.at[tweet[0], tweet[1]] += 1
            # Counting transitions from state (u,v) -> w
            for tag_index in range(1,len(tweet)-1):
                transition_count.at[(tweet[tag_index-1],tweet[tag_index]), tweet[tag_index+1]] += 1
            
            # Counts stop words
            if(len(tweet)>1):
                stop_count[(tweet[len(tweet)-2],tweet[len(tweet)-1])] += 1

        # print(stop_count)
        # print(start_u_count)
        # print(transition_count)
        # Normalises values across rows into probabilities, assigns to self.matrix
        start_u_count["sum"] = start_u_count.sum(axis=1)
        self.start_u_matrix = start_u_count.loc[:, start_u_count.columns.values[0]:start_u_count.columns.values[-2]].div(start_u_count["sum"], axis=0)
        transition_count["sum"] = transition_count.sum(axis=1)
        self.matrix = transition_count.loc[:, transition_count.columns.values[0]:transition_count.columns.values[-2]].div(transition_count["sum"], axis=0)
        # convert nan to 0
        where_are_NaNs = np.isnan(self.start_u_matrix)
        self.start_u_matrix[where_are_NaNs] = 0
        where_are_NaNs = np.isnan(self.matrix)
        self.matrix[where_are_NaNs] = 0
        self.matrix[self.matrix==0] = 0.000000000000000001  # replace 0 with a small value
        
        print("Calculated transition parameter matrix with {} rows and columns.".format(self.matrix.shape[0]))

        # Calculates probabilities of start and stop states
        self.start = self.edge_state_compute(start_start_count)
        self.stop = self.edge_state_compute(stop_count)
        # print(self.start_u_matrix)
        # print(self.matrix)
        # print(self.stop)

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
    def stopwith(self, prev):
        state0 = prev[0]
        state1 = prev[1]
        if (state0,state1) in self.stop:
            return self.stop[(state0,state1)]
        else:
            return 0

    # Probability that (state0, state1) is followed by state2
    # :params prev: (state0, state1) -- the preceding 2 states
    def transit_prob(self, prev, state2):
        state0 = prev[0]
        state1 = prev[1]
        if state0 in self.states and state1 in self.states and state2 in self.states:
            # print("states : ",prev," -> ",state2)
            # print("prob = ",self.matrix.at[(state0, state1), state2])
            return self.matrix.at[(state0, state1), state2]
        else:
            raise RuntimeError("current tag:",(state0,state1), "\n never occurred in train data")
            
    # Probability that state0 is followed by state1 for (START,state0) -> state1 
    def start_u_transit_prob(self, state0, state1):
        if state0 in self.start_u_matrix:
            if state1 in self.start_u_matrix[state0]:
                if state0 in self.states:
                    return self.start_u_matrix.at[state0, state1]
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
    
    def get_start_u_matrix(self):
        return self.start_u_matrix.values




