# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 20:32:07 2018

@author: Yuet Tung, Keshik, Charles 
"""

from transitionOrder2 import Transition2
from smoothed_emission import SmoothedEmission as Emission
import numpy as np


def best_path(transition, emission, sentence):
    """
    Find the likeliest path in a resulting in given sentence.

    :param transition: Transition object -- used to calculate probability of transitioning between states
    :param emission: Emission object -- used to calculate probability of emitting each event in each state
    :param sentence: list of strings (words)

    :return: path: list(int) -- list of states in the most probable path
    :return: p: float -- probability of that path
    """
    # List of arrays giving most likely previous state for each state.
    prev = []

    # List of possible states and words, order here is preserved
    state_list = list(emission.states)
    word_list = emission.get_word_list()

    # Making sentence indexed and iterable for easy traversal
    smoothed_sentence = sentence.copy()
    for i in range(len(sentence)):
        if sentence[i] not in word_list:
            smoothed_sentence[i] = '#UNK#'
    indexed_sentence = [word_list.index(word) for word in smoothed_sentence]
    iter_sentence = iter(indexed_sentence)
    # print(indexed_sentence)

# List of start/stop tag probabilities
    start_start = []
    final = []
    state_list_order2 = [] # list of order2 states = [(O,O),(O,B-INTJ),(O,B-P)...] of length states*states
    for state in state_list:
        start_start.append(transition.startwith(state))
        for state2 in state_list:
            state_list_order2.append((state,state2))
            final.append(transition.stopwith((state,state2)))
    # start_u = transition.get_start_u_matrix()
    transition_matrix = transition.get_transition_matrix()
    emission_matrix = emission.get_emission_param_matrix()
    # print("start_start= ",start_start)
    # print("start_u= ",start_u)
    # print("transition_matrix= ",transition_matrix)
    # print("final= ",final)
    # print("emission matrix = ",emission_matrix)
    
    # Use log-likelihoods to avoid floating-point underflow. Ignore -inf.
    with np.errstate(divide='ignore'):
        start_start = np.log10(start_start)
        # start_u = np.log10(start_u)
        transition_matrix = np.log10(transition_matrix)
        emission_matrix = np.log10(emission_matrix)
        final = np.log10(final)

    # first iteration
    logprob_start = start_start + emission_matrix[:, next(iter_sentence)]
    # print(logprob_start)
    
    # second iteration
    # state_list_order2 = [(O,O),(O,B-INTJ),(O,B-P)...] length states*states
    # where the 0th element indicate the immediate previous state, the 1st element indicate the current state
    # logprob will be the corresponding probability for each element in state_list_order2
    logprob = []
    prev_partial = [] # stores best parent of the each nodes for a word
    second_word = next(iter_sentence)
    for s2 in state_list_order2:
        for s in state_list:
            if (s2[0]==s):
                logprob.append(logprob_start[state_list.index(s)]+np.log10(transition.start_u_transit_prob(s2[0],s2[1]))+emission_matrix[state_list.index(s2[1]), second_word])
                prev_partial.append(state_list.index(s))
    prev.append(prev_partial)
    # print(logprob)
    # print(prev_partial)
    
    # recursive iteration
    for word in iter_sentence:
        # print(len(logprob))
        logprob_prev=logprob
        logprob=[]
        prev_partial = [] # # stores best parent of the each nodes for a word
        for s2 in state_list_order2:
            logprob_partial = [] # stores log-probs for one state of s2 (use to find max and argmax)
            for s2_prev in state_list_order2:
                if (s2[0]==s2_prev[1]): # each node contains (state0, state1), where state 0 represent previous state and state 1 represent current state. state 0 of current node must match state 1 of previous node.
                    # print ("previous logprob from state ",s2_prev," = ", logprob_prev[state_list_order2.index(s2_prev)])
                    # print("transition probability from ",s2_prev," -> ",s2[1]," = ", np.log10(transition.transit_prob(s2_prev,s2[1])))
                    # print("emission probability of word = ", word_list[word], " from state ",s2[1]," = ", emission_matrix[state_list.index(s2[1]), word])
                    logprob_partial.append(logprob_prev[state_list_order2.index(s2_prev)]+np.log10(transition.transit_prob(s2_prev,s2[1]))+emission_matrix[state_list.index(s2[1]), word])
                else:
                    logprob_partial.append(np.log10(0))
            logprob.append(np.max(logprob_partial))
            
            # print("logprob = ",logprob)
            prev_partial.append(np.argmax(logprob_partial))
        prev.append(prev_partial)

    # print("prev = ",prev)
    # Final case
    logprob = logprob + final
    
    # Most likely final state
    best_state = np.argmax(logprob)

    # Reconstruct path by following links and then reversing
    state = best_state
    path = [state]
    for p in reversed(prev):
        state = p[state]
        path.append(state)

    # print(path)
    # Converting path list of ints into states
    state_path = [state_list_order2[i][1] for i in path[::-1]]
    #print(state_path)
    return state_path, 10**logprob[best_state]

def getAllTokens(_inputFile):
    allTokens = []
    
    try:
        tweet_list = open(_inputFile, 'r',  encoding="UTF-8")
        lines = tweet_list.readlines()
    
        for token in lines:
            if token == "\n":
                continue
            else:
                word = token.strip()
                allTokens.append(word)
            
    except IOError:
        print(IOError)
    
    finally:
        tweet_list.close()
        
def label_viterbi(input_file, output_file, emission, transition):
        try:
            tweet_list = open(input_file, 'r',  encoding="UTF-8")
            output_file = open(output_file, "w+", encoding="UTF-8")
            lines = tweet_list.readlines()

            sentence = []
            for line in lines:
                if line == "\n":
                    path, prob = best_path(transition, emission, sentence)
                    for i in range(len(sentence)):
                        output_file.write("{} {}".format(sentence[i], path[i]))
                        output_file.write('\n')
                    sentence.clear()
                    output_file.write('\n')

                else:
                    sentence.append(line.rstrip())

        except IOError:
            print(IOError)

        finally:
            tweet_list.close()
            output_file.close()
            print("Sequence Labelling for", input_file, "completed. Results are saved in", output_file)


from preprocess import Preprocessor
from evaluateResult import evaluate


def train_and_validate_viterbi2(_inputFile, _outputFile, _devFile, _devOutputFile, _validateFile):
    """
    Create the Preprocessor object
    Train using the SG, EN, CN, FR datasets
    Generate the representer, vocabulary and states and feed it into an Emission object
    """
    preprocessor = Preprocessor(_inputFile)
    representer = preprocessor.get_representer()
    vocabulary = preprocessor.get_vocabulary()
    states = preprocessor.get_states()

    listOfWords = getAllTokens(_devFile)
    
    """
    Create the Emission and Transition objects
    Validate using the dev datasets
    Label the input sequence and output the file as dev.p3.out
    """
    emission = Emission(representer, vocabulary, states, listOfWords)
    transition = Transition2()
    transition.compute_params(preprocessor)

    label_viterbi(_devFile, _devOutputFile, emission, transition)

    """
    Calculate Validation Error
    """
    evaluate(_validateFile, _devOutputFile)


 # Define variables
languages = ["FR", "EN", "CN", "SG"]

for language in languages:
     inputFile = "../data/" + str(language) + "/train"
     outputFile = "../data/" + str(language) + "/dev.p5.out"
     devFile = "../data/" + str(language) + "/dev.in" 
     devOutputFile = "../data/" + str(language) + "/dev.p5.out"
     validateFile = "../data/" + str(language) + "/dev.out"

     print("------------------------ " + language + " Training Dataset --------------------------")

     train_and_validate_viterbi2(inputFile, outputFile, devFile, devOutputFile, validateFile)


