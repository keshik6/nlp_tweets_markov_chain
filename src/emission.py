# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:31:00 2018

@author: Keshik
"""

import numpy as np

class Emission:
    """Class Emission.

    The Emission class estimates the emission parameters for Hidden Markov Model based on Maximum
    Likelihood Estimaion. It provides function to encompass smoothing into the model as well.
    
    This class also provides an implementation for sequence labelling system to produce a sequence
    of tags given an input file.
    
    Attributes:
        representer: A dictionary of dictonary indicating the tag and the token counts for the tag
        vocabulary: A dictionary of tokens and their counts generated for the entire document
        states: List to store all possible tags
        smoothing_param: Numeric parameter for smoothing
        matrix: array(states, words) of emission probabilities
        word_list: ordered list of words from vocabulary
    """

    def __init__(self, _representer, _vocabulary, _states, _smoothing_param = 1):
        self.representer = _representer
        self.vocabulary = _vocabulary
        self.states = _states
        self.smoothing_param = _smoothing_param
        self.matrix = []
        self.word_list = list(self.vocabulary.keys())

        self.calc_emission_param_matrix()

    """ 
    @notice Get the total number of tokens for a given tag
    @param state: tag
    @param smooth: boolean indicating whether to apply smoothing or not
    @returns int
    """
    def count_total(self, state, smooth=True):
        words_dict = self.representer[state]
        if smooth == True:  
            return sum(words_dict.values()) + self.smoothing_param
        return sum(words_dict.values())

    """ 
    @notice Get the number of tokens for a given tag
    @param word: token
    @param state: tag
    @param smooth: boolean indicating whether to apply smoothing or not
    @returns int
    """
    def count(self, word, state, smooth=True):
        words_dict = self.representer[state]
        if word not in self.vocabulary or word == '#UNK#':
            if (smooth == True):
                return self.smoothing_param
            return 0
        
        if word not in words_dict:
            return 0
        return words_dict[word]

    """ 
    @notice Set a new value for smoothing parameter
    @param _k: new smoothing parameter
    """
    def setSmoothingParam(self, _k):
        self.smoothing_param = _k

    """ 
    @notice Get the emission paramter for a token given a tag
    @param word: token
    @param state: tag
    @param smooth: boolean indicating whether to apply smoothing or not
    @returns float
    """
    def estimate_emission_param(self, word, state, smooth=True):
        return self.count(word, state, smooth)/self.count_total(state, smooth)

    """
    @notice Calculates the entire emission param matrix for all words and states
    @param word: token
    @param state: tag
    @param smooth: boolean indicating whether to apply smoothing or not
    """
    def calc_emission_param_matrix(self, smooth=True):
        print("Building emission parameter matrix...")
        emission_matrix = []
        if smooth:
            self.word_list.append('#UNK#')
            self.vocabulary['#UNK#'] = len(self.states)
            for state in self.states:
                self.representer[state]['#UNK#'] = 1
        for state in self.states:
            row = []
            for word in self.word_list:
                row.append(self.estimate_emission_param(word, state, True))
            emission_matrix.append(row)
        self.matrix = np.array(emission_matrix)
        print("Calculated emission parameter matrix with {} states (rows) and {} words (columns).".format(self.matrix.shape[0], self.matrix.shape[1]))

    def get_emission_param_matrix(self):
        return self.matrix

    def get_word_list(self):
        return self.word_list
    
    """ 
    @notice Given the input file, label the word sequence using the tag that returns the maximum emission probability
    @param _inputFile: Location of input file
    @param _outputFile: Name of output file to be created
    @returns None
    """
    def labelSequence(self, _inputFile, _outputFile):
        tags = self.states
        try:
            tweet_list = open(_inputFile, 'r',  encoding="UTF-8")
            output_file= open(_outputFile,"w+", encoding="UTF-8")
            lines = tweet_list.readlines()
            
            length = 0
            for token in lines:
                if token == "\n":
                    output_file.write("\n")
                    continue
                else:
                    length = length+1
                    emission_prob = 0
                    best_tag = None
                    for tag in tags:
                        prob = self.estimate_emission_param(token.strip(), tag, True)
                        if prob > emission_prob:
                            #print(prob)
                            emission_prob = prob
                            best_tag = tag
                    
                    labelled_token = " ".join([token.strip(), best_tag])
                    output_file.write(labelled_token)
                    
                    if (length != len(lines)):
                        output_file.write("\n")

        except IOError:
            print(IOError)

        finally:
            tweet_list.close()
            output_file.close()
            print("Sequence Labelling for", _inputFile, "completed. Results are saved in", _outputFile)


from preprocess import Preprocessor
from evaluateResult import evaluate


def train_and_validate_emission(_inputFile, _outputFile, _devFile, _devOutputFile, _validateFile):
    """
    Create the Preprocessor object
    Train using the SG, EN, CN, FR datasets
    Generate the representer, vocabulary and states and feed it into an Emission object 
    """
    preprocessor = Preprocessor(_inputFile)
    representer = preprocessor.get_representer()
    vocabulary = preprocessor.get_vocabulary()
    states = preprocessor.get_states()
    
    
    """
    Create the Emission Object
    Validate using the dev datasets
    Label the input sequence and output the file as dev.p2.out
    """
    emission = Emission(representer, vocabulary, states)
    emission.labelSequence(_devFile, _devOutputFile)
    
    
    """
    Calculate Validation Error
    """
    evaluate(_validateFile, _devOutputFile)


#Define variables
languages = ["CN", "FR", "EN", "SG"]

#for language in languages:
#     inputFile = "../data/" + str(language) + "/train"
#     outputFile = "../data/" + str(language) + "/dev.p2.out"
#     devFile = "../data/" + str(language) + "/dev.in"
#     devOutputFile = "../data/" + str(language) + "/dev.p2.out"
#     validateFile = "../data/" + str(language) + "/dev.out"
#
#     print("------------------------ " + language + " Training Dataset --------------------------")
#
#     train_and_validate_emission(inputFile, outputFile, devFile, devOutputFile, validateFile)
