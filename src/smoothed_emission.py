# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:55:50 2018

@author: Keshik
"""

from good_turing_estimate import SimpleGoodTuring
from emission import Emission
from preprocess import Preprocessor
from evaluateResult import evaluate

class SmoothedEmission(Emission):
    """Class ImprovedEmission.

    The Emission class estimates the emission parameters for Hidden Markov Model based on Maximum
    Likelihood Estimaion. It provides function to encompass smoothing into the model as well.
    
    The Improved Emission Class implements the following smoothing technique on top of Emission Class.
        Good-Turing estimation: Reallocate probability of n-grams that occur c times.
    
    The reason for not implementing this class seperately is to avoid code smell. As a result of Class Inheritence
    there will be some atributes of the original Emission class unused here.
    
    Attributes:
        representer: A dictionary of dictonary indicating the tag and the token counts for the tag
        vocabulary: A dictionary of tokens and their counts generated for the entire document
        states: List to store all possible tags
        smoothing_param: Numeric parameter for smoothing (This value is not used in this class and deprecated)
        matrix: array(states, words) of emission probabilities
        word_list: ordered list of words from vocabulary
        sgt: A dictionary storing the Good-Turing-Smoothed estimation values 
    """
    
    def __init__(self, _representer, _vocabulary, _states, unseen_words, _smoothing_param = 1):
        self.representer = _representer
        self.vocabulary = _vocabulary
        self.states = _states
        self.matrix = []
        self.word_list = list(self.vocabulary.keys())
        self.sgt = {}
        
        for i in self.states:
            self.sgt[i] = self.instantiate_sgt(i, unseen_words)
        
        
        self.calc_emission_param_matrix()
        
        
    """ 
    @notice Get the emission paramter for a token given a tag
    @param word: token
    @param state: tag
    @param smooth: boolean indicating whether to apply smoothing or not
    @returns float
    """
    def estimate_emission_param(self, word, state, smooth=True):
        if word in self.sgt[state]:
            return self.sgt[state][word]
        else:
            return 1/(len(self.vocabulary)**2)
    
    
    """ 
    @notice Instantiate Simple Good Turing Estimates for all the vocabulary as well as unseen words
    @param state: tag 
    @param unseen_words: list of words not occuring in the representer of the particular state 
    @returns Dictionary
    """
    def instantiate_sgt(self, state, unseen_words):
        state_vocab = self.representer[state]
        s = SimpleGoodTuring(state_vocab, max(state_vocab.values()))
        r, n = s.count_of_counts(state_vocab, max(state_vocab.values()))
        Z_arr = s.Z_smoothing(r, n)
        a, b = s.linregres_coeffs(r, Z_arr)
        S =  s.compute_S(a, b, r)
        r_star = s.smooth_counts(r, n, S)
        P0, sgt_probs= s.sgt_discount(r, r_star, n)
        spec_prob = s.species_probs(state_vocab, r, P0, sgt_probs, unseen_words)
        return spec_prob
    
    
    """ 
    @notice Given the state, get the Good-Turing Estimates
    @param state: Dictionary 
    """
    def getSmoothDict(self, state):
        return self.sgt[state]
    
    
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
                        #print(prob)
                        if prob >= emission_prob:
                            emission_prob = prob
                            best_tag = tag
                            
                    #print(best_tag)
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
    
    listOfWords = getAllTokens(_devFile)
    """
    Create the Emission Object
    Validate using the dev datasets
    Label the input sequence and output the file as dev.p2.out
    """
    emission = SmoothedEmission(representer, vocabulary, states, listOfWords)
    emission.labelSequence(_devFile, _devOutputFile)
 
    
    """
    Calculate Validation Error
    """
    evaluate(_validateFile, _devOutputFile)


def getAllTokens(_inputFile):
    
    """
    Given the input file get a list of all words where set operation will be used to get the unseen words
    in Good-Turing Estimation
    """
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
        return allTokens

# Define variables
languages = ["CN", "EN", "FR", "SG"]

#for language in languages:
#      inputFile = "../data/" + str(language) + "/train"
#      outputFile = "../data/" + str(language) + "/dev.custom.out"
#      devFile = "../data/" + str(language) + "/dev.in"
#      devOutputFile = "../data/" + str(language) + "/dev.custom.out"
#      validateFile = "../data/" + str(language) + "/dev.out"
#
#      print("------------------------ " + language + " Training Dataset --------------------------")
#
#      train_and_validate_emission(inputFile, outputFile, devFile, devOutputFile, validateFile)
     
     
     
#        r, n = s.count_of_counts(state_vocab, 2723)
#        Z_arr = s.Z_smoothing(r, n)
#        a, b = s.linregres_coeffs(r, Z_arr)
#        S =  s.compute_S(a, b, r)
#        r_star = s.smooth_counts(r, n, S)
#        P0, sgt_probs= s.sgt_discount(r, r_star, n)
#        spec_prob = s.species_probs(state_vocab, r, P0, sgt_probs)
# 
        # Define the paramteres here
#        s.__P0 = P0
#        s._r = r
#        s._n = n
#        s._Z = Z_arr
#        s.__sgt_probs = spec_prob
#        s.__S = S