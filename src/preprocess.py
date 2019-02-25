# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 20:32:07 2018

@author: Keshik, Charles
"""


class Preprocessor:
    """Class Preprocessor.

    The Preprocessor class reads text input file following the format
    of one token per line with token and tag separated by whitespace and a single
    empty line that separates sentences.
    
    Attributes:
        representer: A dictionary of dictonary indicating the tag and the counts for the tag.
        vocabulary: A dictionary of tokens and their counts generated for the entire document.
        data:
        ordered_states:
    """
    
    def __init__(self, input_file):
        self.data = []
        self.ordered_states = []
        self.representer = {}
        self.vocabulary = {}
        self.read_data(input_file)
        
        
    """ 
    @notice Read data to generate the vocabulary, representer, ordered_states and data 
    @param line: Location of input file
    @returns None
    """
    def read_data(self, input_file):
        try:
            tweet_list = open(input_file, encoding = "UTF-8")
            lines = tweet_list.readlines()
            sentence = []
            sentence_labels = []

            for line in lines:
                if line == '\n':
                    self.data.append(sentence)
                    self.ordered_states.append(sentence_labels)
                    sentence = []
                    sentence_labels = []
                    continue
                else:
                    word, state = self.process_token(line)
                    self.update_vocabulary(word)
                    sentence.append(word)
                    sentence_labels.append(state)
                    if state not in self.representer:
                        word_dict = {word:1}
                        self.representer[state] = word_dict
                    else:
                        if word not in self.representer[state]:
                            word_dict = {word:1}
                            self.representer[state].update(word_dict)
                        else:
                            self.representer[state][word] = self.representer[state][word] + 1

        except IOError:
            print(IOError)

        finally:
            tweet_list.close()
            print('Data set at {} loaded with {} tweets and {} unique labels.'.format(input_file, len(self.data), len(self.representer)))

    """ 
    @notice Process token to generate the word and tag 
    @param line: Tweet Token
    @returns word (string) and state (string)
    """
    def process_token(self, line):
        tokens = line.strip().split()
        state = tokens[-1]
        word = " ".join(tokens[0:len(tokens)-1])
        # print(word, state)
        return word, state

    """ 
    @notice Returns data as a list of lists in original positional indexing
    @returns list
    """
    def get_data(self):
        return self.data

    """ 
    @notice Returns states in same positional indexing as original data
    @returns list
    """
    def get_ordered_states(self):
        return self.ordered_states

    """ 
    @notice Returns states in training data
    @returns list
    """
    def get_states(self):
        return self.representer.keys()

    """ 
    @notice Update vocabulary 
    """
    def update_vocabulary(self, word):
        if word in self.vocabulary:
            self.vocabulary[word] += 1
        else:
            self.vocabulary[word] = 1

    # Return vocabulary in dataset
    """ 
    @notice Returns vocabulary corresponding to training data
    @returns dictionary
    """
    def get_vocabulary(self):
        return self.vocabulary

    """ 
    @notice Returns size of vocabulary corresponding to training data
    @returns int
    """
    def get_vocabulary_size(self):
        return len(self.vocabulary)

    """ 
    @notice Return the most frequent words
    @returns list
    """
    def get_most_frequent_word(self):
        val = max(self.vocabulary.values())
        output = []
        for i in self.vocabulary.items():
            if i[1] == val:
                output.append(i[0])
        return output

    """ 
    @notice Return the representer of training data
    @returns dictionary
    """
    def get_representer(self):
        return self.representer


#preprocess = Preprocessor('../data/SG/train')
#representer = preprocess.get_representer()
#vocabulary = preprocess.get_vocabulary()
#states = preprocess.get_states()
#print(preprocess.get_states())
#print(preprocess.get_data())

