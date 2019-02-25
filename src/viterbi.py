from transition import Transition
from emission import Emission
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
    initial = []
    final =[]
    for state in state_list:
        initial.append(transition.startwith(state))
        final.append(transition.stopwith(state))

    transition_matrix = transition.get_transition_matrix()
    emission_matrix = emission.get_emission_param_matrix()

    # Use log-likelihoods to avoid floating-point underflow. Ignore -inf.
    with np.errstate(divide='ignore'):
        initial = np.log10(initial)
        transition_matrix = np.log10(transition_matrix)
        emission_matrix = np.log10(emission_matrix)
        final = np.log10(final)

    logprob = initial + emission_matrix[:, next(iter_sentence)]

    for word in iter_sentence:
        p = logprob[:, np.newaxis] + transition_matrix + emission_matrix[:, word]
        prev.append(np.argmax(p, axis=0))
        logprob = np.max(p, axis=0)

    # print(prev)
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

    # Converting path list of ints into states
    state_path = [state_list[i] for i in path[::-1]]
    # print(state_path)
    return state_path, 10**logprob[best_state]


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


def train_and_validate_viterbi(_inputFile, _outputFile, _devFile, _devOutputFile, _validateFile):
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
    Create the Emission and Transition objects
    Validate using the dev datasets
    Label the input sequence and output the file as dev.p3.out
    """
    emission = Emission(representer, vocabulary, states)
    transition = Transition()
    transition.compute_params(preprocessor)

    label_viterbi(_devFile, _devOutputFile, emission, transition)

    """
    Calculate Validation Error
    """
    evaluate(_validateFile, _devOutputFile)


# # Define variables
# languages = ["EN", "CN", "FR", "SG"]
#
# for language in languages:
#     inputFile = "../data/" + str(language) + "/train"
#     outputFile = "../data/" + str(language) + "/dev.p3.out"
#     devFile = "../data/" + str(language) + "/dev.in"
#     devOutputFile = "../data/" + str(language) + "/dev.p3.out"
#     validateFile = "../data/" + str(language) + "/dev.out"
#
#     print("------------------------ " + language + " Training Dataset --------------------------")
#
#     train_and_validate(inputFile, outputFile, devFile, devOutputFile, validateFile)


