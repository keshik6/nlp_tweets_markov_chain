import argparse

from emission import train_and_validate_emission
from viterbi import train_and_validate_viterbi
from viterbiOrder2 import train_and_validate_viterbi2


def evaluate():
    languages = ["EN", "CN", "FR", "SG"]
    models = ['emission', 'viterbi', 'viterbi2', 'custom', 'all']

    parser = argparse.ArgumentParser()
    parser.add_argument("language", help="EN, CN, FR, SG or all.")
    parser.add_argument("model", help="emission, viterbi, viterbi2, custom or all.")
    args = parser.parse_args()

    if args.language == 'all':
        pass
    elif args.language not in languages:
        raise ValueError('Invalid language selected.')
    else:
        languages = [args.language]

    if args.model not in models:
        raise ValueError('Invalid model selected.')

    for language in languages:
        print("------------------------ " + language + " Training Dataset --------------------------")
        inputFile = "../data/" + str(language) + "/train"
        devFile = "../data/" + str(language) + "/dev.in"

        if args.model == 'emission' or 'all':
            print("------------------------ " + "Part 2 Emission Model --------------------------")
            outputFile = "../data/" + str(language) + "/dev.p2.out"
            devOutputFile = "../data/" + str(language) + "/dev.p2.out"
            validateFile = "../data/" + str(language) + "/dev.out"
            train_and_validate_emission(inputFile, outputFile, devFile, devOutputFile, validateFile)

        if args.model == 'viterbi' or 'all':
            print("------------------------ " + "Part 3 Viterbi Model --------------------------")
            outputFile = "../data/" + str(language) + "/dev.p3.out"
            devOutputFile = "../data/" + str(language) + "/dev.p3.out"
            validateFile = "../data/" + str(language) + "/dev.out"
            train_and_validate_viterbi(inputFile, outputFile, devFile, devOutputFile, validateFile)

        if args.model == 'viterbi2' or 'all':
            print("------------------------ " + "Part 4 Viterbi2 Model --------------------------")
            outputFile = "../data/" + str(language) + "/dev.p4.out"
            devOutputFile = "../data/" + str(language) + "/dev.p4.out"
            validateFile = "../data/" + str(language) + "/dev.out"
            train_and_validate_viterbi2(inputFile, outputFile, devFile, devOutputFile, validateFile)

        # if args.model == 'custom' or 'all':
        #     print("------------------------ " + "Part 5 Custom Model --------------------------")
        #     outputFile = "../data/" + str(language) + "/dev.p5.out"
        #     devOutputFile = "../data/" + str(language) + "/dev.p5.out"
        #     validateFile = "../data/" + str(language) + "/dev.out"
        #     train_and_validate_custom(inputFile, outputFile, devFile, devOutputFile, validateFile)


if __name__ == "__main__":
    evaluate()
