
# DEPRECATING IN FAVOUR OF PREPROCESS.PY
class Dataset:
    def __init__(self, path):
        self.data = []  # list of tweets broken down word by word
        self.labels = []  # sentiment labels for each word in order
        self.tags = []
        self.file = open(path, 'r', encoding="utf-8")
        for i in self.file.read().split("\n\n"):
            sentence = []
            sentence_label = []
            for j in i.split("\n"):
                if j != "":
                    temp = j.split(" ")
                    sentence.append(temp[0])
                    sentence_label.append(temp[-1])
            if len(sentence) > 0:
                self.data.append(sentence)
                self.labels.append(sentence_label)
        self.file.close()

        self.tags = list(set([item for sublist in self.labels for item in sublist]))
        self.tags.sort()

        print('Tags:', self.tags)
        print('Data set loaded with {} tweets and {} unique labels.'.format(len(self.data), len(self.tags)))

