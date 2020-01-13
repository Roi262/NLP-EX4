from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from random import shuffle
# nltk.download()
from nltk.corpus import dependency_treebank

TRAIN_PERC = .9
TEST_PERC = .1

sents = dependency_treebank.parsed_sents()
#  divide the obtained corpus into training set
# and test set such that the test set is formed by the last 10% of the sentences.
shape = int(len(sents)*TRAIN_PERC)
# train, test = sents[:int(len(sents)*TRAIN_PERC)], sents[int(len(sents)*TRAIN_PERC):]
train, test = np.split(sents, [shape])

# print(train)


def feature_func(u, v, sent, pos_tags):
    """A boolean feature functionthat encodes word and POS bigrams.
    
    Arguments:
        v1 {string} -- one node (word) v1 in V={w1,...,wn,ROOT}
        v2 {string} -- second node (word) v2 in V={w1,...,wn,ROOT}
        sent {list of strings} -- a sentance {w1,...,wn}
        dim {int} -- the dimension of the boolean output vector
    """
    n = len(sent)
    dim = n * n
    pos_tags = pos_tags
    if not pos_tags:
        pos_tags = [tag for word, tag in nltk.pos_tag(sent)]
    output = np.zeros(shape=dim)
    for i in range(n):
        for j in range(n):
            position = n*i + j
            # word bigram
            if sent[i] == u and sent[j] == v: #or (sent[i] == v2 and sent[j] == v1):
                output[position] = 1
            # pos bigram
            if pos_tags[i] == u and pos_tags[j] == v: #or (sent[i] == v2 and sent[j] == v1):
                output[position] = 1

    # TODO im not dealing with the case where u or v are the POS tag ROOT


def perceptron(train_set, iterations=2, lr=1):
    shuffled_train_set = train_set
    for i in range(iterations):
        shuffle(shuffled_train_set)
        pass


    pass

def score():
    w = perceptron()
    feat_vec = feature_func()


