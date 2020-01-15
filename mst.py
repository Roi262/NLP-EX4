from sklearn.model_selection import train_test_split
import numpy as np
import nltk
import os
from random import shuffle
# nltk.download()
from Chu_Liu_Edmonds_algorithm import Arc, min_spanning_arborescence
from nltk.corpus import dependency_treebank
from scipy.sparse import csr_matrix
from tqdm import tqdm
from pickles import save_pickle, load_pickle
from math import sqrt

TRAIN_PERC = .9
TEST_PERC = .1
SEP = ' - '
AUGMENT_FACTOR = 4

def get_augmented_features(u, v, sent, last_index=0):
    """Q4"""
    augmented_features = [0] * AUGMENT_FACTOR
    u_indeces = []
    v_indeces = []
    for i, word in enumerate(sent):
        if word == u:
            u_indeces.append(i)
        if word == v:
            v_indeces.append(i)

    for u_i in u_indeces:
        for v_i in v_indeces:
            if v_i - u_i == 1:  # u preceds v with 0 words in between
                augmented_features[0] = 1
            if v_i - u_i == 2:  # u preceds v with 1 word in between
                augmented_features[1] = 1
            if v_i - u_i == 3:  # u precedes v with 2 words in between
                augmented_features[2] = 1
            if v_i - u_i > 3:  # u preceds v with 3 or more words in between
                augmented_features[4] = 1
    positive_indeces = []
    for i, val in enumerate(augmented_features):
        if val == 1:
            positive_indeces.append(i + last_index)
    return positive_indeces


def feature_func(u, v, word_matrix_size, tag_matrix_size, word_map, tag_map, sent=None, augmented=False):
    """Implementation of the feature function.
    :param u: [string]: the first token
    :param v: [string]: the second token
    :param word_matrix: [np.darray]: a 2D array
    :param tag_matrix: [np.darray]: a 2D array
    :param word_map:
    :param tag_map:
    :return: [np.array]: the indeces with value 1 in the feature vector: a tuple consisting of the indeces
     where the feature function has the value 1
    """
    positive_indeces = []

    # bigram
    i, j = word_map[u], word_map[v]
    positive_indeces.append(i * sqrt(word_matrix_size) + j)

    # POS
    tup_place, pos_place = 0, 1
    POS_u = nltk.pos_tag([u])[tup_place][pos_place]
    POS_v = nltk.pos_tag([v])[tup_place][pos_place]
    i2, j2 = tag_map[POS_u], tag_map[POS_v]
    positive_indeces.append(word_matrix_size + (i2 * sqrt(tag_matrix_size) + j2))

    if augmented:
        aug_pos_indeces = get_augmented_features(
            u, v, sent, last_index=word_matrix_size + tag_matrix_size)
        positive_indeces.extend(aug_pos_indeces)
    return np.array(positive_indeces)



def get_corpus_and_tags_from_trees(trees):
    """

    :param trees: [list]: a list of dependency trees
    :return: [tup]: a list of the corpus taken from the trees, a list of the POS tags taken from the trees
    """
    corpus = set()
    POS_tags = set()
    for tree in trees:
        for node_index in tree.nodes:
            # NOTE that if we are at the root node, then the word is None
            word = tree.nodes[node_index]['word']
            tag = tree.nodes[node_index]['tag']
            corpus.add(word)
            POS_tags.add(tag)
    return list(corpus), list(POS_tags)


def create_matrix(tokens):
    """
    create the matrix from the given corpus
    corpus: [list] a list of tokens
    :return:
    """
    V_Size = len(tokens)
    word_to_index_map = {}
    for i in range(V_Size):
        word_to_index_map[tokens[i]] = i
    matrix = csr_matrix((V_Size, V_Size)).toarray()
    # TODO create scipy or numpy array
    return matrix, word_to_index_map


def make_sparse_vec(matrix):
    np_arr = np.asarray(matrix)
    length = np_arr.shape[0] * np_arr.shape[1]
    return np_arr.reshape((1, length))[0]

# TESTER LINES
# mat = np.arange(9).reshape((3, 3))


# l = csr_matrix((3, 4)).toarray()
# l = np.arange(15).reshape((3,5))
# # print(l)
# x = make_sparse_vec(mat)
# y = make_sparse_vec(l)
# print(x)
# print(y)
# added = np.append(x,y)
# print(added)

# def get_indeces()





def create_graph(corpus, word_map, theta):
    arcs = []
    for i, head in enumerate(corpus):
        for j, tail in enumerate(corpus):
            arc = Arc(head, 0, tail)
            arcs.append(arc)
    return arcs


def get_arcs(sent_feat_dict, sentence, theta):
    arcs = []
    for head in sentence:
        for tail in sentence:
            weight = np.sum(theta.take(sent_feat_dict[(head, tail)]))
            # weight = np.sum([theta[s] for s in feature_dict[(head, tail)]])
            arc = Arc(head, weight, tail)
            arcs.append(arc)
    return arcs

def create_list_of_feature_dic(train_sents, word_matrix, word_map, tag_matrix, tag_map):
    list_feat_dict_path = "list_feat_dict.pkl"
    if not os.path.exists(list_feat_dict_path):
        list_feat_dics = []
        for sent in tqdm(train_sents):
            feat_dict = {}
            for i, word1 in enumerate(sent):
                for j, word2 in enumerate(sent):
                    if (i == j):
                        continue
                    feat_dict[(word1,word2)] = feature_func(word1,word2,word_matrix,tag_matrix,word_map,tag_map)
            list_feat_dics.append(feat_dict)
        save_pickle(list_feat_dics,list_feat_dict_path)
    else:
        list_feat_dics = load_pickle(list_feat_dict_path)
    return list_feat_dics

def get_diff(mst_tag,mst_i_tups ,theta, sent_feat_dict):
    """

    :param mst_tag: mst as coming back from CLE algorithem
    :param mst_i_tups: tuples from helper function each tup is (head,tail)
    :param theta: current theta vector
    :param sent_feat_dict: feature dictionary of specific sentnce.
    :return:
    """
    mst_tag_score = np.zeros(theta.shape)
    mst_i_score = np.zeros(theta.shape)
    for arc in mst_tag:
        for ind in sent_feat_dict[(mst_tag[arc].head,mst_tag[arc].tail)]:
            mst_tag_score[ind] += 1
    # print(mst_tag_score)
    for tup in mst_i_tups:
        for ind in sent_feat_dict[tup]:
            mst_tag_score[ind] += 1
    return (mst_tag_score-mst_i_score)


def list_of_word_tup_per_tree(trees_train):
    list_tup_per_tree_path = "list_tup_per_tree.pkl"
    if not os.path.exists(list_tup_per_tree_path):
        tree_list_of_tup_words = []
        for tree in trees_train:
            tree_list = []
            for node in range(len(tree.nodes)):
                tail = tree.nodes[node]['word']
                head_idx = tree.nodes[node]['head']
                head = tree.nodes[head_idx]['word']
                if tail == None or head == None:
                    continue
                tree_list.append((head,tail))
                l = (head,tail)
            tree_list_of_tup_words.append(tree_list)
        save_pickle(tree_list_of_tup_words,list_tup_per_tree_path)
    else:
        tree_list_of_tup_words = load_pickle(list_tup_per_tree_path)
    return tree_list_of_tup_words


def perceptron(trees_train,sents_train, word_matrix, word_map, tag_matrix, tag_map, N_iterations=2, lr=1):
    vec_size = word_matrix.shape[0]**2 + tag_matrix.shape[0]**2
    theta = np.zeros(vec_size)
    N_sentences = len(trees_train) # get the number of sentences
    # theta_vectors = np.zeros((N_iterations * N_sentences, vec_size))
    theta_sum = np.zeros(vec_size)
    list_feat_dict = create_list_of_feature_dic(sents_train,word_matrix, word_map, tag_matrix, tag_map)
    trees_orderes_tups = list_of_word_tup_per_tree(trees_train)
    for r in range(N_iterations):
        print("Epoch ", r)
        for i in tqdm(range(N_sentences)):
            mst_i_tups = trees_orderes_tups[i]
            sentence = sents_train[i]
            sent_feat_dict = list_feat_dict[i]
            theta_index = (r-1)*N_sentences + i
            arcs = get_arcs(sent_feat_dict, sentence, theta)
            # TODO check where to negativize the theta (or the arcs)
            MST_TAG = min_spanning_arborescence(arcs=arcs, sink=0)
            print(arcs)
            feat_diff = get_diff(MST_TAG,mst_i_tups,theta, sent_feat_dict)
            new_theta = theta + (lr * feat_diff)
            theta_sum += new_theta
            theta = new_theta
    out_theta = theta_sum/(N_iterations*N_sentences)
    return out_theta


# def score():
#     w = perceptron()


if __name__ == '__main__':
    trees_train = dependency_treebank.parsed_sents()
    sents_train = list(dependency_treebank.sents())
    # for sent in sents_train:
    #     print(sent)
    corpus, POS_tags = get_corpus_and_tags_from_trees(trees_train)
    word_matrix, word_map = create_matrix(corpus)
    tag_matrix, tag_map = create_matrix(POS_tags)
    f = 0
    w = perceptron(trees_train,sents_train,word_matrix,word_map,tag_matrix,tag_map)



    # feature_dict = create_or_load_feature_dict(word_matrix, tag_matrix, word_map, tag_map)
    # k = feature_func('Pierre','Vinken',word_matrix,tag_matrix,word_map,tag_map)
    # print(np.argwhere(k==1))
    p = 0
    #
    # # print(trees[0])
    # #  divide the obtained corpus into training set
    # # and test set such that the test set is formed by the last 10% of the sentences.
    # shape = int(len(trees) * TRAIN_PERC)
    # # train, test = sents[:int(len(sents)*TRAIN_PERC)], sents[int(len(sents)*TRAIN_PERC):]
    # train, test = np.split(trees, [shape])
