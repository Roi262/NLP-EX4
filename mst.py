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
    :return: [tuple] (the size of the 'matrix', w2i dict)
    """
    V_Size = len(tokens)
    w2i = dict(zip(tokens, range(V_Size)))
    
    # matrix = csr_matrix((V_Size, V_Size)).toarray()
    # TODO create scipy or numpy array
    return V_Size**2, w2i


def make_sparse_vec(matrix):
    np_arr = np.asarray(matrix)
    length = np_arr.shape[0] * np_arr.shape[1]
    return np_arr.reshape((1, length))[0]


# def create_or_load_feature_dict(word_matrix, tag_matrix, word_map, tag_map):
#     """

#     Creates feature function dictionary, loads it from pkl file if already cached and
#     :param word_matrix:
#     :param tag_matrix:
#     :param word_map:
#     :param tag_map:
#     :param cache_w2v:
#     :return:
#     """
#     feat_dict_path = "feature_dict.pkl"
#     if not os.path.exists(feat_dict_path):
#         feat_dic = {}
#         for word1 in tqdm(corpus):
#             for word2 in tqdm(corpus):
#                 # the feature vectors will now save only the indices with value 1:
#                 feat_dic[(word1, word2)] = feature_func(word1, word2,
#                                                         word_matrix, tag_matrix, word_map, tag_map)
#         # if cache_w2v:
#         save_pickle(feat_dic, feat_dict_path)
#     else:
#         feat_dic = load_pickle(feat_dict_path)
#     return feat_dic

# def get_tree_sum(tree, vector_size, word_matrix, tag_matrix, word_map, tag_map):
#     # TODO - check function
#     sum = np.zeros(vector_size)
#     words = [tree.nodes[node_index]['word'] for node_index in tree.nodes]
#     for i in range(len(words)):
#         for j in range(len(words)):
#             sum += feature_func(words[i], words[j], word_matrix, tag_matrix, word_map, tag_map)
#     return sum
#
#
# def get_max_trees_score(trees, theta):
#     max_score = 0
#     for i in range:
#         pass
#
# def update_theta(theta, tree1, tree2, lr=1):
#     return theta + lr * (get_tree_sum(tree1) - get_tree_sum(tree2))
#
# def perceptron(train_set, word_matrix, word_map, tag_matrix, tag_map, N_iterations=2, lr=1):
#     vec_size = len(word_matrix)**2 + len(word_matrix)**2
#     theta = np.zeros(vec_size) # TODO do we initialize this as zeros?
#     N_sentences = len(train_set) # get the number of sentences
#     theta_vectors = np.array((N_iterations * N_sentences, vec_size))
#     theta_vectors[0] = theta
#     for r in range(N_iterations):
#         for i in range(N_sentences):
#             # TODO send one tree from trees? and find the mst for it?
#             T_tag = get_max_trees_score(arcs , theta_vectors[(r-1)*N_sentences + i - 1])
#             theta_vectors[(r-1)*N_sentences + i] = update_theta()
#     return np.mean(theta_vectors)


def create_graph(corpus, word_map, theta):
    arcs = []
    for head in corpus:
        for tail in corpus:
            arc = Arc(head, 0, tail)
            arcs.append(arc)
    return arcs


def get_arcs(sent_feat_dict, sentence, theta):
    arcs = []
    for head in sentence:
        for tail in sentence:
            weight = np.sum(theta.take(sent_feat_dict[(head, tail)][1]))
            arc = Arc(head, weight, tail)
            arcs.append(arc)
    return arcs


def create_list_of_feature_dic(train_sents, word_matrix_size, word_map, tag_matrix_size, tag_map):
    """Creates a list of dictionaries for all the sentences. 
    Arguments:
        train_sents {[type]} -- [description]
        word_matrix {[type]} -- [description]
        word_map {[type]} -- [description]
        tag_matrix {[type]} -- [description]
        tag_map {[type]} -- [description]
    
    Returns:
        [dict] --  holds the feature function values for every
                   pair of words in the sentence.
    """
    list_feat_dict_path = "list_feat_dict.pkl"
    if not os.path.exists(list_feat_dict_path):
        list_feat_dics = []
        for sent in tqdm(train_sents):
            feat_dict = {}
            for word1 in sent:
                for word2 in sent:
                    feat_dict[(word1, word2)] = feature_func(
                        word1, word2, word_matrix_size, tag_matrix_size, word_map, tag_map)
            list_feat_dics.append(feat_dict)
    else:
        list_feat_dics = load_pickle(list_feat_dict_path)
    return list_feat_dics


def get_real_tree(tree, sent_feat):
    """returns feature vector only on the train edges of the tree (the tree from the train set)"""
    feat_dict = {}
    for node in tree.nodes:
        tail = tree.nodes[node]['word']
        head_idx = tree.nodes[node]['head']
        head = tree.nodes[head_idx]['word']
        feat_dict[(head, tail)] = sent_feat[(head, tail)]
    return feat_dict


def get_diff(mst_tag, mst_i_tups, theta, sent_feat_dict):
    """

    :param mst_tag: mst as coming back from CLE algorithem
    :param mst_i_tups: tuples from helper function each tup is (head,tail)
    :param theta: current theta vector
    :param sent_feat_dict: feature dictionary of specific sentnce.
    :return:
    """
    mst_tag_score = 0
    mst_i_score = 0
    for arc in mst_tag:
        mst_tag_score += mst_tag[arc].weight
    for tup in mst_i_tups:
        mst_tag_score += np.sum(theta.take(sent_feat_dict[tup]))
    return (mst_tag_score-mst_i_score)


def list_of_word_tup_per_tree(trees_train):
    tree_list_of_tup_words = []
    for tree in trees_train:
        tree_list = []
        for node in tree.nodes:
            tail = tree.nodes[node]['word']
            head_idx = tree.nodes[node]['head']
            head = tree.nodes[head_idx]['word']
            tree_list.append((head, tail))
        tree_list_of_tup_words.append(tree_list)
    return tree_list_of_tup_words


def perceptron(trees_train, sents_train, word_matrix_size, word_map, tag_matrix_size, tag_map, N_iterations=2, lr=1, augmented=False):
    """Handles run of the averaged Perceptron algorithm
    
    Arguments:
        trees_train {[type]} -- All trees in the training set.
        sents_train {[type]} -- All sentences in the training set.
        word_matrix {[type]} -- 
        word_map {[type]} -- [description]
        tag_matrix {[type]} -- [description]
        tag_map {[type]} -- [description]
    
    Keyword Arguments:
        N_iterations {int} -- [description] (default: {2})
        lr {int} -- [description] (default: {1})
        augmented {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """
    vec_size = word_matrix_size + tag_matrix_size
    if augmented:
        vec_size += AUGMENT_FACTOR
    theta = np.zeros(vec_size)
    theta_sum = np.zeros(vec_size)

    N_sentences = len(trees_train)  # get the number of sentences
    list_feat_dict = create_list_of_feature_dic(
        sents_train, word_matrix_size, word_map, tag_matrix_size, tag_map)
    trees_orderes_tups = list_of_word_tup_per_tree(trees_train)
    for r in tqdm(range(N_iterations)):
        for i in tqdm(range(N_sentences)):
            sentence = sents_train[i]
            sent_feat_dict = list_feat_dict[i]
            arcs = get_arcs(sent_feat_dict, sentence, theta)
            T_tag = min_spanning_arborescence(arcs=arcs, sink=0)
            T_i = trees_orderes_tups[i]  # check
            # TODO check where to negativize the theta (or the arcs)
            feat_diff = get_diff(T_tag, T_i, theta, sent_feat_dict)  # check
            theta += (lr * feat_diff)
            theta_sum += theta
    return float(theta_sum/(N_iterations*N_sentences))


if __name__ == '__main__':
    trees_train = dependency_treebank.parsed_sents()
    sents_train = list(dependency_treebank.sents())
    # for sent in sents_train:
    #     print(sent)
    corpus, POS_tags = get_corpus_and_tags_from_trees(trees_train)
    word_matrix_size, word_map = create_matrix(corpus)
    tag_matrix_size, tag_map = create_matrix(POS_tags)
    f = 0
    # Q 3
    w = perceptron(trees_train, sents_train, word_matrix_size,
                   word_map, tag_matrix_size, tag_map)
    
    # Q4
    w = perceptron(trees_train, sents_train, word_matrix_size,
                   word_map, tag_matrix_size, tag_map, augmented=True)
    