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

TRAIN_PERC = .9
TEST_PERC = .1
SEP = ' - '


# print(train)


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

def feature_func(u, v, word_matrix, tag_matrix, word_map, tag_map):
    """

    :param u:
    :param v:
    :param word_matrix: [np.darray]:
    :param tag_matrix: [np.darray]:
    :param word_map:
    :param tag_map:
    :return: - the feature vector: [np. array] an array representing the computedd feature vector
             - the indeces: [tup(int, int)]: a tuple consisting of the indeces where the feature function has the
             value 1
    """
    # init size values
    word_matrix, tag_matrix = np.copy(word_matrix), np.copy(tag_matrix) #todo maybe it can be removed in the new method.
    word_matrix_rows, word_matrix_cols = word_matrix.shape
    word_matrix_size = word_matrix_rows * word_matrix_cols
    tag_matrix_rows, tag_matrix_cols = tag_matrix.shape

    # bigram
    i, j = word_map[u], word_map[v]
    word_matrix[i][j] = 1
    word_vec = make_sparse_vec(word_matrix)
    word_index = i * word_matrix_cols + j

    # POS
    tup_place, pos_place = 0, 1
    POS_u = nltk.pos_tag([u])[tup_place][pos_place]
    POS_v = nltk.pos_tag([v])[tup_place][pos_place]
    i2, j2 = tag_map[POS_u], tag_map[POS_v]
    tag_matrix[i2][j2] = 1
    POS_index = word_matrix_size + (i2 * tag_matrix_cols + j2)

    tag_vec = make_sparse_vec(tag_matrix)
    sparse_vec = np.append(word_vec, tag_vec)
    return sparse_vec, (word_index, POS_index)



def create_or_load_feature_dict(word_matrix, tag_matrix, word_map, tag_map):
    """

    Creates feature function dictionary, loads it from pkl file if already cached and
    :param word_matrix:
    :param tag_matrix:
    :param word_map:
    :param tag_map:
    :param cache_w2v:
    :return:
    """
    feat_dict_path = "feature_dict.pkl"
    if not os.path.exists(feat_dict_path):
        feat_dic = {}
        for i, word1 in tqdm(enumerate(corpus)):
            for j, word2 in tqdm(enumerate(corpus)):
                # the feature vectors will now save only the indices with value 1:
                feat_dic[(word1, word2)] = feature_func(word1, word2, word_matrix, tag_matrix, word_map, tag_map)[1]
        # if cache_w2v:
        save_pickle(feat_dic, feat_dict_path)
    else:
        feat_dic = load_pickle(feat_dict_path)
    return feat_dic

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
    for i, head in enumerate(corpus):
        for j, tail in enumerate(corpus):
            arc = Arc(head, 0, tail)
            arcs.append(arc)
    return arcs


def get_arcs(sent_feat_dict, sentence, theta):
    arcs = []
    for head in enumerate(sentence):
        for tail in enumerate(sentence):
            weight = np.sum(theta.take(sent_feat_dict[(head, tail)][1]))
            # weight = np.sum([theta[s] for s in feature_dict[(head, tail)]])
            arc = Arc(head, weight, tail)
            arcs.append(arc)
    return arcs

def create_list_of_feature_dic(train_sents, word_matrix, word_map, tag_matrix, tag_map):
    list_feat_dics = []
    for sent in enumerate(train_sents):
        feat_dict = {}
        for word1 in sent:
            for word2 in sent:
                feat_dict[(word1,word2)] = feature_func(word1,word2,word_matrix,tag_matrix,word_map,tag_map)
        list_feat_dics.append(feat_dict)
    return list_feat_dics

def get_real_tree(tree, sent_feat):
    """returns feature vector only on the train edges of the tree (the tree from the train set)"""
    feat_dict = {}
    for node in tree.nodes:
        tail = tree.nodes[node]['word']
        head_idx = tree.nodes[node]['head']
        head = tree.nodes[head_idx]['word']
        feat_dict[(head,tail)] = sent_feat[(head,tail)]
    return feat_dict


def get_diff(mst_tag,mst_i_tups ,theta, sent_feat_dict):
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
    for i,tree in enumerate(trees_train):
        tree_list = []
        for node in tree.nodes:
            tail = tree.nodes[node]['word']
            head_idx = tree.nodes[node]['head']
            head = tree.nodes[head_idx]['word']
            tree_list.append((head,tail))
        tree_list_of_tup_words.append(tree_list)
    return tree_list_of_tup_words


def perceptron(trees_train,sents_train, word_matrix, word_map, tag_matrix, tag_map, N_iterations=2, lr=1):
    vec_size = word_matrix.shape[0]**2 + tag_matrix.shape[0]**2
    theta = np.zeros(vec_size)
    N_sentences = len(trees_train) # get the number of sentences
    theta_vectors = np.array((N_iterations * N_sentences, vec_size))
    theta_vectors[0] = theta
    list_feat_dict = create_list_of_feature_dic(trees_train,word_matrix, word_map, tag_matrix, tag_map)
    trees_orderes_tups = list_of_word_tup_per_tree(trees_train)
    for r in range(N_iterations):
        for i in range(N_sentences):
            mst_i_tups = trees_orderes_tups[i]
            sentence = sents_train[i]
            sent_feat_dict = list_feat_dict[i]
            theta_index = (r-1)*N_sentences + i
            arcs = get_arcs(sent_feat_dict, sentence, theta_vectors[theta_index - 1])
            # TODO check where to negativize the theta (or the arcs)
            MST_TAG = min_spanning_arborescence(arcs=arcs, sink=0)
            feat_diff = get_diff(MST_TAG,mst_i_tups,theta_vectors[theta_index-1], sent_feat_dict)
            theta_vectors[theta_index] = theta_vectors[theta_index-1] + lr * feat_diff
    return np.mean(theta_vectors)


def score():
    w = perceptron()


if __name__ == '__main__':
    trees = dependency_treebank.parsed_sents()
    sents = dependency_treebank.sents()
    # nd = n.nodes
    # x = n.tree()
    # print(x)
    corpus, POS_tags = get_corpus_and_tags_from_trees(trees)
    word_matrix, word_map = create_matrix(corpus)
    tag_matrix, tag_map = create_matrix(POS_tags)
    f = 0



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
