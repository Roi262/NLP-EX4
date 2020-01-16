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

FULL_DATA = 3914
DATA_SIZE = FULL_DATA

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


def feature_func(u, v, word_matrix_size, tag_matrix_size, word_map, tag_map,tag_u, tag_v, sent=None, augmented=False):
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
    #todo move to the tags they already hold:


    i2, j2 = tag_map[tag_u], tag_map[tag_v]
    positive_indeces.append(word_matrix_size + (i2 * sqrt(tag_matrix_size) + j2))

    if augmented:
        aug_pos_indeces = get_augmented_features(
            u, v, sent, last_index=word_matrix_size + tag_matrix_size)
        positive_indeces.extend(aug_pos_indeces)
    # a = (np.array(positive_indeces))
    return np.array(positive_indeces).astype(np.int)



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
    corpus.add("ROOT") #todo added here ROOT
    POS_tags.add("ROOT")
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


def create_graph(corpus, word_map, theta):
    arcs = []
    for i, head in enumerate(corpus):
        for j, tail in enumerate(corpus):
            arc = Arc(tail, 0, head)
            arcs.append(arc)
    return arcs


def get_arcs(sent_feat_dict, sentence, theta):
    arcs = []
    for i, head in enumerate(sentence):
        for j, tail in enumerate(sentence):
            if i==j:
                continue
            weight = np.sum(theta.take(sent_feat_dict[(head, tail)]))
            # weight = np.sum([theta[s] for s in feature_dict[(head, tail)]])
            arc = Arc(tail, weight, head) #todo changed here tail and head
            arcs.append(arc)
    return arcs

def create_list_of_feature_dic(train_sents, word_matrix_size, word_map, tag_matrix_size, tag_map, tagged_sents,
                               pkl_path,
                               pickles=True, augmented=False):
    if pickles:
        list_feat_dict_path = pkl_path
        if not os.path.exists(list_feat_dict_path):
            list_feat_dics = []
            for g,sent in tqdm(enumerate(train_sents)):
                tagged = tagged_sents[g]
                feat_dict = {}
                for i, word1 in enumerate(sent):
                    for j, word2 in enumerate(sent):
                        if (i == j):
                            continue
                        tag_word1 = tagged[i][1]
                        tag_word2 = tagged[j][1]
                        feat_dict[(word1,word2)] = feature_func(word1, word2, word_matrix_size, tag_matrix_size,
                                                                word_map, tag_map,tag_word1, tag_word2,
                                                                augmented=augmented)
                list_feat_dics.append(feat_dict)
            save_pickle(list_feat_dics,list_feat_dict_path)
        else:
            list_feat_dics = load_pickle(list_feat_dict_path)
    else:
        list_feat_dics = []
        for g, sent in tqdm(enumerate(train_sents)):
            tagged = tagged_sents[g]
            feat_dict = {}
            for i, word1 in enumerate(sent):
                for j, word2 in enumerate(sent):
                    if (i == j):
                        continue
                    tag_word1 = tagged[i][1]
                    tag_word2 = tagged[j][1]
                    feat_dict[(word1,word2)] = feature_func(word1, word2, word_matrix_size, tag_matrix_size,
                                                            word_map, tag_map,tag_word1,tag_word2,  augmented=augmented)
            list_feat_dics.append(feat_dict)

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
    return (mst_i_score-mst_tag_score)


def list_of_word_tup_per_tree(trees_train, pkl_path):
    list_tup_per_tree_path = pkl_path
    if not os.path.exists(list_tup_per_tree_path):
        tree_list_of_tup_words = []
        for tree in trees_train:
            tree_list = []
            for node in range(len(tree.nodes)):
                tail = tree.nodes[node]['word']
                head_idx = tree.nodes[node]['head']
                head = tree.nodes[head_idx]['word']
                if not head:
                    head = "ROOT"
                if tail == None: #todo changed here remove head= none
                    continue
                tree_list.append((head,tail))
                l = (head,tail)
            tree_list_of_tup_words.append(tree_list)
        save_pickle(tree_list_of_tup_words,list_tup_per_tree_path)
    else:
        tree_list_of_tup_words = load_pickle(list_tup_per_tree_path)
    return tree_list_of_tup_words


def perceptron(trees_train, sents_train, word_matrix_size, word_map, tag_matrix_size, tag_map,
               tagged_sents, N_iterations=2,lr=1, augmented=False):
    vec_size = word_matrix_size + tag_matrix_size
    if augmented:
        vec_size += AUGMENT_FACTOR
    theta = np.zeros(vec_size)
    N_sentences = len(trees_train) # get the number of sentences
    # theta_vectors = np.zeros((N_iterations * N_sentences, vec_size))
    theta_sum = np.zeros(vec_size)
    list_feat_dict = create_list_of_feature_dic(sents_train, word_matrix_size, word_map, tag_matrix_size, tag_map,
                                                tagged_sents,
                                                pickles=True,
                                                pkl_path="list_feat_dict_train"+str(DATA_SIZE)+".pkl",
                                                augmented=augmented)
    trees_orderes_tups = list_of_word_tup_per_tree(trees_train, "list_tup_per_tree_train"+str(DATA_SIZE)+".pkl")
    permutation = np.random.permutation(N_sentences)
    permutation = np.arange(N_sentences) #todo delete for permutation
    for r in range(N_iterations):
        print("Epoch ", r)
        for i in tqdm(permutation):
            mst_i_tups = trees_orderes_tups[i]
            sentence = sents_train[i]
            sent_feat_dict = list_feat_dict[i]
            theta_index = (r-1)*N_sentences + i
            arcs = get_arcs(sent_feat_dict, sentence, theta)
            MST_TAG = min_spanning_arborescence(arcs=arcs, sink=0)
            # print(arcs)
            feat_diff = get_diff(MST_TAG,mst_i_tups,theta, sent_feat_dict)
            new_theta = theta + (lr * feat_diff)
            pre_theta_sum = theta_sum
            theta_sum += new_theta
            theta = new_theta
            # print(np.all(np.equal(pre_theta_sum,theta_sum)))
    out_theta = theta_sum.astype(np.float)/(N_iterations*N_sentences)
    return out_theta


def compare_tree(pred_tree_arcs, y_tree_tups):
    counter = 0
    for arc in pred_tree_arcs:
        pred_tup = (pred_tree_arcs[arc].head,pred_tree_arcs[arc].tail)
        if pred_tup in y_tree_tups:
            counter += 1
    return counter


def evaluate(theta, tree_test, sent_test, t_list_feat_dict, augmented = False):

    trees_orderes_tups = list_of_word_tup_per_tree(tree_test, "list_tup_per_tree_test"+str(DATA_SIZE)+".pkl")
    all_scores = 0
    for i,sentence in enumerate(sent_test):
        y_tree_tups = trees_orderes_tups[i]
        sent_feat_dict = t_list_feat_dict[i]
        arcs = get_arcs(sent_feat_dict, sentence, theta)
        pred_tree = min_spanning_arborescence(arcs=arcs, sink=0)
        num_of_sim_arcs = compare_tree(pred_tree,y_tree_tups)
        all_scores += float(num_of_sim_arcs)/(len(sentence)-1) #todo del -1
    return float(all_scores)/len(sent_test)

def add_root(sents):
    root_sents = []
    for s in sents:
        root_sents.append(s+["ROOT"])
    return root_sents

def add_root_tup(sents):
    root_sents = []
    for i,s in enumerate(sents):
        root_sents.append(s)
        root_sents[i].append(tuple(("ROOT","ROOT")))
    return root_sents

def main():
    trees = dependency_treebank.parsed_sents()
    sents = list(dependency_treebank.sents())
    tagged_sents = list(dependency_treebank.tagged_sents())
    x= len(tagged_sents)

    # for sent in sents_train:
    #     print(sent)
    #
    # TESTER LINES
    sents = sents[:DATA_SIZE]
    trees = trees[:DATA_SIZE]
    tagged_sents = tagged_sents[:DATA_SIZE]
    #############
    split_index = int(len(sents) * 0.9)
    trees_train = trees[:split_index]
    trees_test = trees[split_index:]
    sents_train = sents[:split_index]
    sents_test = sents[split_index:]
    tagged_sents_test = tagged_sents[split_index:]
    tagged_sents_train = tagged_sents[:split_index]

    sents_test = add_root(sents_test) #todo added roots
    sents_train= add_root(sents_train)
    tagged_sents = add_root_tup(tagged_sents_train)

    corpus, POS_tags = get_corpus_and_tags_from_trees(trees)
    word_matrix_size, word_map = create_matrix(corpus)
    tag_matrix_size, tag_map = create_matrix(POS_tags)

    tagged_sents_test = add_root_tup(tagged_sents_test)

    t_list_feat_dict = create_list_of_feature_dic(sents_test, word_matrix_size, word_map, tag_matrix_size, tag_map,
                                                  tagged_sents_test,
                                                  pickles=True,
                                                  pkl_path="list_feat_dict_test"+str(DATA_SIZE)+".pkl")
    # Q3
    w = perceptron(trees_train, sents_train, word_matrix_size, word_map, tag_matrix_size, tag_map,tagged_sents_train)
    # print(np.sum(w))

    #Evaluate:
    score = evaluate(theta=w,tree_test= trees_test,sent_test= sents_test, t_list_feat_dict=t_list_feat_dict)
    print("Accuracy :",score)

    # Q4
    w = perceptron(trees_train, sents_train, word_matrix_size, word_map, tag_matrix_size, tag_map,tagged_sents,
                   augmented=True)

    score = evaluate(theta=w,tree_test= trees_test,sent_test= sents_test, t_list_feat_dict=t_list_feat_dict)
    print("Accuracy augumented: ", score)

if __name__ == '__main__':
    main()
