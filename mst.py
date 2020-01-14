from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from random import shuffle
# nltk.download()
from nltk.corpus import dependency_treebank
from scipy.sparse import csr_matrix

TRAIN_PERC = .9
TEST_PERC = .1


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

mat = np.arange(9).reshape((3,3))

# l = csr_matrix((3, 4)).toarray()
l = np.arange(15).reshape((3,5))
# print(l)
x = make_sparse_vec(mat)
y = make_sparse_vec(l)
print(x)
print(y)
added = np.append(x,y)
print(added)


def feature_func(u,v, word_matrix, tag_matrix, word_map, tag_map):
    word_matrix, tag_matrix = np.copy(word_matrix), np.copy(tag_matrix)
    word_matrix[word_map[u]][word_map[v]] = 1
    word_vec = make_sparse_vec(word_matrix)
    tup_place = 0
    pos_place = 1
    POS_u = nltk.pos_tag([u])[tup_place][pos_place]
    POS_v = nltk.pos_tag([v])[tup_place][pos_place]
    x = tag_map[POS_u]
    y = tag_map[POS_v]
    tag_matrix[x][y] = 1

    tag_vec = make_sparse_vec(tag_matrix)

    return np.append(word_vec,tag_vec)


# def feature_func(u, v, sent, pos_tags):
#     """A boolean feature functionthat encodes word and POS bigrams.
#
#     Arguments:
#         v1 {string} -- one node (word) v1 in V={w1,...,wn,ROOT}
#         v2 {string} -- second node (word) v2 in V={w1,...,wn,ROOT}
#         sent {list of strings} -- a sentance {w1,...,wn}
#         dim {int} -- the dimension of the boolean output vector
#     """
#     n = len(sent)
#     dim = n * n
#     pos_tags = pos_tags
#     if not pos_tags:
#         pos_tags = [tag for word, tag in nltk.pos_tag(sent)]
#     output = np.zeros(shape=dim)
#     for i in range(n):
#         for j in range(n):
#             position = n * i + j
#             # word bigram
#             if sent[i] == u and sent[j] == v:  # or (sent[i] == v2 and sent[j] == v1):
#                 output[position] = 1
#             # pos bigram
#             if pos_tags[i] == u and pos_tags[j] == v:  # or (sent[i] == v2 and sent[j] == v1):
#                 output[position] = 1
#
#     # TODO im not dealing with the case where u or v are the POS tag ROOT


def perceptronXX(train_set, word_matrix, word_map, tag_matrix, tag_map, iterations=2, lr=1):
    shuffled_train_set = train_set


    for i in range(iterations):
        shuffle(shuffled_train_set)
        pass

    pass

def get_tree_sum(tree, vector_size, word_matrix, tag_matrix, word_map, tag_map):
    # TODO - check function
    sum = np.zeros(vector_size)
    words = [tree.nodes[node_index]['word'] for node_index in tree.nodes]
    for i in range(len(words)):
        for j in range(len(words)):
            sum += feature_func(words[i], words[j], word_matrix, tag_matrix, word_map, tag_map)
    return sum


def get_max_trees_score(trees, theta):
    max_score = 0
    for i in range:
        pass

def update_theta(theta, tree1, tree2, lr=1):
    return theta + lr * (get_tree_sum(tree1) - get_tree_sum(tree2))

def perceptron(train_set, word_matrix, word_map, tag_matrix, tag_map, N_iterations=2, lr=1):
    vec_size = len(word_matrix)**2 + len(word_matrix)**2
    theta = np.zeros(vec_size) # TODO do we initialize this as zeros?
    N_sentences = len(train_set) # get the number of sentences
    theta_vectors = np.array((N_iterations * N_sentences, vec_size))
    theta_vectors[0] = theta
    for r in range(N_iterations):
        for i in range(N_sentences):
            # TODO send one tree from trees? and find the mst for it?
            T_tag = get_max_trees_score(arcs , theta_vectors[(r-1)*N_sentences + i - 1])
            theta_vectors[(r-1)*N_sentences + i] = update_theta()
    return np.mean(theta_vectors)




def score():
    w = perceptron()

if __name__ == '__main__':
    trees = dependency_treebank.parsed_sents()
    corpus, POS_tags = get_corpus_and_tags_from_trees(trees)
    word_matrix, word_map = create_matrix(corpus)
    tag_matrix, tag_map = create_matrix(POS_tags)

    k = feature_func('Pierre','Vinken',word_matrix,tag_matrix,word_map,tag_map)
    print(np.argwhere(k==1))
    p=0

    # print(trees[0])
    #  divide the obtained corpus into training set
    # and test set such that the test set is formed by the last 10% of the sentences.
    shape = int(len(trees) * TRAIN_PERC)
    # train, test = sents[:int(len(sents)*TRAIN_PERC)], sents[int(len(sents)*TRAIN_PERC):]
    train, test = np.split(trees, [shape])
