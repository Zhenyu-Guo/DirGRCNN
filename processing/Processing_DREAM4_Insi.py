import os
import csv
import numpy as np
import pandas as pd
import cmath
from sklearn.metrics import roc_auc_score
import time
import csv
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as ssp
import random
import pickle
import argparse
import util_functions as uf
from util_functions import DGCNN_classifer
import math
import torch
parser = argparse.ArgumentParser(description='Gene Regulatory Graph Neural Network in ensemble')
parser.add_argument('--traindata-name', default='data3', help='train network name')
parser.add_argument('--traindata-name2', default=None, help='also train another network')
parser.add_argument('--testdata-name', default='data4', help='test network name')
parser.add_argument('--max-train-num', type=int, default=100000,
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=43, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--training-ratio', type=float, default=1.0,
                    help='ratio of used training set')
parser.add_argument('--neighbors-ratio', type=float, default=1.0,
                    help='ratio of neighbors used')
parser.add_argument('--nonezerolabel-flag', default=False,
                    help='whether only use nonezerolabel flag')
parser.add_argument('--nonzerolabel-ratio', type=float, default=1.0,
                    help='ratio for nonzero label for training')
parser.add_argument('--zerolabel-ratio', type=float, default=0.0,
                    help='ratio for zero label for training')
# For debug
parser.add_argument('--feature-num', type=int, default=4,
                    help='feature num for debug')
# Pearson correlation
parser.add_argument('--embedding-dim', type=int, default=1,
                    help='embedding dimmension')
parser.add_argument('--pearson_net', type=float, default=0.8, #1
                    help='pearson correlation as the network')
parser.add_argument('--mutual_net', type=int, default=3, #3
                    help='mutual information as the network')
# model settings
parser.add_argument('--hop', type=int, default=1,
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None,
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=True,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=True,
                    help='whether to use node attributes')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


#**********************************
# 获取当前文件的目录
cur_path = os.path.abspath(os.path.dirname(__file__))
# 获取根目录
root_path = cur_path[:cur_path.find("DirGRGNN\\") + len("DirGRGNN\\")]
#DREAM4_insi数据路径
PATH_ECOLI_insilico_size100_1 = root_path + os.sep + "data" + os.sep + "insilico_size100_1_multifactorial.tsv"

#DREAM_Ecoli基因调控网络金标准
PATH_ECOLI_GoldStandards = root_path + os.sep + "data" + os.sep + "DREAM4_GoldStandard_InSilico_Size100_1.tsv"
#数据保存路径
PATH_eCOLI_process = root_path + os.sep + "data" + os.sep + "Result" +os.sep + "Lxixj.csv"
PATH_eCOLI_process1 = root_path + os.sep + "data" + os.sep + "Result" +os.sep + "SSEik.csv"
PATH_eCOLI_process2 = root_path + os.sep + "data" + os.sep + "Result" +os.sep + "Rik.csv"
PATH_eCOLI_process3 = root_path + os.sep + "data" + os.sep + "Result" +os.sep + "Wei.csv"
PATH_eCOLI_process4 = root_path + os.sep + "data" + os.sep + "Result" +os.sep + "Wei_Stand.csv"

dreamTFdict={}
dreamTFdict['dream4'] = 20

def Data_processing(file, node):
    load_data = pd.read_csv(filepath_or_buffer=file, sep='\t')
    load_data = load_data.values
    #去掉字母G，将Sting数据转换为int数据
    for i in range(load_data.shape[0]):
        for j in range(load_data.shape[1]-1):
            load_data[i][j] = int(load_data[i][j].strip("G"))
    #数据转换为有向图
    A = np.ones((node, node), dtype="int32")  #非负邻接矩阵
    D = np.zeros((node, node), dtype="int32")  #度矩阵
    for i in range(load_data.shape[0]):
            if load_data[i][2] == 0:
                A[load_data[i][0]-1][load_data[i][1]-1] = 0

    for i in range(node):
        degree = 0
        for j in range(node):
            if i == j:
                A[i][j] = 0
            if A[i][j] == 1:
                degree += 1
        D[i][i] = degree
    # A = pd.DataFrame(data=A)
    # D = pd.DataFrame(data=D)
    # A.to_csv(PATH_A, encoding="UTF-8")
    # D.to_csv(PATH_D, encoding="UTF-8")
    return A, D
def draw_directed_graph(my_graph):
    G = nx.DiGraph()  # 建立一个空的无向图G
    for node in my_graph.vertices:
        G.add_node(int(node))
    G.add_weighted_edges_from(my_graph.edges_array)
    print("nodes:", G.nodes())  # 输出全部的节点
    print("edges:", G.edges())  # 输出全部的边
    print("number of edges:", G.number_of_edges())  # 输出边的数量
    # position is stored as node attribute data for random_geometric_graph
    # pos = nx.layout.spring_layout(G)
    #
    # node_sizes = [3 + 0.1 * i for i in range(len(G))]
    # M = G.number_of_edges()
    # edge_colors = range(2, M + 2)
    # edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    #
    # nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="blue")
    # edges = nx.draw_networkx_edges(
    #     G,
    #     pos,
    #     node_size=node_sizes,
    #     arrowstyle="->",
    #     arrowsize=10,
    #     edge_color=edge_colors,
    #     edge_cmap=plt.cm.Blues,
    #     width=2,
    # )
    # # set alpha value for each edge
    # for i in range(M):
    #     edges[i].set_alpha(edge_alphas[i])
    #
    # pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    # pc.set_array(edge_colors)
    # plt.colorbar(pc)
    #
    # ax = plt.gca()
    # ax.set_axis_off()
    # plt.show()
    # G = G.to_undirected()
    color_list = ["gold", "violet", "violet", "violet",
                  "limegreen", "limegreen", "darkorange"]
    nx.draw(G, pos=nx.circular_layout(G), with_labels=True, node_size=100,
            node_color="deepskyblue", font_size=12,
            edge_color="teal", font_color="black", width=1, node_shape="o")
    plt.savefig("graph.png")
    plt.show()
class Graph_Matrix:
    """
    Adjacency Matrix
    """
    def __init__(self, vertices=[], matrix=[]):
        """

        :param vertices:a dict with vertex id and index of matrix , such as {vertex:index}
        :param matrix: a matrix
        """
        self.matrix = matrix
        self.edges_dict = {}  # {(tail, head):weight}
        self.edges_array = []  # (tail, head, weight)
        self.vertices = vertices
        self.num_edges = 0

        # if provide adjacency matrix then create the edges list
        if len(matrix) > 0:
            if len(vertices) != len(matrix):
                raise IndexError
            self.edges = self.getAllEdges()
            self.num_edges = len(self.edges)

        # if do not provide a adjacency matrix, but provide the vertices list, build a matrix with 0
        elif len(vertices) > 0:
            self.matrix = [[0 for col in range(len(vertices))] for row in range(len(vertices))]

        self.num_vertices = len(self.matrix)

    def isOutRange(self, x):
        try:
            if x >= self.num_vertices or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")

    def isEmpty(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices == 0

    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices[key] = len(self.vertices) + 1

        # add a vertex mean add a row and a column
        # add a column for every row
        for i in range(self.getVerticesNumbers()):
            self.matrix[i].append(0)

        self.num_vertices += 1

        nRow = [0] * self.num_vertices
        self.matrix.append(nRow)

    def getVertex(self, key):
        pass

    def add_edges_from_list(self, edges_list):  # edges_list : [(tail, head, weight),()]
        for i in range(len(edges_list)):
            self.add_edge(edges_list[i][0], edges_list[i][1], edges_list[i][2], )

    def add_edge(self, tail, head, cost=0):
        # if self.vertices.index(tail) >= 0:
        #   self.addVertex(tail)
        if tail not in self.vertices:
            self.add_vertex(tail)
        # if self.vertices.index(head) >= 0:
        #   self.addVertex(head)
        if head not in self.vertices:
            self.add_vertex(head)

        # for directory matrix
        self.matrix[self.vertices.index(tail)][self.vertices.index(head)] = cost
        # for non-directory matrix
        # self.matrix[self.vertices.index(fromV)][self.vertices.index(toV)] = \
        #   self.matrix[self.vertices.index(toV)][self.vertices.index(fromV)] = cost

        self.edges_dict[(tail, head)] = cost
        self.edges_array.append((tail, head, cost))
        self.num_edges = len(self.edges_dict)

    def getEdges(self, V):
        pass

    def getVerticesNumbers(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices


    def getAllVertices(self):
        return self.vertices

    def getAllEdges(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if 0 < self.matrix[i][j] < float('inf'):
                    self.edges_dict[self.vertices[i], self.vertices[j]] = self.matrix[i][j]
                    self.edges_array.append([self.vertices[i], self.vertices[j], self.matrix[i][j]])

        return self.edges_array

    def __repr__(self):
        return str(''.join(str(i) for i in self.matrix))

    def to_do_vertex(self, i):
        print('vertex: %s' % (self.vertices[i]))

    def to_do_edge(self, w, k):
        print('edge tail: %s, edge head: %s, weight: %s' % (self.vertices[w], self.vertices[k], str(self.matrix[w][k])))

def create_directed_matrix(Matrix):
    nodes = np.linspace(0, int(Matrix.shape[0])-1, int(Matrix.shape[0]))
    for i in range(len(nodes)):
        nodes[i] = int(nodes[i])
    matrix = Matrix
    my_graph = Graph_Matrix(nodes, matrix)
    return my_graph

def Regression_Relation(data_filename):
    print("**************【processing】**************")
    input_data = pd.read_csv(filepath_or_buffer=data_filename, sep="\t")    #导入数据
    data = input_data.values    #数据转换
    R_average = np.mean(data, axis=0) #R_average基因在多次稳态实验下的均值,每列的均值
    R_var = np.var(data, axis=0)    #R_var基因在多次稳态实验下的方差,每列的方差
    Lxixk = []
    for i in range(data.shape[1]):
        Lxixk.append([])
        for k in range(data.shape[1]):
            temp = 0
            for j in range(data.shape[0]):
                temp += (data[j, i]-R_average[i])*(data[j, i]-R_average[k])
            Lxixk[i].append(temp)
    Lxixk = pd.DataFrame(data=Lxixk)
    Lxixk.to_csv(PATH_eCOLI_process, encoding="UTF-8")
    print("**************【over】**************")
def CalculateSSE(filename):     #计算残差平方和与相关系数
    print("**************【calculating】**************")
    Lxixk = pd.read_csv(filepath_or_buffer=filename, sep=",")
    Lxixk = Lxixk.values
    Lxixk = Lxixk[:, 1:]
    SSEik = []
    Rik = []
    for i in range(Lxixk.shape[1]):
        SSEik.append([])
        Rik.append([])
        for k in range(Lxixk.shape[1]):
            #计算残差和
            SSEik[i].append(Lxixk[k][k] - np.power((Lxixk[i][k]/Lxixk[i][i]), 2)*Lxixk[i][i])
            #计算相关系数
            Rik[i].append(Lxixk[i][k]/(np.sqrt(Lxixk[i][i]) * np.sqrt(Lxixk[k][k])))
    SSEik = pd.DataFrame(data=SSEik)
    Rik = pd.DataFrame(data=Rik)
    #保存数据
    SSEik.to_csv(PATH_eCOLI_process1, encoding="UTF-8")
    Rik.to_csv(PATH_eCOLI_process2, encoding="UTF-8")
    print("**************【over】**************")

def Weight(filename1, filename2):
    SSE = pd.read_csv(filepath_or_buffer=filename1, sep=",")
    REL = pd.read_csv(filepath_or_buffer=filename2, sep=",")
    SSE =SSE.values
    REL = REL.values
    SSE = SSE[:, 1:]
    REL = REL[:, 1:]
    Wei = []
    for i in range(SSE.shape[1]):
        Wei.append([])
        for k in range(SSE.shape[1]):
            if i == k:
                Wei[i].append(0)
            else:
                Wei[i].append(abs(REL[i][k] * cmath.exp(-SSE[i][k])))
    Wei = pd.DataFrame(data=Wei)
    Wei.to_csv(PATH_eCOLI_process3, encoding="UTF-8")
#q范数归一化
def Standardized(filename, q = 2, Threshold=0.5):
    print("**************【Standardized】**************")
    Wei = pd.read_csv(filepath_or_buffer=filename, sep=",")
    Wei = Wei.values
    Wei = Wei[:, 1:]
    Wei_Stand = []
    for i in range(Wei.shape[1]):
        Wei_Stand.append([])
        for k in range(Wei.shape[1]):
            temp = 0
            if i != k:
                for p in range(Wei.shape[1]):
                    temp += np.power(Wei[p][k], q)
                Wei_Stand[i].append(Wei[i][k] / np.power(temp, 1 / q))
            else:
                Wei_Stand[i].append(0)

    # sum = 0
    # for i in range(Wei.shape[1]):
    #     for k in range(Wei.shape[1]):
    #         if Wei_Stand[i][k] < Threshold:
    #             Wei_Stand[i][k] = 0
    #         else:
    #             Wei_Stand[i][k] = 1
    #             sum += 1
    # print("The total edges are : ", sum)
    Wei_Stand = pd.DataFrame(data=Wei_Stand)
    Wei_Stand.to_csv(PATH_eCOLI_process4, encoding="UTF-8")
    print("**************【over】**************")
def processing():
    Regression_Relation(PATH_ECOLI_insilico_size100_1)
    CalculateSSE(PATH_eCOLI_process)
    Weight(PATH_eCOLI_process1, PATH_eCOLI_process2)
    Standardized(PATH_eCOLI_process3, q=2, Threshold=0.35)
def main():
    # processing()
    A, _ = Data_processing(PATH_ECOLI_GoldStandards, 100)
    graph = create_directed_matrix(A)
    ori_A = A
    A = ssp.csc_matrix(A)
    ori_A = ssp.csc_matrix(ori_A)
    pickle.dump(A, open("../data/Result/ins.csc", "wb"))
    pickle.dump(ori_A, open("../data/Result/ins.allx", "wb"))
    pickle.dump(A, open("../data/Result/ins.npy", "wb"))
    trainNet_ori = np.load(os.path.join(cur_path, "../data/Result/ins.csc"), allow_pickle=True)
    trainGroup = np.load(os.path.join(cur_path, "../data/Result/ins.allx"), allow_pickle=True)
    trainNet_agent0 = np.load(os.path.join(cur_path, "../data/Result/ins.npy"), allow_pickle=True)
    trainNet_agent1 = np.load(os.path.join(cur_path, "../data/Result/ins.npy"), allow_pickle=True)
    print("trainNet_agent0: ", trainNet_agent0.shape)
    print("trainNet_agent1: ", trainNet_agent1.shape)

    allx =trainGroup.toarray().astype("float32")
    """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
    trainAttributes = uf.genenet_attribute(allx, dreamTFdict["dream4"])


    testNet_ori = np.load(os.path.join(cur_path, "../data/Result/ins.csc"), allow_pickle=True)
    testGroup = np.load(os.path.join(cur_path, "../data/Result/ins.allx"), allow_pickle=True)
    testNet_agent0 = np.load(os.path.join(cur_path, "../data/Result/ins.npy"), allow_pickle=True)
    testNet_agent1 = np.load(os.path.join(cur_path, "../data/Result/ins.npy"), allow_pickle=True)
    print("testNet_agent0: ", testNet_agent0.shape)
    print("testNet_agent1: ", testNet_agent1.shape)

    allx =testGroup.toarray().astype("float32")
    """traintArreibutes: 显性特征，将数据转换成标准的正太分布后，横向计算均值用作显性特征。"""
    trainAttributes = uf.genenet_attribute(allx, dreamTFdict["dream4"])


    train_pos, train_neg, _, _ = uf.sample_neg_TF(trainNet_ori, 0.0, TF_num=dreamTFdict["dream4"], max_train_num=args.max_train_num)
    use_pos_size = math.floor(len(train_pos[0]) * args.training_ratio)
    use_neg_size = math.floor(len(train_neg[0]) * args.training_ratio)
    train_pos = (train_pos[0][:use_pos_size], train_pos[1][:use_pos_size])
    train_neg = (train_neg[0][:use_neg_size], train_neg[1][:use_neg_size])
    _, _, test_pos, test_neg = uf.sample_neg_TF(testNet_ori, 1.0, TF_num=dreamTFdict["dream4"], max_train_num=args.max_train_num)

    '''Train and apply classifier'''
    Atrain_agent0 = trainNet_agent0.copy()  # the observed network
    Atrain_agent1 = trainNet_agent1.copy()
    Atest_agent0 = testNet_agent0.copy()  # the observed network
    Atest_agent1 = testNet_agent1.copy()
    Atest_agent0[test_pos[0], test_pos[1]] = 0  # mask test links
    Atest_agent0[test_pos[1], test_pos[0]] = 0  # mask test links
    Atest_agent1[test_pos[0], test_pos[1]] = 0  # mask test links
    Atest_agent1[test_pos[1], test_pos[0]] = 0  # mask test links

    # train_node_information = None
    # test_node_information = None
    if args.use_embedding:
        train_embeddings_agent0 = uf.generate_node2vec_embeddings(Atrain_agent0, args.embedding_dim, True,
                                                                              train_neg)  # ?
        train_node_information_agent0 = train_embeddings_agent0
        test_embeddings_agent0 = uf.generate_node2vec_embeddings(Atest_agent0, args.embedding_dim, True,
                                                                             test_neg)  # ?
        test_node_information_agent0 = test_embeddings_agent0

        train_embeddings_agent1 = uf.generate_node2vec_embeddings(Atrain_agent1, args.embedding_dim, True,
                                                                              train_neg)  # ?
        train_node_information_agent1 = train_embeddings_agent1
        test_embeddings_agent1 = uf.generate_node2vec_embeddings(Atest_agent1, args.embedding_dim, True,
                                                                             test_neg)  # ?
        test_node_information_agent1 = test_embeddings_agent1

    # Original: 获取子图，用于训练，子图为无向图
    train_graphs_agent0, test_graphs_agent0, max_n_label_agent0 = uf.extractLinks2subgraphs(Atrain_agent0,
                                                                                            Atest_agent0, train_pos,
                                                                                            train_neg, test_pos, test_neg,
                                                                                            args.hop, args.max_nodes_per_hop,
                                                                                            train_node_information_agent0,
                                                                                            test_node_information_agent0)
    # for Graph in train_graphs_agent0:
    #     nx.draw(Graph.graph, with_labels=True)
    #     plt.show()

    train_graphs_agent1, test_graphs_agent1, max_n_label_agent1 = uf.extractLinks2subgraphs(Atrain_agent1,
                                                                                            Atest_agent1,train_pos,
                                                                                            train_neg, test_pos, test_neg,
                                                                                            args.hop, args.max_nodes_per_hop,
                                                                                            train_node_information_agent1,
                                                                                            test_node_information_agent1)

    print('# train: %d, # test: %d' % (len(train_graphs_agent0), len(test_graphs_agent0)))
    print("**************【max_n_label_agent0】*******************")
    print(max_n_label_agent0)
    print("**************【max_n_label_agent1】*******************")
    print(max_n_label_agent1)

    # Agent 0
    _, _, test_neg_agent0, _, test_prob_agent0 = \
        DGCNN_classifer(train_graphs_agent0, test_graphs_agent0, train_node_information_agent0,
                        max_n_label_agent0, set_epoch=50, eval_flag=True)
    # Agent 1
    _, _, test_neg_agent1, _, test_prob_agent1 = \
        DGCNN_classifer(train_graphs_agent1, test_graphs_agent1, train_node_information_agent1,
                        max_n_label_agent1, set_epoch=50, eval_flag=True)

    # Generate
    trueList = []
    for i in range(len(test_pos[0])):
        trueList.append(1)
    for i in range(len(test_neg[0])):
        trueList.append(0)

    ensembleProb = []

    dic_agent0 = {}
    for i in test_neg_agent0:
        dic_agent0[i] = 0
    dic_agent1 = {}
    for i in test_neg_agent1:
        dic_agent1[i] = 0
    bothwrong = 0
    corrected = 0
    uncorrected = 0
    count = 0

    tp0 = 0
    tp1 = 0
    tn0 = 0
    tn1 = 0
    tp = 0
    tn = 0
    eprob = 0
    testpos_size = len(test_pos[0])
    for i in np.arange(len(test_prob_agent0)):
        if i < testpos_size:  # positive part
            if i in dic_agent0 or i in dic_agent1:
                if test_prob_agent0[i] * test_prob_agent1[i] > 0:
                    # both wrong
                    bothwrong = bothwrong + 1
                    eprob = -test_prob_agent0[i] * test_prob_agent1[i]
                else:
                    if abs(test_prob_agent0[i]) > abs(test_prob_agent1[i]):
                        if i in dic_agent0 and i not in dic_agent1:
                            uncorrected = uncorrected + 1
                            tp1 = tp1 + 1
                            eprob = test_prob_agent0[i] * test_prob_agent1[i]
                        else:
                            corrected = corrected + 1
                            count = count + 1
                            tp = tp + 1
                            tp0 = tp0 + 1
                            eprob = -test_prob_agent0[i] * test_prob_agent1[i]
                    else:
                        if i in dic_agent0 and i not in dic_agent1:
                            corrected = corrected + 1
                            count = count + 1
                            tp = tp + 1
                            tp1 = tp1 + 1
                            eprob = -test_prob_agent0[i] * test_prob_agent1[i]
                        else:
                            uncorrected = uncorrected + 1
                            tp0 = tp0 + 1
                            eprob = test_prob_agent0[i] * test_prob_agent1[i]
            else:
                count = count + 1
                tp = tp + 1
                tp0 = tp0 + 1
                tp1 = tp1 + 1
                eprob = test_prob_agent0[i] * test_prob_agent1[i]
        else:  # negative part
            if i in dic_agent0 or i in dic_agent1:
                if test_prob_agent0[i] * test_prob_agent1[i] > 0:
                    # both wrong
                    bothwrong = bothwrong + 1
                    eprob = -test_prob_agent0[i] * test_prob_agent1[i]
                else:
                    if abs(test_prob_agent0[i]) > abs(test_prob_agent1[i]):
                        if i in dic_agent0 and i not in dic_agent1:
                            uncorrected = uncorrected + 1
                            tn1 = tn1 + 1
                            eprob = -test_prob_agent0[i] * test_prob_agent1[i]
                        else:
                            corrected = corrected + 1
                            count = count + 1
                            tn = tn + 1
                            tn0 = tn0 + 1
                            eprob = test_prob_agent0[i] * test_prob_agent1[i]
                    else:
                        if i in dic_agent0 and i not in dic_agent1:
                            corrected = corrected + 1
                            count = count + 1
                            tn = tn + 1
                            tn1 = tn1 + 1
                            eprob = test_prob_agent0[i] * test_prob_agent1[i]
                        else:
                            uncorrected = uncorrected + 1
                            tn0 = tn0 + 1
                            eprob = -test_prob_agent0[i] * test_prob_agent1[i]
            else:
                count = count + 1
                tn = tn + 1
                tn0 = tn0 + 1
                tn1 = tn1 + 1
                eprob = -test_prob_agent0[i] * test_prob_agent1[i]

        ensembleProb.append(eprob)

    print("Both agents right: " + str(count))
    print("Both agents wrong: " + str(bothwrong))
    print("Corrected by Ensembl: " + str(corrected))
    print("Not corrected by Ensembl: " + str(uncorrected))

    allstr = str(float((tp + tn) / len(test_graphs_agent0))) + "\t" + str(tp) + "\t" + str(
        len(test_pos[0]) - tp) + "\t" + str(tn) + "\t" + str(len(test_neg[0]) - tn) + "\t" + str(
        roc_auc_score(trueList, ensembleProb))
    agent0_str = str(float((tp0 + tn0) / len(test_graphs_agent0))) + "\t" + str(tp0) + "\t" + str(
        len(test_pos[0]) - tp0) + "\t" + str(tn0) + "\t" + str(len(test_neg[0]) - tn0) + "\t" + str(
        roc_auc_score(trueList, test_prob_agent0))
    agent1_str = str(float((tp1 + tn1) / len(test_graphs_agent0))) + "\t" + str(tp1) + "\t" + str(
        len(test_pos[0]) - tp1) + "\t" + str(tn1) + "\t" + str(len(test_neg[0]) - tn1) + "\t" + str(
        roc_auc_score(trueList, test_prob_agent1))
    result = str(float(count / len(test_graphs_agent0)))
    print("Ensemble:Accuracy tp fn tn fp AUC")
    print(allstr + "\n")
    print("Agent0:Accuracy\ttp\tfn\ttn\tfp\tAUC")
    print(agent0_str + "\n")
    print("Agent1:Accuracy tp fn tn fp AUC")
    print(agent1_str + "\n")

    # Output results
    with open('acc_result.txt', 'a+') as f:
        f.write(allstr + "\t" + agent0_str + "\t" + agent1_str + '\n')

    print("**************************************************")
    print("**                                              **")
    print("**                  【Over】                     **")
    print("**                                              **")
    print("**************************************************")
if __name__ == "__main__":
    main()




