from __future__ import print_function
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import os, sys, pdb, math
import _pickle as cp
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from software.node2vec.src import node2vec
from gensim.models.word2vec import Word2Vec
from sklearn import metrics
from software.pytorch_DGCNN.main import *
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='DGCNN', help='gnn model to use')
cmd_opt.add_argument('-data', default="ecoli_data", help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-edge_feat_dim', type=int, default=0, help='dimension of edge features')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0, help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=10, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='64', help='dimension(s) of latent layers')
cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-conv1d_activation', type=str, default='ReLU', help='which nn activation layer to use')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='graph embedding output size')
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of mlp hidden layer')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
cmd_opt.add_argument('-printAUC', type=bool, default=False, help='whether to print AUC (for binary classification only)')
cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')

cmd_args, _ = cmd_opt.parse_known_args()

# cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]

class Classifier(nn.Module):
    def __init__(self, regression=False):
        super(Classifier, self).__init__()
        self.regression = regression
        print(cmd_args)
        if cmd_args.gm == 'DGCNN':
            model = DGCNN
            print('DGCNN')
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.gnn = model(latent_dim=cmd_args.latent_dim,
                             output_dim=cmd_args.out_dim,
                             num_node_feats=cmd_args.feat_dim,
                             num_edge_feats=cmd_args.edge_feat_dim,
                             k=cmd_args.sortpooling_k,
                             conv1d_activation=cmd_args.conv1d_activation)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.gnn.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class,
                                 with_dropout=cmd_args.dropout)
        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=cmd_args.hidden, with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        if cmd_args.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)
        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, 1)
            node_tag.scatter_(0, concat_tag, 1.0)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            if edge_feat_flag == True:
                edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels
        return node_feat, labels

    def forward(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels
def DGCNN_classifer(train_graphs, test_graphs, train_node_information, max_n_label, set_epoch=50, eval_flag=True):
    # DGCNN configurations
    cmd_args.gm = 'DGCNN'
    cmd_args.sortpooling_k = 0.6
    cmd_args.latent_dim = [32, 32, 32, 1]
    cmd_args.hidden = 128
    cmd_args.out_dim = 0
    cmd_args.dropout = True
    cmd_args.num_class = 4    #分类类别数
    cmd_args.mode = 'cpu'
    cmd_args.num_epochs = set_epoch
    cmd_args.learning_rate = 1e-4
    cmd_args.batch_size = 50
    cmd_args.printAUC = True
    cmd_args.feat_dim = 2
    cmd_args.attr_dim = 0
    if train_node_information is not None:
        cmd_args.attr_dim = train_node_information.shape[1]
    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))
    print(cmd_args)
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss, train_neg_idx, train_prob_results = loop_dataset(train_graphs, classifier, train_idxes,
                                                                   optimizer=optimizer)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
        epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

        test_loss = []
        test_neg_idx = []
        test_prob_results = []
        if eval_flag:
            classifier.eval()
            test_loss, test_neg_idx, test_prob_results = loop_dataset(test_graphs, classifier,
                                                                      list(range(len(test_graphs))))
            if not cmd_args.printAUC:
                test_loss[2] = 0.0
            print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
            epoch, test_loss[0], test_loss[1], test_loss[2]))

    return test_loss, train_neg_idx, test_neg_idx, train_prob_results, test_prob_results
class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())
        self.graph = g

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(edge_features.values()[0]) == np.ndarray)
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)
# Generate explicit features for inductive learning, get trends features
def genenet_attribute(allx, tfNum):
    # 1: average to one dimension
    allx_ = StandardScaler().fit_transform(allx)
    trainAttributes = np.average(allx_, axis=1).reshape((len(allx), 1))
    # 2: std,min,max as the attribute
    # meanAtt = np.average(allx, axis=1).reshape((len(allx_), 1))
    # stdAtt = np.std(allx, axis=1).reshape((len(allx_), 1))
    # minVal = np.min(allx, axis=1).reshape((len(allx_), 1))
    # expAtt = allx[:,:536]

    # #2 folder
    # qu1Val = np.quantile(allx,0.5, axis=1).reshape((len(allx_),1))
    # maxVal = np.max(allx,axis=1).reshape((len(allx_),1))

    # qu1Att = (qu1Val-minVal)/(maxVal-minVal)
    # qu2Att = (maxVal-qu1Val)/(maxVal-minVal)

    # quantilPerAtt =np.concatenate([qu1Att,qu2Att],axis=1)
    # quantilValAtt =np.concatenate([minVal, qu1Val, maxVal],axis=1)

    # 4 folder
    # qu1Val = np.quantile(allx, 0.25, axis=1).reshape((len(allx_), 1))
    # qu2Val = np.quantile(allx, 0.5, axis=1).reshape((len(allx_), 1))
    # qu3Val = np.quantile(allx, 0.75, axis=1).reshape((len(allx_), 1))
    # maxVal = np.max(allx, axis=1).reshape((len(allx_), 1))

    # qu1Att = (qu1Val - minVal) / (maxVal - minVal)
    # qu2Att = (qu2Val - qu1Val) / (maxVal - minVal)
    # qu3Att = (qu3Val - qu2Val) / (maxVal - minVal)
    # qu4Att = (maxVal - qu3Val) / (maxVal - minVal)

    # quantilPerAtt = np.concatenate([qu1Att, qu2Att, qu3Att, qu4Att], axis=1)
    # quantilValAtt = np.concatenate([minVal, qu1Val, qu2Val, qu3Val, maxVal], axis=1)

    # 10 folder
    # qu1Val = np.quantile(allx,0.1, axis=1).reshape((len(allx_),1))
    # qu2Val = np.quantile(allx,0.2, axis=1).reshape((len(allx_),1))
    # qu3Val = np.quantile(allx,0.3, axis=1).reshape((len(allx_),1))
    # qu4Val = np.quantile(allx,0.4, axis=1).reshape((len(allx_),1))
    # qu5Val = np.quantile(allx,0.5, axis=1).reshape((len(allx_),1))
    # qu6Val = np.quantile(allx,0.6, axis=1).reshape((len(allx_),1))
    # qu7Val = np.quantile(allx,0.7, axis=1).reshape((len(allx_),1))
    # qu8Val = np.quantile(allx,0.8, axis=1).reshape((len(allx_),1))
    # qu9Val = np.quantile(allx,0.9, axis=1).reshape((len(allx_),1))
    # maxVal = np.max(allx,axis=1).reshape((len(allx_),1))

    # qu1Att = (qu1Val-minVal)/(maxVal-minVal)
    # qu2Att = (qu2Val-qu1Val)/(maxVal-minVal)
    # qu3Att = (qu3Val-qu2Val)/(maxVal-minVal)
    # qu4Att = (qu4Val-qu3Val)/(maxVal-minVal)
    # qu5Att = (qu5Val-qu4Val)/(maxVal-minVal)
    # qu6Att = (qu6Val-qu5Val)/(maxVal-minVal)
    # qu7Att = (qu7Val-qu6Val)/(maxVal-minVal)
    # qu8Att = (qu8Val-qu7Val)/(maxVal-minVal)
    # qu9Att = (qu9Val-qu8Val)/(maxVal-minVal)
    # qu10Att = (maxVal-qu9Val)/(maxVal-minVal)

    # quantilPerAtt =np.concatenate([qu1Att,qu2Att,qu3Att,qu4Att,qu5Att,qu6Att,qu7Att,qu8Att,qu9Att,qu10Att],axis=1)
    # quantilValAtt =np.concatenate([minVal, qu1Val,qu2Val,qu3Val,qu4Val, qu5Val, qu6Val, qu7Val, qu8Val, qu9Val, maxVal],axis=1)

    # 5: TF or not, vital
    tfAttr = np.zeros((len(allx), 1))
    for i in np.arange(tfNum):
        tfAttr[i] = 1.0

    # 2. PCA to 3 dimensions
    allx_ = StandardScaler().fit_transform(allx)
    pca = PCA(n_components=3)
    pcaAttr = pca.fit_transform(allx_)

    # trainAttributes = np.concatenate([trainAttributes, stdAtt, minAtt, qu1Att, qu3Att, maxAtt, tfAttr], axis=1)
    # trainAttributes = np.concatenate([trainAttributes, stdAtt, tfAttr], axis=1)
    # Describe the slope
    # Best now:
    # trainAttributes = np.concatenate([trainAttributes, stdAtt, quantilPerAtt, tfAttr], axis=1)

    trainAttributes = np.concatenate([trainAttributes], axis=1)
    # trainAttributes = np.concatenate([trainAttributes, stdAtt, quantilPerAtt, quantilValAtt, tfAttr], axis=1)

    # trainAttributes = np.concatenate([tfAttr], axis=1)

    return trainAttributes
def sample_neg_TF(net, test_ratio=0.1, TF_num=20, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = net
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))     # 取整
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    #TODO
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    recordDict={}
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, TF_num), random.randint(0, n-1)
        if i < j and net[i, j] == 0 and str(i)+"_"+str(j) not in recordDict:
            neg[0].append(i)
            neg[1].append(j)
            recordDict[str(i)+"_"+str(j)]=''
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg
def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size = emd_size, window=10, min_count=0, sg=1,
            workers=8, epochs = 1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings
def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = net
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    recordDict={}
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0 and str(i)+"_"+str(j) not in recordDict:
            neg[0].append(i)
            neg[1].append(j)
            recordDict[str(i)+"_"+str(j)]=''
        else:
            continue
    train_neg = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg
def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc
def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)
def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels
# original version
def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res
def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph, create_using=nx.DiGraph())
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    return g, labels.tolist(), features
def extractLinks2subgraphs(Atrain, Atest, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, train_node_information=None, test_node_information=None):
    # automatically select h from {1, 2}
    if h == 'auto': # TODO
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(Atrain, 0.1)
        val_A = Atrain.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label, node_information):
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            if i in links[0] and j in links[1]:
                if g_label == 1:
                    if j in links[0] and i in links[1]:
                        g_label = 2
                    if j not in links[0] and i not in links[1]:
                        g_label = -1
            # nx.draw(g, with_labels=True, pos=nx.circular_layout(g))
            # plt.show()
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features))
        return g_list
    print('Extract enclosed subgraph...')
    train_graphs = helper(Atrain, train_pos, 1, train_node_information) + helper(Atrain, train_neg, 0, train_node_information)
    test_graphs = helper(Atest, test_pos, 1, test_node_information) + helper(Atest, test_neg, 0, test_node_information)
    print(max_n_label)
    return train_graphs, test_graphs, max_n_label['value']