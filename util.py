import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.nxg = g
        self.node_tags = node_tags
        self.neighbors = []  # 节点标签

        self.edge_mat = 0
        self.node_features = node_features
        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    print('dataset/%s/%s.txt' % (dataset, dataset))
    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())  # n_g 图的数目

        for i in range(n_g):  # 对于每张图
            node_features = []
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]  # n是图上的节点数目，l是图的标签
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped

            g = nx.Graph()
            node_tags = []
            n_edges = 0
            for j in range(n):  # 对于图上的每个节点
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2  # 每一行的第二个数字是该节点邻接节点的数目
                if tmp == len(row):  # 如果节点没有属性
                    # no node attributes
                    print('Error: Node has no attribute!!')  # 对于正常的ppi网络，不会没有节点特征
                    row = [int(w) for w in row]
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    node_features.append(attr)

                if not row[0] in feat_dict:  # row[0]是节点的标签，未来可以考虑把已知癌症基因的节点标签设置为1
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped

                node_tags.append(feat_dict[row[0]])
                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                # node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            # print(node_features)
            # print('96:',node_features[i])
            # graph_feature = torch.Tensor(node_features[i])

            g_list.append(S2VGraph(g, l, node_tags, node_features))

    # 添加边和节点
    for g in g_list:  # 对于Datset中的每一张图
        g.neighbors = [[] for i in range(len(g.nxg))]
        for i, j in g.nxg.edges():  # 处理无向图方法 只需要一个边就行
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)

        degree_list = []
        for i in range(len(g.nxg)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))

        g.max_neighbor = max(degree_list)
        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.nxg.edges()]
        edges.extend([[i, j] for j, i in edges])

        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)
    torch.set_default_tensor_type(torch.DoubleTensor)
    # i = 0
    for g in g_list:
        i += 1
        g.node_features = torch.from_numpy(np.array(g.node_features))

    #         g.node_features = torch.zeros(len(g.node_tags), len(tagset))
    #         g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
    #         print("After"+str(g.node_features)+'shape:'+str(g.node_features.shape))
    #         a = [0.2,0.3]
    #         g.node_features = torch.tensor([0.2,0.3])
    #         print('here: 129'+str(g.node_features)+'shape:'+str(g.node_features.shape))

    print('# classes: %d' % len(label_dict))
    # print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))
    return g_list, len(label_dict)


def separate_data(graph_list, seed, fold_idx):

    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list
