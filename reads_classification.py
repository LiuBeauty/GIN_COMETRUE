import pandas as pd
import math
import csv
import os
import numpy as np
#作用是把ppi网络加组学数据变成txt文件

#创建目标txt
def _text_create(raw_dir,name):
    # 新创建的txt文件的存放路径
    full_path = os.path.join(raw_dir,name,str(name+'.txt'))# 也可以创建一个.doc的word文档
    file = open(full_path, 'w')
    file.close()
    return os.path.isfile(full_path)

#向txt里写行msg
def _write_msg(raw_dir,name,msg):
    file_path = os.path.join(raw_dir,name,str(name+'.txt'))
    if not os.path.exists(file_path):
        fd = open(file_path, mode="w", encoding="utf-8")
    if os.path.isfile(file_path):
        file = open(file_path, 'a')
        text = file.writelines(msg+'\n')
        file.close()
        return True
    else:
        return False

def _write_msg2(raw_dir,name,name2,msg):
    file_path = os.path.join(raw_dir,name,name2)
    if not os.path.exists(file_path):
        fd = open(file_path, mode="w", encoding="utf-8")
    if os.path.isfile(file_path):
        file = open(file_path, 'a')
        text = file.writelines(msg+'\n')
        file.close()
        return True
    else:
        return False



def mk_dgltxt(raw_dir,name):
    #raw_dir 数据存放总文件夹 Ppi网络
    #如表达矩阵：./Dataset/name/name_expression.csv
    #网络骨架：./Dataset/name/name_ppi.tsv
    #raw_dir = ./Dataset
    #name2 = ./Dataset/name/name_graph_labels.txt
    name2 = name+'_graph_labels.txt'
    name3 = name+'_node_attributes.txt'
    name4 = name+'_graph_indicator.txt'
    name5 = name+'_A.txt'

    id_label = {}
    label_class ={}
    i=0
    total_label_path = "D:\\postgraduate\\DNA甲基化挖掘项目\\论文复现\\GIN_COMETRUE\\reads_dataset\\train\\test_label.xls"

    with open(total_label_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while True:
            i+=1
            content = f.readline()
            if len(content) == 0:
                break
            id,label = content.split('	')
            label = label.split('\n')[0]
            # 当读取到文件末尾的时候，跳出循环
            if not label in label_class:
                mapped = len(label_class)
                label_class[label] = mapped
            id_label[id] = label_class[label]

    n_graph = len(id_label)
    # graphs[i] = [[0,1,2],[2,3,4]]
    #前面是源节点集合，后面是目标节点集合
    graphs = []
    # 表示第i个图的label是2
    # graph_labels[i]=2
    graph_labels = []
    graph_id = []
    each_graph = []
    for i in range(n_graph):
        graphs.append(each_graph)

    #读入ppi网络(图的结构)
    total_Network_path = "D:\\postgraduate\\DNA甲基化挖掘项目\\论文复现\\GIN_COMETRUE\\reads_dataset\\train\\test_Network.xls"
    i = 0
    with open(total_Network_path,'r',encoding='utf-8') as f:
        while True:
            content = f.readline()
            if len(content) == 0:
                break
        # 当读取到文件末尾的时候，跳出循环
            id,s_node,t_node,weight = content.split('\t')
            s_node = int(s_node)
            t_node = int(t_node)
            if id not in graph_id:
                #如果不是第一张图
                #代表从这一行开始是新的一张图
                if len(graph_id) != 0:
                    graphs[len(graph_labels) - 1].append(s_node_set)
                    graphs[len(graph_labels) - 1].append(t_node_set)
                    graph_id.append(id)
                    graph_labels.append(id_label[id])
                    s_node_set = []
                    t_node_set = []
                #如果是第一张图
                else:
                    graph_id.append(id)
                    graph_labels.append(id_label[id])
                    s_node_set = []
                    t_node_set = []

            s_node_set.append(s_node)
            t_node_set.append(t_node)

    _write_msg(raw_dir,name,str(n_graph))


    #node_edges[i]= [0,1,2] 代表和节点i邻接的节点ID为0，1，2
    for l in range(n_graph):
        node_edges = []
        edges = graphs[l]
        s_node_set = edges[0]
        t_node_set = edges[1]
        node_label = 0
        n_node = len(list(set(s_node_set)|set(t_node_set)))
        n_edge = len(s_node_set)
        #写第一行 图的节点数和图的类别
        _write_msg(raw_dir,name,str(n_node)+' '+ str(graph_labels[l]))
        node_feature = {}
        # 一张图存在一种映射关系
        for each_node in list(set(s_node_set)|set(t_node_set)):
            if not each_node in node_feature:
                mapped = len(node_feature)
                node_feature[mapped] = each_node

        for i in range(len(s_node_set)):
            s_node_set[i] = list(node_feature.keys())[list(node_feature.values()).index(s_node_set[i])]
        for j in range(len(t_node_set)):
            t_node_set[j] = list(node_feature.keys())[list(node_feature.values()).index(s_node_set[j])]

        for i in range(0,n_node):
            node_edges.append([])
        #确定图骨架
        for i in range(0, n_node):
            #在node1列表中找有无邻接边
            for j in range (0,n_edge):
                #如果在源节点集合中有节点i  代表存在一条边的源节点是i，此时节点i的一个邻接点是Node2_ID_name[j]
                if s_node_set[j] == i:
                    node_edges[i].append(t_node_set[j])
            #在node2列表中找有无邻接边
            for k in range(0,n_edge):
                if t_node_set[k] == i:
                    node_edges[i].append(s_node_set[k])
            print(msg)
            msg = str(node_label) + " " + str(len(node_edges[i]))+" "+" ".join(node_edges[i])+" "+str(node_feature[i])
            _write_msg(raw_dir, name, msg)

if __name__ == '__main__':
    work_dir = 'D:/postgraduate/DNA甲基化挖掘项目/论文复现/GIN_COMETRUE/reads_dataset'
    mk_dgltxt(work_dir, 'train')


