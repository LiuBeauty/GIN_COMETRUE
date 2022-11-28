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
    edge_tag = 0

    #读入ppi网络(图的结构)
    total_gene_expression_path = os.path.join(raw_dir,name,str(name+"_ppi_Network.csv"))
#     print(total_gene_expression_path)
    ppi_network = pd.read_csv(total_gene_expression_path,header=0)
    print(ppi_network.head())

    #读取节点特征1：基因表达  读入矩阵每一行是一个病人。每一列是节点特征
    total_gene_expression_path = os.path.join(raw_dir,name,str(name+"_gene_expression.csv"))
    total_gene_expression_feature = pd.read_csv(total_gene_expression_path, header=0)
    total_label = total_gene_expression_feature['subtype']
    print(total_gene_expression_feature.head())
    #图的标签集

    #读取节点特征2:BODY的甲基化水平 加在下面
    total_bodyMeth_path = os.path.join(raw_dir,name,str(name+"_bodyMeth.csv"))
    total_bodyMeth_feature = pd.read_csv(total_bodyMeth_path, header=0)
    print(total_bodyMeth_feature.head())

    total_ProMeth_path = os.path.join(raw_dir,name,str(name+"_proMeth.csv"))
    total_ProMeth_feature =  pd.read_csv(total_ProMeth_path, header=0)
    print(total_ProMeth_feature.head())

    # total_cnv_path = os.path.join(raw_dir, name, str(name + "_cnv.csv"))
    # total_cnv_feature = pd.read_csv(total_cnv_path, header=0)
    #图数据集包含的图的数目,表达矩阵的行数是图的数目
    n_graph = total_gene_expression_feature.shape[0]
    #每张图含有的节点数相同,去掉标签列
    n_node = total_gene_expression_feature.shape[1] -1
    print('n_node:'+str(n_node))

    #写图相关  建立ensemble-ID 字典
    Gene_dict = {}
    i = 0
    for item in total_gene_expression_feature.columns:
        Gene_dict[item] = i
        i += 1
    #替换表达矩阵的行名
    total_gene_expression_feature = total_gene_expression_feature.rename(columns=Gene_dict)
    total_bodyMeth_feature = total_bodyMeth_feature.rename(columns=Gene_dict)
    total_proMeth_feature = total_ProMeth_feature.rename(columns=Gene_dict)
    # total_cnv_feature = total_cnv_feature.rename(columns=Gene_dict)


    #Node1 节点的ID转换1 ppi文件的基因ID转换
    Node1_gene_name =  ppi_network['#node1']
    Node1_ID_name = []
    for key in Node1_gene_name:
       Node1_ID_name.append(Gene_dict[key])

    #Node2 节点的ID转换2
    Node2_gene_name =  ppi_network['node2']
    Node2_ID_name = []
    for key in Node2_gene_name:
        Node2_ID_name.append(Gene_dict[key])

    #每张图的结构是一样的，每张图的边的数目相同
    n_edge = len(Node2_ID_name)

    #处理图关系

    #node_edges[i]= [0,1,2] 代表和节点i邻接的节点ID为0，1，2
    #node_attribute[i]= [a,b,c] 代表节点i的特征。有几个特征代表用到了几种组学数据
    node_edges = []
    for i in range(0,n_node):
        node_edges.append([])

    #确定图骨架 每一张图写入txt时只需要加入特异的特征信息 未来可以考虑加入边的权重信息
    for i in range(0,n_node):
        #在node1列表中找有无邻接边
        for j in range (0,n_edge):
            #如果在源节点集合中有节点i  代表存在一条边的源节点是i，此时节点i的一个邻接点是Node2_ID_name[j]
            if Node1_ID_name[j] == i:
                node_edges[i].append(Node2_ID_name[j])

        #在node2列表中找有无邻接边
        for k in range(0,n_edge):
            if Node2_ID_name[k] == i:
                node_edges[i].append(Node1_ID_name[k])
    node_label = 0
#     print('here129'+str(n_graph))
    #txt文件第一行，图的数量
    _write_msg(raw_dir,name,str(n_graph))


    #先处理图的骨架信息
    for i in range(0,n_node):
        node_edges[i] = [str(w) for w in node_edges[i]]

    for i in range(0,n_graph):
        #对于每一张图 第一行  图的节点数  图的类别
        edge_tag+=1
        _write_msg(raw_dir,name,str(n_node)+' '+ str(total_label[i]))
        #写DS_graph_labels.txt
        _write_msg2(raw_dir,name,name2,str(total_label[i]))
        node_indicator = i + 1
        #再写每一个节点的信息
        #把亚型一列放在最后，n_node就不会读到亚型列
        for j in range(0,n_node):
            #msg写主文件
            msg = str(node_label)+' '+ str(len(node_edges[j]))+' '+ " ".join(node_edges[j])+' ' +str(total_gene_expression_feature.iloc[i,j])+' '+str(total_proMeth_feature.iloc[i,j])+' '+str(total_bodyMeth_feature.iloc[i,j])
            # msg2写DS_node_attributes.txt
            msg2 = str(total_gene_expression_feature.iloc[i,j])+' '+str(total_proMeth_feature.iloc[i,j])+' '+str(total_bodyMeth_feature.iloc[i,j])
            # msg3写DS_graph_indicator.txt
            msg3 = str(node_indicator)
            _write_msg(raw_dir, name, msg)
            _write_msg2(raw_dir, name, name3, msg2)
            _write_msg2(raw_dir, name, name4, msg3)

            if edge_tag == 1:
                for adj_node in node_edges[j]:
                    if int(adj_node) > j:
                        mark_adj_node = int(adj_node)
                        msg4 = str(j ) + ', ' + str(mark_adj_node)
                        _write_msg2(raw_dir, name, name5, msg4)
                        msg4 = str(mark_adj_node) + ', ' + str(j)
                        _write_msg2(raw_dir, name, name5, msg4)


if __name__ == '__main__':
    work_dir = './dataset'
    mk_dgltxt(work_dir,'BRCA_NODE420')



