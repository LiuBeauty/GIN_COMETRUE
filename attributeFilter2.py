import pandas as pd
import math
import csv
import os
import numpy as np
"""
这个脚本用于合并原始表达矩阵与通过SRING 数据库筛选后的基因
6/7进度  脚本完成 问题在于从ppi网络中得到的部分基因 超出了基因表达矩阵的基因 待解决
"""
def mergeMatrix(exp_path,ppi_path):

    exp_mat = pd.read_csv(exp_path,sep = ',',header = 0,index_col=0 )
    print(exp_mat.head())
    ppi = pd.read_csv(ppi_path, sep='\t', header=0)

    ppi_gene_list = list(set(ppi['#node1']).union(set(ppi['node2'])))
    print(ppi_gene_list)

    final_exp = exp_mat[['subtype']]
    all_gene = exp_mat.columns
    common_gene = list(set(ppi_gene_list) & set(all_gene))
    diff =  list(set(ppi_gene_list).difference(common_gene))
    print('diff_gene length',diff)
    for item in all_gene:
        if item in ppi_gene_list:
            final_exp[item] = exp_mat[item]


    final_exp['subtype'] = np.array(exp_mat['subtype'])

    print(final_exp.shape)

if __name__ == '__main__':
    # work_dir原表达矩阵
    Exp = 'D:/postgraduate/ALL_code/BRCA_test/GNN_BRCA_GNNExpression.csv'
    ppi_Gene = 'D:/postgraduate/ALL_code/BRCA_test/ppi/ppi_Network.tsv'
    mergeMatrix(Exp,ppi_Gene)