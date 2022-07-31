from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math
import csv
import os
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler  # sklearn归一化API
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
import pylab as pl
from sklearn.feature_selection import SelectKBest, VarianceThreshold, SelectPercentile
from sklearn.feature_selection import f_regression

#将DataFrame中的每一列分别做归一化处理的函数实现

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2
"""
这个文件用于把从R下载整理后的表达的矩阵的处理
包括筛选差异大的基因、以及表达矩阵的归一化
"""


#创建基因名称的csv
def _text_create(raw_dir,name):
    # 新创建的txt文件的存放路径
    full_path = os.path.join(raw_dir,name,str(name+'.txt'))# 也可以创建一个.doc的word文档
    file = open(full_path, 'w')
    file.close()
    return os.path.isfile(full_path)

#向txt里写行msg
def _write_msg(raw_dir,name,msg):
    file_path = os.path.join(raw_dir,name,str(name+'.txt'))
    if os.path.isfile(file_path):
        file = open(file_path, 'a')
        text = file.writelines(msg+'\n')
        file.close()
        return True
#特征筛选第二步 使用卡方分布筛选特征
def Chi2Func(data_final,flag,spec_score):
    #保留百分比的最高得分的特征
    selector = SelectPercentile (chi2, percentile=spec_score)
    selector.fit(data_final,flag)
    return selector
# 加载及调用
def attriFilter2(df1):
    print(df1.info())
    print('卡方检验前特征维度')
    print(df1.shape)
    print(df1.head(5))

    natt = df1.shape[1]-1
    x_before = df1.iloc[:,0:natt]
    x = np.array(x_before)
    y = np.array(df1['subtype'])
    print('查看初始数据信息 : \n',x_before.head())
    selector = Chi2Func(x, y, spec_score=2.62)# 保留2.62%的最高得分特征
    print('特征得分 : \n', selector.scores_)
    print('特征得分的p_value值 : \n', selector.pvalues_)
    print('筛选后保留特征 : \n', selector.get_support())
    df2 = pd.DataFrame(selector.transform(x), columns=x_before.columns[selector.get_support()], index = x_before.index)
    attri =  df2.columns

    _text_create('D:\\postgraduate\\ALL_code','GeneName')
    for item in attri:
        _write_msg('D:\\postgraduate\\ALL_code','GeneName',str(item))

    df2['subtype'] = y
    print(df2.head())
    return df2

def beiyong():
    raw = 'D:/postgraduate/ALL_code/BRCA_test/GNN_BRCA_GNNExpression.csv'
    df = pd.read_csv(raw, sep = ',',header=0, index_col=0)
    attri = df.columns
    _text_create('D:\\postgraduate\\ALL_code', 'GeneName')
    for item in attri:
        _write_msg('D:\\postgraduate\\ALL_code', 'GeneName', str(item))

def regularit(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame




# 对矩阵进行归一化
def normalization(df):
    """
    归一化处理
    :return: NOne
    """
    print('传入卡方检验前的数据')
    print(df.head(10))
    #去除提取出的 “cancer.type"等两列

    #先查看有无缺失数据
    # print(df.isnull().sum())

    subtype = df['subtype']

    attridf = df.iloc[:, 0:df.shape[1]-1]
    print('归一化前的特征矩阵')
    print(attridf.head(10))
    #特征归一化后的矩阵
    df2 = regularit(attridf)
    print('归一化后的特征矩阵')

    df2['subtype']= np.array(df['subtype'])
    print(df2.head())
    df2.to_csv("D:/postgraduate/ALL_code/GNN_BRCA_GNNExpression.csv")

    return df2

#根据信息增益选取特征（待解决问题：根据数据集划分，特征的重要性变化很剧烈
def attriFilter1(raw_path):
    df = pd.read_csv(raw_path, sep=',', header=0, index_col=0)
    print(df.info())

    # Feature Importance with Extra Trees Classifier

    natt = df.shape[1]-1
    X = df.iloc[:, 0:natt]

    Y = df['subtype']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    n = len(df.columns) -1
    feat_label = df.columns[0:n]
# feature extraction
    model = RandomForestClassifier()
    model.fit(X, Y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    #筛选出重要程度前300的特征
    imp_Gene = []
    for f in range(300):
        # imp_Gene.append(feat_label[indices[f]])
        # print(feat_label[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, feat_label[indices[f]], importances[indices[f]]))

#保守特征筛选方法:递归特征消除法
def attriFilter3(raw_path):
    df = pd.read_csv(raw_path, sep=',', header=0, index_col=0)
    print(df.info())

    rfe = RFE(estimator = LinearRegression(), n_features_to_select=300)

    natt = df.shape[1] - 1
    #特征列
    X = df.iloc[:, 0:natt]
    #标签列
    Y = df['subtype']
    sFeature = rfe.fit_transform(X, Y)

    print(rfe.get_support())


#特征筛选第一步 筛选过滤小方差特征
def attriFilter(raw_path):
    # 删除方差为0的特征
    df = pd.read_csv(raw_path, sep=',', header=0, index_col=0)
    # print(df.info())
    # print(df.head(10))

    natt = df.shape[1]
    print('处理前特征维度')
    #特征列
    X = df.iloc[:, 2:natt]
    # print(X.shape)
    #标签列

    #过滤掉特征值在中位数以下的方差
    thre = (np.median(X.var().values))
    vt = VarianceThreshold(threshold=thre)
    X_vt = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()], index = X.index)

    print('处理后维度')
    # print(X_vt.shape)
    subtype = df['Subtype_mRNA']
    sublist = []

    subtypedict = {'Normal': 1, 'LumA': 2, 'LumB': 3, 'Basal': 4, 'Her2': 0}
    for item in subtype:
        sublist.append(subtypedict[item])
    X_vt['subtype'] = sublist
    return X_vt





if __name__ == '__main__':
    # work_dir原表达矩阵
    work_dir = 'D:/postgraduate/ALL_code/BRCA_test/BRCA_GNNExpressionwithSubtype.csv'
    #通过方差 初步筛选特征
    # df1 = attriFilter(work_dir)
    #通过卡方检验筛选特征
    # df2 = attriFilter2(df1)
    #特征归一化
    # df3 = normalization(df2)

    #提取特征基因
    beiyong()




