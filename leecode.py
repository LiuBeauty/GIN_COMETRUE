from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
from functools import reduce
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import os

# np.set_printoptions(threshold=np.inf)
#
cancer_type = 'Our_KIPAN'
num_class = 2
raw_dir='D:/postgraduate/DNA甲基化挖掘项目/论文复现/GIN_COMETRUE/dataset'
name = 'LUNG_NODE300'

data_train = list()
data_test = list()
label_train = list()
label_test = list()
#
# total_expression_path = os.path.join(raw_dir, name, str(name + "_gene_expression.csv"))
# total_expression_feature = pd.read_csv(total_expression_path, header=0)
#
# total_bodyMeth_path = os.path.join(raw_dir, name, str(name + "_bodyMeth.csv"))
# total_bodyMeth_feature = pd.read_csv(total_bodyMeth_path, header=0)
#
# total_proMeth_path = os.path.join(raw_dir, name, str(name + "_proMeth.csv"))
# total_proMeth_feature = pd.read_csv(total_proMeth_path, header=0)
#
# ncol = total_proMeth_feature.shape[1] -1
#
# expression_feature = (total_expression_feature.iloc[:,0:ncol]).to_numpy()
# label = (total_expression_feature.iloc[:,-1]).to_numpy()
#
# label2 = (total_proMeth_feature.iloc[:,-1]).to_numpy()
# proMeth_feature = (total_proMeth_feature.iloc[:,0:ncol]).to_numpy()
#
# label3 = (total_bodyMeth_feature.iloc[:,-1]).to_numpy()
# bodyMeth_feature = (total_bodyMeth_feature.iloc[:,0:ncol]).to_numpy()
#
#
#
# indice = np.arange(0,total_expression_feature.shape[0]-1 )
# data_train_index, data_test_index, label_train_index, label_test_index = train_test_split(indice, indice, random_state=3, test_size=0.21)
# data_train.append(expression_feature[data_train_index])
# label_train = label[data_train_index]
# data_test.append(expression_feature[data_test_index])
# label_test = label[data_test_index]
#
# data_train.append(proMeth_feature[data_train_index])
# data_test.append(proMeth_feature[data_test_index])
#
# data_train.append(bodyMeth_feature[data_train_index])
# data_test.append(bodyMeth_feature[data_test_index])

#-----------MOGONET dataset
train_expression_path = 'D:/MOGONET-main/MOGONET-main/BRCA/1_tr.csv'
train_expression_feature = pd.read_csv(train_expression_path, header=None)
data_train.append(train_expression_feature.values)
train_expression_label_path = 'D:/MOGONET-main/MOGONET-main/BRCA/labels_tr.csv'
train_expression_label = ((pd.read_csv(train_expression_label_path,header=None)).iloc[:,0]).values.astype(np.uint8)
label_train = train_expression_label

test_expression_path = 'D:/MOGONET-main/MOGONET-main/BRCA/1_te.csv'
test_expression_feature = pd.read_csv(test_expression_path, header=None)
data_test.append(test_expression_feature.values)
test_expression_label_path = 'D:/MOGONET-main/MOGONET-main/BRCA/labels_te.csv'
test_expression_label = ((pd.read_csv(test_expression_label_path,header=None))).iloc[:,0].values.astype(np.uint8)
label_test = test_expression_label

train_meth_path = 'D:/MOGONET-main/MOGONET-main/BRCA/2_tr.csv'
train_meth_feature = pd.read_csv(train_meth_path, header=None)
data_train.append(train_meth_feature.values)

test_meth_path = 'D:/MOGONET-main/MOGONET-main/BRCA/2_te.csv'
test_meth_feature = pd.read_csv(test_meth_path, header=None)
data_test.append(test_meth_feature.values)

train_miRNA_path = 'D:/MOGONET-main/MOGONET-main/BRCA/3_tr.csv'
train_miRNA_feature = pd.read_csv(train_miRNA_path, header=None)
data_train.append(train_miRNA_feature.values)

test_miRNA_path = 'D:/MOGONET-main/MOGONET-main/BRCA/3_te.csv'
test_miRNA_feature = pd.read_csv(test_miRNA_path, header=None)
data_test.append(test_miRNA_feature.values)
#-----------MOGONET dataset

def SelectModel(modelname):
    if modelname == "SVM":
        from sklearn.svm import SVC
        model = SVC(kernel='rbf',probability=True)

    elif modelname == "GBDT":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()

    elif modelname == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()

    elif modelname == "XGBOOST":
        import xgboost as xgb
        model = xgb()

    elif modelname == "KNN":
        from sklearn.neighbors import KNeighborsClassifier as knn
        model = knn()
    else:
        pass
    return model


def get_oof(clf, n_folds, X_train, y_train, X_test):
    #ntrain 返回行数，即训练集中的样本数
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]

    classnum = len(np.unique(y_train))

    kf = KFold(n_splits=n_folds, shuffle=False)
    oof_train = np.zeros((ntrain, classnum))
    oof_test = np.zeros((ntest, classnum))

    pd.set_option('display.max_rows', None)  # 显示全部行
    pd.set_option('display.max_columns', None)  # 显示全部列
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):

        kf_X_train = X_train[train_index]  # 数据
        kf_y_train = y_train[train_index]  # 标签

        kf_X_test = X_train[test_index]  # k-fold的验证集

        clf.fit(kf_X_train, kf_y_train)

        oof_train[test_index] = clf.predict_proba(kf_X_test)
        oof_test += clf.predict_proba(X_test)


    oof_test = oof_test / float(n_folds)
    return oof_train, oof_test



# 使用stacking方法的时候
# 第一级，重构特征当做第二级的训练集
modelist = ['SVM', 'GBDT', 'RF', 'KNN']

modelname = 'KNN'
newfeature_list = []
newtestdata_list = []
newfeature = []
newtestdata=[]
print('-------First layer calculating--------')

for dataset_index in range(0,3):
    clf_first = SelectModel(modelname)
    oof_train_, oof_test_ = get_oof(clf=clf_first, n_folds=10, X_train=data_train[dataset_index], y_train=label_train,
                                    X_test=data_test[dataset_index])
    newfeature_list.append(oof_train_)
    newtestdata_list.append(oof_test_)

newfeature = reduce(lambda x, y: np.concatenate((x, y), axis=1), newfeature_list)
newtestdata = reduce(lambda x, y: np.concatenate((x, y), axis=1), newtestdata_list)

# 特征组合
# 第二级，使用上一级输出的当做训练集
# clf_second1 = RandomForestClassifier()
print('---------Third layer caculating--------')
clf_second1 = SelectModel(modelname)
clf_second1.fit(newfeature, label_train)
pred = clf_second1.predict(newtestdata)

if num_class == 2:
    F1_score = f1_score(label_test, pred)
    accuracy = accuracy_score(label_test, pred)
    score = roc_auc_score(label_test, pred)
    print(f'{cancer_type} : {modelname} accuracy:  {accuracy}\n  F1_macro: {F1_score} \n roc_auc_score: {score}')

else:
    F1_score = f1_score(label_test,pred,average='macro')
    F1_weight = f1_score(label_test,pred,average='weighted')
    accuracy = accuracy_score(label_test,pred)
    print(f'{cancer_type} : {modelname} accuracy:  {accuracy}\n  F1_macro: {F1_score} \n F1_weight: {F1_weight} ')
