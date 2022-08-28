import random

import pandas as pd
import math
import csv
import os

#PartI检测已有的十折交叉验证的数据集
# list = []
# num_list =[]
# fileHandler  =  open  ("dataset/Synthie/10fold_idx/train_idx-2.txt",  "r")
# listOfLines  =  fileHandler.readlines()
# for line in listOfLines:
#    list.append(line.strip())
# num_list = [int(each) for each in list]
# num_list.sort()

#PartII生成自己的十折交叉验证数据
def _write_msg(raw_dir,name,msg):
    file_path = os.path.join(raw_dir, name)
    with open(file_path, 'a') as file:
        file.writelines(msg+'\n')
        file.close()
        return True

list1 = list(range(0,872))
sample = 87
for i in range(1,11):
    test_idx = random.sample(list1, sample)
    train_idx = list(set(list1) - set(test_idx))
    random.shuffle(train_idx)

    for each in train_idx:
        _write_msg('dataset/BRCA/10fold_idx', 'train_idx-%s.txt' % str(i), str(each))
    for each in test_idx:
        _write_msg('dataset/BRCA/10fold_idx', 'test_idx-%s.txt' % str(i), str(each))




