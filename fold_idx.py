import random
import os


#PartII生成自己的十折交叉验证数据
def _write_msg(raw_dir,name,msg):
    file_path = os.path.join(raw_dir, name)
    with open(file_path, 'a') as file:
        file.writelines(msg+'\n')
        file.close()
        return True

list1 = list(range(0,342))
test_sample = 69
for i in range(1,11):
    test_idx = random.sample(list1, test_sample)
    train_idx = list(set(list1) - set(test_idx))
    random.shuffle(train_idx)

    for each in train_idx:
        _write_msg('dataset/STAD/10fold_idx', 'train_idx-%s.txt' % str(i), str(each))
    for each in test_idx:
        _write_msg('dataset/STAD/10fold_idx', 'test_idx-%s.txt' % str(i), str(each))




