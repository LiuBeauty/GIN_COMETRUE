import numpy as np
import pandas as pd
import os

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

    else:
        return False


if __name__ == '__main__':

    data = [0,1,2,3,4]
    print(data[10:])