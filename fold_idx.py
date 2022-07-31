import pandas as pd
import math
import csv
import os

list = []
num_list =[]
# Open file
fileHandler  =  open  ("dataset/Synthie/10fold_idx/train_idx-2.txt",  "r")
# Get list of all lines in file
listOfLines  =  fileHandler.readlines()
# Close file
# Iterate over the lines
for line in listOfLines:

    list.append(line.strip())

num_list = [int(each) for each in list]
num_list.sort()



