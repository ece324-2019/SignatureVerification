# ===================================== IMPORT ============================================#
import argparse
from time import time
import math
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.signal import savgol_filter

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from data_prep_lib import *

'''
    path of images (name-forg)  1-0 1-1  2-0 2-2  3-0 3-3 ......
    randomly choose 6 names for validation and 9 names for test
    for loop generate triplet_train.txt triplet_valid.txt triplet_test.txt
    
    each triplet: load three images -> convert to binary -> do calculations -> cut images -> bucketIterator -> load 
'''
seed = 1
num_of_names = 55
base_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'
iter_dir = []
num = []
dataset_name = []
random.seed(seed)

for i in range(1, num_of_names + 1):
    iter_dir.append(["/Users/yizezhao/PycharmProjects/ece324/sigver/signatures/name_" + str(i) + "/0",
                     "/Users/yizezhao/PycharmProjects/ece324/sigver/signatures/name_" + str(i) + "/1"])
    num.append(i)
    dataset_name.append(["name_" + str(i) + "_auth", "name_" + str(i) + "_forg"])
    print(iter_dir[i-1])

print(num)
random.shuffle(num)
print(num)
print(dataset_name)
'''
    after shuffle, name 1-40: train     41-50: validation       51-55: train
'''


print("lol")
print(iter_dir[2][1])

for j in num:
    for i in [0, 1]:
