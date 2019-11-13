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


trainning_dir = "signature/train"
trainning_csv = "signature/train_data.csv"

valid_dir = "signature/valid"
valid_csv = "signature/valid_data.csv"

test_dir = "signature/test"
test_csv = "signature/test_data.csv"






