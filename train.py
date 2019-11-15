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
from sigverDataset import *
from helpers import *
from loss import *
from models import *

'''
    path of images (name-forg)  1-0 1-1  2-0 2-2  3-0 3-3 ......
    randomly choose 6 names for validation and 9 names for test
    for loop generate triplet_train.txt triplet_valid.txt triplet_test.txt
    
    each triplet: load three images -> convert to binary -> do calculations -> cut images -> bucketIterator -> load 
'''


def baseline_train(args, sigVerNet, dataloader):
    counter = []
    loss_history = []
    iteration_number = 0

    optimizer = optim.RMSprop(sigVerNet.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
    criterion = ContrastLoss()

    for epoch in range(0, args.epochs):
        for i, data in enumerate(dataloader, 0):
            img0, img1, label = data
            #img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            img0, img1, label = img0, img1, label
            optimizer.zero_grad()
            output1, output2 = sigVerNet(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            #if i % 50 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
    return sigVerNet



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 4)
    parser.add_argument('--valid_size', type=int, default=4)
    parser.add_argument('--split_coefficient', type=int, default=0.2)
    parser.add_argument('--lr', type=float, default = 0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--loss_type', choices=['mse', 'ce'], default='ce')
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--if_batch', type=bool, default=False)
    parser.add_argument('--num_kernel', type=int, default=30)
    parser.add_argument('--model_type', choices=['small', 'test', 'best', 'best_small'], default='test')
    args = parser.parse_args()

    num_of_names = 55
    data_base_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'



    train_dir = "/Users/yizezhao/PycharmProjects/ece324/sigver"
    train_csv = "train_paried_list.csv"

    valid_dir = "/Users/yizezhao/PycharmProjects/ece324/sigver"
    valid_csv = "valid_paried_list.csv"

    test_dir = "/Users/yizezhao/PycharmProjects/ece324/sigver"
    test_csv = "test_paried_list.csv"

    #define transformer
    sig_transformations = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()

    ])

    # Load the the dataset from raw image folders
    siamese_dataset = SiameseNetworkDataset(csv=train_csv, dir=train_dir,
                                            transform=sig_transformations)
    train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            batch_size=args.batch_size)

    valid_dataset = SiameseNetworkDataset(csv=valid_csv, dir=valid_dir,
                                            transform=sig_transformations)
    valid_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            batch_size=args.batch_size)
    test_dataset = SiameseNetworkDataset(csv=test_csv, dir=test_dir,
                                            transform=sig_transformations)
    test_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            batch_size=args.batch_size)

    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())


    sigVerNet = SiameseNetwork()
    net_after = baseline_train(args, sigVerNet, train_dataloader)



if __name__ == "__main__":
    main()
