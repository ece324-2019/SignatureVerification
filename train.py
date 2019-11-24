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
import torch.nn.functional as F

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


def plot_loss_acc(n, loss_train, m, acc_train):
    n_array = np.arange(n) + 1
    plt.subplot(2, 1, 1)
    line1 = plt.plot(n_array, loss_train, label='training loss')
    plt.legend(loc='upper right')
    plt.xlabel('number of mini-batches')
    plt.ylabel('training loss')
    plt.title('loss of training dataset')

    m_array = np.arange(m) + 1
    plt.subplot(2, 1, 2)
    line3 = plt.plot(m_array, acc_train, label='training accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('number of epochs')
    plt.ylabel('training accuracy')
    plt.title('prediction accuracy of training dataset')

    plt.show()


def eval_baseline(args, model, dataloader):
    tot_num = 0
    corr_num = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            img0, img1, label = data
            output1, output2 = model(img0, img1)
            euclidean_distance = F.pairwise_distance(output1, output2)
            predictions = []
            print("distance: ", euclidean_distance)
            for j in range(output1.shape[0]):
                if euclidean_distance[j] > args.baseline_margin:
                    predictions.append(1)
                else:
                    predictions.append(0)
            print("predictions: ", predictions)
            for j in range(len(predictions)):
                if predictions[j] == label[j]:
                    tot_num += 1
                    corr_num += 1
                else:
                    tot_num += 1
    return float(corr_num) / tot_num


def eval_triplet_valid(args, model, dataloader):
    tot_num = 0
    corr_num = 0
    loss_accum = 0
    criterion = torch.nn.TripletMarginLoss(margin=1.5)
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            anchor, pos, question, label = data
            output1, output2, output3 = model(anchor, pos, question)
            loss_triplet = criterion(output1, output2, output3)
            dist_pos = F.pairwise_distance(output1, output2)
            dist_neg = F.pairwise_distance(output1, output3)
            print("distance: ", dist_pos, dist_neg)
            for j in range(output1.shape[0]):
                if (dist_neg[j] - dist_pos[j] > args.triplet_margin and label[j] == 1) \
                        or (dist_neg[j] - dist_pos[j] <= args.triplet_margin and label[j] == 0):
                    corr_num += 1
                    tot_num += 1
                else:
                    tot_num += 1
        loss_accum += loss_triplet
    print('corr_num: {} | tot_num: {}'.format(corr_num, tot_num))
    return float(corr_num) / tot_num, loss_accum / tot_num


def baseline_train(args, sigVerNet, dataloader, eval_dataloader):
    counter = []
    loss_history = []
    iteration_number = 0

    # optimizer = optim.RMSprop(sigVerNet.parameters(), lr=args.lr, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
    optimizer = optim.SGD(sigVerNet.parameters(), lr=args.lr)
    criterion = ContrastLoss(margin=1.5)

    train_loss_list = []
    train_acc_list = []
    for epoch in range(0, args.epochs):
        for i, data in enumerate(dataloader, 0):
            img0, img1, label = data
            # img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            img0, img1, label = img0, img1, label
            optimizer.zero_grad()
            output1, output2 = sigVerNet(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            print("Epoch number {} batch number {}\n Current loss {}".format(epoch + 1, i + 1, loss_contrastive.item()))
            train_loss_list += [loss_contrastive.item()]
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

        train_acc = eval_baseline(args, sigVerNet, eval_dataloader)
        print(" training accuracy {}\n".format(train_acc))
        train_acc_list += [train_acc]
    plot_loss_acc(len(train_loss_list), train_loss_list, len(train_acc_list), train_acc_list)

    return sigVerNet


def triplet_train(args, sigVerNet, dataloader, eval_dataloader):
    counter = []
    loss_history = []
    iteration_number = 0

    criterion = torch.nn.TripletMarginLoss(margin=1.5)
    # optimizer = optim.SGD(sigVerNet.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(sigVerNet.parameters(), lr=args.lr, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
    optimizer = torch.optim.Adam(sigVerNet.parameters(), lr=args.lr)

    train_loss_list = []
    train_acc_list = []

    train_corr_num = 0
    train_tot_num = 0
    eval_loss_list = []
    eval_acc_list = []

    for epoch in range(0, args.epochs):
        for i, data in enumerate(dataloader, 0):
            anchor, pos, neg = data
            optimizer.zero_grad()
            output1, output2, output3 = sigVerNet(anchor, pos, neg)

            dist_pos = F.pairwise_distance(output1, output2)
            dist_neg = F.pairwise_distance(output1, output3)

            for j in range(output1.shape[0]):
                if (dist_neg[j] - dist_pos[j] > args.triplet_margin):
                    train_corr_num += 1
                    train_tot_num += 1
                else:
                    train_tot_num += 1


            loss_triplet = criterion(output1, output2, output3)
            loss_triplet.backward()
            optimizer.step()

            print("Epoch number {} batch number {}\n Current loss {}".format(epoch + 1, i + 1, loss_triplet.item()))
            train_loss_list += [loss_triplet.item()]
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_triplet.item())

        eval_acc, eval_loss = eval_triplet_valid(args, sigVerNet, eval_dataloader)
        print(" training accuracy {}\n".format(eval_acc))
        if epoch % 5 == 0 and epoch != 0:
            torch.save(sigVerNet, 'D:/1_Study/EngSci_Year3/ECE324_SigVer_project/triplet_sigVerNet_ep{}.pt'.format(epoch+1))
        eval_acc_list += [eval_acc]
        eval_loss_list += [eval_loss]
    plot_loss_acc(len(eval_loss_list), eval_loss_list, len(eval_acc_list), eval_acc_list)

    return sigVerNet


def main():
    if torch.cuda.is_available():
        print("Using cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--valid_size', type=int, default=4)
    parser.add_argument('--split_coefficient', type=int, default=0.2)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--loss_type', choices=['mse', 'ce'], default='ce')
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--if_batch', type=bool, default=False)
    parser.add_argument('--num_kernel', type=int, default=30)
    parser.add_argument('--model_type', choices=['small', 'test', 'best', 'best_small'], default='test')
    parser.add_argument('--baseline_margin', type=float, default=0.75)
    parser.add_argument('--triplet_margin', type=float, default=0.75)
    parser.add_argument('--computer', type=str, default='terry')

    args = parser.parse_args()

    num_of_names = 55

    # tri_train_csv = '20_train_triplet_list.csv'

    # tri_train_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'
    # tri_train_dir = '/content'

    # train_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'
    # train_dir = "D:/1_Study/EngSci_Year3/ECE324_SigVer_project"
    # train_dir = '/content'

    # tri_train_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'
    # tri_train_dir = "D:/1_Study/EngSci_Year3/ECE324_SigVer_project"

    # train_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'
    # train_dir = "D:/1_Study/EngSci_Year3/ECE324_SigVer_project"

    # train_csv = "train_paried_list.csv"
    # train_csv = "20_overfit_list.csv"
    # eval_csv = "20_overfit_list.csv"

    # valid_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'
    # valid_dir = "D:/1_Study/EngSci_Year3/ECE324_SigVer_project"
    # valid_csv = "valid_paried_list.csv"

    # test_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'
    # test_dir = "D:/1_Study/EngSci_Year3/ECE324_SigVer_project"
    # test_csv = "test_paried_list.csv"

    if args.computer == 'terry':
        # terry
        data_base_dir = 'D:/1_Study/EngSci_Year3/ECE324_SigVer_project/'

        baseline_train_csv = "20_overfit_list.csv"
        baseline_valied_csv = "20_overfit_list.csv"
        baseline_test_csv = "20_overfit_list.csv"

        # triplet_train_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_train_triplet_list.csv"
        # triplet_valid_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_valid_triplet_list.csv"
        # triplet_test_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_test_triplet_list.csv"

        triplet_train_csv = "20_train_triplet_list.csv"
        triplet_valid_csv = "20_valid_triplet_list.csv"
        # triplet_test_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_test_triplet_list.csv"

    elif args.computer == 'google':
        # google
        pass

    elif args.computer == 'yize':
        # yize
        data_base_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'

        baseline_train_csv = "20_overfit_list.csv"
        baseline_valied_csv = "20_overfit_list.csv"
        baseline_test_csv = "20_overfit_list.csv"

        # triplet_train_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_train_triplet_list.csv"
        # triplet_valid_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_valid_triplet_list.csv"
        # triplet_test_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_test_triplet_list.csv"

        triplet_train_csv = "20_train_triplet_list.csv"
        triplet_valid_csv = "20_valid_triplet_list.csv"
        triplet_test_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_test_triplet_list.csv"

    # define transformer
    sig_transformations = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()

    ])

    # data pipeline for siamese
    siamese_dataset = SiameseNetworkDataset(csv=baseline_train_csv, dir=data_base_dir,
                                            transform=sig_transformations)
    baseline_train_dataloader = DataLoader(siamese_dataset,
                                           shuffle=True,
                                           batch_size=args.batch_size)

    valid_dataset = SiameseNetworkDataset(csv=baseline_valied_csv, dir=data_base_dir,
                                          transform=sig_transformations)
    valid_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  batch_size=args.batch_size)

    test_dataset = SiameseNetworkDataset(csv=baseline_test_csv, dir=data_base_dir,
                                         transform=sig_transformations)
    test_dataloader = DataLoader(siamese_dataset,
                                 shuffle=True,
                                 batch_size=args.batch_size)

    # data pipeline for triplet
    triplet_dataset = TripletDataset(csv=triplet_train_csv, dir=data_base_dir,
                                     transform=sig_transformations)
    triplet_train_dataloader = DataLoader(triplet_dataset,
                                          shuffle=True,
                                          batch_size=args.batch_size)

    triplet_valid_dataset = Triplet_Eval_Dataset(csv=triplet_valid_csv, dir=data_base_dir,
                                                 transform=sig_transformations)
    triplet_valid_dataloader = DataLoader(triplet_valid_dataset,
                                          shuffle=True,
                                          batch_size=args.batch_size)

    # triplet_test_dataset = Triplet_Eval_Dataset(csv = triplet_test_csv, dir = data_base_dir,
    #                                              transform = sig_transformations)
    # triplet_test_dataloader = DataLoader(triplet_test_dataset,
    #                                       shuffle=True,
    #                                       batch_size=args.batch_size)

    vis_dataloader = DataLoader(triplet_dataset,
                                shuffle=True,
                                batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1], example_batch[2]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[3].numpy())

    sigVerNet = SiameseNetwork()
    vggNet = VGG_SiameseNet()

    tripletNet = TripletNetwork()
    vgg_tripletNet = VggTriplet()
    if torch.cuda.is_available():
        vgg_tripletNet = vgg_tripletNet.cuda()

    # net_after = baseline_train(args, sigVerNet, train_dataloader, eval_dataloader)
    # net_after = baseline_train(args, vggNet, train_dataloader, eval_dataloader)
    # trp_after = triplet_train(args, tripletNet, tri_train_dataloader, tri_eval_dataloader)
    trp_after = triplet_train(args, vgg_tripletNet, triplet_train_dataloader, triplet_valid_dataloader)


if __name__ == "__main__":
    main()
