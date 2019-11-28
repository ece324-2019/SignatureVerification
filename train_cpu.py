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


def plot_loss_acc(n, train_loss, valid_loss, m, train_acc, valid_acc, step):
    n_array = np.arange(n) + 1
    plt.subplot(2, 1, 1)
    line1 = plt.plot(n_array, valid_loss, label='validation loss')
    line2 = plt.plot(n_array, train_loss, label='training loss')
    plt.legend(loc='upper right')
    plt.xlabel('number of mini-batches')
    plt.ylabel('training loss')
    plt.title('loss of training dataset')

    m_array = np.arange(m) + 1
    plt.subplot(2, 1, 2)
    line3 = plt.plot(m_array, valid_acc, label='validation accuracy')
    line4 = plt.plot(m_array, train_acc, label='training accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('number of epochs')
    plt.ylabel('training accuracy')
    plt.title('prediction accuracy of training dataset')

    plt.savefig('/content/plots/triplet_sigVerNet_step{}.png'.format(step + 1))
    plt.close("all")
    # plt.show()


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
    criterion = torch.nn.TripletMarginLoss(margin=args.triplet_margin)
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            anchor, pos, question, label = data
            output1, output2, output3 = model(anchor, pos, question)
            loss_triplet = criterion(output1, output2, output3)

            dist = torch.nn.PairwiseDistance(p=2)
            dist_pos = dist(output1, output2)
            dist_neg = dist(output1, output3)
            print("distance: ", dist_pos, dist_neg)
            for j in range(output1.shape[0]):
                if (dist_neg[j] - dist_pos[j] > args.triplet_eval_margin and label[j] == 1):
                    print("pos, neg, prediction: ", dist_pos[j], dist_neg[j], "forgeries", "correct 1")
                    corr_num += 1
                    tot_num += 1
                elif (dist_neg[j] - dist_pos[j] <= args.triplet_eval_margin and label[j] == 0):
                    print("pos, neg, prediction: ", dist_pos[j], dist_neg[j], "authentic", "correct 0")
                    corr_num += 1
                    tot_num += 1
                else:
                    tot_num += 1
                    if (label[j] == 1):
                        print("pos, neg, prediction: ", dist_pos[j], dist_neg[j], "authentic", "incorrect 1")
                    else:
                        print("pos, neg, prediction: ", dist_pos[j], dist_neg[j], "forgeries", "incorrect 0")

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
    batch_train_acc_list = []
    iteration_number = 0

    criterion = torch.nn.TripletMarginLoss(margin=args.triplet_margin, p=2)
    # optimizer = optim.SGD(sigVerNet.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(sigVerNet.parameters(), lr=args.lr, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
    optimizer = torch.optim.Adam(sigVerNet.parameters(), lr=args.lr)

    train_loss_list = []
    train_acc_list = []

    train_corr_num = 0
    train_tot_num = 0
    valid_loss_list = []
    valid_acc_list = []

    train_acc = 0
    train_loss = 0

    for epoch in range(0, args.epochs):

        for i, data in enumerate(dataloader, 0):
            train_corr_num = 0
            train_tot_num = 0
            # concatenated = torch.cat((example_batch[0], example_batch[1], example_batch[2]), 0)
            # imshow(torchvision.utils.make_grid(concatenated))

            anchor, pos, neg = data

            # for i in range (data[0].shape[0]):
            #    concat = torch.cat((anchor[i], pos[i], neg[i]), 0)
            #    imshow(torchvision.utils.make_grid(concat))

            optimizer.zero_grad()
            output1, output2, output3 = sigVerNet(anchor, pos, neg)

            dist = torch.nn.PairwiseDistance(p=2)

            dist_pos = dist(output1, output2)
            dist_neg = dist(output1, output3)

            # print("pos_dist and neg_dist: ", dist_pos, dist_neg)

            for j in range(output1.shape[0]):
                if (dist_neg[j] - dist_pos[j] > args.triplet_eval_margin):
                    print("pos, neg, prediction: ", dist_pos[j], dist_neg[j], "forgeries")
                    train_corr_num += 1
                    train_tot_num += 1
                else:
                    print("pos, neg, prediction: ", dist_pos[j], dist_neg[j], "authentic")
                    train_tot_num += 1

            loss_triplet = criterion(output1, output2, output3)
            loss_triplet.backward()
            optimizer.step()

            train_acc = train_corr_num / train_tot_num
            train_loss = loss_triplet.item() / data[0].shape[0]

            if i % 50 == 0 and i != 0:
                eval_acc, eval_loss = eval_triplet_valid(args, sigVerNet, eval_dataloader)
                valid_acc_list += [eval_acc]
                valid_loss_list += [eval_loss]
                train_acc_list += [train_acc]
                train_loss_list += [train_loss]
                plot_loss_acc(len(valid_loss_list), train_loss_list, valid_loss_list, len(valid_acc_list),
                              train_acc_list, valid_acc_list, i)

                if (eval_acc >= 0.7):
                    torch.save(sigVerNet, '/content/models/triplet_sigVerNet_ep{}_step{}.pt'.format(epoch + 1, i + 1))
            print("Epoch number {} batch number {} running loss {} running acc {}".format(epoch + 1, i + 1,
                                                                                          loss_triplet.item(),
                                                                                          train_acc))

        # print("validation accuracy {}\n".format(eval_acc))

    return sigVerNet


def main():


    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--valid_size', type=int, default=4)
    parser.add_argument('--split_coefficient', type=int, default=0.2)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--loss_type', choices=['mse', 'ce'], default='ce')
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--if_batch', type=bool, default=False)
    parser.add_argument('--num_kernel', type=int, default=30)
    parser.add_argument('--model_type', choices=['small', 'test', 'best', 'best_small'], default='test')
    parser.add_argument('--baseline_margin', type=float, default=0.75)
    parser.add_argument('--triplet_margin', type=float, default=2)
    parser.add_argument('--triplet_eval_margin', type=float, default=0.8)
    parser.add_argument('--computer', type=str, default='yize')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
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
        data_base_dir = '/content/'

        baseline_train_csv = "20_overfit_list.csv"
        baseline_valied_csv = "20_overfit_list.csv"
        baseline_test_csv = "20_overfit_list.csv"

        # triplet_train_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_train_triplet_list.csv"
        # triplet_valid_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_valid_triplet_list.csv"
        # triplet_test_csv = "/Users/yizezhao/PycharmProjects/ece324/sigver/50k_test_triplet_list.csv"

        triplet_train_csv = "/content/50k_train_triplet_list.csv"
        triplet_valid_csv = "/content/20_valid_triplet_list.csv"
        triplet_test_csv = "/content/20_valid_triplet_list.csv"

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
        transforms.Resize((200, 300)),
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

    # sigVerNet = SiameseNetwork()
    # vggNet = VGG_SiameseNet()

    tripletNet = TripletNetwork()
    # vgg_tripletNet = VggTriplet()

    # net_after = baseline_train(args, sigVerNet, train_dataloader, eval_dataloader)
    # net_after = baseline_train(args, vggNet, train_dataloader, eval_dataloader)
    trp_after = triplet_train(args, tripletNet, triplet_train_dataloader, triplet_valid_dataloader)
    # trp_after = triplet_train(args, vgg_tripletNet, triplet_train_dataloader, triplet_valid_dataloader)


if __name__ == "__main__":
    main()
