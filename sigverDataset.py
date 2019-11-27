import os
import torch
from PIL import Image

import argparse
from time import time
import math
import numpy as np
import pandas as pd
import random

class SiameseNetworkDataset():

    def __init__(self, csv=None, dir=None, transform=None):
        # used to prepare the labels and images path
        self.training_df = pd.read_csv(csv)
        self.training_df.columns = ["image1", "image2", "label"]
        self.training_dir = dir
        self.transform = transform

    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.training_dir, self.training_df.iat[index, 0])
        image2_path = os.path.join(self.training_dir, self.training_df.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(self.training_df.iat[index, 2])], dtype=np.float32))

    def __len__(self):
        return len(self.training_df)


class TripletDataset():


    def __init__(self, csv=None, dir=None, transform=None):
        # used to prepare the labels and images path
        self.training_df = pd.read_csv(csv)
        self.training_df.columns = ["anchor", "pos", "neg"]
        self.training_dir = dir
        self.transform = transform

    def __getitem__(self, index):
        # getting the image path
        anchor_path = os.path.join(self.training_dir, self.training_df.iat[index, 0])
        pos_path = os.path.join(self.training_dir, self.training_df.iat[index, 1])
        neg_path = os.path.join(self.training_dir, self.training_df.iat[index, 2])

        # Loading the image
        anchor = Image.open(anchor_path)
        pos = Image.open(pos_path)
        neg = Image.open(neg_path)

        anchor = anchor.convert("L")
        pos = pos.convert("L")
        neg = neg.convert("L")

        # Apply image transformations
        if self.transform is not None:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return anchor, pos, neg

    def __len__(self):
        return len(self.training_df)


class Triplet_Eval_Dataset():

    def __init__(self, csv=None, dir=None, transform=None):
        # used to prepare the labels and images path
        self.training_df = pd.read_csv(csv)
        self.training_df.columns = ["anchor", "pos", "question", "label"]
        self.training_dir = dir
        self.transform = transform

    def __getitem__(self, index):
        # getting the image path
        anchor_path = os.path.join(self.training_dir, self.training_df.iat[index, 0])
        pos_path = os.path.join(self.training_dir, self.training_df.iat[index, 1])
        question_path = os.path.join(self.training_dir, self.training_df.iat[index, 2])


        # Loading the image
        anchor = Image.open(anchor_path)
        pos = Image.open(pos_path)
        question = Image.open(question_path)

        anchor = anchor.convert("L")
        pos = pos.convert("L")
        question = question.convert("L")

        # Apply image transformations
        if self.transform is not None:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            question = self.transform(question)

        return anchor, pos, question, torch.from_numpy(np.array([int(self.training_df.iat[index, 3])], dtype=np.float32))

    def __len__(self):
        return len(self.training_df)