from tkinter import Tk
from tkinter.filedialog import askopenfilename
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


def gray2bi(img, threshold):
    partial_bi = []
    for h in range(img.shape[0]):
        new_h = []
        for w in range(img.shape[1]):
            pix_val = img[h][w][0] / float(img[h][w][1])
            if pix_val < threshold:
                new_h.append(pix_val)
            else:
                new_h.append(1.0)
        partial_bi.append(new_h)

    return np.array(partial_bi)


def convert2bi(input_path):
    gray = Image.open(input_path).convert('LA')
    gray_np = np.array(gray)
    partial_bi = gray2bi(gray_np, threshold=0.88)
    partial_bi = 1 - partial_bi
    return partial_bi


def get_filename():
    Tk().withdraw()
    print("Initializing Dialogue... \nPlease select a file.\n")
    filename = askopenfilename(title='Please select one file')
    return filename


def data_pipeline(anchor, pos, question):
    anchor_bi = convert2bi(anchor)
    pos_bi = convert2bi(pos)
    question_bi = convert2bi(question)

    anchor_path = 'C:/Users/Terry Mei/Desktop/demo/anchor_bi.png'
    pos_path = 'C:/Users/Terry Mei/Desktop/demo/pos_bi.png'
    question_path = 'C:/Users/Terry Mei/Desktop/demo/question_bi.png'

    plt.imsave(anchor_path, anchor_bi, cmap=plt.get_cmap('gray'))
    plt.imsave(pos_path, pos_bi, cmap=plt.get_cmap('gray'))
    plt.imsave(question_path, question_bi, cmap=plt.get_cmap('gray'))

    # anchor_bi = Image.fromarray(np.uint8(anchor_bi))
    # pos_bi = Image.fromarray(np.uint8(pos_bi))
    # question_bi = Image.fromarray(np.uint8(question_bi))

    sig_transformations = transforms.Compose([
        transforms.Resize((200, 300)),
        transforms.ToTensor()
    ])

    anchor_bi = Image.open(anchor_path).convert("L")
    pos_bi = Image.open(pos_path).convert("L")
    question_bi = Image.open(question_path).convert("L")

    anchor_bi = sig_transformations(anchor_bi)
    pos_bi = sig_transformations(pos_bi)
    question_bi = sig_transformations(question_bi)

    anchor_bi = torch.reshape(anchor_bi, (1, anchor_bi.size()[0], anchor_bi.size()[1], anchor_bi.size()[2]))
    pos_bi = torch.reshape(pos_bi, (1, pos_bi.size()[0], pos_bi.size()[1], pos_bi.size()[2]))
    question_bi = torch.reshape(question_bi, (1, question_bi.size()[0], question_bi.size()[1], question_bi.size()[2]))

    return anchor_bi, pos_bi, question_bi


def bot(model, eval_margin):
    while True:
        proceed = False
        while not proceed:
            toggle = input("Select an anchor image. Continue? (Y/N):")
            if toggle in ['y', 'Y']:
                anchor = get_filename()
                proceed = True

        proceed = False
        while not proceed:
            toggle = input("Select a positive image. Continue? (Y/N):")
            if toggle in ['y', 'Y']:
                pos = get_filename()
                proceed = True

        proceed = False
        while not proceed:
            toggle = input("Select the questioned signature image. Continue? (Y/N):")
            if toggle in ['y', 'Y']:
                question = get_filename()
                proceed = True

        anchor_bi, pos_bi, question_bi = data_pipeline(anchor, pos, question)
        output1, output2, output3 = model(anchor_bi, pos_bi, question_bi)
        dist = torch.nn.PairwiseDistance(p=2)
        dist_pos = dist(output1, output2)
        dist_q = dist(output1, output3)
        if (dist_q - dist_pos > eval_margin):
            print("SigVerNet prediction: Signature in question is a forgery")
            print("Forgery index (the larger the more certain it is forgery): ", dist_q.item() - dist_pos.item())
            print(" ")
        else:
            print("SigVerNet prediction: Signature in question is authentic")
            print("Authenticity index (the smaller the more certain it is authentic): ", dist_q.item() - dist_pos.item())
            print(" ")


def main():
    # load models
    model = torch.load('D:/1_Study\EngSci_Year3\ECE324_SigVer_project/triplet_sigVerNet_ep1_step401.pt',
                       map_location=torch.device('cpu'))
    bot(model, eval_margin=0.8)


if __name__ == "__main__":
    main()
