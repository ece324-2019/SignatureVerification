from tkinter import Tk
from tkinter.filedialog import askopenfilename
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


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


def get_filenames():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    print("Initializing Dialogue... \nPlease select a file.\n")
    tk_filenames = askopenfilename(title='Please select one or more files', multiple=True)
    filenames = list(tk_filenames)
    return filenames


def conversion_bot():
    while True:
        proceed = False
        while not proceed:
            toggle = input("Select one or multiple RGB images to be converted. Continue? (Y/N):")
            if toggle in ['y', 'Y']:
                filenames = get_filenames()
                proceed = True

        for image_path in filenames:
            bi_image = convert2bi(image_path)
            plt.imsave(image_path, bi_image, cmap=plt.get_cmap('gray'))


conversion_bot()