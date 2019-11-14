import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy


# first convert all rgb images to grayscale images
def rgb2gray(input_path):
    gray = Image.open(input_path).convert('LA')
    gray_np = np.array(gray)
    return gray_np


# gray = Image.open('original_1_1.png').convert('LA')
# plt.imshow(gray)
# plt.show()
# gray_np = np.array(gray)
# print(gray_np)
# print(gray_np.shape)


# then convert all grayscale images to partial-binary images
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


# partial_bi = gray2bi(gray_np, 0.88)
# partial_bi = 1 - partial_bi
# print'partial_bi shape: ', partial_bi.shape)


# print(partial_bi)
# plt.imshow(partial_bi, cmap=plt.get_cmap('gray'))
# plt.show()
# plt.imsave('original_1_1_bi.png', partial_bi, cmap=plt.get_cmap('gray'))


# then calculate the curvature at each point on the image
def cal_curv(img):
    dx_dt, dy_dt = np.gradient(img)
    d2x_dt2 = np.array(np.gradient(dx_dt))[0]
    d2y_dt2 = np.array(np.gradient(dy_dt))[1]
    # print('dx_dt: ', dx_dt.shape)
    # print('d2x_dt2: ', d2x_dt2.shape)
    # print('dy_dt: ', dx_dt.shape)
    # print('d2y_dt2: ', d2y_dt2.shape)

    curvature = []
    for h in range(img.shape[0]):
        new_h = []
        for w in range(img.shape[1]):
            # numerator = np.abs(d2x_dt2[h][w] * dy_dt[h][w] - dx_dt[h][w] * d2y_dt2[h][w])
            # denominator = (dx_dt[h][w] * dx_dt[h][w] + dy_dt[h][w] * dy_dt[h][w]) ** 1.5
            # if denominator != 0:
            #     new_h.append((numerator/denominator))
            # else:
            #     new_h.append(0.0)

            new_h.append(dy_dt[h][w] / dx_dt[h][w])

        curvature.append(new_h)
    return np.array(curvature)


# curvature = cal_curv(partial_bi)
# curvature = curvature
# print('curvature shape: ', curvature.shape)
# print(curvature)
# plt.imshow(curvature, cmap=plt.get_cmap('gray'))
# plt.show()
# np.savetxt('curvature.csv', curvature, delimiter=',')


def img_cutter(img, n, m, path_prefix):  # n-horizontalcuts, m-verticalcuts
    # first get the tot num of black pixels in the img
    img_ceil = np.ceil(img)
    sum = np.sum(img_ceil)
    n_sub_goal = sum / n
    m_sub_goal = sum / m

    pix_counter = 0
    n_counter = 1
    n_idx_list = []
    for i in range(img_ceil.shape[0]):
        pix_counter += np.sum(img_ceil[i])
        if pix_counter >= (n_counter * n_sub_goal):
            n_idx_list.append(i)
            n_counter += 1
            print('pix_counter: ', pix_counter)
        if n_counter == n:
            break
    # print('sum: ', sum)
    # print('n_sub_goal: ', n_sub_goal)
    # print('n_counter: ', n_counter)
    # print('n_idx_list: ', n_idx_list)

    img_ceil_tp = np.transpose(img_ceil)
    pix_counter = 0
    m_counter = 1
    m_idx_list = []
    for i in range(img_ceil_tp.shape[0]):
        pix_counter += np.sum(img_ceil_tp[i])
        if pix_counter >= (m_counter * m_sub_goal):
            m_idx_list.append(i)
            m_counter += 1
            print('pix_counter: ', pix_counter)
        if m_counter == m:
            break
    # print('sum: ', sum)
    # print('m_sub_goal: ', m_sub_goal)
    # print('m_counter: ', m_counter)
    # print('m_idx_list: ', m_idx_list)

    n_counter = 1
    m_counter = 1
    n_idx_list.append(img.shape[0])
    m_idx_list.append(img.shape[1])
    for i in range(len(n_idx_list)):
        for j in range(len(m_idx_list)):
            out_path = path_prefix + '_%d_%d.png' % (i, j)
            if i == 0:
                prev_n = 0
            else:
                prev_n = n_idx_list[i - 1]
            if j == 0:
                prev_m = 0
            else:
                prev_m = m_idx_list[j - 1]
            this_n = n_idx_list[i]
            this_m = m_idx_list[j]
            img_crop = img[prev_n:this_n, prev_m:this_m]
            plt.imsave(out_path, img_crop, cmap=plt.get_cmap('gray'))


def img12_cutter(img, n, m, path_prefix_list):  # n-horizontalcuts, m-verticalcuts
    # first get the tot num of black pixels in the img
    img_ceil = np.ceil(img)
    sum = np.sum(img_ceil)
    n_sub_goal = sum / n
    m_sub_goal = sum / m

    pix_counter = 0
    n_counter = 1
    n_idx_list = []
    for i in range(img_ceil.shape[0]):
        pix_counter += np.sum(img_ceil[i])
        if pix_counter >= (n_counter * n_sub_goal):
            n_idx_list.append(i)
            n_counter += 1
        if n_counter == n:
            break

    img_ceil_tp = np.transpose(img_ceil)
    pix_counter = 0
    m_counter = 1
    m_idx_list = []
    for i in range(img_ceil_tp.shape[0]):
        pix_counter += np.sum(img_ceil_tp[i])
        if pix_counter >= (m_counter * m_sub_goal):
            m_idx_list.append(i)
            m_counter += 1
        if m_counter == m:
            break

    n_idx_list.append(img.shape[0])
    m_idx_list.append(img.shape[1])

    cropped = []
    for i in range(len(n_idx_list)):
        row = []
        for j in range(len(m_idx_list)):
            if i == 0:
                prev_n = 0
            else:
                prev_n = n_idx_list[i - 1]
            if j == 0:
                prev_m = 0
            else:
                prev_m = m_idx_list[j - 1]
            this_n = n_idx_list[i]
            this_m = m_idx_list[j]
            img_crop = img[prev_n:this_n, prev_m:this_m]
            row.append(img_crop)
        cropped.append(row)

    for i in range(n):
        for j in range(m):
            out_path = path_prefix_list[i*m+j] + '_%d_%d.png' % (i, j)
            plt.imsave(out_path, cropped[i][j], cmap=plt.get_cmap('gray'))


def gray_bi_cut(input_path, path_prefix_list, num_horizontal_cut, num_vertical_cut):
    gray = Image.open(input_path).convert('LA')
    # plt.imshow(gray)
    # plt.show()
    gray_np = np.array(gray)
    partial_bi = gray2bi(gray_np, threshold=0.88)
    partial_bi = 1 - partial_bi  # strokes to 1 and background to 0
    img12_cutter(partial_bi, num_horizontal_cut, num_vertical_cut, path_prefix_list)

# path = 'original_1_1_bi'
# img_cutter(partial_bi, 4, 6, out_path=path)


def gray_bi(input_path):
    gray = Image.open(input_path).convert('LA')
    gray_np = np.array(gray)
    partial_bi = gray2bi(gray_np, threshold=0.88)
    partial_bi = 1 - partial_bi
    plt.imsave(input_path, partial_bi, cmap=plt.get_cmap('gray'))