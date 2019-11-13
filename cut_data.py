from data_prep_lib import *
import os


train_input_path = 'D:/1_Study/EngSci_Year3/ECE324_SigVer_project/split_signature/train/'
valid_input_path = 'D:/1_Study/EngSci_Year3/ECE324_SigVer_project/split_signature/valid/'
test_input_path = 'D:/1_Study/EngSci_Year3/ECE324_SigVer_project/split_signature/test/'
input_path_list = [train_input_path, valid_input_path, test_input_path]

# for i in range(16, 56):
#     print('handling data in %d -th folder' % i)
#     input_path = train_input_path
#
#     for j in range(1, 6):
#         auth_image_path = input_path + 'name_%d_auth/original_%d_%d.png' % (i, i, j*4)
#         forg_image_path = input_path + 'name_%d_forg/forgeries_%d_%d.png' % (i, i, j*4)
#
#         auth_output_path_prefix = []
#         forg_output_path_prefix = []
#         for x in range(1, 13):
#             os.mkdir(input_path + 'name_%d_auth/' % (55 + x + 12*(j-1) + 60*(i-16)))
#             auth_output_path_prefix.append(input_path + 'name_%d_auth/original_%d_%d' % (55+x+12*(j-1)+60*(i-16), i, j*4))
#             os.mkdir(input_path + 'name_%d_forg/' % (55+x+12*(j-1)+60*(i-16)))
#             forg_output_path_prefix.append(input_path + 'name_%d_forg/original_%d_%d' % (55+x+12*(j-1)+60*(i-16), i, j*4))
#
#         gray_bi_cut(auth_image_path, auth_output_path_prefix, num_horizontal_cut=3, num_vertical_cut=4)
#         gray_bi_cut(forg_image_path, forg_output_path_prefix, num_horizontal_cut=3, num_vertical_cut=4)


for i in range(16, 56):
    print('handling data in %d-th folder' % i)
    input_path = train_input_path

    # make folders for 5 sub-images
    for x in range(1, 5):
        os.mkdir(input_path + 'name_%d_auth/' % (55 + x + 4*(i-16)))
        os.mkdir(input_path + 'name_%d_forg/' % (55 + x + 4*(i-16)))

    auth_prefix_all = []
    forg_prefix_all = []
    for j in range(1, 6):
        for x in range(1, 5):
            auth_prefix_all.append(input_path + 'name_%d_auth/original_%d_%d' % ((55 + x + 4*(i-16)), i, j*4))
            forg_prefix_all.append(input_path + 'name_%d_forg/forgeries_%d_%d' % ((55 + x + 4*(i-16)), i, j*4))

    # cut 5 images and put the right sub-images into the right folder
    for j in range(1, 6):
        auth_image_path = input_path + 'name_%d_auth/original_%d_%d.png' % (i, i, j*4)
        forg_image_path = input_path + 'name_%d_forg/forgeries_%d_%d.png' % (i, i, j*4)

        auth_prefix = auth_prefix_all[4*(j-1):4*j]
        forg_prefix = forg_prefix_all[4*(j-1):4*j]
        gray_bi_cut(auth_image_path, auth_prefix, num_horizontal_cut=2, num_vertical_cut=2)
        gray_bi_cut(forg_image_path, forg_prefix, num_horizontal_cut=2, num_vertical_cut=2)



