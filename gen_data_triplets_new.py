import re

'''
    this script will generate data pairs(in 2 or 3) that will be used for baseline model
    pre:
    generate following 6 txtfiles (in terminal) contrains all trainning samples and stores in workinf directory:

(base) Yizes-MacBook-Pro-2:sigver yizezhao$ ls -r split_signature/train/*_auth/*.png > splitted_train_auth.txt
(base) Yizes-MacBook-Pro-2:sigver yizezhao$ ls -r split_signature/valid/*_auth/*.png > splitted_valid_auth.txt
(base) Yizes-MacBook-Pro-2:sigver yizezhao$ ls -r split_signature/test/*_auth/*.png > splitted_test_auth.txt
(base) Yizes-MacBook-Pro-2:sigver yizezhao$ ls -r split_signature/test/*_forg/*.png > splitted_test_forg.txt
(base) Yizes-MacBook-Pro-2:sigver yizezhao$ ls -r split_signature/valid/*_forg/*.png > splitted_valid_forg.txt
(base) Yizes-MacBook-Pro-2:sigver yizezhao$ ls -r split_signature/train/*_forg/*.png > splitted_train_forg.txt

    open auth and forg files
    iterate all images, figure out the name number and stored in a seperate
'''

'''
command for getting smaller dataset: 
(base) Yizes-MacBook-Pro-2:SignatureVerification yizezhao$ awk 'NR % 4 == 0' train_paried_list.csv > 50k_train_paried_list.csv
(base) Yizes-MacBook-Pro-2:SignatureVerification yizezhao$ vim 50k_train_paried_list.csv 
(base) Yizes-MacBook-Pro-2:SignatureVerification yizezhao$ awk 'NR % 4 == 0' valid_paried_list.csv > 50k_valid_paried_list.csv
(base) Yizes-MacBook-Pro-2:SignatureVerification yizezhao$ awk 'NR % 4 == 0' test_paried_list.csv > 50k_test_paried_list.csv

(base) Yizes-MacBook-Pro-2:SignatureVerification yizezhao$ awk 'NR % 50 == 0' train_paried_list.csv > small_train_paried_list.csv
(base) Yizes-MacBook-Pro-2:SignatureVerification yizezhao$ head -921 small_train_paried_list.csv > 921_train_paried_list.csv

'''

base_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'
test_auth_path = base_dir + 'sigcomp_cn_train_orig.txt'
test_forg_path = base_dir + 'sigcomp_cn_train_forg.txt'



test_triplet_path = '/Users/yizezhao/PycharmProjects/ece324/sigver/train_triplet_list_cn.csv'

names = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']


train_forg_names = []
train_auth_names = []

for i, id in enumerate(names):
    train_forg_names.append(base_dir + 'temp_lists/train_forg_' + id + '.txt')
    print(train_forg_names[i])
    open(train_forg_names[i], 'a').close()
    train_auth_names.append(base_dir + 'temp_lists/train_auth_' + id + '.txt')
    print(train_auth_names[i])
    open(train_auth_names[i], 'a').close()
    print(i)

'''
the following lnes generate lists of data according to names
and will be used for paring
please only run the following code once

START GENERATING NAME LISIS
'''

# with open(train_auth_path, 'r') as t_auth_path:
#     for cnt, line in enumerate(t_auth_path):
#         line = line.rstrip()
#         #signatures/train/name_55_auth/original_55_5.png
#         name = re.match(
#             r'split_signature/.*name_(.*?)_auth.*.png', line)
#         print("index", int(name.group(1))-1)
#         with open(auth_names[int(name.group(1))-1], 'a') as temp:
#             temp.write(line + '\n')
#             print(auth_names[int(name.group(1))-1])
#             print(line)
#
# with open(train_forg_path, 'r') as t_forg_path:
#     for cnt, line in enumerate(t_forg_path):
#         line = line.rstrip()
#         #signatures/train/name_55_forg/forgeries_55_9.png
#         name = re.match(
#             r'split_signature/.*name_(.*?)_forg.*.png', line)
#         print("index", int(name.group(1))-1)
#         with open(forg_names[int(name.group(1))-1], 'a') as temp:
#             temp.write(line + '\n')
#             print(forg_names[int(name.group(1))-1])
#             print(line)
#
# with open(valid_auth_path, 'r') as v_auth_path:
#     for cnt, line in enumerate(v_auth_path):
#         line = line.rstrip()
#         #signatures/train/name_55_auth/original_55_5.png
#         name = re.match(
#             r'split_signature/.*name_(.*?)_auth.*.png', line)
#         print("index", int(name.group(1))-1)
#         with open(auth_names[int(name.group(1))-1], 'a') as temp:
#             temp.write(line + '\n')
#             print(auth_names[int(name.group(1))-1])
#             print(line)
#
# with open(valid_forg_path, 'r') as v_forg_path:
#     for cnt, line in enumerate(v_forg_path):
#         line = line.rstrip()
#         #signatures/train/name_55_forg/forgeries_55_9.png
#         name = re.match(
#             r'split_signature/.*name_(.*?)_forg.*.png', line)
#         print("index", int(name.group(1))-1)
#         with open(forg_names[int(name.group(1))-1], 'a') as temp:
#             temp.write(line + '\n')
#             print(forg_names[int(name.group(1))-1])
#             print(line)

with open(test_auth_path, 'r') as t_auth_path:
    for cnt, line in enumerate(t_auth_path):
        line = line.rstrip()
        #sigComp2011-trainingSet-bi/016_forg/0202016_04.png
        #sigComp2011-trainingSet-bi/016_orig/016_24.PNG
        name = re.match(
            r'sigComp2011-trainingSet-CN/cn_(.*?)_orig/.*', line)
        print("index", name.group(1))
        with open(base_dir + 'temp_lists/train_auth_' + name.group(1) + '.txt', 'a') as temp:
            temp.write(line + '\n')
            print(base_dir + 'temp_lists/train_auth_' + name.group(1) + '.txt')
            print(line)

with open(test_forg_path, 'r') as t_forg_path:
    for cnt, line in enumerate(t_forg_path):
        line = line.rstrip()
        #sigComp2011-trainingSet-bi/016_forg/0202016_04.png
        #sigComp2011-trainingSet-bi/016_orig/016_24.PNG
        name = re.match(
            r'sigComp2011-trainingSet-CN/cn_(.*?)_forg/.*', line)
        print("index", name.group(1))
        with open(base_dir + 'temp_lists/train_forg_' + name.group(1) + '.txt', 'a') as temp:
            temp.write(line + '\n')
            print(base_dir + 'temp_lists/train_forg_' + name.group(1) + '.txt')
            print(line)
'''
    END OF GENERATING NAME LISTS
'''

'''
    the next session of the code will iterate all lists and generating parings (positive and negative) that comes from the same name
'''
# auth_last = None
# with open(test_triplet_path, 'a') as test_triplet:
#     for i in range(0, len(test_forg_names)):
#
#         with open(test_forg_names[i], 'r') as forg_lists, \
#                 open(test_auth_names[i], 'r') as auth_lists:
#             auth_lists_itr = [line.strip() for line in auth_lists]
#             auth_lists_itr_2 = auth_lists_itr.copy()
#             auth_lists_itr_3 = auth_lists_itr.copy()
#
#             forg_lists_itr = [line.strip() for line in forg_lists]
#
#             # write forgery pair (1)
#             for c1, auth_line in enumerate(auth_lists_itr):
#                 for c2, auth_line_2 in enumerate(auth_lists_itr_2):
#                     for c3, auth_line_3 in enumerate(auth_lists_itr_3):
#                         test_triplet.write(auth_line + ',' + auth_line_2 + ',' + auth_line_3 + ',' + '0' + '\n')
#
#             for c1, auth_line in enumerate(auth_lists_itr):
#                 for c2, auth_line_2 in enumerate(auth_lists_itr_2):
#                     for c3, forg_line in enumerate(forg_lists_itr):
#                         test_triplet.write(auth_line + ',' + auth_line_2 + ',' + forg_line + ',' + '1' + '\n')

with open(test_triplet_path, 'a') as test_triplet:
    for i in range(0, len(train_forg_names)):

        with open(train_forg_names[i], 'r') as forg_lists, \
                open(train_auth_names[i], 'r') as auth_lists:
            auth_lists_itr = [line.strip() for line in auth_lists]
            auth_lists_itr_2 = auth_lists_itr.copy()
            auth_lists_itr_3 = auth_lists_itr.copy()

            forg_lists_itr = [line.strip() for line in forg_lists]

            # write forgery pair (1)
            # for c1, auth_line in enumerate(auth_lists_itr):
            #     for c2, auth_line_2 in enumerate(auth_lists_itr_2):
            #         for c3, auth_line_3 in enumerate(auth_lists_itr_3):
            #             test_triplet.write(auth_line + ',' + auth_line_2 + ',' + auth_line_3 + ',' + '0' + '\n')

            for c1, auth_line in enumerate(auth_lists_itr):
                for c2, auth_line_2 in enumerate(auth_lists_itr_2):
                    for c3, forg_line in enumerate(forg_lists_itr):
                        test_triplet.write(auth_line + ',' + auth_line_2 + ',' + forg_line + '\n')