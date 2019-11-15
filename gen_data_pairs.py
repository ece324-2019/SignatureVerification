import re

'''
    this script will generate data pairs(in 2) that will be used for baseline model
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

base_dir = '/Users/yizezhao/PycharmProjects/ece324/sigver/'
train_auth_path = base_dir + 'splitted_train_auth.txt'
train_forg_path = base_dir + 'splitted_train_forg.txt'
valid_auth_path = base_dir + 'splitted_valid_auth.txt'
valid_forg_path = base_dir + 'splitted_valid_forg.txt'
test_auth_path = base_dir + 'splitted_test_auth.txt'
test_forg_path = base_dir + 'splitted_test_forg.txt'

train_paired_path = 'train_paried_list.csv'
valid_paried_path = 'valid_paried_list.csv'
test_paried_path = 'test_paried_list.csv'

total_name = 215

forg_names = []
auth_names = []
valid_forg_names = []
valid_auth_names = []
test_forg_names = []
test_auth_names = []

for i in range(1, total_name + 1):

    if (i <= 6):  # valid
        forg_names.append(base_dir + 'temp_lists/valid_forg_' + str(i) + '.txt')
        print(forg_names[i - 1])
        # open(forg_names[i - 1], 'a').close()
        auth_names.append(base_dir + 'temp_lists/valid_auth_' + str(i) + '.txt')
        print(auth_names[i - 1])
        # open(valid_forg_names[i - 1], 'a').close()

    elif (i <= 15):  # test
        forg_names.append(base_dir + 'temp_lists/test_forg_' + str(i) + '.txt')
        print(forg_names[i - 1])
        # open(test_forg_names[i], 'a').close()
        auth_names.append(base_dir + 'temp_lists/test_auth_' + str(i) + '.txt')
        print(auth_names[i - 1])
        # open(test_forg_names[i - 6 - 1], 'a').close()
    else:
        forg_names.append(base_dir + 'temp_lists/train_forg_' + str(i) + '.txt')
        print(forg_names[i - 1])
        # open(train_forg_names[i - 15 - 1], 'a').close()
        auth_names.append(base_dir + 'temp_lists/train_auth_' + str(i) + '.txt')
        print(auth_names[i - 1])
        # open(train_forg_names[i - 15 - 1], 'a').close()
        print(i)

'''
the following lnes generate lists of data according to names
and will be used for paring
please only run the following code once

START GENERATING NAME LISIS
'''

with open(train_auth_path, 'r') as t_auth_path:
    for cnt, line in enumerate(t_auth_path):
        line = line.rstrip()
        #signatures/train/name_55_auth/original_55_5.png
        name = re.match(
            r'split_signature/.*name_(.*?)_auth.*.png', line)
        print("index", int(name.group(1))-1)
        with open(auth_names[int(name.group(1))-1], 'a') as temp:
            temp.write(line + '\n')
            print(auth_names[int(name.group(1))-1])
            print(line)

with open(train_forg_path, 'r') as t_forg_path:
    for cnt, line in enumerate(t_forg_path):
        line = line.rstrip()
        #signatures/train/name_55_forg/forgeries_55_9.png
        name = re.match(
            r'split_signature/.*name_(.*?)_forg.*.png', line)
        print("index", int(name.group(1))-1)
        with open(forg_names[int(name.group(1))-1], 'a') as temp:
            temp.write(line + '\n')
            print(forg_names[int(name.group(1))-1])
            print(line)

with open(valid_auth_path, 'r') as v_auth_path:
    for cnt, line in enumerate(v_auth_path):
        line = line.rstrip()
        #signatures/train/name_55_auth/original_55_5.png
        name = re.match(
            r'split_signature/.*name_(.*?)_auth.*.png', line)
        print("index", int(name.group(1))-1)
        with open(auth_names[int(name.group(1))-1], 'a') as temp:
            temp.write(line + '\n')
            print(auth_names[int(name.group(1))-1])
            print(line)

with open(valid_forg_path, 'r') as v_forg_path:
    for cnt, line in enumerate(v_forg_path):
        line = line.rstrip()
        #signatures/train/name_55_forg/forgeries_55_9.png
        name = re.match(
            r'split_signature/.*name_(.*?)_forg.*.png', line)
        print("index", int(name.group(1))-1)
        with open(forg_names[int(name.group(1))-1], 'a') as temp:
            temp.write(line + '\n')
            print(forg_names[int(name.group(1))-1])
            print(line)

with open(test_auth_path, 'r') as t_auth_path:
    for cnt, line in enumerate(t_auth_path):
        line = line.rstrip()
        #signatures/train/name_55_auth/original_55_5.png
        name = re.match(
            r'split_signature/.*name_(.*?)_auth.*.png', line)
        print("index", int(name.group(1))-1)
        with open(auth_names[int(name.group(1))-1], 'a') as temp:
            temp.write(line + '\n')
            print(auth_names[int(name.group(1))-1])
            print(line)

with open(test_forg_path, 'r') as t_forg_path:
    for cnt, line in enumerate(t_forg_path):
        line = line.rstrip()
        #signatures/train/name_55_forg/forgeries_55_9.png
        name = re.match(
            r'split_signature/.*name_(.*?)_forg.*.png', line)
        print("index", int(name.group(1))-1)
        with open(forg_names[int(name.group(1))-1], 'a') as temp:
            temp.write(line + '\n')
            print(forg_names[int(name.group(1))-1])
            print(line)
'''
    END OF GENERATING NAME LISTS
'''

'''
    the next session of the code will iterate all lists and generating parings (positive and negative) that comes from the same name
'''
# auth_last = None
with open(train_paired_path, 'a') as train_pair, open(valid_paried_path, 'a') as valid_pair, \
        open(test_paried_path, 'a') as test_pair:
    for i in range(1, total_name + 1):
        if (i <= 6):
            writing = valid_pair
        elif (i <= 15):
            writing = test_pair
        else:
            writing = train_pair

        with open(forg_names[i - 1], 'r') as forg_lists, \
                open(auth_names[i - 1], 'r') as auth_lists:

            # write forgery pair (1)
            for c1, auth_line in enumerate(auth_lists):
                forg_lists.seek(0)
                for c2, forg_line in enumerate(forg_lists):
                    auth_line = auth_line.rstrip()
                    forg_line = forg_line.rstrip()
                    # print(auth_line + ',' + forg_line + ',' + '1')
                    writing.write(auth_line + ',' + forg_line + ',' + '1' + '\n')

            # write forgery pair (0)
            auth_lists.seek(0)
            auth_lists_itr = [line.strip() for line in auth_lists]
            auth_lists_itr_2 = auth_lists_itr.copy()
            for c1, auth_line in enumerate(auth_lists_itr):
                for c2, auth_line_2 in enumerate(auth_lists_itr_2):
                    # print(auth_line + ',' + auth_line_2 + ',' + '0')
                    writing.write(auth_line + ',' + auth_line_2 + ',' + '0' + '\n')