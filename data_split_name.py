# ===================================== IMPORT ============================================#
import os
import shutil
import re

num_of_names = 55

'''
    1. get the path for forged signatures: ls -R /Users/yizezhao/PycharmProjects/ece324/sigver/signatures/full_forg/*.png > forg_path.txt
    2. get the path for original signatures: ls -R /Users/yizezhao/PycharmProjects/ece324/sigver/signatures/full_org/*.png > auth_path.txt
'''
target_dir = []
forg_target_dir = []
auth_target_dir = []

# forgery data
for i in range(1, 56):
    forg_target_dir.append("/Users/yizezhao/PycharmProjects/ece324/sigver/signatures/name_" + str(i) + "_forg")
    print(str(i))

    os.mkdir(forg_target_dir[i - 1])

with open('forg_path.txt', 'r') as forg_path:
    for cnt, line in enumerate(forg_path):
        line = line.rstrip()
        name = re.match(
            r'/Users/yizezhao/PycharmProjects/ece324/sigver/signatures/full_forg/forgeries_(.*?)_(.*).png', line)
        # print(name.group(1))
        dest = forg_target_dir[int(name.group(1)) - 1]
        print(line)
        print(dest)
        shutil.move(line, dest)

# authetic data
for i in range(1, num_of_names + 1):
    auth_target_dir.append("/Users/yizezhao/PycharmProjects/ece324/sigver/signatures/name_" + str(i) + "_auth")

    os.mkdir(auth_target_dir[i - 1])
    print(auth_target_dir)

with open('auth_path.txt', 'r') as auth_path:
    for cnt, line in enumerate(auth_path):
        line = line.rstrip()
        name = re.match(
            r'/Users/yizezhao/PycharmProjects/ece324/sigver/signatures/full_org/original_(.*?)_(.*).png', line)
        print(name.group(1))
        dest = auth_target_dir[int(name.group(1)) - 1]
        print(line)
        print(dest)
    shutil.move(line, dest)