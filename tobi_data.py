from data_prep_lib import *


train_input_path = 'D:/1_Study/EngSci_Year3/ECE324_SigVer_project/split_signature/train/'
valid_input_path = 'D:/1_Study/EngSci_Year3/ECE324_SigVer_project/split_signature/valid/'
test_input_path = 'D:/1_Study/EngSci_Year3/ECE324_SigVer_project/split_signature/test/'

for i in range(1, 56):
    print('handling data in %d-th folder' % i)
    if i <= 6:
        input_path = valid_input_path
    elif i <= 15:
        input_path = test_input_path
    else:
        input_path = train_input_path

    for j in range(1, 25):
        auth_image_path = input_path + 'name_%d_auth/original_%d_%d.png' % (i, i, j)
        forg_image_path = input_path + 'name_%d_forg/forgeries_%d_%d.png' % (i, i, j)

        gray_bi(auth_image_path)
        gray_bi(forg_image_path)
