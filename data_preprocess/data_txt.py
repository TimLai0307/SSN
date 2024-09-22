import os

path = 'D:/Lai/counting_dataset/test/preprocess_data/SHHA/'
cls = 'train'

img_list = os.listdir(path + cls + '/')

with open(path + cls + '.txt', 'w') as f:

    num = len(img_list)
    for i in range(0, num, 3):
        # f.write('/home/twsahaj458/Lai/NWPU/' + cls + '/' + img_list[i+1] + ' ' + '/home/twsahaj458/Lai/NWPU/' + cls + '/' + img_list[i] + '\n')
        # f.write(target_path + cls + '/' + img_list[i + 1] + ' ' + target_path + cls + '/' + img_list[i] + '\n')
        f.write(path + cls + '/' + img_list[i + 1] + ' ' + path + cls + '/' + img_list[i+2] + '\n')



