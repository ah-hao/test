import os
import numpy as np


# path = "./data/data_classmate"
# dirs = os.listdir( path )

# # 输出所有文件和文件夹
# for file in dirs:
#    print (file)


# x = np.zeros([10, 2, 1, 32, 32])

# y = np.zeros([10, 1])


# print(x)

# model_path = 'C:\\Users\\Jeff\\Documents\\WorkSpace\\darrel\\software\\model'
# for m in os.listdir(model_path):
#     print(m)
#     k = os.path.join(model_path, m)
#     print(k)


def read_label_txt(number):
    '''
    read the label 
    '''
    map_characters = []
    label_path = './data/anchor_imgs_' + str(number)
    read_label = os.listdir(label_path)
    for label in read_label:
        name = label.split('_')[-3]
        map_characters.append(name)

    return map_characters

print(read_label_txt(1))