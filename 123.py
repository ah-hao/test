import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D
from keras import models
import keras
from keras.preprocessing import image
from cv2_haar_face_detection import face_detection, box_reduce


# total_clusters = 4

# anchor_path = './data/anchor_list_path'

# kmeans_anchor_path = './data/kmeans_anchor'

# # anchor_name = {'0':'jeff', '1':'jiawei', '2':'donna', '3':'peng', '4':'fannie'}

# anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie']

# print(len(anchor_name))

# # print(anchor_name[0])

# anchor_image = []


# for count in range(len(anchor_name)):
#     for m in os.listdir(anchor_path + '/' + anchor_name[count]):
#         img = cv2.imread(anchor_path + '/' + anchor_name[count] + '/' + m, cv2.IMREAD_UNCHANGED)
#         # img = image.load_img(anchor_path+'/'+m, target_size=(32, 32), grayscale=True)
#         # x = image.img_to_array(img)
#         anchor_image.append(img)
#     anchor_image_array = np.array(anchor_image, dtype='float32')

#     anchor_image = []  

#     anchor_image_array = anchor_image_array[:, np.newaxis, :, :]  #(40,1,32,32)
#     print(anchor_image_array.shape)

# a = np.array([[1, 2], [3, 4]])

# b = np.array([[7, 8], [9,10]])

# c = np.add(a, b)

# print(c)

# b1 = np.ones([2, 1, 2, 4, 4])

# s = np.swapaxes(b1, 0, 1)

# print(s.shape)

# b0 = np.zeros([1, 2, 2, 4, 4])

# d = np.concatenate([s, b0], axis=1)

# print(d)

# print(a.shape)

# c = np.concatenate([a, b], axis=1)

# print(c)
# print(c.shape)


# a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# a_np = np.array(a)

# c = [1, 2, 3]

# d = [4, 5, 6]

# b = []

# b.append([x + y for x, y in zip(a[0], a[1])])

# b.append([x + y for x, y in zip(a[1], a[2])])

# print(b)

# b = np.array(b, dtype='float32')

# print(b/2)
anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie', 'cheih', 'sam', 'tim', 'mjun']
# map_characters = []
# test_data = './test'
# list_dir = os.listdir(test_data)

# img_32_list = []

# for m in list_dir:
#     imagePath = os.path.join(test_data, m)
#     print(imagePath)
#     image = cv2.imread(imagePath)
#     success, local = face_detection(image)
#     if success:
#         local = box_reduce(local, 30)
#         # img_32 = image_pre_processing(image, local)
#         # img_32_list.append(img_32)
#         if m.split('_')[-3] == anchor_name[0] or m.split('_')[-3] == anchor_name[1] or m.split('_')[-3] == anchor_name[2] or m.split('_')[-3] == anchor_name[3] or m.split('_')[-3] == anchor_name[4]:
#             name =  m.split('_')[-3]
#             map_characters.append(anchor_name.index(name))
#         else:
#             name =  5
#             map_characters.append(5)
    
# print(map_characters)

# for number in range(len(anchor_name)):
#     name_str(number) = []

# label_ = np.load('./npz/save_label.npy')

# print('label=', int(label_[len(anchor_name)-1]))                 


a = int(26/3)

print(a)