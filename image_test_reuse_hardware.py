'''
This code is to test the image
Written by Darrel (2020/07/29)
function: check_output_dir, image_pre_processing, sigmoid, img_test, read_label_txt
'''
import os
def check_output_dir(out_dir_list):
    '''
    Check output dir
    '''
    if not os.path.isdir(out_dir_list):
        os.mkdir(out_dir_list)
import cv2
import time
import numpy as np
from keras import backend
from keras.models import load_model

from cv2_haar_face_detection import face_detection, box_reduce
from model import build_base_network, model_path
from keras.models import Model
from keras import backend as K
from keras.layers import Input
import math

model = build_base_network((1, 32, 32))
model.load_weights('./model/siamesenet.h5')

# load anchor weight
npy_dir_path = './data/anchor_npy/'
anchor_npy = os.listdir(npy_dir_path)
feats = np.zeros(0)
for anchor_person in anchor_npy:
    npy_feat = np.load(npy_dir_path + anchor_person, mmap_mode='r')
    feats = np.append(feats, npy_feat)
feats = np.reshape(feats, (-1, 7))
dense_weight = np.load('./data/dense_npy/reuse_dense_2kernel0.npy')
# print(dense_weight.shape)

def image_pre_processing(image, f_lc):
    '''
    input:

        image: BGR image
        f lc: face bounding box: [left, top, right, bottom]

    output: 

        32 x 32 Gray face image
    '''

    image_face = image[f_lc[1]: f_lc[3], f_lc[0]: f_lc[2]]
    image_gray = cv2.cvtColor(image_face, cv2.COLOR_BGR2GRAY)
    image_32 = cv2.resize(image_gray, (32, 32), interpolation=cv2.INTER_CUBIC)

    return image_32

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
def relu(x):
    return np.maximum(x, 0)

def img_test(image):
    image = image / 255  # normalize in 0 ~ 1
    input_img = image[np.newaxis, np.newaxis, :, :]

    conv_model = Model(inputs=model.input, outputs=model.get_layer('sequential_1').get_output_at(1))
    conv_out = conv_model.predict([input_img, input_img])   # model.predict
    all_score = np.zeros(0)
    for feat in feats:
        abs_sub = np.absolute(conv_out - feat)
        dense = np.sum(np.dot(abs_sub, dense_weight))
        score = sigmoid(dense)
        all_score = np.append(all_score, score)
    argmax = np.argmax(all_score)
    guess = all_score[argmax]

    return argmax, guess

def read_label_txt():
    '''
    read the label 
    '''
    map_characters = []
    label_path = './data/anchor_npy/'
    read_label = os.listdir(label_path)
    for label in read_label:
        name = label.split('_')[-3]
        map_characters.append(name)
    return map_characters

if __name__ == "__main__":
    map_characters = read_label_txt()

    # write predict result name
    pd_result_name = ''

    timestamp = time.strftime('_%m%d%H%M%S', time.localtime())
    output_dir = './test_result' + timestamp
    check_output_dir(output_dir)
    
    test_data = './test'
    list_dir = os.listdir(test_data)
    for target in list_dir:
        # read one image 
        imagePath = os.path.join(test_data, target)
        print(imagePath)
        image = cv2.imread(imagePath)
        success, local = face_detection(image)
        if success:
            local = box_reduce(local, 30)
            img_32 = image_pre_processing(image, local)
            # img_test(img_32)  # predict
            result, guess = img_test(img_32)  # predict
            pd_result_name = map_characters[result]
            print('index:', result)
            print('name:', pd_result_name)

            cv2.rectangle(
                image, (local[0], local[1]), (local[2], local[3]), (0, 255, 0), 4, cv2.LINE_AA)

            cv2.putText(image, pd_result_name, (local[0], local[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            timestamp = str(int(round(time.time() * 1000)))
            cv2.imwrite(output_dir + '/' + str(guess) + '_' + str(target) + '.png', image)

        else:
            print("No face in picture.")
