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
from model import build_base_network, build_reuse_network
from resnet_face_model import conv_layer, resnet
from SiameseResNet_test2 import building_block, create_model, create_CaptureFeature_model
from model_2 import ResBlock, build_res_network
from keras.models import Model
from keras import backend as K
from keras.layers import Input
import math

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
  

import glob
from PIL import Image 





# def img_test(origin_model, model, image):
#     # load anchor imgs
#     imgs_dir_path = './data/kmeans_anchor/'
#     imgs_name = os.listdir(imgs_dir_path)

#     image = image / 255  # normalize in 0 ~ 1
#     input_img = image[np.newaxis, np.newaxis, :, :]  
#     print(input_img.shape)
#     all_score = np.zeros(0)   
    
    
#     anchor_imgs = np.array([np.array(Image.open(anchor_image)) for anchor_image in glob.glob('./data/kmeans_anchor/*.png')]).astype(np.float)
#     for anchor_img in anchor_imgs:
#         anchor_img = anchor_img[np.newaxis, np.newaxis, :, :]     
#         score = model.predict([input_img, anchor_img])
#         all_score = np.append(all_score, score)
#     print(all_score)
#     argmax = np.argmax(all_score)
#     guess = all_score[argmax]

#     print(input_img.shape)

#     return argmax, guess

total_clusters = 3

# anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie', 'dummy', 'dummy', 'dummy', 'dummy']

anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie', 'chieh', 'sam', 'tim', 'mjun']

# anchor_name = ['chieh', 'jeff', 'jiawei', 'donna', 'peng', 'fannie', 'sam', 'tim', 'mjun']

# anchor_name = ['chieh', 'fannie', 'jeff', 'jiawei', 'peng', 'dummy', 'dummy', 'dummy', 'dummy']# new

def read_label_txt():
    '''
    read the label 
    '''
    map_characters = []
    label_path = './data/kmeans_anchor/'
    read_label = os.listdir(label_path)
    for label in read_label:
        name = label.split('_')[-3]
        map_characters.append(name) 
    return map_characters

import time    

if __name__ == "__main__":


#-------------------------------------- Load_weights --------------------------------------------------  

    origin_model = load_model('./model_with5/SiameseResnet_mc2_model/SiameseResnet_mc2_stable_.h5')

    fix_model = create_CaptureFeature_model((1, 32, 32))
    fix_model.set_weights(origin_model.get_weights())#.layers[3]

    relation_model = load_model('./model_with5/test_by_test.h5') # SiameseResnet_with5_model
    

#------------------------------------------------------------------------------------------------------

    map_characters = read_label_txt()

    # write predict result name
    pd_result_name = ''
    
    timestamp = time.strftime('_%m%d%H%M%S', time.localtime())
    output_dir = './test_result/test_result' + timestamp
    check_output_dir(output_dir)
    
    test_data = './test'
    list_dir = os.listdir(test_data)

    img_32_list = []

    for m in list_dir:
        imagePath = os.path.join(test_data, m)
        print(imagePath)
        image = cv2.imread(imagePath)
        success, local = face_detection(image)
        if success:
            local = box_reduce(local, 30)
            img_32 = image_pre_processing(image, local)
            img_32_list.append(img_32)

    
    img_32_np = np.array(img_32_list, dtype='float32')
    img_32_np = img_32_np[:,np.newaxis, :,:] / 255.0
    print(img_32_np.shape)

    test_feature = fix_model.predict(img_32_np)  ##
    test_feature = test_feature[np.newaxis, :,:,:,:]

    print(test_feature.shape)

    # fix_img = np.load('./npz/fix_img_np.npy')
    fix_img = np.load('./npz/fix_image_sum_np.npy')
    fix_img = fix_img / 255.0

    fix_feature = fix_model.predict(fix_img) ##
    fix_feature = fix_feature[np.newaxis, :,:,:,:]

    print(fix_feature.shape)

    fix_feature_repeat = np.repeat(fix_feature, test_feature.shape[1], axis=0)

    test_feature_repeat = np.repeat(test_feature, fix_feature.shape[1], axis=0)
    test_feature_repeat = np.swapaxes(test_feature_repeat, 0, 1)

    print(fix_feature_repeat.shape)
    print(test_feature_repeat.shape)

    cat_feature = np.concatenate([fix_feature_repeat, test_feature_repeat], axis=2)

    input_feature = np.reshape(cat_feature, (-1, 64, 4, 4))

    print(input_feature.shape)

    score = relation_model.predict(input_feature)

    score = np.reshape(score, (-1, len(anchor_name))) #####len(anchor_name)*3

    print(score.shape)
    # print(score[1])
    # print(np.argmax(score[1]))
    # print(score[1][np.argmax(score[1])])
    

    

    i=0
    for target in list_dir:
        # read one image 
        imagePath = os.path.join(test_data, target)
        print(imagePath)
        image = cv2.imread(imagePath)
        success, local = face_detection(image)
        
        if success:
            local = box_reduce(local, 30)
            # img_32 = image_pre_processing(image, local)

            result = np.argmax(score[i]) # predict ***
            guess = score[i][result]
            pd_result_name = anchor_name[result] # int(result / total_clusters)
            print('index:', result)
            print('name:', pd_result_name)

            cv2.rectangle(
                image, (local[0], local[1]), (local[2], local[3]), (0, 255, 0), 4, cv2.LINE_AA)

            cv2.putText(image, pd_result_name, (local[0], local[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            timestamp = str(int(round(time.time() * 1000)))
            cv2.imwrite(output_dir + '/' + str(guess) + '_' + str(target), image)

            print('picture_index:', i)
            i+=1
            

        else:
            print("No face in picture.")
    

