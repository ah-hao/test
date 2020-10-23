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
from SiameseResNet_test2 import building_block, create_model
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


def img_test(model, image):
    # load anchor imgs
    imgs_dir_path = './data/kmeans_anchor/'
    imgs_name = os.listdir(imgs_dir_path)

    image = image / 255  # normalize in 0 ~ 1
    input_img = image[np.newaxis, np.newaxis, :, :]  
    print(input_img.shape)
    all_score = np.zeros(0)   
    
    
    anchor_imgs = np.array([np.array(Image.open(anchor_image)) for anchor_image in glob.glob('./data/kmeans_anchor/*.png')]).astype(np.float)
    for anchor_img in anchor_imgs:
        anchor_img = anchor_img[np.newaxis, np.newaxis, :, :]     
        score = model.predict([input_img, anchor_img])
        all_score = np.append(all_score, score)
    print(all_score)
    argmax = np.argmax(all_score)
    guess = all_score[argmax]

    print(input_img.shape)

    return argmax, guess

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

    # Load_model
    # model = build_base_network((1, 32, 32))
    model = create_model((1, 32, 32))
    # model = build_res_network((1, 32, 32))

############################################################ Load_weights #################################################################  

    # model.load_weights('./model_with5/SiameseLenet_model/siamesenet_with5_stable.h5') # SiameseLenet_model

    # model.load_weights('./model_with5/SiameseLenet_mc2_model/siamesenet_mc2_stable.h5') # SiameseLenet_model

    # model = load_model('./model_with5/SiameseResnet_with5_model/SiameseResnet_with5_0.93.h5') # SiameseResnet_with5_model
    
    model.load_weights('./model_with5/SiameseResnet_mc2_model/SiameseResnet_mc2_stable_.h5') # SiameseResnet_mc2_model

###########################################################################################################################################

    map_characters = read_label_txt()

    # write predict result name
    pd_result_name = ''
    
    timestamp = time.strftime('_%m%d%H%M%S', time.localtime())
    output_dir = './test_result/test_result' + timestamp
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
            result, guess = img_test(model, img_32)  # predict
            pd_result_name = map_characters[result]
            print('index:', result)
            print('name:', pd_result_name)

            cv2.rectangle(
                image, (local[0], local[1]), (local[2], local[3]), (0, 255, 0), 4, cv2.LINE_AA)

            cv2.putText(image, pd_result_name, (local[0], local[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            timestamp = str(int(round(time.time() * 1000)))
            cv2.imwrite(output_dir + '/' + str(guess) + '_' + str(target), image)

        else:
            print("No face in picture.")
