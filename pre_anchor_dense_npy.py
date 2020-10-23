'''
This code is to create the anchor feature and model dense weight
Written by Darrel (2020/07/29)
function: check_output_dir, imgfeature_tonpy
'''
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from model import build_base_network, build_reuse_network

import os
def check_output_dir(out_dir_list):
    '''
    Check output dir
    '''
    if not os.path.isdir(out_dir_list):
        os.mkdir(out_dir_list)

import cv2
from keras.models import Model
def imgfeature_tonpy(input_shape=(1, 32, 32), model_path='./model/siamesenet.h5'):
    model = build_base_network(input_shape)
    model.load_weights(model_path)
    img_dir_path = './data/anchor_imgs/'
    anker_imgs = os.listdir(img_dir_path)
    print('load iamge file to image feature!\n' + '-'*50)
    for image_file in anker_imgs:    
        print(img_dir_path + image_file)
        image_name = image_file.split(".")[0]

        image = cv2.imread(img_dir_path + image_file, 0)
        image - cv2.resize(image,(32,32))
        image = np.array(image).astype(np.float32)
        image = image / 255  # normalize in 0 ~ 1
        image = image[np.newaxis, np.newaxis, :, :]  # shape(1, 1, 32, 32,)    
        layer_model = Model(inputs=model.input, outputs=model.get_layer('sequential_1').get_output_at(1))
        layer_output = layer_model.predict([image, image])
        print(layer_output)
        check_output_dir('./data/anchor_npy/')
        np.save('./data/anchor_npy/' + image_name, layer_output)
        print('-'*50)   

def base_dense_weight_tonpy(input_shape=(1, 32, 32), model_path='./model/siamesenet.h5'):
    model = build_base_network(input_shape)
    model.load_weights(model_path)
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    print('moedel layer\n----------------')
    for name, weight in zip(names, weights):
        print(name, weight.shape)
        if 'dense' in name:
            check_output_dir('./data/dense_npy/')
            name = name.replace('/', '')
            name = name.replace(':', '')
            np.save('./data/dense_npy/base_' + name, weight)
            print('\nsave dense_npy!\nfile path:./data/dense_npy/' + name + '\n')

def reuse_dense_weight_tonpy(input_shape=([7]), model_path='./model/siamesenet_dense.h5'):
    model = build_reuse_network(input_shape)
    model.load_weights(model_path)
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    print('moedel layer\n----------------')
    for name, weight in zip(names, weights):
        print(name, weight.shape)
        if 'dense' in name:
            check_output_dir('./data/dense_npy/')
            name = name.replace('/', '')
            name = name.replace(':', '')
            np.save('./data/dense_npy/reuse_' + name, weight)
            print('\nsave dense_npy!\nfile path:./data/dense_npy/' + name + '\n')

if __name__ == "__main__": 
    # to extract image feature
    imgfeature_tonpy(model_path='./model/siamesenet_darrel_judy_1000.h5')

    # to extract model dense weight(frist model)
    base_dense_weight_tonpy(model_path='./model/siamesenet_darrel_judy_1000.h5')

    # to extract model dense weight(reuse model) 
    # reuse_dense_weight_tonpy(model_path='./model/siamesenet_dense_1_mark_li.h5')
