'''
This code is to train the siamese model
Written by Darrel (2020/07/29)
function: check_output_dir, get_data, show_train_history
'''
import warnings
warnings.filterwarnings('ignore')
import os

def check_output_dir(out_dir_list):
    '''
    Check output dir
    '''
    if not os.path.isdir(out_dir_list):
        os.mkdir(out_dir_list)

import numpy as np
import cv2
from itertools import combinations
from itertools import combinations_with_replacement

def load_data(data):
    count = 0
    geuine_size = 0
    for i in os.listdir(data):
        geuine_size += len([c for c in  combinations_with_replacement(range(len(os.listdir(os.path.join(data, i)))), 2)])
        count += len(os.listdir(os.path.join(data, i)))
        # for j in os.listdir(os.path.join(data, i)):
    print(geuine_size)
    total_count = len([c for c in  combinations_with_replacement(range(count), 2)])
    print(total_count)
    imposite_size = total_count - geuine_size
    count_genuine = 0

    #initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([geuine_size, 2, 1, 32, 32]) # 2 is for pairs
    y_genuine = np.zeros([geuine_size, 1])


    person_path_list = []
    for person_dir in os.listdir(data):
        person_path = os.path.join(data, person_dir)
        person_path_list.append(person_path)

        # read images from same directory (genuine pair)
        same_person = os.listdir(person_path)
        combins_imgs = [c for c in  combinations_with_replacement((same_person), 2)]
        for a, b in combins_imgs:    
            path_a, path_b = os.path.join(person_path, a), os.path.join(person_path, b)
            img1, img2 = cv2.imread(path_a, 0), cv2.imread(path_b, 0)

            #store the images to the initialized numpy array
            x_geuine_pair[count_genuine, 0, 0, :, :] = img1
            x_geuine_pair[count_genuine, 1, 0, :, :] = img2

            #as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count_genuine] = 1
            count_genuine += 1
    print('count_genuine:',count_genuine)

    count_imposite = 0

    x_imposite_pair = np.zeros([imposite_size, 2, 1, 32, 32])
    y_imposite = np.zeros([imposite_size, 1])

    combins = [c for c in  combinations((person_path_list), 2)]

    # read images from different directory (imposite pair)
    for different_person_path in combins:
        person_path_a, person_path_b = different_person_path
        for a in os.listdir(person_path_a):
            for b in os.listdir(person_path_b):
                path_a, path_b = os.path.join(person_path_a, a), os.path.join(person_path_b, b)
                img1, img2 = cv2.imread(path_a, 0), cv2.imread(path_b, 0)

                #store the images to the initialized numpy array
                x_imposite_pair[count_imposite, 0, 0, :, :] = img1
                x_imposite_pair[count_imposite, 1, 0, :, :] = img2
                
                #as we are drawing images from the different directory we assign label as 0. (imposite pair)
                y_imposite[count_imposite] = 0
                count_imposite += 1
    print('count_imposite:', count_imposite)
    print('-'*100)

    #now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)/255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y
 
import matplotlib.pyplot as plt

def show_train_history(train_history, train):
    # plot train set accuarcy / loss function value ( determined by what parameter 'train' you pass )
    # The type of train_history.history is dictionary (a special data type in Python)
    plt.plot(train_history.history[train])
    # set the title of figure you will draw
    plt.title('Train History')
    # set the title of y-axis
    plt.ylabel(train)
    # set the title of x-axis
    plt.xlabel('Epoch')
    # Places a legend on the place you set by loc
    plt.legend(['train'], loc='upper left')
    plt.show()

from keras import backend as K
import time
from keras.models import Model
from model import build_base_network, build_reuse_network

if __name__ == "__main__":
    data = './data/data_37p_mark_li'
    X, Y = load_data(data)

    print('X.shape:', X.shape)
    print('Y.shape:', Y.shape)
    print('-'*100)

    input_dim = X.shape[2:]

    X_1 = X[:, 0]
    X_2 = X[:, 1] 

    model = build_base_network(input_dim)
    model.load_weights('./model/siamesenet_35p.h5')

    # using the feature extractor to get feature
    conv_model = Model(inputs=model.input, outputs=model.get_layer('sequential_1').get_output_at(1))
    conv_out_1 = conv_model.predict([X_1, X_1])   
    conv_out_2 = conv_model.predict([X_2, X_2])    

    # using the feature to train model
    siamese_model = build_reuse_network(([7]))
    siamese_model.compile(loss='mse', optimizer='adam')
    train_history = siamese_model.fit([conv_out_1, conv_out_2], Y, batch_size=512, verbose=2, nb_epoch=50)

    check_output_dir('./model')
    siamese_model.save('./model/siamesenet_dense_1_mark_li.h5')

    show_train_history(train_history, 'loss')