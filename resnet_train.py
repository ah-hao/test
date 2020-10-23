'''
This code is to train the siamese model
Written by Darrel (2020/07/29)
function: check_output_dir, get_data, show_train_history
'''
import warnings
warnings.filterwarnings('ignore')
import os
# close GPU memory & device information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from keras.preprocessing.image import ImageDataGenerator
from resnet_face_model import conv_layer, resnet
from keras.callbacks import ReduceLROnPlateau

np.random.seed(666)


train_datagen = ImageDataGenerator(horizontal_flip=True,
                                   rotation_range=0, 
                                   width_shift_range=0.15, 
                                   height_shift_range=0.15,
                                   shear_range=0,
                                   zoom_range=0, 
                                   data_format='channels_last',
                                   fill_mode='constant',
                                   cval=0.)


lr_function = ReduceLROnPlateau(monitor='loss',
                                patience=2,
                                verbose=1,
                                factor=0.5,
                                min_lr=0.00001)


def feature_normalize(train_data):
    # global mean, std
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)

    return np.nan_to_num((train_data - mean) / (std + 1e-7))


def load_data(data):
    '''
    load face database to label
    '''
    count = 0
    geuine_size = 0
    for i in os.listdir(data):
        geuine_size += len([c for c in  combinations_with_replacement(range(len(os.listdir(os.path.join(data, i)))), 2)])
        count += len(os.listdir(os.path.join(data, i)))
    # print(geuine_size)
    total_count = len([c for c in  combinations_with_replacement(range(count), 2)])
    # print(total_count)
    imposite_size = total_count - geuine_size

    count_genuine = 0

    #initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([geuine_size, 2, 1, 32, 32]) # 2 is for pairs 3//
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
from sklearn.model_selection import train_test_split
from model import build_base_network


# optim = SGD(lr=0.001, decay=.01, momentum=0.9, nesterov=True)

if __name__ == "__main__":
    data = './data/data_classmate_with5'
    # X, Y = load_data(data)

    # np.savez('./npz/classmate_with5.npz', x=X, y=Y)
    data = np.load('./npz/classmate_with5.npz')
    X = data['x']
    Y = data['y']

    # train_datagen.fit(X)

    print('X.shape:', X.shape)
    print('Y.shape:', Y.shape)
    print('-'*100)

    input_dim = X.shape[2:]
    # siamese_model = build_base_network(input_dim)

    resnet_model = resnet(input_dim)

    resnet_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    X_1 = X[:, 0]
    # X_1 = feature_normalize(X_1)
    # print(X_1.shape)
    X_2 = X[:, 1] 
    # X_2 = feature_normalize(X_2)

    # train_datagen.fit(X_1)

    train_history = resnet_model.fit([X_1, X_2], Y, batch_size=256, verbose=1, nb_epoch=15, callbacks=[lr_function])

    # train_history = resnet_model.fit_generator(train_datagen.flow([X_1, X_2], Y, batch_size=128), 
    #                                            steps_per_epoch=X_1.shape[0]/128, 
    #                                            epochs=20, 
    #                                            validation_data=None, 
    #                                            validation_steps=None, 
    #                                            callbacks=None)



    check_output_dir('./model')
    resnet_model.save('./model_with5/resnet_model.h5')
    print('\nFinished!\n./model/resnet_model.h5')
    show_train_history(train_history, 'loss')