'''
This code is to train the siamese model
Written by Darrel (2020/07/29)
function: check_output_dir, get_data, show_train_history
'''
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import cv2
from itertools import combinations, combinations_with_replacement
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras import backend as K
import time
from sklearn.model_selection import train_test_split
from model import build_base_network, euclidean_distance, build_base_network # siamese
from SiameseResNet_test2 import building_block, create_model # resnet
from keras.optimizers import RMSprop, SGD

# close GPU memory & device information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def check_output_dir(out_dir_list):
    '''
    Check output dir
    '''
    if not os.path.isdir(out_dir_list):
        os.mkdir(out_dir_list)


# tf.compat.v1.set_random_seed(10)
# np.random.seed(10)

lr_function = ReduceLROnPlateau(monitor='loss',
                                patience=3,
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

    x_imposite_pair = np.zeros([imposite_size, 2, 1, 32, 32]) #//
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




def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


'''
Loss function
'''
rms = RMSprop()
sgd = SGD(momentum=0.9)


def model_function(model, input_size=None, path=None): ###

    def siamese():
        print('model = Siamese Model')
        return build_base_network(input_size)
    
    def resnet():
        print('model = ResNet Model')
        return create_model(input_size)


    switcher = {'siamese':siamese, 
                 'resnet':resnet }
    
    func = switcher.get(model, "nothing")

    return func()



if __name__ == "__main__":
    data = './data/data_classmate_mc2_'
    # X, Y = load_data(data)

    # np.savez('./npz/data_classmate_mc2_.npz', x=X, y=Y)
    data = np.load('./npz/data_classmate_mc2_.npz')
    X = data['x']
    Y = data['y']

    # print('X.shape:', X.shape)
    # print('Y.shape:', Y.shape)
    # print('-'*100)

    input_dim = X.shape[2:]

    #--------------------- Load model ------------------------

    siamese_model = model_function('resnet', input_size=input_dim)

    siamese_model.compile(loss='mse', optimizer='adam')

    X_1 = X[:, 0]

    X_2 = X[:, 1] 

    train_history = siamese_model.fit([X_1, X_2], Y, batch_size=512, verbose=1, nb_epoch=12, callbacks=None)#[lr_function]

    check_output_dir('./model')

    #------------------------------------------ Save model path ---------------------------------------------------

    # siamese_model.save('./model_with5/SiameseLenet_model/siamesenet_with5_stable.h5') # SiameseLenet_model

    # siamese_model.save('./model_with5/SiameseLenet_mc2_model/siamesenet_mc2_stable.h5') # SiameseLenet_mc2_model

    # siamese_model.save('./model_with5/SiameseResnet_with5_model/SiameseResnet_stable.h5') # SiameseResnet_with5_model

    siamese_model.save('./model_with5/SiameseResnet_mc2_model/SiameseResnet_mc2_stable_.h5') # SiameseResnet_mc2_model

    #---------------------------------------------------------------------------------------------------------------

    print('\nFinished!\n./model/siamesenet.h5')
    show_train_history(train_history, 'loss')

