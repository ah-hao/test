from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, BatchNormalization, AveragePooling2D
from keras.layers import Add
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras import backend
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD



def building_block(X, filters, stride=1):

    X_shortcut = X

    if stride > 1:
        X_shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', data_format='channels_first', use_bias=False)(X_shortcut)
        # X_shortcut = BatchNormalization(axis=1)(X_shortcut)

    # First layer of the block
    X = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first', use_bias=False)(X)
    # X = BatchNormalization(axis=1)(X)#momentum=0.9
    X = Activation('relu')(X)

    # Second layer of the block
    X = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first', use_bias=False)(X)
    # X = BatchNormalization(axis=1)(X)# momentum=0.9
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)


    return X

def create_model(input_dim):


    img_input1 = Input(shape=input_dim)
    img_input2 = Input(shape=input_dim)

    prep = Conv2D(8, (3, 3), input_shape=input_dim, data_format='channels_first', padding='same', use_bias=False)(img_input1)
    # prep = BatchNormalization(axis=1)(prep)#momentum=0.9
    prep = Activation('relu')(prep)

    pooling_1 = MaxPooling2D((2, 2), data_format='channels_first')(prep)

    X = building_block(pooling_1, filters=8, stride=1)
    X = Conv2D(16, (3, 3), data_format='channels_first', padding='same', use_bias=False)(X)
    X = Activation('relu')(X)

    pooling_2 = MaxPooling2D((2, 2), data_format='channels_first')(X)

    X = building_block(pooling_2, filters=16, stride=1)
    X = Conv2D(32, (3, 3), data_format='channels_first', padding='same', use_bias=False)(X)
    X = Activation('relu')(X)

    pooling_3 = MaxPooling2D((2, 2), data_format='channels_first')(X)

    X = building_block(pooling_3, filters=32, stride=1)

    pooling_last = MaxPooling2D((4, 4), data_format='channels_first')(X)
    feature = Flatten()(pooling_last)
    feature = Dense(12, activation='sigmoid', use_bias=False)(feature)

    fc = Model(inputs=img_input1, outputs=feature)

    fc.summary()

    # feature
    feat_vecs_a = fc(img_input1)
    feat_vecs_b = fc(img_input2)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([feat_vecs_a, feat_vecs_b])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    dense1 = Dense(6, activation='relu', use_bias=False)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(dense1)

    resnet = Model(inputs=[img_input1, img_input2], outputs=prediction) 

    return resnet


def create_kmeans_model(input_dim):

    img_input = Input(shape=input_dim)

    prep = Conv2D(8, (3, 3), input_shape=input_dim, data_format='channels_first', padding='same', use_bias=False)(img_input)
    # prep = BatchNormalization(axis=1)(prep)#momentum=0.9
    prep = Activation('relu')(prep)

    pooling_1 = MaxPooling2D((2, 2), data_format='channels_first')(prep)

    X = building_block(pooling_1, filters=8, stride=1)
    X = Conv2D(16, (3, 3), data_format='channels_first', padding='same', use_bias=False)(X)
    X = Activation('relu')(X)

    pooling_2 = MaxPooling2D((2, 2), data_format='channels_first')(X)

    X = building_block(pooling_2, filters=16, stride=1)
    X = Conv2D(32, (3, 3), data_format='channels_first', padding='same', use_bias=False)(X)
    X = Activation('relu')(X)

    pooling_3 = MaxPooling2D((2, 2), data_format='channels_first')(X)

    X = building_block(pooling_3, filters=32, stride=1)

    pooling_last = MaxPooling2D((4, 4), data_format='channels_first')(X)
    feature = Flatten()(pooling_last)
    feature = Dense(12, activation='sigmoid', use_bias=False)(feature)

    fc = Model(inputs=img_input, outputs=feature)

    fc.summary()


    return fc


def create_CaptureFeature_model(input_dim):

    img_input = Input(shape=input_dim)

    prep = Conv2D(8, (3, 3), input_shape=input_dim, data_format='channels_first', padding='same', use_bias=False)(img_input)
    # prep = BatchNormalization(axis=1)(prep)#momentum=0.9
    prep = Activation('relu')(prep)

    pooling_1 = MaxPooling2D((2, 2), data_format='channels_first')(prep)

    X = building_block(pooling_1, filters=8, stride=1)
    X = Conv2D(16, (3, 3), data_format='channels_first', padding='same', use_bias=False)(X)
    X = Activation('relu')(X)

    pooling_2 = MaxPooling2D((2, 2), data_format='channels_first')(X)

    X = building_block(pooling_2, filters=16, stride=1)
    X = Conv2D(32, (3, 3), data_format='channels_first', padding='same', use_bias=False)(X)
    X = Activation('relu')(X)

    pooling_3 = MaxPooling2D((2, 2), data_format='channels_first')(X)

    feature = building_block(pooling_3, filters=32, stride=1)

    fc = Model(inputs=img_input, outputs=feature)

    fc.summary()


    return fc




if __name__ == '__main__':
    resnet_model = create_model((1, 32, 32))
    resnet_model.summary()
    kmeans_model = create_CaptureFeature_model((1, 32, 32))
    # resnet_model.summary()

    