from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, BatchNormalization
from keras.layers import add
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras import backend
from sklearn.model_selection import train_test_split


def conv_layer(input_tensor, filters, name):

    filters_1 = 2*filters
    filters_2 = filters

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(filters_1, (3, 3), use_bias=True, padding='same',
                                  kernel_initializer='he_normal', 
                                  kernel_regularizer=regularizers.l2(1e-4), name=str(name)+'_conv')(input_tensor)

    x = BatchNormalization(axis=bn_axis, momentum=0.9,
                                         epsilon=1e-5, name=str(name)+'_bn')(x)

    x = Activation('relu')(x)

    pooling = MaxPooling2D((2, 2), strides=(2, 2), name=str(name)+'_pool')(x)

    shortcut = Conv2D(filters_2, (1, 1), use_bias=True, padding='same', 
                                  kernel_initializer='he_normal', 
                                  kernel_regularizer=regularizers.l2(1e-4), name=str(name)+'_shortcut')(pooling)

    shortcut = BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-5)(shortcut)
    
    x = Conv2D(filters_1, (3, 3), use_bias=True, padding='same', 
                                  kernel_initializer='he_normal', 
                                  kernel_regularizer=regularizers.l2(1e-4), name=str(name)+'_res1_conv')(pooling)

    x = BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-5, name=str(name)+'_res1_bn')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters_2, (3, 3), use_bias=True, padding='same', 
                                  kernel_initializer='he_normal', 
                                  kernel_regularizer=regularizers.l2(1e-4), name=str(name)+'_res2_conv')(pooling)

    x = BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-5, name=str(name)+'_res2_bn')(x)
    x = Activation('relu')(x)
    x = add([x, shortcut], name=str(name)+'_add')

    return x

def resnet(input_dim):

    img_input = Input(shape=input_dim)

    prep = Conv2D(512, (3, 3),input_shape=input_dim, use_bias=True, padding='same', 
                             kernel_initializer='he_normal', 
                             kernel_regularizer=regularizers.l2(1e-4))(img_input)

    prep = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(prep)
    prep = Activation('relu')(prep)

    layer_1 = conv_layer(prep, 128, name='layer_1')

    layer_2 = Conv2D(128, (3, 3), use_bias=True, padding='same', 
                                 kernel_initializer='he_normal', 
                                 kernel_regularizer=regularizers.l2(1e-4))(layer_1)

    layer_2 = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(layer_2)
    layer_2 = Activation('relu')(layer_2)

    layer_3 = conv_layer(layer_2, 64, name='layer_3')

    classifier = MaxPooling2D((2, 2), strides=(2, 2))(layer_3)
    classifier = Activation('linear')(classifier)
    classifier = Flatten()(classifier)

    # scale = Dense(64, activation='relu', use_bias=False)(classifier)

    scale =  Dense(10, activation='softmax', use_bias=True, kernel_initializer='he_normal')(classifier)

    resnet = Model(inputs=img_input, outputs=scale) 

    return resnet


if __name__ == '__main__':
    resnet_model = resnet((32, 32, 3))
    resnet_model.summary()