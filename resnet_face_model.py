from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
# from keras.datasets import cifar10
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
from keras.optimizers import SGD



def conv_layer(input_tensor, filters, name):

    filters_1 = filters
    filters_2 = filters

    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    x = Conv2D(filters_1, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', data_format='channels_first',
                                  kernel_regularizer=regularizers.l2(1e-4), name=str(name)+'_conv')(input_tensor)

    x = BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-5, name=str(name)+'_bn')(x)
    x = Activation('relu')(x)
    pooling = MaxPooling2D((2, 2), data_format='channels_first', name=str(name)+'_pool')(x)
    shortcut = Conv2D(filters_2, (1, 1), use_bias=True, padding='valid', kernel_initializer='he_normal', data_format='channels_first',
                                  kernel_regularizer=regularizers.l2(1e-4), name=str(name)+'_shortcut')(pooling)
    
    shortcut = BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-5)(shortcut)

    x = Conv2D(filters_1, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', data_format='channels_first',
                                  kernel_regularizer=regularizers.l2(1e-4), name=str(name)+'_res1_conv')(pooling)

    x = BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-5, name=str(name)+'_res1_bn')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters_2, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', data_format='channels_first',
                                  kernel_regularizer=regularizers.l2(1e-4), name=str(name)+'_res2_conv')(pooling)

    x = BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-5, name=str(name)+'_res2_bn')(x)
    
    x = add([x, shortcut], name=str(name)+'_add')
    x = Activation('relu')(x)

    return x

def resnet(input_dim):

    img_input1 = Input(shape=input_dim)
    img_input2 = Input(shape=input_dim)

    prep = Conv2D(16, (3, 3),input_shape=input_dim, use_bias=True, padding='same', kernel_initializer='he_normal', data_format='channels_first', 
                              kernel_regularizer=regularizers.l2(1e-4))(img_input1)
    prep = BatchNormalization(axis=1, momentum=0.9, epsilon=1e-5)(prep)
    prep = Activation('relu')(prep)

    layer_1 = conv_layer(prep, 32, name='layer_1')

    layer_2 = Conv2D(32, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', data_format='channels_first', 
                                  kernel_regularizer=regularizers.l2(1e-4))(layer_1)

    layer_2 = BatchNormalization(axis=1, momentum=0.9, epsilon=1e-5)(layer_2)
    layer_2 = Activation('relu')(layer_2)

    layer_3 = conv_layer(layer_2, 64, name='layer_3')

    classifier = MaxPooling2D((2, 2), data_format='channels_first')(layer_3)
    classifier = Activation('linear')(classifier)
    classifier = Flatten()(classifier)

    # classifier = Dropout(0.2)(classifier)

    scale = Dense(512, activation='relu')(classifier)
    # scale = Dropout(0.2)(scale)

    scale =  Dense(10, activation='sigmoid', use_bias=True, kernel_initializer='he_normal')(scale) ##1&2

    feat_10 = Model(inputs=img_input1, outputs=scale)

    feat_10.summary()

    # feature
    feat_vecs_a = feat_10(img_input1)
    feat_vecs_b = feat_10(img_input2)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([feat_vecs_a, feat_vecs_b])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid', use_bias=False)(L1_distance)

    resnet = Model(inputs=[img_input1, img_input2], outputs=prediction) 

    return resnet


if __name__ == '__main__':
    resnet_model = resnet((1, 32, 32))
    resnet_model.summary()