import tensorflow as tf
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D
from keras.models import Sequential, Model
from keras import backend as K
import numpy as np




def for_kmeans_network(input_dim):

    # one input
    img = Input(shape=input_dim)

    conv_net = Sequential()

    # convolutional layer 1
    conv_net.add(Conv2D(6, (5, 5), padding='valid', input_shape=input_dim, data_format="channels_first", activation='relu', name='c1'))
    conv_net.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first", name='s2')) 

    # convolutional layer 2
    conv_net.add(Conv2D(12, (5, 5), padding='valid', data_format="channels_first", activation='relu', name='c3'))
    conv_net.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first", name='s4')) 

    # flatten 
    conv_net.add(Flatten())

    # dense
    conv_net.add(Dense(7, activation='sigmoid', name='f5'))

    # feature
    feat_vecs = conv_net(img)

    feat_kmeans = tf.constant(value=[0,0,0,0,0,0,0], dtype=tf.float32 )

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([feat_vecs, feat_kmeans])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid', use_bias=False)(L1_distance)
    
    # Connect the inputs with the outputs
    # siamese_net = Model(inputs=[img_a,img_b],outputs=prediction)

    siamese_net = Model(inputs=img, outputs=prediction)
    
    return siamese_net


if __name__ == "__main__":
    model = for_kmeans_network((1, 32, 32))
    model.summary()

    