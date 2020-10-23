'''
This code is to create the siamese model
Written by Darrel (2020/07/29)
function: build_base_network, build_reuse_network
'''
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D
from keras.models import Sequential, Model
from keras import backend as K

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))



def build_base_network(input_dim):

    img_a = Input(shape=input_dim)
    img_b = Input(shape=input_dim)

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
    feat_vecs_a = conv_net(img_a)
    feat_vecs_b = conv_net(img_b)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([feat_vecs_a, feat_vecs_b])

    # add 
    # L1_distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid', use_bias=False)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[img_a,img_b],outputs=prediction)
    
    return siamese_net   

def build_reuse_network(input_dim):
    # feature
    feat_vecs_a = Input(shape=input_dim)
    feat_vecs_b = Input(shape=input_dim)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([feat_vecs_a, feat_vecs_b])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid', use_bias=False)(L1_distance)
    
    # Connect the inputs with the outputs
    reuse_siamese_net = Model(inputs=[feat_vecs_a, feat_vecs_b],outputs=prediction)
    return reuse_siamese_net

def model_path():
    return './model/siamese_with_dummy.h5'

if __name__ == '__main__':
    siamese_model = build_base_network((1, 32, 32))
    siamese_model.summary()
    reuse_siamese_model = build_reuse_network(([7]))
    reuse_siamese_model.summary()
    # weights = siamese_model.get_weights()
    # print(weights)