import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, \
    Activation, Lambda, BatchNormalization, ReLU, Input, Add, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model


np.random.seed(121212)
# tf.random.set_seed(216)
tf.compat.v1.set_random_seed(212121)


# ------------------------------------------------------------------------------------------------------------------
# define the residual block
# ------------------------------------------------------------------------------------------------------------------
def ResBlock(X, filters, strides=1):
    x = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False, data_format='channels_first')(X)
    # x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False, data_format='channels_first')(x)
    # x = BatchNormalization()(x)

    if strides != 1:
        identity = Conv2D(filters, (1, 1), strides=strides, use_bias=False, data_format='channels_first')(X)
    else:
        identity = X

    outputs = Add()([x, identity])
    outputs = ReLU()(outputs)

    return outputs


# ------------------------------------------------------------------------------------------------------------------
# make the network model
# ------------------------------------------------------------------------------------------------------------------
def build_res_network(input_dim):
    img_input1 = Input(shape=input_dim)
    img_input2 = Input(shape=input_dim)

    prep = Conv2D(input_shape=input_dim, filters=8, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, data_format='channels_first')(img_input1)
    # prep = BatchNormalization()(prep)
    prep = ReLU()(prep)
    prep = MaxPooling2D((2, 2), data_format='channels_first')(prep)
    prep = ResBlock(prep, 8, strides=1)

    layer1 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, data_format='channels_first')(prep)
    # layer1 = BatchNormalization()(layer1)
    layer1 = ReLU()(layer1)
    layer1 = MaxPooling2D((2, 2), data_format='channels_first')(layer1)
    layer1 = ResBlock(layer1, 16, strides=1)

    layer2 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, data_format='channels_first')(layer1)
    # layer2 = BatchNormalization()(layer2)
    layer2 = ReLU()(layer2)
    layer2 = MaxPooling2D((2, 2), data_format='channels_first')(layer2)
    layer2 = ResBlock(layer2, 32, strides=1)

    other_layer = MaxPooling2D((4, 4), data_format='channels_first')(layer2)
    other_layer = Flatten()(other_layer)
    outputs = Dense(12, activation='sigmoid', use_bias=False)(other_layer)

    feature_net = Model(inputs=img_input1, outputs=outputs)

    # feature
    feat_vecs_a = feature_net(img_input1)
    feat_vecs_b = feature_net(img_input2)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([feat_vecs_a, feat_vecs_b])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    dense1 = Dense(6, activation='relu', use_bias=False)(L1_distance)

    prediction = Dense(1, activation='sigmoid')(dense1)

    resnet = Model(inputs=[img_input1, img_input2], outputs=prediction)

    return resnet


def build_feature_network(input_dim):
    img_input1 = Input(shape=input_dim)

    prep = Conv2D(input_shape=input_dim, filters=8, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, data_format='channels_first')(
        img_input1)
    # prep = BatchNormalization()(prep)
    prep = ReLU()(prep)
    prep = MaxPooling2D((2, 2), data_format='channels_first')(prep)
    prep = ResBlock(prep, 8, strides=1)

    layer1 = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, data_format='channels_first')(prep)
    # layer1 = BatchNormalization()(layer1)
    layer1 = ReLU()(layer1)
    layer1 = MaxPooling2D((2, 2), data_format='channels_first')(layer1)
    layer1 = ResBlock(layer1, 16, strides=1)

    layer2 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, data_format='channels_first')(layer1)
    # layer2 = BatchNormalization()(layer2)
    layer2 = ReLU()(layer2)
    layer2 = MaxPooling2D((2, 2), data_format='channels_first')(layer2)
    layer2 = ResBlock(layer2, 32, strides=1)

    other_layer = MaxPooling2D((4, 4))(layer2)
    other_layer = Flatten()(other_layer)
    outputs = Dense(12, activation='sigmoid', use_bias=False)(other_layer)

    feature_net = Model(inputs=img_input1, outputs=outputs)

    return feature_net


# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    siamese_model = build_res_network((1, 32, 32))
    feature_model = build_feature_network((1, 32, 32))
    siamese_model.summary()
    # feature_model.summary()
    # plot_model(siamese_model, to_file='siamese_model.png', show_shapes=True)
    # plot_model(feature_model, to_file='feature_model.png', show_shapes=True)

