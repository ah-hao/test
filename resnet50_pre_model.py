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
from keras.applications.resnet50 import ResNet50


def create_base_model():
    # img_input = Input(shape=input_dim)
    base_model = ResNet50( include_top=False, 
                           weights="imagenet", 
                           input_shape=(32, 32, 3),
                           input_tensor=None,
                           pooling='max')

    # base_model.layers.pop()
    # base_model.output = [base_model.layers[-1].output]
    # base_model.layers[-1].outbound_nodes = []
    base_model.trainable = False


    x = base_model.output
    print(x.shape)
    # x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(rate=0.2)(x)
    x = Dense(18, activation='sigmoid')(x)
    # x = Dropout(rate=0.2)(x)

    # x.summary()

    # x = base_model(img_input, training=False)

    return base_model.input, x


def create_siamese_model():
    input_a, output_a = create_base_model()
    input_b, output_b = create_base_model()

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([output_a, output_b])
    L1_prediction = Dense(1, activation='sigmoid')(L1_distance)

    # prediction = Dropout(0.2)(L1_prediction)

    siamese_model = Model(inputs=[input_a, input_b], outputs=L1_prediction)
    siamese_model.summary()
    return siamese_model


a = create_siamese_model()

a.summary()

# a.summary()
