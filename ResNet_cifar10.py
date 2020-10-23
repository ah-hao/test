from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D
from keras.layers import add
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K

np.random.seed(513)

# load dataset
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# summarize loaded dataset
print('Train: x = %s, y = %s' % (train_x.shape, train_y.shape))
print('Test: x = %s, y = %s' % (test_x.shape, test_y.shape))


# plot first few images
for i in range(9):
    #define subplot
    pyplot.subplot(330 + 1 + i)
    #plot raw pixel data
    pyplot.imshow(train_x[i])

    print(train_y[i])

# show the figure

pyplot.show()

train_x = train_x/255.0
test_x = test_x/255.0

train_y_oneHot = to_categorical(train_y, num_classes = 10)
test_y_oneHot = to_categorical(test_y, num_classes = 10)
# print(train_y_oneHot)


def build_resnet_network(input_dim):

    inputs = Input(shape=input_dim)
    # img_b = Input(shape=input_dim)

    # resnet = Sequential()

    prep = Conv2D(64, (3, 3), padding='same', input_shape=input_dim, data_format=None, activation=None, name='covn_1')(inputs)
    prep = BatchNormalization(name='b1')(prep)
    prep = Activation('relu', name='relu_1')(prep)

    layer_1 = Conv2D(64, (3, 3), padding='same', data_format=None, activation=None, name='covn_2')(prep)
    layer_1 = BatchNormalization(name='b2')(layer_1)
    layer_1 = Activation('relu', name='relu_2')(layer_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), data_format=None, name='pool_1')(layer_1)

    shortcut_1 = Conv2D(32, (1, 1), padding='same', data_format=None, activation=None, name='shortcut_1')(pool_1)
    
    res_1_1 = Conv2D(32, (3, 3), padding='same', data_format=None, activation=None, name='covn_3')(pool_1)
    res_1_1 = BatchNormalization(name='b3')(res_1_1)
    res_1_1 = Activation('relu', name='relu_3')(res_1_1)

    res_1_2 = Conv2D(32, (3, 3), padding='same', data_format=None, activation=None, name='covn_4')(res_1_1)
    res_1_2 = BatchNormalization(name='b4')(res_1_2)
    res_1_2 = Activation('relu', name='relu_4')(res_1_2)

    add_1 =  add([shortcut_1, res_1_2])

    layer_2 = Conv2D(16, (3, 3), padding='same', data_format=None, activation=None, name='covn_5')(add_1)
    layer_2 = BatchNormalization(name='b5')(layer_2)
    layer_2 = Activation('relu', name='relu_5')(layer_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2), data_format=None, name='pool_2')(layer_2)

    shortcut_2 = Conv2D(16, (1, 1), padding='same', data_format=None, activation=None, name='shortcut_2')(pool_2)

    res_2_1 = Conv2D(16, (3, 3), padding='same', data_format=None, activation=None, name='covn_6')(pool_2)
    res_2_1 = BatchNormalization(name='b6')(res_2_1)
    res_2_1 = Activation('relu', name='relu_6')(res_2_1)

    res_2_2 = Conv2D(16, (3, 3), padding='same', data_format=None, activation=None, name='covn_7')(res_2_1)
    res_2_2 = BatchNormalization(name='b7')(res_2_2)
    res_2_2 = Activation('relu', name='relu_7')(res_2_2)

    add_2 =  add([shortcut_2, res_2_2])

    classifier = MaxPooling2D(pool_size=(2, 2), data_format=None, name='pool_3')(add_2)
    classifier = Activation('linear')(classifier)

    scale = Flatten()(classifier)
    scale = Dense(10, activation='sigmoid', use_bias=False)(scale)
    # scale = Dense(1, activation='sigmoid', use_bias=False)(scale)


    resnet = Model(inputs=inputs, outputs=scale)
    
    return resnet


# plot train set accuarcy / loss function value ( determined by what parameter 'train' you pass )
# The type of train_history.history is dictionary (a special data type in Python)



if __name__ == '__main__':
    resnet_model = build_resnet_network(train_x.shape[1:])
    resnet_model.summary()
    resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_history = resnet_model.fit(x=train_x, y=train_y_oneHot, shuffle=True, validation_split=0.12, epochs=50, batch_size=256, verbose=1)
    resnet_model.save('resnet_model.h5')
    # show_train_history(train_history)

    # test_loss, test_acc = resnet_model.evaluate(test_x, test_y_oneHot)
    # print('test loss:', test_loss)
    # print('test acc:', test_acc)
    
    # plt.subplot(211)
    plt.plot(train_history.history['loss'], color='red', label='Training Loss')
    plt.plot(train_history.history['val_loss'], color='blue', label='Validation Loss')
    plt.plot(train_history.history['accuracy'], color='green', label='Training Accuracy')
    plt.plot(train_history.history['val_accuracy'], color='orange', label='Validation Accuracy')
    plt.legend()
    plt.title('Training/Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Train History')

    # plt.subplot(212)  
    # plt.plot(train_history.history['acc'], color='green', label='Training Accuracy')
    # plt.plot(train_history.history['val_acc'], color='orange', label='Validation Accuracy')
    # plt.title('Training/Validation Accuracy')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('Accuracy')
    plt.show()
    
    
        