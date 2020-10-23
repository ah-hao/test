from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
# from keras import backend as K
from keras import backend
from sklearn.model_selection import train_test_split

from resnet_cifar_model import conv_layer, resnet

np.random.seed(666)

# load dataset
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# summarize loaded dataset
print('Train: x = %s, y = %s' % (train_x.shape, train_y.shape))
print('Test: x = %s, y = %s' % (test_x.shape, test_y.shape))


train_datagen = ImageDataGenerator(horizontal_flip=True,
                                   rotation_range=0, 
                                   width_shift_range=0.15, 
                                   height_shift_range=0.15,
                                   shear_range=0,
                                   zoom_range=0, 
                                   data_format='channels_last',
                                   fill_mode='constant',
                                   cval=0.)


lr_function = ReduceLROnPlateau(monitor='val_accuracy',
                                patience=4,
                                verbose=1,
                                factor=0.75,
                                min_lr=0.00001)
                                                       

def feature_normalize(train_data):
    # global mean, std
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)

    return np.nan_to_num((train_data - mean) / (std + 1e-7))


# train_x = train_x.astype('float32')/255.0
# test_x = test_x.astype('float32')/255.0

train_x = feature_normalize(train_x)
test_x = feature_normalize(test_x)


train_y_oneHot = to_categorical(train_y, num_classes = 10)
test_y_oneHot = to_categorical(test_y, num_classes = 10)

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y_oneHot, test_size=0.2, random_state=7)

print(train_x.shape)
print(valid_x.shape)

train_datagen.fit(train_x)



if __name__ == '__main__':
    resnet_model = resnet(train_x.shape[1:])
    resnet_model.summary()
    resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_history = resnet_model.fit_generator(train_datagen.flow(train_x, train_y, batch_size=128), 
                                               steps_per_epoch=train_x.shape[0]/128, 
                                               epochs=100, 
                                               validation_data=(valid_x, valid_y), 
                                               validation_steps=valid_x.shape[0]/128, 
                                               callbacks=[lr_function])

    resnet_model.save('./resnet_model/resnet_model.h5')


    test_loss, test_acc = resnet_model.evaluate(test_x, test_y_oneHot)
    print('test loss:', test_loss)
    print('test acc:', test_acc)


    #matplotlib loss & accuracy
    plt.subplot(211)
    plt.plot(train_history.history['loss'], color='red', label='Training Loss')
    plt.plot(train_history.history['val_loss'], color='blue', label='Validation Loss')
    plt.legend()
    plt.title('Training/Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title('Train History')

    plt.subplot(212)  
    plt.plot(train_history.history['accuracy'], color='green', label='Training Accuracy')
    plt.plot(train_history.history['val_accuracy'], color='orange', label='Validation Accuracy')
    plt.title('Training/Validation Accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')

    plt.show()
    

    # # plot first few images
    # for i in range(9):
    #     #define subplot
    #     pyplot.subplot(330 + 1 + i)
    #     #plot raw pixel data
    #     pyplot.imshow(train_x[i])

    #     print(train_y[i])

    # # show the figure

    # pyplot.show()
    
        