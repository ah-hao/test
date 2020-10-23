from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, BatchNormalization, AveragePooling2D, Reshape
from keras.layers import Add, concatenate
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras import backend
from sklearn.model_selection import train_test_split
from SiameseResNet_test2 import building_block, create_model, create_CaptureFeature_model
import os
from keras.utils.np_utils import to_categorical

ori_image_path = './data/data_test'

mode = 'one'

anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie', 'chieh', 'sam', 'tim', 'mjun']

len_list = []

total_clusters = 3

# anchor_name = ['chieh', 'fannie', 'jeff', 'jiawei', 'peng', 'donna', 'sam', 'tim', 'mjun'] # new



def mode_function(mode): 

    def OneShot():
        print('model = OneShot Model')

        return OneShot_function()
    
    def FewShot():
        print('model = FewShot Model')
        return FewShot_function()

    switcher = {'one': OneShot, 
                'few': FewShot }
    
    func = switcher.get(mode, "nothing")

    return func()


#---------- Create Relation Model -----------


def create_relation_model(input_dim):

    input_img = Input(input_dim)

    layer1 = Conv2D(12, (3, 3),input_shape=input_dim, data_format='channels_first', padding='same')(input_img)
    layer1 = BatchNormalization(axis=1, momentum=0.9)(layer1)
    layer1 = Activation('relu')(layer1)
    pooling_1 = MaxPooling2D((2, 2), data_format='channels_first')(layer1)

    layer2 = Conv2D(24, (3, 3),input_shape=input_dim, data_format='channels_first', padding='same')(pooling_1)
    layer2 = BatchNormalization(axis=1, momentum=0.9)(layer2)
    layer2 = Activation('relu')(layer2)
    pooling_2 = MaxPooling2D((2, 2), data_format='channels_first')(layer2)

    fc0 = Flatten()(pooling_2)
    # fc1 = Dense(12, activation='relu')(fc0)
    fc2 = Dense(1, activation='sigmoid')(fc0)

    # re = Reshape((6,))(fc2)

    relation_net = Model(inputs=input_img, outputs=fc2)

    relation_net.summary()

    return relation_net


#------------ Load base_model --------------

# origin_siamese_model = load_model('./model_with5/SiameseResnet_with5_model/SiameseResnet_with5_0.93.h5')
origin_siamese_model = load_model('./model_with5/SiameseResnet_mc2_model/SiameseResnet_mc2_stable_.h5')
fix_model = create_CaptureFeature_model((1, 32, 32))
fix_model.set_weights(origin_siamese_model.get_weights())#.layers[3]


#------------ One Shot Labeling -------------

def OneShot_function():

    OneShot_label = []

    for i in os.listdir(ori_image_path):
        for j in os.listdir(ori_image_path + '/' + i):
            check = 0
            for number in range(len(anchor_name)):
                if i == anchor_name[number]:
                    check = 1
                    OneShot_label.append(number)

                elif number == len(anchor_name) and check == 0 :
                    OneShot_label.append(number) ###

    OneShot_label = np.array(OneShot_label)
    OneShot_label_onehot = to_categorical(OneShot_label, num_classes=len(anchor_name))

    print('OneShot_label_onehot:\n', OneShot_label_onehot)

    fix_img = np.load('./npz/fix_image_sum_np.npy') # one shot anchor image
    fix_img = fix_img / 255.0
    print('fix_img.shape:', fix_img.shape)

    return OneShot_label_onehot, fix_img

    
#------------ Few Shot Labeling -------------  有比重問題 比重偏向最後一個dummy

def FewShot_function():

    FewShot_label = []
    label_ = np.load('./npz/save_label.npy')
    print('label=', label_)

    for i in os.listdir(ori_image_path):
        len_ = 0  
        for j in os.listdir(ori_image_path + '/' + i):
            check = 0
            for number in range(len(anchor_name)):
                if i == anchor_name[number]: # while image is anchor or dummy
                    FewShot_label.append((int(label_[ 60 * (number) + len_])) + (number * total_clusters) ) # 在訓練集中，每個人皆60張影像
                    len_ = len_ + 1
                    check = 1
                elif number == (len(anchor_name)-1) and check == 0 : # while image is stranger
                    FewShot_label.append( (len(anchor_name) * total_clusters) - 1 )

    FewShot_label = np.array(FewShot_label)
    FewShot_label_onehot = to_categorical(FewShot_label, num_classes=( len(anchor_name) * total_clusters ))

    print('FewShot_label_onehot', FewShot_label_onehot)

    fix_img = np.load('./npz/fix_img_np.npy') # few shot anchoer image
    fix_img = fix_img / 255.0
    print('fix_img.shape:', fix_img.shape)

    return FewShot_label_onehot, fix_img


#----- Select OneShot or FewShot -----

train_label_onehot, fix_img = mode_function(mode)

train_label_y = np.reshape(train_label_onehot, (-1,1))
print('train_label_y.shape', train_label_y.shape)


#------------ fix_img ------------

fix_feature = fix_model.predict(fix_img) ##
print('fix_feature.shape', fix_feature.shape)

fix_feature = fix_feature[np.newaxis, :,:,:,:] # new axis for repeat


#------------ fix_img_sum3 ------------

# print(int(len(fix_feature) / 3))

# fix_feature_sum = []
# for i in range(int(len(fix_feature) / 3)):#len(fix_feature) / 3
#     fix_feature_sum.append([x + y + z for x, y, z in zip(fix_feature[3 * i], fix_feature[3 * i + 1], fix_feature[3 * i + 2])])

# fix_feature_sum = np.array(fix_feature_sum) / 3.0 ###
# fix_feature_sum = fix_feature_sum[np.newaxis, :,:,:,:]
# print(fix_feature_sum.shape)


#------------ ori_img ------------

ori_img_array = []

for i in os.listdir(ori_image_path): 
    for j in os.listdir(ori_image_path + '/' + i):

        ori_img = cv2.imread(ori_image_path + '/' + i + '/' + j, cv2.IMREAD_UNCHANGED)
        ori_img_array.append(ori_img)


#-------------- Data Preproccessing ----------------

ori_img_np = np.array(ori_img_array, dtype='float32')
ori_img_np = ori_img_np[:, np.newaxis,:,:]
ori_img_np = ori_img_np / 255.0

# print('ori_img_np.shape', ori_img_np.shape)


ori_feature = fix_model.predict(ori_img_np) ##
print('ori_feature.shape', ori_feature.shape)

ori_feature = ori_feature[np.newaxis,:,:,:,:] # new axis for repeat




#-------------- Repeat ----------------

fix_feature_repeat = np.repeat(fix_feature, ori_feature.shape[1], axis=0)

ori_feature_repeat = np.repeat(ori_feature, fix_feature.shape[1], axis=0)
ori_feature_repeat = np.swapaxes(ori_feature_repeat, 0, 1)

# train_label_repeat = np.repeat(train_label_onehot, fix_feature.shape[1], axis=0)

print('fix_feature_repeat.shap', fix_feature_repeat.shape)
print('ori_feature_repeat.shape', ori_feature_repeat.shape)
# print(train_label_repeat.shape)


#------------ Concatenate -------------

cat_feature = np.concatenate([fix_feature_repeat, ori_feature_repeat], axis=2)

# print(cat_feature.shape)

input_feature = np.reshape(cat_feature, (-1, cat_feature.shape[2], cat_feature.shape[3], cat_feature.shape[4]))

print('input_feature.shape', input_feature.shape)


#------------ Train --------------

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


if __name__ == '__main__':

    test = create_relation_model((64, 4, 4))

    test.summary()

    test.compile(loss='mse', optimizer='adam', metrics=['acc'])

    train_history = test.fit(input_feature, train_label_y, batch_size=128, verbose=1, nb_epoch=10, shuffle=True)

    test.save('./model_with5/test_by_test.h5')

    show_train_history(train_history, 'loss')







