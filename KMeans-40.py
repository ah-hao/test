import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, BatchNormalization, AveragePooling2D, Activation, add
from keras import models
import keras
from keras.preprocessing import image
from SiameseResNet_test import  create_CaptureFeature_model #create_kmeans_model, building_block, create_model,
from SiameseResNet_test2 import create_kmeans_model


total_clusters = 3

anchor_path = './data/data_classmate_mc2_'

kmeans_anchor_path = './data/kmeans_anchor'

# anchor name list
anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie', 'chieh', 'sam', 'tim', 'mjun']#, 'johnson'

# anchor_name = ['chieh', 'fannie', 'jeff', 'jiawei', 'peng', 'donna', 'sam', 'tim', 'mjun']# new

len_list = []


save_label = np.zeros(0)

anchor_image = []

real_center_list = []

R_real_center_list = []

fix_image = []



def build_base_network(input_dim):

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

    siamese_net = Model(inputs=img, outputs=feat_vecs)
    
    return siamese_net

def building_block(X, filters, stride=1):

    # if backend.image_data_format() == 'channels_first':
    #     bn_axis = 1
    # else:
    #     bn_axis = 3

    X_shortcut = X

    # if stride > 1:
    X_shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', data_format='channels_first')(X_shortcut)
    X_shortcut = BatchNormalization(axis=1)(X_shortcut)

    # First layer of the block
    X = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(X)
    X = BatchNormalization(axis=1, momentum=0.9)(X)
    X = Activation('relu')(X)

    # Second layer of the block
    X = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', data_format='channels_first')(X)
    X = BatchNormalization(axis=1, momentum=0.9)(X)
    X = add([X, X_shortcut])
    X = Activation('relu')(X)

    # X.summary()

    return X

def create_model(input_dim):


    img_input = Input(shape=input_dim)

    prep = Conv2D(6, (3, 3),input_shape=input_dim, kernel_initializer='he_normal', data_format='channels_first', padding='same')(img_input)
    # prep = BatchNormalization(axis=1, momentum=0.9)(prep)
    prep = Activation('relu')(prep)

    pooling_1 = MaxPooling2D((2, 2), data_format='channels_first')(prep)
    X = building_block(pooling_1, filters=12, stride=1)

    # pooling_2 = MaxPooling2D((2, 2), data_format='channels_first')(X)
    # X = building_block(pooling_2, filters=64, stride=1)

    avg_pool = MaxPooling2D((2, 2), data_format='channels_first')(X)
    feature = Flatten()(avg_pool)
    feature = Dense(12, activation='sigmoid')(feature)

    fc = Model(inputs=img_input, outputs=feature)


    return fc


if __name__ == "__main__":
     
    anchor_image = []

    real_center_list = []



    # siamese = build_base_network((1, 32, 32)) 

    siamese = create_kmeans_model((1, 32, 32))

    # siamese = create_kmeans_model((1, 32, 32))

    CaptureFeature_model = create_CaptureFeature_model((1, 32, 32))

    ############################################################# Load_model ##################################################################

    # origin_model = keras.models.load_model('./model_with5/SiameseLenet_model/siamesenet_with5_stable.h5') # SiameseLenet_model

    # origin_model = keras.models.load_model('./model_with5/SiameseLenet_mc2_model/siamesenet_mc2_stable.h5') # SiameseLenet_model

    # origin_model = keras.models.load_model('./model_with5/SiameseResnet_with5_model/SiameseResnet_with5_0.93.h5') # SiameseResnet_with5_model
    
    origin_model = keras.models.load_model('./model_with5/SiameseResnet_mc2_model/SiameseResnet_mc2_stable_.h5') # SiameseResnet_mc2_model

    ###########################################################################################################################################

    # load_model.layers.pop()
    # load_model.layers.pop()
    # load_model.layers.pop()
    origin_model.summary()

    siamese.set_weights(origin_model.get_weights()) #.layers[3]
    CaptureFeature_model.set_weights(origin_model.layers[3].get_weights()) #capture 

# feature_capture_map = CaptureFeature_model.predict(anchor_image_array)##

    for count in range(len(anchor_name)):
        len_ = 0
        for m in os.listdir(anchor_path + '/' + anchor_name[count]):
            len_ = len_ + 1
            img = cv2.imread(anchor_path + '/' + anchor_name[count] + '/' + m, cv2.IMREAD_UNCHANGED)
            # img = image.load_img(anchor_path+'/'+m, target_size=(32, 32), grayscale=True)
            # x = image.img_to_array(img)
            anchor_image.append(img)

        len_list.append(len_)

        anchor_image_array = np.array(anchor_image, dtype='float32')

        anchor_image_array = anchor_image_array[:, np.newaxis, :, :]  #(?,1,32,32)

        print(anchor_image_array.shape)

        # anchor_rollaxis = np.rollaxis(anchor_image_array, 3, 1)

        anchor_image_array = anchor_image_array/255.0

        # print(anchor_rollaxis.shape[1:])

        feature_map = siamese.predict(anchor_image_array)
        

        print(feature_map.shape)



        # Initialize the K-Means model
        kmeans = MiniBatchKMeans(n_clusters=total_clusters)
        # Fitting the model to training set
        kmeans.fit(feature_map)


        #labels_
        kmeans_labels = kmeans.labels_
        print(kmeans_labels)
        unique, counts = np.unique(kmeans_labels, return_counts=True)
        cluster_dict = dict(zip(unique, counts))
        print(cluster_dict)
        save_label = np.append(save_label, kmeans_labels)


        #cluster_centers_
        cluster_centers = kmeans.cluster_centers_
        print(cluster_centers)

        #pairwise distances argmin min
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feature_map)

        print(closest)

        for i in range(total_clusters):
            real_center_list.append(feature_map[closest[i]])

            cv2.imwrite(kmeans_anchor_path +'/'+ 'kmeans_anchor_' + str(anchor_name[count]) + '_frame_'+str(i)+'.png', anchor_image[closest[i]])
         
        anchor_image = [] # reset anchor_image list 


#------------------------ Save feature ---------------------------
        real_center_np = np.array(real_center_list)
        np.save('./npz/kmeans_np.npy', real_center_np)

        print(real_center_np.shape)


        # k = np.load('./npz/kmeans_np.npy')
        # print(k.shape)
        
        
    '''
        # #show k-means anchor image

        # cv2.imshow('test', anchor_image[closest[0]])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    '''
    print(save_label)
    fix_image_clusters = []
    fix_image_sum = []
#------------------------ fix_image ---------------------------


    for name in range(len(anchor_name)):
        for i in range(total_clusters): #test
            fix_img = cv2.imread(kmeans_anchor_path +'/'+ 'kmeans_anchor_' + str(anchor_name[name]) + '_frame_'+ str(i) +'.png', cv2.IMREAD_UNCHANGED)
            fix_image.append(fix_img)
            fix_image_clusters.append(fix_img)

            if i == (total_clusters-1):
                fix_image_sum.append([(x + y + z) for x, y, z in zip(fix_image_clusters[0], fix_image_clusters[1], fix_image_clusters[2])])
                fix_image_clusters = []
        

    fix_image = np.array(fix_image, dtype='float32')
    fix_image = fix_image[:, np.newaxis, :, :] #(15, 1, 32, 32)
    print(fix_image.shape)

    #----- num_3 -----
    fix_image_sum = np.array(fix_image_sum, dtype='float32') / total_clusters #取平均
    fix_image_sum = fix_image_sum[:, np.newaxis, :, :]
    print(fix_image_sum.shape)
   

    #----- save -----

    np.save('./npz/fix_img_np.npy', fix_image)

    np.save('./npz/fix_image_sum_np.npy', fix_image_sum)

    np.save('./npz/save_label.npy', save_label)

    # np.savez('')

    print(len_list)



