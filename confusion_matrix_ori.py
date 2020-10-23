import glob
import os
import numpy as np
from PIL import Image 
import time
import matplotlib.pyplot as plt
from model import build_base_network, build_reuse_network  
from resnet_face_model import conv_layer, resnet
from SiameseResNet_test2 import building_block, create_model
import cv2
from sklearn.metrics import confusion_matrix
import itertools

allow_path = './data/auc_mc2/allow'
reject_path = './data/auc_mc2/reject'

number_of_centers = 3

anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie']

x = []
y_true = []
y_pred = []

# allow = np.array([np.array(Image.open(allow_img)) for allow_img in allow_imgs]).astype(np.float)
# reject = np.array([np.array(Image.open(reject_img)) for reject_img in reject_imgs]).astype(np.float)

# allow = np.stack((allow,)*1, axis=1)
# reject = np.stack((reject,)*1, axis=1) 

# val_allow_x = allow[:int(len(allow))]
# val_allow_y = np.ones(len(val_allow_x))

# val_reject_x = reject[:int(len(reject))]
# val_reject_y = np.zeros(len(val_reject_x))

# val_x = np.concatenate((val_allow_x, val_reject_x))
# val_y = np.concatenate((val_allow_y, val_reject_y))


# print('val_allow_y:',val_allow_y)

# print('\tval.shape:\t', val_x.shape, val_y.shape)



for allow in os.listdir(allow_path):  # allow
    for i in os.listdir(allow_path + '/' + allow):
        img = cv2.imread(allow_path + '/' + allow + '/' + i, cv2.IMREAD_UNCHANGED)
        x.append(img)
        y_true.append(anchor_name.index(allow))


for reject in os.listdir(reject_path): # reject
    for i in os.listdir(reject_path + '/' + reject):
        img = cv2.imread(reject_path + '/' + reject + '/' + i, cv2.IMREAD_UNCHANGED)
        x.append(img)
        y_true.append(5)
# print(x)
x = np.array(x, dtype='float32') / 255
y_true = np.array(y_true)

print('x.shape:', x.shape)
print('y_true.shape:', y_true.shape)

# LeNet Architecture
# model = build_base_network((1, 32, 32)) 

# Siamese-ResNet Architecture
model = create_model((1, 32, 32))

# model.load_weights('./model_with5/siameseResnet_stable.h5')

###########################################################################################################################################

#Load_weights

# model.load_weights('./model_with5/SiameseLenet_model/siamesenet_with5_stable.h5') # SiameseLenet_model

# model.load_weights('./model_with5/SiameseLenet_mc2_model/siamesenet_mc2_stable.h5') # SiameseLenet_model

# model.load_weights('./model_with5/SiameseResnet_with5_model/SiameseResnet_with5_0.93.h5') # SiameseResnet_with5_model

model.load_weights('./model_with5/SiameseResnet_mc2_model/SiameseResnet_mc2_stable_.h5') # SiameseResnet_mc2_model

###########################################################################################################################################



# Use Reuse
# reuse_model = build_reuse_network([7])
# reuse_model.load_weights('./model/siamesenet_dense_1_mark_li.h5')
print('Generating AUC ROC!')
predy = []
from tqdm import tqdm


def read_label_txt():
    '''
    read the label 
    '''
    map_characters = []
    label_path = './data/kmeans_anchor/'
    read_label = os.listdir(label_path)
    for label in read_label:
        name = label.split('_')[-3]
        map_characters.append(name) 
    return map_characters

map_characters = read_label_txt()


# def img_test(model, image):
# load anchor imgs
imgs_dir_path = './data/kmeans_anchor/'




for i in range(len(x)):
    # i = np.squeeze(i)
    # test base model
    imgs_name = os.listdir(imgs_dir_path)
    # x[i] = x[i] / 255  # normalize in 0 ~ 1
    input_img = x[i][np.newaxis, np.newaxis, :, :]  
    print(input_img.shape)

    # all_score = []  
    all_score = np.zeros(0) 

    anchor_imgs = np.array([np.array(Image.open(anchor_image)) for anchor_image in glob.glob('./data/kmeans_anchor/*.png')]).astype(np.float)

    
    thresholds = 0.6

    for anchor_img in anchor_imgs:
        anchor_img = anchor_img / 255
        anchor_img = anchor_img[np.newaxis, np.newaxis, :, :]     
        score = model.predict([input_img, anchor_img])
        # print('score:', score)
        # all_score = all_score.append(score)
        all_score = np.append(all_score, score)

    # print(all_score)
    # all_score = np.array(all_score)
    argmax = np.argmax(all_score)
    guess = all_score[argmax]
    # print(input_img.shape)


    if map_characters[argmax] == anchor_name[0] and guess >= thresholds :
        y_pred.append(anchor_name.index(map_characters[argmax]))

    elif map_characters[argmax] == anchor_name[1] and guess >= thresholds :
        y_pred.append(anchor_name.index(map_characters[argmax]))

    elif map_characters[argmax] == anchor_name[2] and guess >= thresholds :
        y_pred.append(anchor_name.index(map_characters[argmax]))

    elif map_characters[argmax] == anchor_name[3] and guess >= thresholds :
        y_pred.append(anchor_name.index(map_characters[argmax]))

    elif map_characters[argmax] == anchor_name[4] and guess >= thresholds :
        y_pred.append(anchor_name.index(map_characters[argmax]))

    else:
        y_pred.append(5)

print(y_pred)


#------------ confusion_matrix -----------------


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
anchor_name_all = ['jeff', 'jiawei', 'donna', 'peng', 'fannie', 'dummy']

plt.figure()
cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cnf_matrix, classes=anchor_name_all, normalize=True,
                    title='confusion matrix')

plt.show()