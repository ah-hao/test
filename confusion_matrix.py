from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from keras.models import load_model
from SiameseResNet_test2 import building_block, create_model, create_CaptureFeature_model
import itertools

allow_path = './data/auc_mc2/allow'
reject_path = './data/auc_mc2/reject'



# anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie']
# anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie']
anchor_name = ['chieh', 'fannie', 'jeff', 'jiawei', 'peng']# new


y_true = []
y_pred = []


#----------- create y_true -------------

for allow in os.listdir(allow_path):  # allow
    for i in os.listdir(allow_path + '/' + allow):
        y_true.append(anchor_name.index(allow))


for reject in os.listdir(reject_path): # reject
    for i in os.listdir(reject_path + '/' + reject):
        y_true.append(5)
        
# print(y_true)

#----------- create y_pred -------------

ori_img_array = []

origin_model = load_model('./model_with5/SiameseResnet_mc2_model/SiameseResnet_mc2_stable_.h5')

fix_model = create_CaptureFeature_model((1, 32, 32))
fix_model.set_weights(origin_model.get_weights()) #.layers[3]
relation_model = load_model('./model_with5/test_old.h5') 

#--- fix_feature ---

fix_img = np.load('./npz/fix_image_sum_np.npy')
fix_img = fix_img / 255.0

fix_feature = fix_model.predict(fix_img) 
fix_feature = fix_feature[np.newaxis, :,:,:,:]
print(fix_feature.shape)

print(len(fix_img))


#--- ori_feature ---

for i in os.listdir(allow_path):
    for j in os.listdir(allow_path + '/' + i):
        ori_img = cv2.imread(allow_path + '/' + i + '/' + j, cv2.IMREAD_UNCHANGED)
        ori_img_array.append(ori_img)

for i in os.listdir(reject_path):
    for j in os.listdir(reject_path + '/' + i):
        ori_img = cv2.imread(reject_path + '/' + i + '/' + j, cv2.IMREAD_UNCHANGED)
        ori_img_array.append(ori_img)

ori_img_np = np.array(ori_img_array, dtype='float32')
ori_img_np = ori_img_np[:, np.newaxis,:,:]
ori_img_np = ori_img_np / 255.0

print(ori_img_np.shape)

ori_feature = fix_model.predict(ori_img_np)

ori_feature = ori_feature[np.newaxis,:,:,:,:]

print(ori_feature.shape)

#--- repeat ---

fix_feature_repeat = np.repeat(fix_feature, ori_feature.shape[1], axis=0)

ori_feature_repeat = np.repeat(ori_feature, fix_feature.shape[1], axis=0)
ori_feature_repeat = np.swapaxes(ori_feature_repeat, 0, 1) # 轉置

print(fix_feature_repeat.shape)
print(ori_feature_repeat.shape)

#--- concatenate ---

cat_feature = np.concatenate([fix_feature_repeat, ori_feature_repeat], axis=2)

input_feature = np.reshape(cat_feature, (-1, 64, 4, 4))

print(input_feature.shape)

score = relation_model.predict(input_feature)
score = np.reshape(score, (-1, len(fix_img)))


thresholds = 0.3

for i in range(len(score)):
    result = np.argmax(score[i])
    guess = score[i][result]
    if result < len(anchor_name) and guess >= thresholds  :
        y_pred.append(result)
    
    else:  
        y_pred.append(len(anchor_name))
# print(y_pred)


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
    
anchor_name_all = ['chieh', 'fannie', 'jeff', 'jiawei', 'peng', 'dummy'] 

plt.figure()
cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cnf_matrix, classes=anchor_name_all, normalize=True,
                    title='confusion matrix')

plt.show()

