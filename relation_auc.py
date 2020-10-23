from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from keras.models import load_model
import sklearn.metrics as metrics
from SiameseResNet_test2 import building_block, create_model, create_CaptureFeature_model
from sklearn.preprocessing import label_binarize
import time

allow_path = './data/auc_mc2/allow'
reject_path = './data/auc_mc2/reject'



anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie']
# anchor_name = ['chieh', 'jeff', 'jiawei', 'donna', 'peng']
nb_classes = len(anchor_name) + 1

y_true = []
y_pred = []


#----------- create y_true -------------

for allow in os.listdir(allow_path):  # allow
    for i in os.listdir(allow_path + '/' + allow):
        y_true.append(anchor_name.index(allow))


for reject in os.listdir(reject_path): # reject
    for i in os.listdir(reject_path + '/' + reject):
        y_true.append(5)

y_true = np.array(y_true)
Y_true = label_binarize(y_true, classes=[i for i in range(nb_classes)])

print('k=', y_true.shape)

#----------- create y_pred -------------

ori_img_array = []

origin_model = load_model('./model_with5/SiameseResnet_mc2_model/SiameseResnet_mc2_stable_.h5')

fix_model = create_CaptureFeature_model((1, 32, 32))
fix_model.set_weights(origin_model.get_weights()) #.layers[3]
relation_model = load_model('./model_with5/test_by_test.h5') 

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

print(score)


for i in range(len(score)):
    result = np.argmax(score[i])
    guess = score[i][result]
    if result >= len(anchor_name):
        y_pred.append(len(anchor_name))
    
    else:
        y_pred.append(result)

# print(y_pred)
y_pred = np.array(y_pred)
Y_pred = label_binarize(y_pred, classes=[i for i in range(nb_classes)])

#------
fpr, tpr, thresholds = metrics.roc_curve(Y_true.ravel(), Y_pred.ravel())
auc = metrics.auc(fpr, tpr)
print('fpr:', fpr)
print('tpr:', tpr)

timestamp = time.strftime('_%m%d%H%M%S', time.localtime())
txt = open('./auc/roc_curve'+ timestamp +'.txt', 'w')
txt.write('AUC: %.3f' % auc)
print('AUC: %.3f' % auc)
for i in range(len(thresholds)):
    txt.write('\nthresholds:' + str(thresholds[i]))
    txt.write('\nfpr:' + str(fpr[i]))
    txt.write('\ntpr:' + str(tpr[i]))
txt.write('\ncount:' + str(len(Y_true)))
txt.write('\nthresholds count:' + str(len(thresholds)))
txt.close() 

# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('./auc/roc_curve'+ timestamp +'.png')
plt.show()  

    

