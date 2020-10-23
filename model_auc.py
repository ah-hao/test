'''
This code is to benchmark the siamese model
Written by Darrel (2020/07/29)
'''
import glob

allow_imgs = glob.glob('./data/auc_mc2/allow/*/*.png')
reject_imgs = glob.glob('./data/auc_mc2/reject/*/*.png')

import numpy as np
from PIL import Image 
import time

allow = np.array([np.array(Image.open(allow_img)) for allow_img in allow_imgs]).astype(np.float)
reject = np.array([np.array(Image.open(reject_img)) for reject_img in reject_imgs]).astype(np.float)

allow = np.stack((allow,)*1, axis=1)
reject = np.stack((reject,)*1, axis=1) 

val_allow_x = allow[:int(len(allow))]
val_allow_y = np.ones(len(val_allow_x))

val_reject_x = reject[:int(len(reject))]
val_reject_y = np.zeros(len(val_reject_x))


print('\tval_allow.ahape:\t', val_allow_x.shape, val_allow_y.shape)
print('\tval_reject.shape:\t', val_reject_x.shape, val_reject_y.shape)


val_x = np.concatenate((val_allow_x, val_reject_x))
val_y = np.concatenate((val_allow_y, val_reject_y))

print('\tval.shape:\t', val_x.shape, val_y.shape)

import os
def check_output_dir(out_dir_list):
    '''
    Check output dir
    '''
    if not os.path.isdir(out_dir_list):
        os.mkdir(out_dir_list)

# using first time training model
from image_test import img_test

# using reusable model
# from image_test_resue import img_test

from model import build_base_network, build_reuse_network  
from resnet_face_model import conv_layer, resnet
from SiameseResNet_test2 import building_block, create_model

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


for i in tqdm(val_x):
    i = np.squeeze(i)
    # test base model
    argmax, guess = img_test(model, i) 

    # test reuse model
    # argmax, guess = img_test(model, reuse_model, i)    
    predy += list(guess.ravel())
    # time.sleep(0.1)


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import time    
timestamp = time.strftime('_%m%d%H%M%S', time.localtime())
check_output_dir('./auc')
txt = open('./auc/roc_curve'+ timestamp +'.txt', 'w')
auc = roc_auc_score(val_y, predy)
txt.write('AUC: %.3f' % auc)
print('AUC: %.3f' % auc)
fpr, tpr, thresholds = roc_curve(val_y, predy)
for i in range(len(thresholds)):
    txt.write('\nthresholds:' + str(thresholds[i]))
    txt.write('\nfpr:' + str(fpr[i]))
    txt.write('\ntpr:' + str(tpr[i]))
txt.write('\ncount:' + str(len(val_y)))
txt.write('\nthresholds count:' + str(len(thresholds)))
txt.close()    
    
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('./auc/roc_curve'+ timestamp +'.png')
plt.show()   