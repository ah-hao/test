'''
This code is realtime face detection
Written by Darrel (2020/07/29)
'''
import numpy as np
import time
import cv2
import os
from keras.models import load_model
from image_test import img_test, image_pre_processing
# from image_test_resue import read_label_txt, img_test, image_pre_processing
from cv2_haar_face_detection import face_detection, box_reduce
from model import build_base_network, build_reuse_network  
from resnet_face_model import conv_layer, resnet 
# from SiameseResNet_test import building_block, create_model
from SiameseResNet_test2 import building_block, create_model, create_CaptureFeature_model


# anchor_name = ['chieh', 'fannie', 'jeff', 'jiawei', 'peng', 'dummy', 'dummy', 'dummy', 'dummy']
anchor_name = ['jeff', 'jiawei', 'donna', 'peng', 'fannie', 'dummy', 'dummy', 'dummy', 'dummy']

# def read_label_txt(number):
#     '''
#     read the label 
#     '''
#     map_characters = []
#     label_path = './data/anchor_imgs_' + str(number)
#     read_label = os.listdir(label_path)
#     for label in read_label:
#         name = label.split('_')[-3]
#         map_characters.append(name)

#     return map_characters

def get_relation_input(model, fix_feature, image):

    image = np.array(image, dtype='float32') #** need finish


if __name__ == "__main__":


    ori_model = load_model('./model_with5/SiameseResnet_mc2_model/SiameseResnet_mc2_stable_.h5')

    fix_model = create_CaptureFeature_model((1, 32, 32))
    fix_model.set_weights(ori_model.get_weights()) 

    relation_model = load_model('./model_with5/test_old.h5')

    #-------- fix_image ----------

    fix_img = np.load('./npz/fix_image_sum_np.npy')
    fix_img = fix_img / 255.0

    fix_feature = fix_model.predict(fix_img) ##
    fix_feature = fix_feature[np.newaxis, :,:,:,:]

    #-----------------------------


    count_in, count_out = 0, 0
    # camera (input) configuration
    frame_in_w = 640
    frame_in_h = 480
    videoIn = cv2.VideoCapture(0)
    videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w)
    videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h)
    print("capture device is open: " + str(videoIn.isOpened()))

    # from label txt path read label
    print('anchor_name', anchor_name)
    
    # write predict result name
    pd_result_name = ''

    count_list = [0] * len(anchor_name) ###

    while videoIn.isOpened():
        # read video from camera
        ret, outframe = videoIn.read()
        if (ret):
            # keyboard input value
            key = cv2.waitKey(1) & 0xFF

            success, local = face_detection(outframe)
            if success:
                local = box_reduce(local)
                img_32 = image_pre_processing(outframe, local)

                #--------------- predict ----------------

                img_32_np = np.array(img_32, dtype='float32')
                img_32_np = img_32_np[np.newaxis, np.newaxis, :, : ] / 255.0
                img_feature = fix_model.predict(img_32_np)
                img_feature = img_feature[np.newaxis, :, :, :, :]

                fix_feature_repeat = np.repeat(fix_feature, img_feature.shape[1], axis=0)

                img_feature_repeat = np.repeat(img_feature, fix_feature.shape[1], axis=0)
                img_feature_repeat = np.swapaxes(img_feature_repeat, 0, 1)

                cat_feature = np.concatenate([fix_feature_repeat, img_feature_repeat], axis=2)

                input_feature = np.reshape(cat_feature, (-1, 64, 4, 4))


                score = relation_model.predict(input_feature) ##
                score = np.reshape(score, (-1, len(anchor_name)))

                result = np.argmax(score)
                guess = score[0][result]
                # pd_result_name = anchor_name[result]

                if result != -1:    # predict success                    
                    print('index:', result)
                    print('score:', guess)
                    print('name:', anchor_name[result])
                    count_list[result] += 1
                    threshold = 0.5
                    for i in count_list:
                        if i > 3 and guess > threshold:      # if one person is discriminateed 50 times
                            pd_result_name = anchor_name[result]     # it will show his/her name
                        
                        if i > 8 and guess > threshold:     
                            count_list = [0] * len(anchor_name)

                        if i > 5 and guess < threshold:
                            pd_result_name = 'unknown'     
                            count_list = [0] * len(anchor_name)

                    # save image
                    out_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                    file_dir = './realtime_output/' + out_time + '.png'
                    cv2.imwrite(file_dir, outframe)
                else:   # predict false
                    pd_result_name = ''

                # draw on moniter
                cv2.rectangle(
                    outframe, (local[0], local[1]), (local[2], local[3]), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(outframe, pd_result_name, (local[0], local[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                pd_result_name = ''

            cv2.imshow('show', outframe)

            if key == ord('q'):     # press 'q' to leave while
                break
        
        else:
            raise RuntimeError("Error while reading from camera.")

    print('Video Capture end, release camera.')
    videoIn.release()
    cv2.destroyAllWindows()
