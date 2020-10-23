'''
This code is realtime face detection
Written by Darrel (2020/07/29)
'''
import numpy as np
import time
import cv2
import os
from image_test import img_test, image_pre_processing
# from image_test_resue import read_label_txt, img_test, image_pre_processing
from cv2_haar_face_detection import face_detection, box_reduce
from model import build_base_network, build_reuse_network    

def read_label_txt(number):
    '''
    read the label 
    '''
    map_characters = []
    label_path = './data/anchor_imgs_' + str(number)
    read_label = os.listdir(label_path)
    for label in read_label:
        name = label.split('_')[-3]
        map_characters.append(name)

    return map_characters



if __name__ == "__main__":
    model = build_base_network((1, 32, 32))
    # model.load_weights('./model/siamesenet.h5')
    reuse_model = build_reuse_network([7])
    # reuse_model.load_weights('./model/siamesenet_dense.h5')
    
    # map_characters = read_label_txt()    
    # print(map_characters)

    count_in, count_out = 0, 0
    # camera (input) configuration
    frame_in_w = 640
    frame_in_h = 480
    videoIn = cv2.VideoCapture(0)
    videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w)
    videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h)
    print("capture device is open: " + str(videoIn.isOpened()))

    # from label txt path read label
    # print('map_characters', map_characters)
    
    # write predict result name
    # pd_result_name = ''

    # count_list = [0] * len(map_characters)

    model_path = './model_test'
    
    number = 0

    for m in os.listdir(model_path): ##
        
        number += 1
        map_characters = read_label_txt(number)
        print('map_characters', map_characters)

        print('Start model'+' '+ m)

        pd_result_name = ''
        count_list = [0] * len(map_characters)

        while videoIn.isOpened():
            # read video from camera
            ret, outframe = videoIn.read()
            if (ret):
                # keyboard input value
                key = cv2.waitKey(1) & 0xFF

                success, local = face_detection(outframe)
                if success:
                    local = box_reduce(local)
                    img_32 = image_pre_processing(outframe, local) #./model/siamesenet_'+ str(m)+ '.h5'

                    # predict
                    model.load_weights(os.path.join(model_path, m))   #
                    # print("start" + m)
                    result, guess = img_test(model, img_32)
                    # result, guess = img_test(model, reuse_model, img_32)
                    if result != -1:    # predict success                    
                        print('index:', result)
                        print('score:', guess)
                        print('name:', map_characters[result])
                        count_list[result] += 1
                        threshold = 0.45
                        for i in count_list:
                            if i > 4 and guess > threshold:      # if one person is discriminateed 50 times
                                pd_result_name = map_characters[result]     # it will show his/her name
                            
                            if i > 8 and guess > threshold:     
                                count_list = [0] * len(map_characters)

                            if i > 5 and guess < threshold:
                                pd_result_name = 'unknown'     
                                count_list = [0] * len(map_characters)

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
                

                cv2.putText(outframe, 'Test' + ' ' + str(number), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow('show', outframe)

                # cv2.waitKey(10000) 

                if key == ord('q'):     # press 'q' to leave while
                    break
            
            else:
                raise RuntimeError("Error while reading from camera.")

    print('Video Capture end, release camera.')
    videoIn.release()
    cv2.destroyAllWindows()
