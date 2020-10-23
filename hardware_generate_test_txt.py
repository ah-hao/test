'''
This code is to transform the picture pixel to value
Written by Darrel (2020/07/29)
'''
import struct
def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])[2:]

import os
def check_output_dir(out_dir_list):
    '''
    Check output dir
    '''
    if not os.path.isdir(out_dir_list):
        os.mkdir(out_dir_list)

import cv2
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from model import build_base_network
from keras.models import Model

if __name__ == "__main__":
    output_dir = './hardware_test_data'
    check_output_dir(output_dir)
    img_path = './data/anchor_imgs/D20200324_144436_GN_10_darrel_frame_0.png'
    image = cv2.imread(img_path, 0)/255

    # txt for rtl simulation test file (input_txt.txt)
    input_hex_txt = open(output_dir +'/input_hex.txt', 'w')
    # txt for fpga test file 
    c_txt = open(output_dir + '/c_input_hex.txt', 'w')
    c_txt.write('{')
    for image_pixel_x in range(len(image)):
        c_txt.write('{')
        for image_pixel_y in range(len(image[image_pixel_x])):
            image_pixel_hex = float_to_hex(image[image_pixel_x][image_pixel_y])
            if image_pixel_hex == '0':
                image_pixel_hex == '00000000'
            input_hex_txt.write(image_pixel_hex + '\n')
            
            if image_pixel_x == range(len(image[image_pixel_x]))[-1] and image_pixel_y == range(len(image[image_pixel_y]))[-1]:
                c_txt.write('0x' + image_pixel_hex + '}};')
            elif image_pixel_y == range(len(image[image_pixel_y]))[-1]:
                c_txt.write('0x' + image_pixel_hex + '},\n')
            else:
                c_txt.write('0x' + image_pixel_hex + ',')
    # close txt
    c_txt.close()
    input_hex_txt.close()    
    # load mdoel
    model = build_base_network((1, 32, 32))
    model.load_weights('./model/siamesenet.h5')
    image = image[np.newaxis, np.newaxis, :, :]  # shape(1, 1, 32, 32,)        
    try:
        layer_model = Model(inputs=model.input, outputs=model.get_layer('sequential_1').get_output_at(1))
    except:
        layer_model = Model(inputs=model.input, outputs=model.get_layer('sequential_1').get_output_at(2))
    layer_output = layer_model.predict([image, image])
    print('predict image:', img_path)
    print('\npredict result:', layer_output)
    txt_result = open(output_dir + '/input_hex_result.txt', 'w')
    for i in layer_output.ravel():
        txt_result.write(str(i) + '\n')
    txt_result.close()