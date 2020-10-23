'''
This code is for the model weight to Max vivado coe file.
Written by Darrel (2020/07/29) 
'''
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import os
def check_output_dir(out_dir_list):
    '''確認路徑資料夾'''
    if not os.path.isdir(out_dir_list):
        os.mkdir(out_dir_list)

def model_structure_list(model):
    '''讀取模型內部結構'''
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    for name, weight in zip(names, weights):        
        print(name, weight.shape)
        # print(weight)

import struct
def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])[2:]

def conv2d_to_verilog(model, input_layer_name):
    '''將卷積層參數提出並轉成verilog寫法'''
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    check_output_dir(output_dir)
    txt_kernel = open(output_dir + input_layer_name + '_kernel_verilog.txt', 'w')
    txt_kernel_dec = open(output_dir + input_layer_name + '_kernel_dec.txt', 'w')
    txt_bias = open(output_dir + input_layer_name + '_bias_verilog.txt', 'w')
    txt_bias_dec = open(output_dir + input_layer_name + '_bias_dec.txt', 'w')
    # 進入模型
    for name, weight in zip(names, weights):
        # 進入卷積層kernel
        if input_layer_name in name and '/kernel:0' in name:
            # (5, 5, 6, 12) ==> (12, 5, 5, 6)
            weight = np.rollaxis(weight, 3, 0)  
            # (12, 5, 5, 6) ==> (6, 12, 5, 5)
            weight = np.rollaxis(weight, 3, 0)  
            txt_kernel.write(str(name) + str(weight.shape) + '\n')
            txt_kernel_dec.write(str(name) + str(weight.shape) + '\n')
            # 進入維度一('6', 12, 5, 5)
            for dim1 in range(len(weight)):
                # 進入維度二(6, '12', 5, 5)
                for dim2 in range(len(weight[dim1])):
                    txt_kernel.write('parameter bit [31:0] w' + str(dim1) + '_' + str(dim2) + ' [24:0] = \'{')
                    txt_kernel_dec.write(str(dim1) + '_' + str(dim2) + '\n')
                    kernel_size_1, kernel_size_2 = weight[dim1][dim2].shape
                    # 進入維度三(6, 12, '5', 5),此(5, 5)矩陣為我們所需要的 5x5 kernel參數
                    for dim3 in range(len(weight[dim1][dim2][:][:])):
                        for dim4 in range(len(weight[dim1][dim2][dim3][:])):
                            if dim3 == kernel_size_1 - 1 and dim4 == kernel_size_2 - 1:
                                txt_kernel.write('32\'h' + float_to_hex(weight[dim1][dim2][dim3][dim4]) + '};')
                            else:    
                                txt_kernel.write('32\'h' + float_to_hex(weight[dim1][dim2][dim3][dim4]) + ',')
                            txt_kernel_dec.write(str((weight[dim1][dim2][dim3][dim4])) + ',')
                        txt_kernel_dec.write('\n')
                    txt_kernel.write('\n')
        # 進入卷積層bias
        if input_layer_name in name and '/bias:0' in name:
            txt_bias.write(str(name) + str(weight.shape) + '\n')
            txt_bias_dec.write(str(name) + str(weight.shape) + '\n')
            # 依序提出參數
            for bias_num in range(len(weight)):
                txt_bias.write('parameter b' + str(bias_num) + ' = 32\'h' + float_to_hex(weight[bias_num]) + ';\n')
                txt_bias_dec.write(str(weight[bias_num]) + '\n')

    txt_kernel.close()
    txt_kernel_dec.close()
    txt_bias.close()
    txt_bias_dec.close()

def float_to_binary(f):
    return format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')

def dense_to_verilog(model, input_name):
    '''將全連接層參數提出並轉成verilog寫法'''
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    check_output_dir(output_dir)
    max_kernel = open(output_dir + input_name + '_max.txt', 'w')
    coe_kernel = open(output_dir + input_name + '.coe', 'w')
    mif_kernel = open(output_dir + input_name + '.mif', 'w')
    txt_kernel_dec = open(output_dir + input_name + '_kernel_dec.txt', 'w')
    txt_bias = open(output_dir + input_name + '_bias_verilog.txt', 'w')
    txt_bias_dec = open(output_dir + input_name + '_bias_dec.txt', 'w')
    # 進入模型
    for name, weight in zip(names, weights): 
        # 進入全連接層kernel
        if input_name in name and '/kernel:0' in name:
            txt_kernel_dec.write(str(name) + str(weight.shape) + '\n')
            txt_kernel_dec.write(str(weight))
            # 依序放入全連接層參數即可
            max_kernel.write(str(weight.shape) + '\n')
            for dim1_num in range(len(weight)):
                max_kernel.write('DIMENSION:(' + str(dim1_num) + ')\n')
                for dim2_num in range(len(weight[dim1_num])):
                   max_kernel.write(str(weight[dim1_num][dim2_num]) + ' ')
                max_kernel.write('\n')
            weight_list = weight.flatten().tolist()
            mif_kernel.write('DEPTH = ' + str(len(weight_list)) + ';\nWIDTH = 32;\nADDRESS_RADIX = HEX;\nDATA_RADIX = BIN;\nCONTENT\nBEGIN\n')
            coe_kernel.write('memory_initialization_radix = 2;\nmemory_initialization_vector = \n')
            for index in range(len(weight_list)):
                mif_kernel.write(str(f"{index:X}") + ' : ' + float_to_binary(weight_list[index]) + ';\n')
                if index == range(len(weight_list))[-1]:
                    coe_kernel.write(float_to_binary(weight_list[index]) + ';')
                else:                    
                    coe_kernel.write(float_to_binary(weight_list[index]) + ', ')
        # 進入全連接層bias        
        if input_name in name and '/bias:0' in name:
            txt_bias.write(str(name) + str(weight.shape) + '\n')
            txt_bias_dec.write(str(name) + str(weight.shape) + '\n')
            # 依序提出參數
            for c1_b_num in range(len(weight)):
                txt_bias.write('\t\t' + str(c1_b_num + 1) + ' : bias_out = 32\'h' + float_to_hex(weight[c1_b_num]) + ';\n')
                txt_bias_dec.write(str(weight[c1_b_num]) + '\n') 
    mif_kernel.write('END;')
    mif_kernel.close()
    coe_kernel.close()
    max_kernel.close()
    txt_kernel_dec.close()
    txt_kernel_dec.close()
    txt_bias_dec.close()

import keras
if __name__ == "__main__":
    model_path = './model/siamesenet_darrel_judy_1000.h5'
    model = keras.models.load_model(model_path)
    output_dir = './'+ model_path +'_weights/'
    model.summary()
    model_structure_list(model)
    conv2d_to_verilog(model, 'c1')
    conv2d_to_verilog(model, 'c3')
    dense_to_verilog(model, 'f5')

    # model_path = './model/siamesenet_dense.h5'
    # model = keras.models.load_model(model_path)
    # output_dir = './'+ model_path +'_weights/'
    # model.summary()
    # model_structure_list(model)
    # dense_to_verilog(model, 'f5')