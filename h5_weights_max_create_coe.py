import struct
import numpy as np
import os
from decimal import *

parent_path = "./"  # 專案所在的位置
model_name = "7p09061104[P7_adj]"
dec_dir = "dec_parameter"  # 存放10進制的參數
hex_dir = "hex_parameter"  # 存放16進制的參數
F6_file = "sequential_1_f5__kernel_0.txt"
person_num = 7

def float_to_bin(f):
    return bin(struct.unpack('<I', struct.pack('<f', f))[0])

def create_coe(input_location, output_location, num):
    F6_weights = open(input_location + '/f6_rom' + str(num+1) + '.txt', 'r')
    lines = F6_weights.readlines()
    F6_weights.close()
    F6_weights_list = []
    for i in range(len(lines)):
        #print(len(lines))
        #print(lines[i])
        F6_weights_list.append(lines[i].strip('\n'))

    print(F6_weights_list)

    output_mif = open(output_location + '/f6_rom' + str(num+1) + '.coe', 'w')
    DEPTH = 300*person_num/4           # The size of memory in words
    WIDTH = 32               # The size of every data in bit
    memory_initialization_radix = 2   #The radix for data values

    output_mif.write('memory_initialization_radix = ' + str(memory_initialization_radix) + ';\n')
    output_mif.write('memory_initialization_vector = \n')

    for i in range(int(DEPTH)):
        if i == 0:
            output_mif.write(str(F6_weights_list[i]))
        else:
            output_mif.write(', ' + str(F6_weights_list[i]) )

    output_mif.write(';\n')
    output_mif.close()

def read_dense_weights(file_name, output_location):
    dense_weights = open(file_name, 'r')
    lines = dense_weights.readlines()
    dense_weights.close()
    dense = []
    for i in range(len(lines)):
        if i == 0:
            dense_shape = lines[i]
        elif i % 2 == 1:
            dimension = lines[i]
        else:
            dense.append(lines[i].split())
    for i in range(300):
        for j in range(person_num):
            dense[i][j] = float_to_bin(Decimal(dense[i][j]))[2:34]

        #print(dense[i])

    #==========================rom1==========================
    rom1_txt = open(output_location + '/f6_rom1.txt', 'w')
    for i in range(75):
        for j in range(0, person_num, 1):
            if len(dense[i][j]) >= 32:
                rom1_txt.write(str(dense[i][j])[:32] + '\n')
            else:
                for k in range(0, 32-len(dense[i][j])):
                    rom1_txt.write('0')
                rom1_txt.write(str(dense[i][j]) + '\n')

    rom1_txt.close()

    #==========================rom2==========================
    rom2_txt = open(output_location + '/f6_rom2.txt', 'w')
    for i in range(75, 150):
        for j in range(0, person_num, 1):
            if len(dense[i][j]) >= 32:
                rom2_txt.write(str(dense[i][j])[:32] + '\n')
            else:
                for k in range(0, 32-len(dense[i][j])):
                    rom2_txt.write('0')
                rom2_txt.write(str(dense[i][j]) + '\n')
    rom2_txt.close()

    #==========================rom3==========================
    rom3_txt = open(output_location + '/f6_rom3.txt', 'w')
    for i in range(150, 225):
        for j in range(0, person_num, 1):
            if len(dense[i][j]) >= 32:
                rom3_txt.write(str(dense[i][j])[:32] + '\n')
            else:
                for k in range(0, 32 - len(dense[i][j])):
                    rom3_txt.write('0')
                rom3_txt.write(str(dense[i][j]) + '\n')
    rom3_txt.close()

    #==========================rom4==========================
    rom4_txt = open(output_location + '/f6_rom4.txt', 'w')
    for i in range(225, 300):
        for j in range(0, person_num, 1):
            if len(dense[i][j]) >= 32:
                rom4_txt.write(str(dense[i][j])[:32] + '\n')
            else:
                for k in range(0, 32 - len(dense[i][j])):
                    rom4_txt.write('0')
                rom4_txt.write(str(dense[i][j]) + '\n')
    rom4_txt.close()


if __name__ == "__main__":

    F6_dec_path = './model/siamesenet_darrel_judy_1000.h5_weights/f5_max.txt'

    make_dir_bin = './model/siamesenet_darrel_judy_1000.h5_weights/max_dense_txt'
    if not os.path.exists(make_dir_bin):
        os.makedirs(make_dir_bin)

    make_dir_coe = './model/siamesenet_darrel_judy_1000.h5_weights/max_coe'
    if not os.path.exists(make_dir_coe):
        os.makedirs(make_dir_coe)
    #create_mif()
    read_dense_weights(F6_dec_path, make_dir_bin)

    for i in range(4):
        create_coe(make_dir_bin, make_dir_coe, i)
