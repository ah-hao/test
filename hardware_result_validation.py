'''
This code is to sigmoid the result value
Written by Jaiwei modified by Darrel (2020/07/29)
'''
import numpy as np
import struct
import glob

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x): 
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def hextobin(num):
    ss=''
    for i in range(len(num)):
        if(num[i]=='0'):
            ss=ss+'0000'
        elif(num[i]=='1'):
            ss=ss+'0001'
        elif(num[i]=='2'):
            ss=ss+'0010'
        elif(num[i]=='3'):
            ss=ss+'0011'
        elif(num[i]=='4'):
            ss=ss+'0100'
        elif(num[i]=='5'):
            ss=ss+'0101'
        elif(num[i]=='6'):
            ss=ss+'0110'
        elif(num[i]=='7'):
            ss=ss+'0111'
        elif(num[i]=='8'):
            ss=ss+'1000'
        elif(num[i]=='9'):
            ss=ss+'1001'
        elif(num[i]=='A' or num[i]=='a'):
            ss=ss+'1010'
        elif(num[i]=='B' or num[i]=='b'):
            ss=ss+'1011'
        elif(num[i]=='C' or num[i]=='c'):
            ss=ss+'1100'
        elif(num[i]=='D' or num[i]=='d'):
            ss=ss+'1101'
        elif(num[i]=='E' or num[i]=='e'):
            ss=ss+'1110'
        elif(num[i]=='F' or num[i]=='f'):
            ss=ss+'1111'
    return ss

def binaryToFloat(value):    
    hx = hex(int(value, 2))   
    return struct.unpack("f", struct.pack("l", int(hx, 16)))[0]

def bintodec(data):
    fg=False
    for i in range(len(data)):
        if data[i][0]=='1':
            data[i]=data[i].replace('1'+data[i][1:],'0'+data[i][1:])
            fg=True
            
        fp=binaryToFloat(data[i])
        if fg==True:
            fp=float('-'+str(fp))
            fg=False
        data[i]=fp
    return data

if __name__ == "__main__":

    files = glob.glob('./hardware_test_data/out_hex_*.txt')

    for file in files:

        f=open(file, "r")
        data=[]

        #讀取檔案
        for i in f:
            data.append(i.replace('\n',""))
        #print(data)

        f.close()

        #將ieee754轉成2進制表示
        for i in range(len(data)):
            data[i]=hextobin(data[i])
        #print(data)


        #將ieee754轉回10進制 且判斷正負值
        bintodec(data) 
        # print(data)
        data = np.array(data)
        print(file)
        # print(data)
        print(sigmoid(data))