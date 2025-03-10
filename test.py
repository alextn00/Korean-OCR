# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:48:33 2020

@author: 원기
"""
import numpy as np
import json
import matplotlib.pylab as plt
import sys
# sys.path.append('C:/Users/원기/Korean/')
from PIL import Image
from keras import models, layers
import cv2

def test_Image(addr,table):
    T_Image = cv2.imread(addr)
    T_Image = cv2.resize(T_Image,(32, 32))
    # T_Image = Image.open(addr)
    # T_Image = T_Image.resize((32,32))
    T_Image_Array = np.array(T_Image,'uint8')
    plt.imshow(T_Image_Array)
    T_Image_Array = T_Image_Array.reshape(1,32,32,3)
    a = CNN.predict(T_Image_Array)
    b = np.argmax(a,axis=1)
    print('예측한 단어: ',table[str(b[0])])

CNN = models.load_model('D:/hanwriting_data/final_data/Korean-OCR-Model-Design-based-on-Keras-CNN/model/Korean_CNN_model(97.8).h5')
with open('D:/hanwriting_data/final_data/Korean-OCR-Model-Design-based-on-Keras-CNN/model/index_to_syllable(97.8).json','r',encoding='utf-8') as f:
    index_to_syllable = json.load(f)
CNN.summary()

T_Image_addr = 'D:/hanwriting_data/final_data/Korean-OCR-Model-Design-based-on-Keras-CNN/test6.png'
test_Image(T_Image_addr,index_to_syllable)
