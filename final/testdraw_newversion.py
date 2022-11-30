import cv2
import numpy as np
import json
import matplotlib.pylab as plt
from tkinter import *
# sys.path.append('C:/Users/원기/Korean/')
from keras import models, layers
from googletrans import Translator
import os

r=Tk()
run = False

def draw(event, x, y, flag, param):
	global run

	if event == cv2.EVENT_LBUTTONDOWN:
		run = True
		cv2.circle(win, (x,y), 10 , (0,0,0), -1)

	if event == cv2.EVENT_LBUTTONUP:
		run = False

	if event == cv2.EVENT_MOUSEMOVE:
		if run == True:
			cv2.circle(win, (x,y), 10 , (0,0,0), -1)


CNN = models.load_model('model/KoreanOCR_ver2.h5')

cv2.namedWindow('window')
cv2.setMouseCallback('window', draw)

win = np.ones((500,500,3), dtype='float64')*255

word =[]

while True:

	
	cv2.imshow('window', win)
	

	k = cv2.waitKey(1)

	if k == ord('c'):
		win = np.ones((500,500,3), dtype='float64') *255

	if k == ord('\n') or k == ord('\r'):
		path = "final_dataset/split_data/train/"
		class_list = os.listdir(path)
		image = cv2.resize(win, dsize=(32, 32))
		image = np.array(image)/255.0
		prediction = CNN.predict(image[np.newaxis,...])
		predicted_class = class_list[np.argmax(prediction[0], axis=-1)]
		word.append(predicted_class)
		win = np.ones((500,500,3), dtype='float64')*255

	if k == ord('q'):
		cv2.destroyAllWindows()
		break


print(''.join(map(str, word)))




translator=Translator()
print(translator.translate(''.join(map(str, word))))