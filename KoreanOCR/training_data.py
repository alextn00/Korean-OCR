import json
import os
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import sys
from keras.utils import to_categorical
from keras import models, layers
from keras import optimizers
import tensorflow as tf

path = "./json/TL/"
file_list = os.listdir(path)

Korean_id = []
Korean_text = []

# Korean_2350 = []
# with open("./KS1001/Korean_2350.txt", "r", encoding='utf-8') as f :
#     Korean_2350_origin = f.read()
# for i in range(len(Korean_2350_origin)) :
#     Korean_2350.append(Korean_2350_origin[i])
# Korean_2350 = Korean_2350[0:2349]

for f in file_list : 
    filename = str(f)
    filename_list = filename.split('.')
    filename = filename_list[0]
    with open('./json/TL/' + filename + '.json','r',encoding='utf-8') as f:
        file = json.load(f)
    # print(file['bbox'][0]['x'])
    # print(file['bbox'][0]['y'])
    bbox = file['bbox']

    for index, box in enumerate(bbox):
        text = file['bbox'][index]['data']
        id = file['bbox'][index]['id']
        Korean_id.append(filename + "_" + str(id))
        Korean_text.append(text)

Korean_train_id = Korean_id[0:214420]
Korean_train_text = Korean_text[0:214420]

Korean_val_id = Korean_id[214420:321630]
Korean_val_text = Korean_text[214420:321630]

Korean_test_id = Korean_id[321630:]
Korean_test_text = Korean_text[321630:]

syllable = list(set(Korean_text))

del(Korean_id)
del(Korean_text)

x_train = np.zeros((214420, 32, 32, 3), 'uint8')
x_val = np.zeros((107210, 32, 32, 3), 'uint8')
x_test = np.zeros((35736, 32, 32, 3), 'uint8')

for i, ID in enumerate(Korean_train_id) :
    Image_addr = "./process_data/" + str(ID) + ".png"
    Korean_Image = Image.open(Image_addr)
    Korean_Image = Korean_Image.resize((32, 32))
    Korean_Image_Array = np.array(Korean_Image, 'uint8')
    x_train[i] = Korean_Image_Array

for i, ID in enumerate(Korean_val_id) :
    Image_addr = "./process_data/" + str(ID) + ".png"
    Korean_Image = Image.open(Image_addr)
    Korean_Image = Korean_Image.resize((32, 32))
    Korean_Image_Array = np.array(Korean_Image, 'uint8')
    x_val[i] = Korean_Image_Array

for i, ID in enumerate(Korean_test_id) :
    Image_addr = "./process_data/" + str(ID) + ".png"
    Korean_Image = Image.open(Image_addr)
    Korean_Image = Korean_Image.resize((32, 32))
    Korean_Image_Array = np.array(Korean_Image, 'uint8')
    x_test[i] = Korean_Image_Array

x_train = x_train.astype('float32')/255.0
x_val = x_val.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

del(Korean_train_id)
del(Korean_val_id)
del(Korean_test_id)

syllable_to_index = {syllable : index for index, syllable in enumerate(syllable)}
index_to_syllable = {index : syllable for index, syllable in enumerate(syllable)}

label_train = []
label_val = []
label_test = []
        
for syllables in Korean_train_text : 
    if syllable_to_index.get(syllables) is not None: 
        label_train.extend([syllable_to_index[syllables]])

for syllables in Korean_val_text : 
    if syllable_to_index.get(syllables) is not None : 
        label_val.extend([syllable_to_index[syllables]])

for syllables in Korean_test_text : 
    if syllable_to_index.get(syllables) is not None : 
        label_test.extend([syllable_to_index[syllables]])

del(Korean_train_text)
del(Korean_val_text)
del(Korean_test_text)

label_train = to_categorical(label_train)
label_val = to_categorical(label_val)
label_test = to_categorical(label_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2349, activation='softmax')
])



model.compile(optimizer=optimizers.RMSprop(lr = 0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, label_train, epochs = 20, batch_size = 128, validation_data=(x_val, label_val))

test_loss, test_acc = model.evaluate(x_test, label_test)
print('test_loss:      ',test_loss)
print('test_accuracy:  ',test_acc)

# CNN = models.Sequential()
# CNN.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# CNN.add(layers.MaxPooling2D((2, 2)))

# CNN.add(layers.Conv2D(256, (3, 3), activation='relu'))
# CNN.add(layers.MaxPooling2D((2, 2)))

# CNN.add(layers.Conv2D(512, (3, 3), activation='relu'))
# CNN.add(layers.Flatten())

# CNN.add(layers.Dense(512, activation='relu'))

# CNN.add(layers.Dense(2349, activation='softmax'))

#CNN.compile(optimizer=optimizers.RMSprop(lr = 0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])

#hist = CNN.fit(x_train, label_train, epochs = 200, batch_size = 128, validation_data=x_val)

#test accuracy 출력
# test_loss, test_acc = CNN.evaluate(x_test, label_test)
# print('test_loss:      ',test_loss)
# print('test_accuracy:  ',test_acc)

#Epoch당 loss / accuracy plot
# loss = hist.history['loss']
# acc = hist.history['accuracy']
# val_loss = hist.history['val_loss']
# val_acc = hist.history['val_accuracy']
# epochs = range(1, len(loss)+1)
# plt.figure(figsize=(10,7))
# plt.subplots_adjust(wspace=0.5)
# plt.subplot(1,2,1)
# plt.plot(epochs, loss, 'bo-', label='Training loss')
# plt.plot(epochs, val_loss, 'rx-', label='Validation loss')
# plt.title('Loss')
# plt.xlabel('Epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(epochs, acc, 'bo-', label='Training accuracy')
# plt.plot(epochs, val_acc, 'rx-', label='Validation accuracy')
# plt.title('Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('accuracy')
# plt.grid()
# plt.legend()

model.save('Korean_CNN_model.h5')

with open("index_to_syllable.json", "w") as json_file:
   json.dump(index_to_syllable, json_file)