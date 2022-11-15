import json
from PIL import Image
import os

path = "./image/TS6/"
file_list = os.listdir(path)

for f in file_list : 
    filename = str(f)
    filename_list = filename.split('.')
    filename = filename_list[0]
    with open('./json/TL/' + filename + '.json','r',encoding='utf-8') as f:
        file = json.load(f)
    # print(file['bbox'][0]['x'])
    # print(file['bbox'][0]['y'])
    bbox = file['bbox']

    img = Image.open('./image/TS6/' + filename + '.png')

    for index, box in enumerate(bbox):
        id = file['bbox'][index]['id']
        left = file['bbox'][index]['x'][0]
        top = file['bbox'][index]['y'][0]
        right = file['bbox'][index]['x'][2]
        bottom = file['bbox'][index]['y'][1]
        coor = (left, top, right, bottom)
        
        crop_img = img.crop(coor)
        crop_img.save('./process_data/' + filename + '_' + str(id) + '.png')
