import cv2
import numpy as np
import os

src_path = "dataset"
dst_path = "final_dataset"

# make final_dataset folder
ldir = os.listdir(src_path)
for dir in ldir :
    os.makedirs(dst_path + "/" + dir)

# image processing
ldir = os.listdir(src_path)
for dir in ldir :
    file_list = os.listdir(src_path + "/" + dir)
    file_list_png = [file for file in file_list if file.endswith('.png')]
    for file in file_list_png : 
        full_path = src_path + "/" + dir + "/" + file
        # image file open
        img_array = np.fromfile(full_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # white pedding
        white = [255, 255, 255]
        dst = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=white)
        # image file save
        save_image_path = dst_path + "/" + dir + "/" + file
        # file extension name
        extension = os.path.splitext(save_image_path)[1]
        result, encoded_img = cv2.imencode(extension, dst)
        if result:
            with open(save_image_path, mode='w+b') as f:
                encoded_img.tofile(f)
