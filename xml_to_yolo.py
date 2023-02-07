import os
import pandas as pd
import glob
import numpy as np
import shutil
from tqdm.auto import tqdm
import  xml.dom.minidom

XML_DIR = r'C:\project\Computer_Vision\term_paper_2023\antiuav_train\xml'
IMAGES_DIR = r'C:\project\Computer_Vision\term_paper_2023\antiuav_train_middle'
YOLO_DIR = r'C:\project\Computer_Vision\term_paper_2023\yolo_drons_test'

YOLO_IMAGES_DIR = os.path.join(YOLO_DIR, 'images')
YOLO_LABELS_DIR = os.path.join(YOLO_DIR, 'labels')


SIZE = 512
CLASS_ID = "0"

xml_df = pd.DataFrame (glob.glob( os.path.join(XML_DIR, '*.xml')), columns=['path_xml'])
xml_df['id'] = xml_df['path_xml'].apply(lambda x : x.split('\\')[-1].split('.')[0])

img_df = pd.DataFrame (glob.glob( os.path.join(IMAGES_DIR, '*.jpg')), columns=['path_img'])
img_df['id'] = img_df['path_img'].apply(lambda x : x.split('\\')[-1].split('.')[0])

if not os.path.exists(YOLO_IMAGES_DIR):
    os.makedirs(YOLO_IMAGES_DIR)

if not os.path.exists(YOLO_LABELS_DIR):
    os.makedirs(YOLO_LABELS_DIR)

for i, row in img_df.iterrows():
    path_image = os.path.join(IMAGES_DIR, str(row["id"]) + '.jpg' )
    path_xml = os.path.join(XML_DIR, str(row["id"]) + '.xml')

    path_image_yolo = os.path.join(YOLO_IMAGES_DIR, str(row["id"]) + '.jpg')
    path_label_yolo = os.path.join(YOLO_LABELS_DIR, str(row["id"]) + '.txt')
    shutil.copyfile(path_image, path_image_yolo)


    dom = xml.dom.minidom.parse(path_xml)
    xmin_arr = dom.getElementsByTagName('xmin')
    ymin_arr = dom.getElementsByTagName('ymin')
    xmax_arr = dom.getElementsByTagName('xmax')
    ymax_arr = dom.getElementsByTagName('ymax')

    width_image = int(dom.getElementsByTagName('width')[0].childNodes[0].data)
    height_image = int(dom.getElementsByTagName('height')[0].childNodes[0].data)

    for i, _ in enumerate(xmin_arr):
        xmin = int(xmin_arr[i].childNodes[0].data) / width_image
        ymin = int(ymin_arr[i].childNodes[0].data) / height_image
        xmax = int(xmax_arr[i].childNodes[0].data) / width_image
        ymax = int(ymax_arr[i].childNodes[0].data) / height_image

        width = xmax - xmin
        height = ymax - ymin
        x_center = (xmax + xmin) / 2
        y_center = (ymax + ymin) / 2

        str_ = CLASS_ID + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height) +"\n"

    with open(path_label_yolo, 'w') as f:
        f.write(str_)