import os
import pandas as pd
import glob
import numpy as np
import shutil
from tqdm.auto import tqdm


CSV_DIR = r'C:\project\Computer_Vision\term_paper_2023\generate\markup.csv'
IMAGES_DIR = r'C:\project\Computer_Vision\term_paper_2023\cyclegan_epoch_10_lr_0.0002'
YOLO_DIR = r'C:\project\Computer_Vision\term_paper_2023\yolo_drons_ts2_train'

YOLO_IMAGES_DIR = os.path.join(YOLO_DIR, 'images')
YOLO_LABELS_DIR = os.path.join(YOLO_DIR, 'labels')


SIZE = 512
CLASS_ID = "0"

df = pd.read_csv(CSV_DIR)
print(df)

if not os.path.exists(YOLO_IMAGES_DIR):
    os.makedirs(YOLO_IMAGES_DIR)

if not os.path.exists(YOLO_LABELS_DIR):
    os.makedirs(YOLO_LABELS_DIR)



for i, row in df.iterrows():
    path_image = os.path.join(IMAGES_DIR, str(row["id_image"]) + '.jpg' )
    path_image_coco = os.path.join(YOLO_IMAGES_DIR, str(row["id_image"]) + '.jpg')
    path_label_coco = os.path.join(YOLO_LABELS_DIR, str(row["id_image"]) + '.txt')

    shutil.copyfile(path_image, path_image_coco)
    xmin = float(row["xmin"])/SIZE
    ymin = float(row["ymin"])/SIZE
    xmax = float(row["xmax"])/SIZE
    ymax = float(row["ymax"])/SIZE
    width = xmax - xmin
    height = ymax - ymin
    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2
    str_ = CLASS_ID + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)
    with open(path_label_coco, 'w') as f:
        f.write(str_)


