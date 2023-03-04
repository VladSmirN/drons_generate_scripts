from PIL import Image
import os
import pandas as pd
import glob
import numpy as np
import shutil
from tqdm.auto import tqdm
import  xml.dom.minidom

OUTPUT_SIZE = 512
ADDITIONAL_CLEARANCE = 15
ADDITIONAL_CLEARANCE_SMALL_DRONE = 80
NUMBER_OUTPUT_IMG = 1000000
MIN_AREA_DRONE = 0.00
AREA_SMALL_DRONE = 0.04
AREA_LARGE_DRONE = 0.6

DATASET_DIR = r"C:\project\Computer_Vision\term_paper_2023\datasets\antiuav_train"
IMAGE_INPUT_DIR = r"C:\project\Computer_Vision\term_paper_2023\datasets\antiuav_train\img"
LABEL_INPUT_DIR = r"C:\project\Computer_Vision\term_paper_2023\datasets\antiuav_train\xml"

OUTPUT_DIR = r"C:\project\Computer_Vision\term_paper_2023\antiuav_with_masks"
IMAGE_OUTPUT_DIR = r"C:\project\Computer_Vision\term_paper_2023\antiuav_with_masks\images"
LABEL_OUTPUT_DIR = r"C:\project\Computer_Vision\term_paper_2023\antiuav_with_masks\labels"
MASK_OUTPUT_DIR = r"C:\project\Computer_Vision\term_paper_2023\antiuav_with_masks\masks"

shutil.rmtree(OUTPUT_DIR)
os.makedirs(os.path.join(IMAGE_OUTPUT_DIR))
os.makedirs(os.path.join(LABEL_OUTPUT_DIR))
os.makedirs(MASK_OUTPUT_DIR)


def get_dron_df():
    df = pd.DataFrame (glob.glob( os.path.join(IMAGE_INPUT_DIR,'*.jpg')), columns=['path_image'])
    df['id'] = df['path_image'].apply(lambda x : x.split('\\')[-1].split('.')[0])
    df['path_label'] = df['id'].apply(lambda x: os.path.join(LABEL_INPUT_DIR,f'{x}.xml'))
    return df

def get_coords_drons(path_xml):
    dom = xml.dom.minidom.parse(path_xml)
    coords_drons = []
    for i in range(len(dom.getElementsByTagName('xmin'))):
        xmin = int(dom.getElementsByTagName('xmin')[i].childNodes[0].data)
        ymin = int(dom.getElementsByTagName('ymin')[i].childNodes[0].data)
        xmax = int(dom.getElementsByTagName('xmax')[i].childNodes[0].data)
        ymax = int(dom.getElementsByTagName('ymax')[i].childNodes[0].data)
        coords_drons.append((xmin, ymin, xmax, ymax))
    return coords_drons

def coords2yolo(xmin, ymin, xmax, ymax ,output_size=512 ):
    width = (xmax - xmin) / output_size
    height = (ymax - ymin) / output_size
    x_center = (xmin + width / 2) / output_size
    y_center = (ymin + height / 2) / output_size
    return (x_center, y_center, width, height)


df_drons = get_dron_df()
dron_statistics = []
mask_statistics = []
id_statistics = []
additional_clearance_statistics = []
for i, row in tqdm(df_drons.sample(frac=1)[:NUMBER_OUTPUT_IMG].iterrows()):

    coords_drons = get_coords_drons(row['path_label'])
    if len(coords_drons) == 0:
        continue;
    xmin, ymin, xmax, ymax = coords_drons[0]
    dron = Image.open(row['path_image'])
    width, height = dron.size

    # YOLO LABEL
    width_yolo = (xmax - xmin) / width
    height_yolo = (ymax - ymin) / height
    #
    # if width_yolo*height_yolo < MIN_AREA_DRONE:
    #     continue

    # не обрабатываем больших дронов
    if  width_yolo*height_yolo > AREA_LARGE_DRONE:
        continue
    x_center = (xmin + width_yolo / 2) / width
    y_center = (ymin + height_yolo / 2) / height
    str_ = "0 " + str(x_center) + " " + str(y_center) + " " + str(width_yolo) + " " + str(height_yolo)
    path_label_output = os.path.join(LABEL_OUTPUT_DIR, row['id'] + '.txt')
    with open(path_label_output, 'w') as f:
        f.write(str_)

    #SAVE MASK
    new_width = int(width_yolo*OUTPUT_SIZE)
    new_height = int(height_yolo*OUTPUT_SIZE)
    if width_yolo*height_yolo < AREA_SMALL_DRONE:
        new_width += 2 * ADDITIONAL_CLEARANCE_SMALL_DRONE
        new_height += 2 * ADDITIONAL_CLEARANCE_SMALL_DRONE
        new_xmin = int(xmin/width*OUTPUT_SIZE) - ADDITIONAL_CLEARANCE_SMALL_DRONE
        new_ymin = int(ymin/height*OUTPUT_SIZE) - ADDITIONAL_CLEARANCE_SMALL_DRONE
        additional_clearance_statistics.append(ADDITIONAL_CLEARANCE_SMALL_DRONE)
    else:
        new_width += 2*ADDITIONAL_CLEARANCE
        new_height += 2 * ADDITIONAL_CLEARANCE
        new_xmin = int(xmin/width*OUTPUT_SIZE) - ADDITIONAL_CLEARANCE
        new_ymin = int(ymin/height*OUTPUT_SIZE) - ADDITIONAL_CLEARANCE
        additional_clearance_statistics.append(ADDITIONAL_CLEARANCE)

    mask = Image.new("RGBA", (OUTPUT_SIZE, OUTPUT_SIZE), (255, 255, 255, 255))
    rect = Image.new("RGBA",  (new_width, new_height), (0, 0, 0, 255))

    mask.paste(rect, (new_xmin, new_ymin))
    mask.save(os.path.join(MASK_OUTPUT_DIR, f"{row['id']}.png"))

    path_image_output = os.path.join(IMAGE_OUTPUT_DIR, row['id'] + '.jpg')
    dron.resize((OUTPUT_SIZE, OUTPUT_SIZE)).save(path_image_output)

    dron_statistics.append(width_yolo*height_yolo)
    id_statistics.append(row['id'])
    mask_statistics.append(new_width*new_height/512/512)
df_statistics = pd.DataFrame({'id': id_statistics,
                              'dron': dron_statistics,
                              'mask' : mask_statistics,
                              'additional clearance': additional_clearance_statistics
                              })
df_statistics.to_csv(os.path.join(OUTPUT_DIR, "statistics.csv"))

