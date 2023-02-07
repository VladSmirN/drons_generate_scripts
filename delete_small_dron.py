import os
import pandas as pd
import glob
import numpy as np
import shutil
from tqdm.auto import tqdm
import  xml.dom.minidom
from PIL import Image

DRONS_DIR = r"C:\project\Computer_Vision\term_paper_2023\antiuav_train"
NEW_DIR_DRONS = r"C:\project\Computer_Vision\term_paper_2023\antiuav_train_middle"




def df_antiuav(PATH):
    df = pd.DataFrame (glob.glob( os.path.join(PATH,'img','*.jpg') ), columns=['path_image'])
    df['id'] = df['path_image'].apply(lambda x : x.split("\\")[-1].split('.')[0])
    df['path_xml'] = df['id'].apply(lambda x : os.path.join(PATH,'xml',x+'.xml') )
    return df

df_train = df_antiuav(DRONS_DIR)
# print(df_train["id"])


if not os.path.exists(NEW_DIR_DRONS):
    os.makedirs(NEW_DIR_DRONS)

for _,row in tqdm(df_train.iterrows()):
    dom = xml.dom.minidom.parse(row['path_xml'])
    try:
        xmin = int(dom.getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin = int(dom.getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax = int(dom.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax = int(dom.getElementsByTagName('ymax')[0].childNodes[0].data)
        if  xmax - xmin > 115  and ymax - ymin > 55 :
            img = Image.open(row['path_image'])
            newsize = (512, 512)
            img = img.resize(newsize)
            img = img.save(os.path.join(NEW_DIR_DRONS, row['id']+'.jpg'))
            # shutil.copy(row['path_image'], os.path.join(NEW_DIR_DRONS, row['id']+'.jpg'))
    except:
        path_dron = ""

