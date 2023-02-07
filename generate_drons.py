from PIL import Image
import os
import pandas as pd
import glob
import numpy as np
import shutil
from tqdm.auto import tqdm

DRONS_DIR = r"C:\project\Computer_Vision\term_paper_2023\dataset_drone_without_background_crop"
BACKGROUND_DIR = r"C:\project\Computer_Vision\term_paper_2023\background"
GENERATE_DIR = r"C:\project\Computer_Vision\term_paper_2023\generate"

def get_dron_df():
    df = pd.DataFrame (glob.glob( os.path.join(DRONS_DIR,'*.png')), columns=['path_image'])
    df['id'] = df['path_image'].apply(lambda x : x.split('/')[-1].split('.')[0])
    return df

def get_background_df():
    df = pd.DataFrame (glob.glob( os.path.join(BACKGROUND_DIR,'*.jpg') ), columns=['path_image'])
    df['id'] = df['path_image'].apply(lambda x : x.split('/')[-1].split('.')[0])
    return df

df_drons = get_dron_df()
df_background = get_background_df()



if not os.path.exists(os.path.join(GENERATE_DIR, "image")):
    os.makedirs(os.path.join(GENERATE_DIR, "image"))



def generate_raw(n):
    list_drons = df_drons.sample(n, replace=True)['path_image'].to_list()
    list_background = df_background.sample(n, replace=True)['path_image'].to_list()
    id_image = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    output_size = 512
    for i in tqdm(range(n)):
        try:
            dron = Image.open(list_drons[i])
            background = Image.open(list_background[i])

            background_width, background_height = background.size
            dron_width, dron_height = dron.size

            x_crop_background = np.random.randint(0, background_width - output_size, 1)[0]
            y_crop_background = np.random.randint(0, (background_height - output_size) * 0.3, 1)[0]

            background = background.crop((x_crop_background,
                                          y_crop_background,
                                          x_crop_background + output_size,
                                          y_crop_background + output_size))

            if dron_width > 256 or dron_height > 256:
                dron_width = np.random.randint(120, 160, 1)[0]
                dron_height = np.random.randint(60, 100, 1)[0]
                dron = dron.resize((dron_width, dron_height))

            x_dron = np.random.randint(0, output_size - dron_width, 1)[0]
            y_dron = np.random.randint(0, output_size - dron_height, 1)[0]

            background.paste(dron, (x_dron, y_dron), dron)
            #             background.paste(dron, (x_dron, y_dron))
            #             background.show()
            #             ImageShow.show(background)
            background.save(os.path.join(GENERATE_DIR, "image", str(i) + '.jpg'))

            xmin.append(x_dron)
            ymin.append(y_dron)
            xmax.append(x_dron + dron_width)
            ymax.append(y_dron + dron_height)
            id_image.append(i)

        except Exception as e:
            print(e)

    df = pd.DataFrame()
    df['xmin'] = xmin
    df['ymin'] = ymin
    df['xmax'] = xmax
    df['ymax'] = ymax
    df['id_image'] = id_image
    df.to_csv(os.path.join(GENERATE_DIR, "markup.csv"))

#delete trash
# for i,row in tqdm(df_background.iterrows()):
#     img = Image.open(row['path_image'])
#     np_img =  np.asarray( img )
#     if  len(np_img.shape) != 3 or np_img.shape[2] != 3:
#         os.remove(row['path_image'])

print(df_drons)
generate_raw(6000)
shutil.make_archive(r"C:\project\Computer_Vision\term_paper_2023\drons_with_background", 'zip', GENERATE_DIR)