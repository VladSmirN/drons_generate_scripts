from PIL import Image
import os
import pandas as pd
import glob
import numpy as np
import shutil
from tqdm.auto import tqdm

DRONS_DIR = r"C:\project\Computer_Vision\term_paper_2023\datasets\dataset_drone_without_background_crop"
BACKGROUND_DIR = r"C:\project\Computer_Vision\term_paper_2023\datasets\background"
GENERATE_DIR = r"C:\project\Computer_Vision\term_paper_2023\generate"
IMAGES_DIR = os.path.join(GENERATE_DIR, "images")
MASKS_DIR = os.path.join(GENERATE_DIR, 'masks')
YOLO_LABELS_DIR = os.path.join(GENERATE_DIR, 'labels')

if not os.path.exists(IMAGES_DIR):
    os.makedirs(os.path.join(IMAGES_DIR))
if not os.path.exists(MASKS_DIR):
    os.makedirs(os.path.join(MASKS_DIR))
if not os.path.exists(YOLO_LABELS_DIR):
    os.makedirs(YOLO_LABELS_DIR)


def get_dron_df():
    df = pd.DataFrame (glob.glob( os.path.join(DRONS_DIR,'*.png')), columns=['path_image'])
    df['id'] = df['path_image'].apply(lambda x : x.split('\\')[-1].split('.')[0])
    return df

def get_background_df():
    df = pd.DataFrame (glob.glob( os.path.join(BACKGROUND_DIR,'*.jpg') ), columns=['path_image'])
    df['id'] = df['path_image'].apply(lambda x : x.split('\\')[-1].split('.')[0])
    return df

df_drons = get_dron_df()
df_background = get_background_df()



def to_binary(img):
    T = 125
    target = (np.array(img.split()[-1])>T).astype(int)
    target = np.stack((target, target, target), axis=2) * 255
    return Image.fromarray(np.uint8(target)).convert('RGBA')
def generate_raw(n):
    drons = df_drons.sample(n, replace=True)
    path_drons = drons['path_image'].to_list()
    id_drons = drons['id'].to_list()
    background = df_background.sample(n, replace=True)
    path_backgrounds = background['path_image'].to_list()
    id_backgrounds = background['id'].to_list()
    id_image = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    output_size = 512
    for i in tqdm(range(n)):
        try:
            dron = Image.open(path_drons[i]).convert("RGBA")
            background = Image.open(path_backgrounds[i]).convert("RGB")

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

            # SAVE MASK
            # mask = Image.new("RGBA", (512, 512), (0, 0, 0, 255))
            # bin_dron = to_binary(dron)
            # mask.paste(bin_dron, (x_dron, y_dron), dron)
            # mask.save(
            #     os.path.join(MASKS_DIR, f"{str(i)}_{str(id_drons[i])}_{str(id_backgrounds[i])}.png"))

            mask = Image.new("RGBA", (512, 512), (0, 0, 0, 255))
            rect = Image.new("RGBA", (dron_width, dron_height), (255, 255, 255, 255))
            mask.paste(rect, (x_dron, y_dron), rect)
            mask.save(
                  os.path.join(MASKS_DIR, f"{str(i)}_{str(id_drons[i])}_{str(id_backgrounds[i])}.png"))

            # SAVE GENERATE IMAGE
            background.save(os.path.join(IMAGES_DIR , f"{str(i)}_{str(id_drons[i])}_{str(id_backgrounds[i])}.jpg"))

            # FOR CSV
            xmin.append(x_dron)
            ymin.append(y_dron)
            xmax.append(x_dron + dron_width)
            ymax.append(y_dron + dron_height)
            id_image.append(i)

            # YOLO LABELS
            width = dron_width / output_size
            height = dron_height / output_size
            x_center = (x_dron + dron_width / 2) / output_size
            y_center = (y_dron + dron_height / 2) / output_size

            str_ = "0 " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)

            path_label_coco = os.path.join(YOLO_LABELS_DIR, str(i) + '.txt')
            with open(path_label_coco, 'w') as f:
                f.write(str_)

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
generate_raw(100)
# mask = Image.open(r"C:\project\Computer_Vision\term_paper_2023\overture-creations-5sI6fQgYIuo_mask.png").convert("RGB")
# dron = Image.open(r"C:\project\Computer_Vision\term_paper_2023\datasets\dataset_drone_without_background_crop\0.png").convert("RGBA")

# print(np.array(mask))


# mask.show()
shutil.make_archive(r"C:\project\Computer_Vision\term_paper_2023\drons_with_background", 'zip', GENERATE_DIR)