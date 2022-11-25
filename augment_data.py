from pathlib import Path
from os.path import exists
import subprocess
import json
from PIL import Image
import numpy as np
import os 
import albumentations as A
from tqdm import tqdm

AUGMENTATION_VERSION = "1"

# print(dir(A.augmentations))
transforms = [
  A.ImageCompression(quality_lower=5,quality_upper=50, always_apply=True),
  A.augmentations.geometric.rotate.SafeRotate(limit=30,value=[0,0,0], border_mode=0,always_apply=True),  
  A.HorizontalFlip(always_apply=True),
  A.augmentations.MotionBlur(always_apply=True)
]


def read_img_file(f):
    img = Image.open(f)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224,224))
    return np.array(img)

# p = Path('./datasets')

all_dirs = ["./datasets/"+el for el in os.listdir('./datasets')]
print(all_dirs)
for dir in all_dirs:
    dir = str(dir)
    if "_aug_" in dir:
        continue
    if not exists(dir+"/meta.data"):
        continue
    with open(dir+"/meta.data") as file:
        data = json.load(file)
        if not "deduplicated" in data and not data["deduplicated"]:
            continue
    print(dir+"_aug_"+AUGMENTATION_VERSION)
    if dir+"_aug_"+AUGMENTATION_VERSION in all_dirs:
        continue
    os.mkdir(dir+"_aug_"+AUGMENTATION_VERSION)
    all_classes = os.listdir(dir)
    for cls in all_classes:
        if cls == "meta.data":
            continue
        os.mkdir(dir+"_aug_"+AUGMENTATION_VERSION+"/"+cls)
        for file_name in tqdm(os.listdir(f"{dir}/{cls}")):
            img = read_img_file(f"{dir}/{cls}/{file_name}")
            Image.fromarray(img).save(f"{dir}_aug_{AUGMENTATION_VERSION}/{cls}/{file_name}")
            for i in range(len(transforms)):
                new_img = transforms[i](image=img)["image"]
                new_img = Image.fromarray(new_img)
                new_img.save(f"{dir}_aug_{AUGMENTATION_VERSION}/{cls}/{i}_{file_name}")
    # print(str(dir))
    # print("generate_phashes.py")
    # subprocess.call(['python', 'generate_phashes.py',"./"+str(dir),"--use_int_filenames_as_id=0"])
    # print("add_to_index.py")
    # subprocess.call(['python', 'add_to_index.py'])
    # print("phash_dedup.py")
    # subprocess.call(['python', 'phash_dedup.py',"./"+str(dir)])
    # with open(str(dir)+"/meta.data", 'w') as file:
    #     data = {"deduplicated":True}
    #     json.dump(data, file)