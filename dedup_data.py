from pathlib import Path
from os.path import exists
import subprocess
import json
import os 

all_dirs = ["./datasets/"+el for el in os.listdir('./datasets')]
print(all_dirs)
for dir in all_dirs:
    if "_aug_" in dir:
        continue
    if exists(str(dir)+"/meta.data"):
        with open(str(dir)+"/meta.data") as file:
            data = json.load(file)
            if "deduplicated" in data and data["deduplicated"]:
                continue
    else:
        for class_folder in os.listdir(dir):
            if class_folder == "meta.data":
                continue
            class_folder =  f"{dir}/{class_folder}"
           
            print(dir)
            print("generate_phashes.py")
            subprocess.call(['python', 'generate_phashes.py', class_folder ,"--use_int_filenames_as_id=0"])
            print("add_to_index.py")
            subprocess.call(['python', 'add_to_index.py'])
        print("phash_dedup.py")
        subprocess.call(['python', 'phash_dedup.py',"./"+str(dir)])
        with open(str(dir)+"/meta.data", 'w') as file:
            data = {"deduplicated":True}
            json.dump(data, file)