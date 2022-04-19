# libaries
import json
import shutil
import os

### paths ####################################### 
js_path = "train/train.json"
image_path = "datasets/unprocessed_datasets/all_rv_and_init_ims/"

new_dir = "train/"
### importing json ##############################
with open(js_path, 'r') as f:
    js_data = json.load(f)

### getting image list ##########################
im_list = []
for key in js_data["images"]:
    im_list.append(key["file_name"])

### making image dir from image list
for img_dir in im_list:
    copy_dir = os.path.join(image_path, img_dir)
    shutil.copy(copy_dir, new_dir)


