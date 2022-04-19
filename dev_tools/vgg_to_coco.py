# vgg to coco format conversion, based on https://stackoverflow.com/questions/61210420/converting-the-annotations-to-coco-format-from-mask-rcnn-dataset-format

# import libraries
import os
from re import I
from unicodedata import name
import skimage
import math
from itertools import chain
import numpy as np
import json

# json encoder class
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# functions
def vgg_to_coco(dataset_dir, vgg_json_path):

    with open(vgg_json_path) as f:
        vgg = json.load(f)

    # image information formatting
    image_ids_dict = {}
    image_info = []
    
    for i, v in enumerate(vgg.values()):
        image_ids_dict[v["filename"]] = I
        image_path = os.path.join(dataset_dir, v["filename"])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]
        image_info.append({"file_name": v["filename"], "id": i, "width": width, "height": height})
 
    
    # class information formatting
    classes = {}
    class_count = 0
    class_log = []

    for v in vgg.values():
        for r in v["regions"]:
            if r["region_attributes"]["name"] not in class_log:
                classes[class_count] = r["region_attributes"]["name"]
                class_log.append(r["region_attributes"]["name"])
                class_count += 1
    
    categories = [{"id": k, "name":v, "supercategory": ""} for k, v in classes.items()]

    # annotation information
    annotations = []
    suffix_zeros = math.ceil(math.log10(len(vgg)))

    for i, v in enumerate(vgg.values()):
        for j, r in enumerate(v["regions"]):
            x, y = r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"]
            cat_tag = 0
            for k, w in classes.items():
                if w == r["region_attributes"]["name"]:
                    cat_tag = k
            annotations.append({
                "segmentation": [list(chain.from_iterable(zip(x, y)))],
                "area": 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))),
                "bbox": [min(x), min(y), max(x)-min(x), max(y)-min(y)],
                "image_id": image_ids_dict[v["filename"]],
                "category_id": cat_tag,
                "id": int(f"{i:0>{suffix_zeros}}{j:0>{suffix_zeros}}"),
                "iscrowd": 0
            })

    ds_dict = {
        "info": {},
        "licence": [],
        "images": image_info,
        "categories": categories,
        "annotations": annotations
    }

    return(ds_dict)

# conversion form vgg to json format


# main function
def main(dataset_dir, vgg_json_path):
    ds_dict = vgg_to_coco(dataset_dir, vgg_json_path)
    
    outfile =  vgg_json_path.replace(".json", "_coco.json")

    with open(outfile, "w") as f:
        json.dump(ds_dict, f, cls=NpEncoder)

# execution
if __name__ == "__main__":
    main("datasets/unprocessed_datasets/cambridge_data/train", "datasets/unprocessed_datasets/cambridge_data/train/potato_d2_dataset.json")