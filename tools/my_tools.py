import numpy as np 
import os, json, cv2, random, copy, PIL

from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.structures import BoxMode
from detectron2.data.datasets import load_coco_json, register_coco_instances

def custom_data_loader(data_format, dataset_name, json_location, data_location, thing_classes_data):
    """
    details
    """
    # internal vgg to coco function
    def vgg_to_coco():
        """
        details
        """
        # open json file containing data annotations
        with open(json_location) as f:
            imgs_anns = json.load(f)
        
        # initialise list for dictionaries   
        dataset_dicts = []
        
        # itterate through data enteries in json
        for idx, v in enumerate(imgs_anns.values()):
            
            # init dict for data
            record = {}
            
            # retrieving file detials
            file_name = os.path.join(data_location, v['filename'])
            height, width = cv2.imread(file_name).shape[:2]
            
            # recording image details
            record["file_name"] = file_name
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width 
            
            # recording objects in image, looping through instances in data
            annos = v["regions"]
            objs = []
            for key in annos:
                for _, anno in key.items():                
                    if _ == "shape_attributes":

                        px = anno["all_points_x"]
                        py = anno["all_points_y"]
                        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                        poly = [p for x in poly for p in x]

                        obj = {
                            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": [poly],
                            "category_id": 0,
                        }
                        objs.append(obj)
            record["annotations"] = objs
            
            # appending image data to dataset dicts list
            dataset_dicts.append(record)
        
        # returning dataset dictionary
        return(dataset_dicts)
            
    # data formate is coco
    if data_format=="coco":
        register_coco_instances(dataset_name, {}, json_location, data_location)
    
    # data format is vgg    
    if data_format=="vgg":
        DatasetCatalog.register(dataset_name,  lambda: vgg_to_coco())
        
    # assigning and collecting meta data
    metadata = MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes_data)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    return(metadata)