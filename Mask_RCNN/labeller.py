import os
import numpy as np
import cv2
import json

from utilities import mask_to_poly

def coco_labeller(predictor, data_dir):


    info_dict = {
        "year": "2022",
        "version": "1.0",
        "description": "Test Coco JSON Production",
        "contributor": "Me",
        "url": "https://non-maybeputmyemailhere",
        "date_created": "function for todays date?"
    }

    licence_dict = [{
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"        
    }]
    
    category_dict = [{
        "id": 1,
        "name": "Jersey Royal",
        "supercategory": ""
    },{
        "id": 2,
        "name": "Handle Bar",
        "supercategory": ""
    }]

    # loop initialisation
    img_count = 0
    anno_count = 0 
    image_dict = []
    anno_dict = []
    
    # looping through images in file
    for file in os.listdir(data_dir):

        # laoding image from file
        img_str = data_dir + str(file)
        img = cv2.imread(img_str)
        
        # making image dict entry
        im_dict = {
            "id": img_count,
            "license": 1,
            "file_name": file,
            "height": img.shape[0],
            "width": img.shape[1],
            "date_captured": None
        }       
        
        # append to dict
        image_dict.append(im_dict)

        # generating predicition
        pred_data = predictor(img)
        
        # getting number of instances in image
        instance_nums = pred_data["instances"].pred_classes.cpu().detach().numpy().shape[0]
        
        # looping through instances in image
        for instance in range(instance_nums):
            
            cat_id = int(pred_data["instances"][instance].pred_classes.cpu().detach().numpy()[0] + 1)
            mask = pred_data["instances"][instance].pred_masks.cpu().detach().numpy()*1
            area = int(np.sum(mask))
            
            points, bbox = mask_to_poly(mask)
            
            # current annotation dict
            ann_dict = {
                "id": anno_count,
                "image_id": img_count,
                "category_id": cat_id,
                "bbox": bbox,
                "segmentation": points,
                "area": area,
                "iscrowd": 0 
            }
            
            # append current dict
            anno_dict.append(ann_dict)
            
            # add 1 to annotation cound            
            anno_count += 1
            
        # add to dict count value
        img_count += 1
        
    coco_dict = {
        "info": info_dict,
        "licenses": licence_dict,
        "categories": category_dict,
        "images": image_dict,
        "annotations": anno_dict, 
    }
    
    with open("labelled.json", "w") as outfile:
        json.dump(coco_dict, outfile)


         