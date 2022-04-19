import json
import math
import statistics as s

def instance_size(json_path):
    
    instance_areas = []
    
    with open(json_path) as f:        
        data = json.load(f)
        
        for key, val in data.items():            
            if key == "annotations":                            
                for i in val:                    
                    for k2, v2 in i.items():                        
                        if k2 == "area":
                            instance_areas.append(math.sqrt(v2))
    
    quart_areas = s.quantiles(instance_areas, n=4)
    range = [round(num) for num in quart_areas]
    
    print(range)
            
if __name__ == "__main__":
    
    json_path = "data/jersey_royal_ds/train/train.json"
    instance_size(json_path)
    
    
