import json
import matplotlib.pyplot as plt
from sys import argv

experiment_folder = argv[1]
# 'SOLOv2_1e3_100_epoch'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
#print(experiment_metrics)

# total loss data
tl_iter_list = []
tl_list = []

# validation loss data
vl_iter_list = []
vl_list = []

for x in experiment_metrics:
    if "total_loss" in x:
        tl_iter_list.append(x['iteration']) 
        tl_list.append(x['total_loss'])

    if "validation_loss" in x:
        vl_iter_list.append(x['iteration']) 
        vl_list.append(x['validation_loss'])   
    
plt.plot(
    tl_iter_list, 
    tl_list)
plt.plot(
    vl_iter_list, 
    vl_list)
plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.show()