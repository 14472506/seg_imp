"""
Details
"""
# ------ Imported libraries for code ------------------------------------- #
"""
Note, all libraries coppied over from bd2.py
"""
# base libraries
from audioop import avg
from importlib.metadata import metadata
import time
import random
from turtle import speed
import numpy as np
import os, json, cv2, random, copy

# setting up detectron logger
from detectron2.utils.logger import setup_logger
import tqdm
setup_logger()

# utilites from detectron2
from detectron2.engine import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import Visualizer, ColorMode

# for data augmentation
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

# from seg imp
from adet.config import get_cfg
from my_tools import custom_data_loader

# ------ functions ------------------------------------------------------- # 
def image_vis(predictor, image_path, output_path, input_metadata):
    """
    detials
    """
    # get predictor and image
    im = cv2.imread(image_path)
    
    # generate prediction
    output = predictor(im)
    
    # initialise visualiser
    v = Visualizer(im[:, :, ::-1], metadata=input_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
    
    # predict and show
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    image = v.get_image()[:, :, ::-1]
    image = cv2.resize(image, (960, 540))

    # save image
    cv2.imwrite(output_path, image)
    
def video_vis(predictor, video_in_path, video_out_path, input_metadata):
    """
    details
    """
    # Extract video properties
    video = cv2.VideoCapture(video_in_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    video_writer = cv2.VideoWriter(video_out_path + ".mp4", fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

    # Initialize visualizer
    v = VideoVisualizer(input_metadata, ColorMode.IMAGE)

    def runOnVideo(video, maxFrames):
        """ Runs the predictor on every frame in the video (unless maxFrames is given),
        and returns the frame with the predictions drawn.
        """

        readFrames = 0
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break

            # Get prediction results for this frame
            outputs = predictor(frame)

            # Make sure the frame is colored
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw a visualization of the predictions using the video visualizer
            visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))

            # Convert Matplotlib RGB format to OpenCV BGR format
            visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

            yield visualization

            readFrames += 1
            if readFrames > maxFrames:
                break

    # Create a cut-off for debugging
    num_frames = 1571

    # Enumerate the frames of the video
    for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):

        # Write test image
        cv2.imwrite(video_out_path + 'POSE_detectron2.png', visualization)

        # Write to video file
        video_writer.write(visualization)

    # Release resources
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
def speed_benchmark(predictor, image_path):
    
    im = cv2.imread(image_path)
    times = []

    for i in range(10):
        start_time = time.time()
        outputs = predictor(im)
        delta = time.time() - start_time
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    return(fps)

def get_predictor(config, weights):

    cfg = get_cfg()
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = weights
    predictor = DefaultPredictor(cfg)
    return(predictor)

def main(config_dir, weight_dir, test_meta, 
        im_eval, image_in_dir, image_out_dir,
        vid_eval, vid_in_dir, vid_out_dir,
        fps_eval):

    # getting predictor
    predictor = get_predictor(config_dir, weight_dir)

    if im_eval == True:
        save_count = 0
        for filename in os.listdir(image_in_dir):
            if filename.endswith(".json"):
                pass
            else:
                # in image path
                in_path = os.path.join(image_in_dir, filename)

                # out image path
                out_file = str(save_count) + ".jpg"
                out_path = os.path.join(image_out_dir, out_file)
                save_count += 1 
                
                # processing images
                image_vis(predictor, in_path, out_path, test_meta)
        print("images processed")

    if vid_eval == True:
        save_count = 0
        for filename in os.listdir(vid_in_dir):
            
            # in path 
            vid_in_path = os.path.join(vid_in_dir, filename)

            # out path
            vid_out_path = os.path.join(vid_out_dir, str(save_count))
            save_count +=1

            #print(vid_out_path)
            # video processing
            video_vis(predictor, vid_in_path, vid_out_path, test_meta)
        print("videos processed")

    if fps_eval == True:
        fps_list = []
        for filename in os.listdir(image_in_dir):
            if filename.endswith(".json"):
                pass
            else:
                # in image path
                in_path = os.path.join(image_in_dir, filename)
                fps_list.append(speed_benchmark(predictor, in_path))
        
        fps = sum(fps_list) / len(fps_list)
        print(fps)

# ----- initialisation --------------------------------------------------- #
if __name__ == "__main__":

    # loading data
    testing_config_dict = {
        "test1": ["coco", "jr_val", "data/jersey_royal_ds/val/val.json", "data/jersey_royal_ds/val"],
        "test2": ["coco", "jr_val", "data/jersey_royal_ds/test/test.json", "data/jersey_royal_ds/test"]
    }
    thing_classes = ["Jersey Royal"]
    test_data = testing_config_dict["test2"]
    test_meta = custom_data_loader(test_data[0], test_data[1], test_data[2], test_data[3], thing_classes)

    # model_configs
    config_dir = "configs/SOLOv2/R50_3x_1e4.yaml" 
    weight_dir = "train_dir/SOLOv2_R50_REFINED/model_final.pth"
    #config_dir = "configs/Mask_RCNN/R50_3x_1e2.yaml" 
    #weight_dir = "train_dir/MRCNN_R50_REFINED/model_final.pth"

    # data_in_dirs
    image_in_dir = "data/jersey_royal_ds/test/"
    video_in_dir = "data/test_videos/"

    # data out dir
    image_out_dir = "data/visual_results/images/SOLOv2"
    video_out_dir = "data/visual_results/video/SOLOv2"
    #image_out_dir = "data/visual_results/images/Mask_RCNN"
    #video_out_dir = "data/visual_results/video/Mask_RCNN"

    im_eval = False
    vid_eval = True
    fps_eval = False
    
    main(config_dir, weight_dir, test_meta, 
        im_eval, image_in_dir, image_out_dir,
        vid_eval, video_in_dir, video_out_dir,
        fps_eval)