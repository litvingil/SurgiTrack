# import
from ultralytics import YOLO
import os
import cv2

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np

from tqdm import tqdm

import heapq

save_best = True

# conda activate dinov2
keypoint_threshold = 0.7
num_keypoint_threshold = 6
model = YOLO("/home/user_219/omer_gil/yolov8n-pose.pt")

def list_folders(directory):
    # List all the entries in the directory
    entries = os.listdir(directory)
    # Filter entries to include only directories
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return folders

def filter_person(image):
    # Detect persons in the image
    results = model(image, verbose=False)
    for result in results:
        num_of_keypoints = 0
        for keypoint in result.keypoints.data.tolist()[0]:
            if keypoint[2] > keypoint_threshold:
                num_of_keypoints += 1
        if num_of_keypoints > num_keypoint_threshold:
            return True
    return False

def get_num_keypoints(image):
    # Detect persons in the image
    results = model(image, verbose=False)
    best_num_of_keypoints = 0
    for result in results:
        num_of_keypoints = 0
        for keypoint in result.keypoints.data.tolist()[0]:
            if keypoint[2] > keypoint_threshold:
                num_of_keypoints += 1
        if num_of_keypoints > best_num_of_keypoints:
            best_num_of_keypoints = num_of_keypoints
    return best_num_of_keypoints


def segment_person(image, predictor):
    outputs = predictor(image)
    # Get masks of detected persons
    masks = outputs["instances"].pred_masks
    classes = outputs["instances"].pred_classes
    person_masks = masks[classes == 0]  # Assuming '0' is the class index for persons

    # Combine masks if multiple persons are detected
    if person_masks.shape[0] > 0:
        combined_mask = np.logical_or.reduce(person_masks.cpu().numpy())
        segmented_image = image * np.expand_dims(combined_mask, axis=2)
        return segmented_image
    else:
        return image  # Return original image if no person is detected
    
def main():
    # Initialize model configuration and predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold for object detection
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    videos_pats = "/home/user_219/omer_gil/BoT-SORT/images"
    # output_folder = "/home/user_219/omer_gil/BoT-SORT/images_segmented"
    # Omer: Tweaked some things here:
    output_folder = "/home/user_219/omer_gil/BoT-SORT/images_best"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    videos = os.listdir(videos_pats)
    for vid in videos:
        print(vid)
        # Get all the folders in the video directory
        ids = os.listdir(os.path.join(videos_pats, vid))
        for id in ids:
            print(id)
            # Get all the images in the id directory
            images = os.listdir(os.path.join(videos_pats, vid, id))
            for image in tqdm(images):
                img = cv2.imread(os.path.join(videos_pats, vid, id, image))
                # import pdb; pdb.set_trace()
                try:
                    if filter_person(img):
                        # Segment the person in the image
                        segmented_image = segment_person(img, predictor=predictor)
                        # Save the segmented image
                        path = os.path.join(output_folder, vid, id)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        # cv2.imwrite(os.path.join(path, image), segmented_image)
                        cv2.imwrite(os.path.join(path, image), image)
                        continue
                except:
                    print("Error in filtering person in image: " + image)
                    exit()

def main_save_max(max_to_save = 5):
    # add filter based on num of key points and remove segment
    # Initialize model configuration and predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold for object detection
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    videos_pats = "/home/user_219/omer_gil/BoT-SORT/images"
    output_folder = "/home/user_219/omer_gil/BoT-SORT/best_images_segmented"



    videos = os.listdir(videos_pats)
    for vid in videos:
        print("Processing video:", vid)
        # Get all the folders in the video directory
        ids = os.listdir(os.path.join(videos_pats, vid))
        for id in ids:
            min_heap = []
            print("Processing person ID:", id)
            # Get all the images in the id directory
            images = os.listdir(os.path.join(videos_pats, vid, id))
            for image in tqdm(images):
                img = cv2.imread(os.path.join(videos_pats, vid, id, image))
                try:
                    n = get_num_keypoints(img)
                    if n > num_keypoint_threshold:
                        if len(min_heap) < max_to_save:
                            heapq.heappush(min_heap, (n, image))
                        else:
                            heapq.heappushpop(min_heap, (n, image))
                except:
                    print("Error in filtering person in image: " + image)
                    exit()
            for i in range(len(min_heap)):
                n, image = min_heap[i]
                img = cv2.imread(os.path.join(videos_pats, vid, id, image))
                # # Segment the person in the image
                # segmented_image = segment_person(img, predictor=predictor)
                # Save the segmented image
                path = os.path.join(output_folder, vid, id)
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path, str(n)+'_keypoints_'+ image), img)




if __name__ == "__main__":
    # main()
    main_save_max(5)