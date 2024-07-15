import cv2
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from PIL import Image
import numpy as np

setup_logger()

# conda activate dinov2

# Initialize model configuration and predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold for object detection
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

def segment_person(image_path):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    # Get masks of detected persons
    masks = outputs["instances"].pred_masks
    classes = outputs["instances"].pred_classes
    person_masks = masks[classes == 0]  # Assuming '0' is the class index for persons

    # Combine masks if multiple persons are detected
    if person_masks.shape[0] > 0:
        combined_mask = np.logical_or.reduce(person_masks.cpu().numpy())
        segmented_image = im * np.expand_dims(combined_mask, axis=2)
        return segmented_image
    else:
        return im  # Return original image if no person is detected

def main():
    input_folder = '/home/user_219/omer_gil/BoT-SORT/images'
    output_folder = '/home/user_219/omer_gil/BoT-SORT/images_segmented'
    feature_folder = '/home/user_219/omer_gil/BoT-SORT/images_feature_extracted'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for vid_id in range(1, 5):
        current_vid_dir = os.path.join(input_folder, str(vid_id))
        output_folder_current_vid = os.path.join(output_folder, str(vid_id))
        if not os.path.exists(output_folder_current_vid):
            os.makedirs(output_folder_current_vid)
        ids_dir = [
        f for f in os.listdir(current_vid_dir) if os.path.isdir(os.path.join(current_vid_dir, f))]
    
        for id in ids_dir:
            id_output = os.path.join(output_folder_current_vid, id)
            if not os.path.exists(id_output):
                os.makedirs(id_output)
            current_id_dir = os.path.join(current_vid_dir, id)
            # First, segment:
            for image_name in os.listdir(current_id_dir):
                print("Segmenting image: " + image_name)
                image_path = os.path.join(current_id_dir, image_name)
                segmented_image = segment_person(image_path)
                output_path = os.path.join(id_output, image_name)
                cv2.imwrite(output_path, segmented_image)
            # Then, feature extract:
            # id_feature_output = os.path.join(feature_folder, id)
            # if not os.path.exists(id_feature_output):
            #     os.makedirs(id_feature_output)
            # for image_name in os.listdir(id_output):
            #     print("Feature extracting image: " + image_name)
            #     image_path = os.path.join(id_output, image_name)
            #     feature_image = extract_features(image_path)
            #     output_path = os.path.join(id_feature_output, image_name)
            #     cv2.imwrite(output_path, feature_image)
    
if __name__ == "__main__":
    main()




# #importing the required module
# import cv2 as cv
# #reading the image whose key points are to detected using imread() function
# imageread = cv.imread('C:/Users/admin/Desktop/Images/logo.png')
# #creating an object of ORB() function to detect the key points in the image
# ORB_object = cv.ORB_create()
# #detecting the key points in the image using ORB_object.detect() function
# keypoints = ORB_object.detect(imageread)
# #computing the descriptors for the input image using ORB_object.compute() function
# keypoints, descriptors = ORB_object.compute(imageread, keypoints)
# #using drawKeypoints() function to draw the detected key points on the image
# imageresult = cv.drawKeypoints(imageread, keypoints, None, color=(255,0,0), flags=0)
# #displaying the resulting image as the output on the screen
# cv.imshow('ORB_image', imageresult)
# cv.waitKey()