# import
from ultralytics import YOLO
import sys
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
aspect_ratio_threshold = 0.25
keypoint_threshold = 0.75
num_keypoint_threshold = 6
num_keypoints = 17
model = YOLO("/home/user_219/omer_gil/yolov8s-pose.pt")

def apply_transform(img, transform_matrix):
  """
  Applies a transformation matrix to an image.

  Args:
      img: NumPy array representing the image.
      transform_matrix: A 3x3 NumPy array representing the transformation matrix.

  Returns:
      A NumPy array representing the transformed image.
  """
  # Use cv2.warpAffine for image transformation
  rows, cols, channels = img.shape
  # Detect if it is a rotation matrix:
  if transform_matrix[0, 2] == 0 and transform_matrix[1, 2] == 0:
      cols = int(cols * transform_matrix[0, 0]) 
      rows = int(rows * transform_matrix[1, 1]) 
  transformed_img = cv2.warpAffine(img, transform_matrix, (cols, rows))
  return transformed_img

def get_image_variance_batched(img, all_keypoint_arr, kpt_thresh):
    var_parts = [0, 0, 0, 0] # head, body, left foot, right foot
    num_parts = len(all_keypoint_arr)
    # First, filter the image, regarding number of keypoints found:
    results = model(img, verbose=False)
    keypoints = results[0].keypoints[0].data[0]
    keypoints_detected_head = 0
    keypoints_detected_body = 0
    # import pdb; pdb.set_trace()
    for idx, keypoint in enumerate(keypoints):
        if keypoint[2] > keypoint_threshold and idx in all_keypoint_arr[0]:
            keypoints_detected_head += 1
        if keypoint[2] > keypoint_threshold and idx in all_keypoint_arr[1]:
            keypoints_detected_body += 1
    if keypoints_detected_head < kpt_thresh[0] or kpt_thresh[0] == 0:
        var_parts[0] = -1
    if keypoints_detected_body < kpt_thresh[1] or kpt_thresh[1] == 0:
        var_parts[1] = -1
        
    rows, cols, channels = img.shape
    rotation_matrix_1 = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 0.6)
    rotation_matrix_2 = cv2.getRotationMatrix2D((cols/2, rows/2), -30, 0.6)
    mirror_matrix = np.array([[-1, 0, cols], [0, 1, 0]], dtype=np.float32)
    # shear_matrix_1 = np.array([[1, 0.5, 0], [0, 1, 0]], dtype=np.float32)
    # shear_matrix_2 = np.array([[1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    stretch_matrix_1 = np.array([[1.5, 0, 0], [0, 1, 0]], dtype=np.float32)
    stretch_matrix_2 = np.array([[1, 0, 0], [0, 1.5, 0]], dtype=np.float32)
    # transform_matrices = [rotation_matrix_1, rotation_matrix_2, mirror_matrix, shear_matrix_1, shear_matrix_2, stretch_matrix_1, stretch_matrix_2]
    transform_matrices = [rotation_matrix_1, rotation_matrix_2, mirror_matrix, stretch_matrix_1, stretch_matrix_2]
    coords = np.zeros((num_parts, num_keypoints, len(transform_matrices), 2))
    transformed_images = []
    for M in transform_matrices:
        transformed_img = apply_transform(img, M)
        transformed_images.append(transformed_img)
    
    results = model(transformed_images, verbose=False)
    
    for j, (result, M) in enumerate(zip(results, transform_matrices)):
        keypoints = result.keypoints[0].cpu()
        inv_matrix = cv2.invertAffineTransform(M)
        keypoints_np = keypoints.data[0].numpy()
        keypoints_2d = keypoints_np[:,:2]
        ones = np.ones((keypoints_2d.shape[0], 1))
        keypoints_homogeneous = np.hstack((keypoints_2d, ones))
        transformed_keypoints_homogeneous = inv_matrix @ keypoints_homogeneous.T
        transformed_keypoints = keypoints_np.copy()
        transformed_keypoints[:, :2] = transformed_keypoints_homogeneous.T
        for part_idx, keypoint_arr in enumerate(all_keypoint_arr):
            for i, kpt in enumerate(transformed_keypoints):
            # import pdb; pdb.set_trace()
                x = kpt[0]
                y = kpt[1]
                if kpt[2] > keypoint_threshold and i in keypoint_arr:  # Confidence threshold
                    # import pdb; pdb.set_trace()
                    point = np.array([x, y])
                    coords[part_idx, i, j] = point
    for part_idx in range(num_parts):
        if var_parts[part_idx] == -1:
            var_parts[part_idx] = sys.maxsize
            continue
        for idx in range(num_keypoints):
            var_parts[part_idx] += np.var(coords[part_idx, idx, :, 0]) + np.var(coords[part_idx, idx, :, 1])
    return var_parts

def get_image_variance(img, keypoint_arr, kpt_thresh):
    # First, filter the image, regarding number of keypoints found:
    results = model(img, verbose=False)
    # import pdb; pdb.set_trace()
    keypoints = results[0].keypoints[0].data[0]
    keypoints_detected = 0
    for idx, keypoint in enumerate(keypoints):
        if keypoint[2] > keypoint_threshold and idx in keypoint_arr:
            keypoints_detected += 1
    if keypoints_detected < kpt_thresh and kpt_thresh != -1:
            return sys.maxsize
    rows, cols, channels = img.shape
    rotation_matrix_1 = cv2.getRotationMatrix2D((cols/2, rows/2), 7, 0.85)
    rotation_matrix_2 = cv2.getRotationMatrix2D((cols/2, rows/2), -7, 0.85)
    mirror_matrix = np.array([[-1, 0, cols], [0, 1, 0]], dtype=np.float32)
    shear_matrix_1 = np.array([[1, 0.75, 0], [0, 1, 0]], dtype=np.float32)
    shear_matrix_2 = np.array([[1, 0, 0], [0.75, 1, 0]], dtype=np.float32)
    transform_matrices = [rotation_matrix_1, rotation_matrix_2, mirror_matrix, shear_matrix_1, shear_matrix_2]
    coords = np.zeros((num_keypoints, len(transform_matrices), 2))
    for j, M in enumerate(transform_matrices):
        tranformed_img = apply_transform(img, M)
        # Omer: Can do it better with batched run
        results = model(tranformed_img, verbose=False)
        keypoints = results[0].keypoints[0].cpu()
        inv_matrix = cv2.invertAffineTransform(M)
        keypoints_np = keypoints.data[0].numpy()
        keypoints_2d = keypoints_np[:,:2]
        ones = np.ones((keypoints_2d.shape[0], 1))
        keypoints_homogeneous = np.hstack((keypoints_2d, ones))
        transformed_keypoints_homogeneous = inv_matrix @ keypoints_homogeneous.T
        transformed_keypoints = keypoints_np.copy()
        transformed_keypoints[:, :2] = transformed_keypoints_homogeneous.T
        for i, kpt in enumerate(transformed_keypoints):
        # import pdb; pdb.set_trace()
            x = kpt[0]
            y = kpt[1]
            if kpt[2] > 0 and i in keypoint_arr:  # Confidence threshold
                # import pdb; pdb.set_trace()
                point = np.array([x, y])
                # import pdb; pdb.set_trace()
                coords[i, j] = point
        total_variance = 0
    for idx in range(num_keypoints):
        total_variance += np.var(coords[idx, :, 0]) + np.var(coords[idx, :, 1])
    return total_variance

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

def filter_image(image):
    if image.shape[0] / image.shape[1] < aspect_ratio_threshold or image.shape[1] / image.shape[0] < aspect_ratio_threshold:
        return False
    return True

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


def get_num_keypoints_part(image, part_keypoints, keypoints = None):
    # import pdb; pdb.set_trace()
    num_of_keypoints = 0
    for idx, keypoint in enumerate(keypoints):
        if keypoint[2] > keypoint_threshold and idx in part_keypoints:
            num_of_keypoints += 1
    return num_of_keypoints

def get_part_confidence(image, part_keypoints):
    # Detect persons in the image
    results = model(image, verbose=False)
    import pdb; pdb.set_trace()
    confidence = 0
    for result in results:
        num_of_keypoints = 0
        for idx, keypoint in enumerate(result.keypoints.data.tolist()[0]):
            if keypoint[2] > keypoint_threshold and idx in part_keypoints:
                num_of_keypoints += 1
                confidence += keypoint[2]
    if num_of_keypoints == 0:
        return -1
    return confidence #/ num_of_keypoints

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


    videos_pats = "/home/user_219/omer_gil/BoT-SORT/images_final_run"
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
                        # segmented_image = segment_person(img, predictor=predictor)
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

def main_save_max(max_to_save = 10):
    # add filter based on num of key points and remove segment
    # Initialize model configuration and predictor
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold for object detection
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # predictor = DefaultPredictor(cfg)
    head_keys = [0, 1, 2, 3, 4]
    # body_keys = [5, 6, 7, 8, 9, 10, 11, 12]
    body_keys = [5, 6, 11, 12]
    left_foot_keys = [14, 16]
    right_foot_keys = [13, 15]
    all_keys = [head_keys, body_keys, left_foot_keys, right_foot_keys]

    videos_pats = "/home/user_219/omer_gil/BoT-SORT/images_final_run"
    # output_folder = "/home/user_219/omer_gil/BoT-SORT/best_images_parts_gil"
    output_folder = "/home/user_219/omer_gil/BoT-SORT/best_images_parts"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    videos = sorted(os.listdir(videos_pats))
    for vid in videos:
        print("Processing video:", vid)
        # Get all the folders in the video directory
        ids = sorted(os.listdir(os.path.join(videos_pats, vid)))
        print("ID's are: ", ids)
        for id in ids:
            min_heap_rf = []
            min_heap_lf = []
            best_head_img = -1
            best_body_img = -1
            best_left_foot_img = -1
            best_right_foot_img = -1
            best_head_keypoints = 0
            best_body_keypoints = 0
            best_head_score = sys.maxsize
            best_body_score = sys.maxsize
            best_left_foot_score = sys.maxsize
            best_right_foot_score = sys.maxsize
            # best_head_score = -1
            # best_body_score = -1
            # best_left_foot_score = -1
            # best_right_foot_score = -1
            print("Processing person ID:", id)
            # Get all the images in the id directory
            images = os.listdir(os.path.join(videos_pats, vid, id))
            # Get max ammt of keypoints for head and body:
            bottom_images = []
            for image in tqdm(images):
                img = cv2.imread(os.path.join(videos_pats, vid, id, image))
                if not filter_image(img):
                    continue
                try:
                    # import pdb; pdb.set_trace()
                    results = model(img, verbose=False)
                    num_keypoints = get_num_keypoints_part(img, head_keys, results[0].keypoints.data.tolist()[0])
                    if num_keypoints > best_head_keypoints:
                        best_head_keypoints = num_keypoints
                    num_keypoints = get_num_keypoints_part(img, body_keys, results[0].keypoints.data.tolist()[0])
                    if num_keypoints > best_body_keypoints:
                        best_body_keypoints = num_keypoints
                    # if get_num_keypoints_part(img, left_foot_keys + right_foot_keys, results[0].keypoints.data.tolist()[0]) > 0:
                    #     bottom_images.append(image)
                except:
                    print("Error in filtering person in image: " + image)
                    exit()
            kpt_thresh = [best_head_keypoints, best_body_keypoints]
            for image in tqdm(images):
                # import pdb; pdb.set_trace()
                img = cv2.imread(os.path.join(videos_pats, vid, id, image))
                if not filter_image(img):
                    continue
                # try:
                var = get_image_variance_batched(img, all_keys, kpt_thresh)
                if var[0] < best_head_score:
                    best_head_score = var[0]
                    best_head_img = image
                if var[1] < best_body_score:
                    best_body_score = var[1]
                    best_body_img = image
                if var[2] < best_left_foot_score:
                    best_left_foot_score = var[2]
                    best_left_foot_img = image
                if len(min_heap_lf) < max_to_save:
                    heapq.heappush(min_heap_lf, (-1 * var[2], image))
                else:
                    heapq.heappushpop(min_heap_lf, (-1 * var[2], image))
                if var[3] < best_right_foot_score:
                    best_right_foot_score = var[3]
                    best_right_foot_img = image
                if len(min_heap_rf) < max_to_save:
                    heapq.heappush(min_heap_rf, (-1 * var[3], image))
                else:
                    heapq.heappushpop(min_heap_rf, (-1 * var[3], image))
                
                    # # Best head image:
                    # if best_head_keypoints > 0:
                    #     var = get_image_variance_batched(img, head_keys, best_head_keypoints)
                    #     if var < best_head_score:
                    #         best_head_score = var
                    #         best_head_img = image
                    # # Best body image:
                    # if best_body_keypoints > 0:
                    #     var = get_image_variance_batched(img, body_keys, best_body_keypoints - 1)
                    #     if var < best_body_score:
                    #         best_body_score = var
                    #         best_body_img = image
                    # # Best left foot image:
                    # var = get_image_variance_batched(img, left_foot_keys, -1)
                    # if var < best_left_foot_score:
                    #     best_left_foot_score = var
                    #     best_left_foot_img = image
                    # # Best right foot image:
                    # var = get_image_variance_batched(img, right_foot_keys, -1)
                    # if var < best_right_foot_score:
                    #     best_right_foot_score = var
                    #     best_right_foot_img = image
                #     # Best head image:
                #     conf = get_part_confidence(img, head_keys)
                #     if conf > best_head_score:
                #         best_head_score = conf
                #         best_head_img = image
                #     # Best body image:
                #     conf = get_part_confidence(img, body_keys)
                #     if conf > best_body_score:
                #         best_body_score = conf
                #         best_body_img = image
                #     # Best left foot image:
                #     conf = get_part_confidence(img, left_foot_keys)
                #     if conf > best_left_foot_score:
                #         best_left_foot_score = conf
                #         best_left_foot_img = image
                #     # Best right foot image:
                #     conf = get_part_confidence(img, right_foot_keys)
                #     if conf > best_right_foot_score:
                #         best_right_foot_score = conf
                #         best_right_foot_img = image
                # except:
                #     print("Error in filtering person in image: " + image)
                #     exit()
            path = os.path.join(output_folder, vid, id)
            if (not os.path.exists(path)) and (best_head_img != -1 or best_body_img != -1):
                os.makedirs(path)
            # Should probably do something better here... cv2 functions just to copy the image to a new location and to rename them is just dumb.
            if best_head_img != -1:
                img = cv2.imread(os.path.join(videos_pats, vid, id, best_head_img))
                # Save the best part's image
                cv2.imwrite(os.path.join(path, 'head_'+ best_head_img), img)
            if best_body_img != -1:
                img = cv2.imread(os.path.join(videos_pats, vid, id, best_body_img))
                # Save the best part's image
                cv2.imwrite(os.path.join(path, 'body_'+ best_body_img), img)
            if best_head_img == -1 and best_body_img == -1:
                continue
            for var, bottom_img in min_heap_rf:
                img = cv2.imread(os.path.join(videos_pats, vid, id, bottom_img))
                cv2.imwrite(os.path.join(path, 'bottom_'+ bottom_img), img)
            for var, bottom_img in min_heap_lf:
                img = cv2.imread(os.path.join(videos_pats, vid, id, bottom_img))
                cv2.imwrite(os.path.join(path, 'bottom_'+ bottom_img), img)
            # for bottom_img in bottom_images:
            #     img = cv2.imread(os.path.join(videos_pats, vid, id, bottom_img))
            #     # Save the best part's image
            #     cv2.imwrite(os.path.join(path, 'bottom_'+ bottom_img), img)
                
            # if best_left_foot_img != -1:
            #     img = cv2.imread(os.path.join(videos_pats, vid, id, best_left_foot_img))
            #     # Save the best part's image
            #     cv2.imwrite(os.path.join(path, 'lf_'+ best_left_foot_img), img)
            # if best_right_foot_img != -1:
            #     img = cv2.imread(os.path.join(videos_pats, vid, id, best_right_foot_img))
            #     # Save the best part's image
            #     cv2.imwrite(os.path.join(path, 'rf_'+ best_right_foot_img), img)




if __name__ == "__main__":
    # main()
    main_save_max(5)