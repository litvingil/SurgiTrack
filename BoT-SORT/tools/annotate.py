# import

import os
import cv2

from ultralytics import YOLO
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np

from tqdm import tqdm

import heapq
  
save_best = True

# conda activate dinov2
keypoint_threshold = 0.65
num_keypoint_threshold = 6
model = YOLO("/home/user_219/omer_gil/yolov8m-pose.pt")
# image = "/home/user_219/omer_gil/Self-Correction-Human-Parsing/test_images/15_keypoints_frame_293.jpg"
image1 = "/home/user_219/omer_gil/BoT-SORT/images/1/1/frame_156.jpg"
image2 = "/home/user_219/omer_gil/BoT-SORT/images/1/1/frame_170.jpg"
image3 = "/home/user_219/omer_gil/BoT-SORT/images/1/1/frame_191.jpg"
image4 = "/home/user_219/omer_gil/BoT-SORT/images/2/13/frame_3680.jpg"
image5 = "/home/user_219/omer_gil/BoT-SORT/images/2/13/frame_3693.jpg"
image6 = "/home/user_219/omer_gil/BoT-SORT/images/2/14/frame_3632.jpg"
image7 = "/home/user_219/omer_gil/BoT-SORT/images/2/44/frame_34217.jpg"
image_annot = "/home/user_219/omer_gil/BoT-SORT/images/1/1/frame_191.jpg"
# image = "/home/user_219/omer_gil/BoT-SORT/best_images_parts/2/3/body_frame_259_rotated.jpg"


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


# Function to apply the transformation 
# def apply_transform(img, transform_matrix):
#     rows, cols, channels = img.shape 
#     # Calculate new dimensions 
#     new_cols = int(cols * transform_matrix[0, 0]) 
#     new_rows = int(rows * transform_matrix[1, 1]) 
#     # Update the transformation matrix to account for the new dimensions 
#     # scale_matrix = np.array([[new_cols / cols, 0, 0], [0, new_rows / rows, 0]], dtype=np.float32) 
#     # combined_matrix = np.dot(scale_matrix, transform_matrix) 
#     # Apply the transformation 
#     transformed_img = cv2.warpAffine(img, transform_matrix, (new_cols, new_rows)) 
#     return transformed_img


def get_image_variance_batched(img, keypoint_arr):
    num_keypoints = 17
    rows, cols, channels = img.shape
    rotation_matrix_1 = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 0.6)
    rotation_matrix_2 = cv2.getRotationMatrix2D((cols/2, rows/2), -30, 0.6)
    mirror_matrix = np.array([[-1, 0, cols], [0, 1, 0]], dtype=np.float32)
    shear_matrix_1 = np.array([[1, 0.5, 0], [0, 1, 0]], dtype=np.float32)
    shear_matrix_2 = np.array([[1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    stretch_matrix_1 = np.array([[1.5, 0, 0], [0, 1, 0]], dtype=np.float32)
    stretch_matrix_2 = np.array([[1, 0, 0], [0, 1.5, 0]], dtype=np.float32)
    # transform_matrices = [rotation_matrix_1, rotation_matrix_2, mirror_matrix, shear_matrix_1, shear_matrix_2, stretch_matrix_1, stretch_matrix_2]
    transform_matrices = [rotation_matrix_1, rotation_matrix_2, mirror_matrix, stretch_matrix_1, stretch_matrix_2]
    coords = np.zeros((num_keypoints, len(transform_matrices), 2))
    transformed_images = []
    for i, M in enumerate(transform_matrices):
        transformed_img = apply_transform(img, M)
        transformed_images.append(transformed_img)
        
        cv2.imwrite("/home/user_219/omer_gil/test_images/transformed_image_" + str(i) + ".jpg", transformed_img)
    
    results = model(transformed_images, verbose=False)
    for j, (result, M) in enumerate(zip(results, transform_matrices)):
        print("j = ", j, ", len matrices = ", len(transform_matrices), "len results = ", len(results))
        keypoints = result.keypoints[0].cpu()
        output_name_skeleton = "/home/user_219/omer_gil/test_images/transformed_image_skeleton_" + str(j) + ".jpg"
        output_name_keypoints = "/home/user_219/omer_gil/test_images/transformed_image_keypoints_" + str(j) + ".jpg"
        plot_skeleton(transformed_images[j], keypoints, output_name_skeleton)
        annotate(transformed_images[j], keypoints, output_name_keypoints)
        inv_matrix = cv2.invertAffineTransform(M)
        keypoints_np = keypoints.data[0].numpy()
        keypoints_2d = keypoints_np[:,:2]
        ones = np.ones((keypoints_2d.shape[0], 1))
        keypoints_homogeneous = np.hstack((keypoints_2d, ones))
        transformed_keypoints_homogeneous = inv_matrix @ keypoints_homogeneous.T
        transformed_keypoints = keypoints_np.copy()
        transformed_keypoints[:, :2] = transformed_keypoints_homogeneous.T
        output_name_skeleton_combined =  "/home/user_219/omer_gil/test_images/transformed_image_skeleton_combined_" + str(j) + ".jpg"
        img = plot_skeleton_2(img, transformed_keypoints, output_name_skeleton_combined, ((j * 50) % 255, (j * 100) % 255, 255), (255, (j * 100) % 255, (j * 50) % 255))
        for i, kpt in enumerate(transformed_keypoints):
        # import pdb; pdb.set_trace()
            x = kpt[0]
            y = kpt[1]
            if kpt[2] > keypoint_threshold and i in keypoint_arr:  # Confidence threshold
                # import pdb; pdb.set_trace()
                point = np.array([x, y])
                # import pdb; pdb.set_trace()
                coords[i, j] = point
        total_variance = 0
    for idx in range(num_keypoints):
        total_variance += np.var(coords[idx, :, 0]) + np.var(coords[idx, :, 1])
    return total_variance


def get_image_variance(img, keypoint_arr):
    num_keypoints = 17
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
    coords = np.zeros((num_keypoints, len(transform_matrices), 2))
    for j, M in enumerate(transform_matrices):
        tranformed_img = apply_transform(img, M)
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
            if kpt[2] > keypoint_threshold and i in keypoint_arr:  # Confidence threshold
                # import pdb; pdb.set_trace()
                point = np.array([x, y])
                # import pdb; pdb.set_trace()
                coords[i, j] = point
        total_variance = 0
        for idx in range(num_keypoints):
            total_variance += np.var(coords[idx, :, 0]) + np.var(coords[idx, :, 1])
        return total_variance

    
    # Now, iterate over the coords array, and compute total variance of all coordinates:
     
                
    # rotated = apply_transform(img, rotation_matrix)
    # cv2.imwrite("/home/user_219/omer_gil/BoT-SORT/best_images_parts/2/3/body_frame_259_skewed.jpg", rotated)
    
    # keypoints2 = results[0].keypoints
    # Plot keypoints on the image
    
def plot_skeleton(image, keypoints = None, img_name = None):
    if not keypoints:
        results = model(image, verbose=False)
        keypoints = results[0].keypoints[0].cpu()
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), 
        (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    # Create a copy of the image to draw on
    img_copy = image.copy()
    # Draw keypoints
    for x, y, conf in keypoints.data[0]:
        if conf > keypoint_threshold:
            cv2.circle(img_copy, (int(x), int(y)), radius = 5, color = (0, 255, 0), thickness = -1)
    # Draw skeleton connections
    for i, j in skeleton:
        # import pdb; pdb.set_trace()
        if len(keypoints.data[0]) <= i or len(keypoints.data[0]) <= j:
            continue
        x1, y1, conf1 = keypoints.data[0][i]
        x2, y2, conf2 = keypoints.data[0][j]
        if conf1 > keypoint_threshold and conf2 > keypoint_threshold:
            cv2.line(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color = (255, 0, 0), thickness = 2)
    # Convert BGR image to RGB for plotting
    if not img_name:
        img_name = "/home/user_219/omer_gil/test_images/skeleton.jpg"
    cv2.imwrite(img_name, img_copy)
    return img_copy


def plot_skeleton_2(image, keypoints = None, img_name = None, color1 = (0, 255, 0), color2 = (255, 0, 0)):
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), 
        (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    # Create a copy of the image to draw on
    img_copy = image.copy()
    # Draw keypoints
    for x, y, conf in keypoints:
        if conf > keypoint_threshold:
            cv2.circle(img_copy, (int(x), int(y)), radius = 5, color = color1, thickness = -1)
    # Draw skeleton connections
    for i, j in skeleton:
        # import pdb; pdb.set_trace()
        if len(keypoints) <= i or len(keypoints) <= j:
            continue
        x1, y1, conf1 = keypoints[i]
        x2, y2, conf2 = keypoints[j]
        if conf1 > keypoint_threshold and conf2 > keypoint_threshold:
            cv2.line(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color = color2, thickness = 2)
    # Convert BGR image to RGB for plotting
    if not img_name:
        img_name = "/home/user_219/omer_gil/test_images/skeleton.jpg"
    cv2.imwrite(img_name, img_copy)
    return img_copy
    
def annotate(img , keypoints = None, img_name = None):    
    if not keypoints:
        results = model(img, verbose=False)
        keypoints = results[0].keypoints[0].cpu()
    # keypoints2 = results[0].keypoints
    # Plot keypoints on the image
    # import pdb; pdb.set_trace()
    img_copy = img.copy()
    for i, kpt in enumerate(keypoints.data[0]):
        x = keypoints.xy[0,i, 0]
        y = keypoints.xy[0,i, 1]
        if kpt[2] > keypoint_threshold:  # Confidence threshold
            cv2.circle(img_copy, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.putText(img_copy, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX ,  
                   fontScale = 1, color=(0, 255, 0), thickness = 3, lineType = cv2.LINE_AA)
    if not img_name:
        img_name = "/home/user_219/omer_gil/test_images/keypoints.jpg" 
    cv2.imwrite(img_name, img_copy)
    
def main():
    # img = cv2.imread(image1)
    # # plot_skeleton(img, img_name = "/home/user_219/omer_gil/test_images/skeleton_frame.jpg")
    # var1 = get_image_variance(img, [0, 1, 2, 3, 4])
    # print("Img 1 var = ", var1)
    # var1 = get_image_variance_batched(img, [0, 1, 2, 3, 4])
    # print("Img 1 var batched = ", var1)
    # img = cv2.imread(image2)
    # var2 = get_image_variance(img, [0, 1, 2, 3, 4])
    # print("Img 2 var = ", var2)
    # var2 = get_image_variance_batched(img, [0, 1, 2, 3, 4])
    # print("Img 2 var batched = ", var2)
    # img = cv2.imread(image3)
    # var3 = get_image_variance(img, [0, 1, 2, 3, 4])
    # print("Img 3 var = ", var3)
    # var3 = get_image_variance_batched(img, [0, 1, 2, 3, 4])
    # print("Img 3 var batched = ", var3)
    # img = cv2.imread(image4)
    # var4 = get_image_variance(img, [0, 1, 2, 3, 4])
    # print("Img 4 var = ", var4)
    # var4 = get_image_variance_batched(img, [0, 1, 2, 3, 4])
    # print("Img 4 var batched = ", var4)
    # img = cv2.imread(image5)
    # var5 = get_image_variance(img, [0, 1, 2, 3, 4])
    # print("Img 3 var = ", var5)
    # var5 = get_image_variance_batched(img, [0, 1, 2, 3, 4])
    # print("Img 3 var batched = ", var5)
    img = cv2.imread(image7)
    var6 = get_image_variance(img, [0, 1, 2, 3, 4])
    print("Img 4 var = ", var6)
    var6 = get_image_variance_batched(img, [0, 1, 2, 3, 4])
    print("Img 4 var batched = ", var6)
    
    
    # img_skeleton = cv2.imread('/home/user_219/omer_gil/test_images/depositphotos_65615121-stock-photo-happy-man-isolated-full-body.jpg')
    # plot_skeleton(img_skeleton)
    # img = cv2.imread(image3)
    # var3 = get_image_variance(img, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    # print("Img 3 var = ", var3)
    # img_annotate = cv2.imread(image_annot)
    # annotate(img_annotate)
    
    

if __name__ == "__main__":
    # main()
    main()