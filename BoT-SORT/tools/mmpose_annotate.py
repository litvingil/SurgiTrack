from mmpose.apis import MMPoseInferencer
img_path = '/home/user_219/omer_gil/Self-Correction-Human-Parsing/test_images/15_keypoints_frame_293.jpg'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('human')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
# import os
# import cv2

# from detectron2.utils.logger import setup_logger
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# import numpy as np

# from tqdm import tqdm

# import heapq

# save_best = True

# # conda activate dinov2
# keypoint_threshold = 0.7
# num_keypoint_threshold = 6
# model = YOLO("/home/user_219/omer_gil/yolov8n-pose.pt")
# image = "/home/user_219/omer_gil/Self-Correction-Human-Parsing/test_images/15_keypoints_frame_293.jpg"
# img = cv2.imread(image)
# def main():
#     results = model(img, verbose=False)
#     keypoints = results[0].keypoints.data.tolist()[0]
#     keypoints2 = results[0].keypoints
#     # Plot keypoints on the image
#     for i, kpt in enumerate(keypoints):
#         import pdb; pdb.set_trace()
#         x = kpt[0]
#         y = kpt[1]
#         if kpt[2] > 0.75:  # Confidence threshold
#             cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
#             cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX ,  
#                    fontScale = 1, color=(0, 255, 0), thickness = 3, lineType = cv2.LINE_AA) 
#     cv2.imwrite("/home/user_219/omer_gil/BoT-SORT/best_images_parts/2/3/body_frame_259_2.jpg", img)
    
    

# if __name__ == "__main__":
#     # main()
#     main()