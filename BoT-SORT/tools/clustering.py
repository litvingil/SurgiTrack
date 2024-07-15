import os
import shutil
from collections import defaultdict

def clustering(features_path):
    if features_path[-1] == '1':
        classes_dict = {
                            0:[20,21],
                            1:[1,7,10,11,14,17],
                            2:[3,4,5,6,16,18],
                            3:[8,9]
                        }
    if features_path[-1] == '2':
        classes_dict = {
                    0:[1,24],
                    1:[2,7,13,21,23],
                    2:[5],
                    3:[6,8,25],
                    4:[12,14],
                    5:[16,20]
                  }
    return classes_dict


def clustering1(features_path):
    if features_path[-1] == '1':
        classes_dict = {
                            0:[20,21],
                            1:[1,7,10,11,14,17],
                            2:[3,4,5,6,16,18],
                            3:[8,9]
                        }
    if features_path[-1] == '2':
        classes_dict = {
                    0:[1,24,44,49,54,57,58,59,61],
                    1:[2,7,13,21,23,29,38,39,47,53,56],
                    2:[5],
                    3:[6,8,25],
                    4:[12,14,35],
                    5:[16,20,31,34,36,55]
                  }
    return classes_dict

def rearrange_folders(cluster_map , input_path, output_path):
    
    # Create and move content to new cluster folders
    for cluster, folders in cluster_map.items():
        cluster_folder_path = os.path.join(output_path,str(cluster))
        os.makedirs(cluster_folder_path, exist_ok=True)  # Ensure the folder exists
        for folder in folders:
            folder_path = os.path.join(input_path, str(folder))
            for item in os.listdir(folder_path):
                src_path = os.path.join(folder_path, item)
                dst_path = os.path.join(cluster_folder_path, item)
                shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    input_path = '/home/user_219/omer_gil/BoT-SORT/images_final_run'
    output_path = '/home/user_219/omer_gil/BoT-SORT/images_clustered'
    os.makedirs(output_path, exist_ok=True)
    folders = os.listdir(input_path)
    for folder in folders:
        # import pdb; pdb.set_trace()
        camera_path = os.path.join(input_path, folder)
        clusterd_camera_path = os.path.join(output_path, folder)
        image_folders = os.listdir(camera_path)
        classes = clustering(folder)
        rearrange_folders(classes, camera_path,clusterd_camera_path) 