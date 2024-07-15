import os
import torch
import torchvision.transforms as T
import hubconf
# import dinov2_test.dinov2.hubconf as hubconf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
batch_size = 5 #50
from tqdm import tqdm
# conda activate dinov2

def main():
    input_folder = '/home/user_219/omer_gil/BoT-SORT/images_segmented'
    feature_folder = '/home/user_219/omer_gil/BoT-SORT/images_feature_extracted'
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    # Load the base dino model
    dino = hubconf.dinov2_vitb14()
    dino = dino.cuda()
    # Preprocess & convert to tensor
    transform = T.Compose([
              T.ToTensor(),
              T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
              T.CenterCrop(560),
              T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    for vid_id in range(4,5):
      print("Extracting features for video " + str(vid_id) + "...")
      current_vid_dir = os.path.join(input_folder, str(vid_id))
      ids_dir = [
      f for f in os.listdir(current_vid_dir) if os.path.isdir(os.path.join(current_vid_dir, f))]
      features_folder_current_vid = os.path.join(feature_folder, str(vid_id))
      if not os.path.exists(features_folder_current_vid):
          os.makedirs(features_folder_current_vid)
      for id in ids_dir:
        # Create output directory:
          id_output = os.path.join(features_folder_current_vid, id)
          if not os.path.exists(id_output):
              os.makedirs(id_output)
              
          current_id_dir = os.path.join(current_vid_dir, id)
          print("Extracting features for ID " + id + "...")
          # Divide into batches and extract features:
          image_files = os.listdir(current_id_dir)
          files_num = len(image_files)
          batches = [list(range(i, min(i + batch_size, files_num))) for i in range(0, files_num, batch_size)]
          print("Number of batches: ", len(batches))
          batch_num = 0

          for batch in tqdm(batches):
            # print("batch number: ", batch_num)
            images = []
            for index in batch:
                img = Image.open(os.path.join(current_id_dir, image_files[index])).convert("RGB") 
                img_array = np.array(img)
                images.append(img_array)

            # Use image tensor to load in a batch of images
            imgs_tensor = torch.zeros(batch_size, 3, 560, 560)

            # import pdb; pdb.set_trace() 
            for j, img in enumerate(images):
                imgs_tensor[j] = transform(img)[:3]

            # Inference
            with torch.no_grad():
                features_dict = dino.forward_features(imgs_tensor.cuda())
                features = features_dict['x_norm_patchtokens']

            # print("i is ",i)
            # print(features.shape)
            # print(features.cpu().numpy().shape)
            output_path = os.path.join(id_output, 'batch' + str(batch_num) + '.npy')
            #save features
            np.save(output_path, features.cpu().numpy())
            batch_num+=1

          # avg_features = np.zeros(0)
          # for batch in batches:
          #   print("Batch number: ", batch_num)
          #   images = []
          #   for index in batch:
          #     img = Image.open(os.path.join(current_id_dir, image_files[index])).convert("RGB") 
          #     img_array = np.array(img)
          #     images.append(img_array)

          #   # Use image tensor to load in a batch of images
          #   imgs_tensor = torch.zeros(batch_size, 3, 560, 560)

          #   for i, img in enumerate(images):
          #     imgs_tensor[i] = transform(img)[:3]

          #   # Inference
          #   with torch.no_grad():
          #     features_dict = dino.forward_features(imgs_tensor.cuda())
          #     features = features_dict['x_norm_patchtokens']

          #   # print(features.shape)
          #   # if avg_features.shape == (0,):
          #   #   avg_features = features.cpu().numpy()
          #   # else:
          #   #   avg_features = avg_features + features.cpu().numpy()
          #   batch_num += 1
          # file_path = os.path.join(id_output, 'features.npy')
          # avg_features /= len(batches)
          # # Save features vector:
          # with open(file_path, 'wb') as f:
          #   np.save(f, features.cpu().numpy())
        
def main_fearutes_body_parts():
    parts = ['head', 'lower', 'upper', 'shoes']
    input_folder = '/home/user_219/omer_gil/Self-Correction-Human-Parsing/outputs2'
    feature_folder = '/home/user_219/omer_gil/Self-Correction-Human-Parsing/outputs2_features'
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    # Load the base dino model
    dino = hubconf.dinov2_vitb14()
    # dino.eval()
    dino = dino.cuda()
    # Preprocess & convert to tensor
    transform = T.Compose([
              T.ToTensor(),
              T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
              T.CenterCrop(560),
              T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
    images_list = os.listdir(input_folder)
      
    for part in parts:
      images_part = [image_name for image_name in images_list if part in image_name]
      images_part = sorted(images_part)

      # Create output directory:
      part_output = os.path.join(feature_folder, part)
      if not os.path.exists(part_output):
          os.makedirs(part_output)
          
      files_num = len(images_part)
      batches = [list(range(i, min(i + batch_size, files_num))) for i in range(0, files_num, batch_size)]
      print("Number of batches: ", len(batches))
      print("batches:")
      print(batches)
      batch_num = 0

      for batch in tqdm(batches):
        # print("batch number: ", batch_num)
        images = []
        for index in batch:
            img = Image.open(os.path.join(input_folder, images_part[index])).convert("RGB") 
            img_array = np.array(img)
            images.append(img_array)

        # Use image tensor to load in a batch of images
        imgs_tensor = torch.zeros(batch_size, 3, 560, 560)

        # import pdb; pdb.set_trace() 
        for j, img in enumerate(images):
            imgs_tensor[j] = transform(img)[:3]

        # Inference
        with torch.no_grad():
            features_dict = dino.forward_features(imgs_tensor.cuda())
            features = features_dict['x_norm_patchtokens']
            # features = dino(imgs_tensor.cuda())

        # print("i is ",i)
        # print(features.shape)
        # print(features.cpu().numpy().shape)
        # import pdb; pdb.set_trace()
        output_path = os.path.join(part_output, 'batch_' + str(part) + '.npy')
        #save features
        np.save(output_path, features.cpu().numpy())


if __name__ == "__main__":
  # main()
  main_fearutes_body_parts()
