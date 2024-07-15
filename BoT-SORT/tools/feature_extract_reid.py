import argparse
import glob
import os
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.backends import cudnn
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
from tqdm import tqdm
import sys
sys.path.append('/home/user_219/omer_gil/BoT-SORT/fast_reid')
sys.path.append('/home/user_219/omer_gil/BoT-SORT/fast_reid/demo')
from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.utils.logger import setup_logger
from fast_reid.fastreid.utils.file_io import PathManager
from predictor import FeatureExtractionDemo

cudnn.benchmark = True
setup_logger(name="fastreid")

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

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
    parts = ['head']
    input_folder = '/home/user_219/omer_gil/Self-Correction-Human-Parsing/outputs2'
    input_folder = '/home/user_219/omer_gil/Self-Correction-Human-Parsing/test3'
    feature_folder = '/home/user_219/omer_gil/Self-Correction-Human-Parsing/outputs2_features'
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    # Load the base dino model
    args = get_parser().parse_args()

    train_data = 'DukeMTMC'
    method = 'sbs_S50'  # bagtricks_S50 | sbs_S50
    seq = 'MOT20-02'

    args.config_file = r'/home/user_219/omer_gil/BoT-SORT/fast_reid/configs/' + train_data + '/' + method + '.yml'
    # args.input = [r'/home/nir/Datasets/MOT20/train/' + seq + '/img1', '*.jpg']
    # args.output = seq + '_' + method + '_' + train_data
    args.opts = ['MODEL.WEIGHTS', '/home/user_219/omer_gil/BoT-SORT/fast_reid/pretrained/duke_bot_S50.pth']

    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=True)
    feature_files = []
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
        vectors = []
        for image in tqdm(images_part):
            path = os.path.join(input_folder, image)
            img = cv2.imread(path)
            feat = demo.run_on_image(img)
            feat = postprocess(feat)
            # import pdb; pdb.set_trace()
            vectors.append(feat)
        feature_mat = np.stack(vectors, axis = 0)
        np.save(os.path.join(part_output, 'batch_' + str(part) + '.npy'), feature_mat)

        # # Inference
        # with torch.no_grad():
        #     features_dict = dino.forward_features(imgs_tensor.cuda())
        #     features = features_dict['x_norm_patchtokens']
        #     features = dino(imgs_tensor.cuda())

        # output_path = os.path.join(part_output, 'batch_' + str(part) + '.npy')
        # #save features
        # np.save(output_path, features.cpu().numpy())


if __name__ == "__main__":
  # main()
  main_fearutes_body_parts()
