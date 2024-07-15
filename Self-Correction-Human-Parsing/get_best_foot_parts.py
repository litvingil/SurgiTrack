#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# python3 simple_extractor.py --dataset 'lip' --model-restore 'checkpoints/final.pth' --input-dir 'test_images' --output-dir 'outputs'

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='checkpoints/final.pth', help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

import matplotlib.pyplot as plt

def get_mask_size(masks_img, label_index):
    mask = (masks_img == label_index)
    # import pdb; pdb.set_trace()
    return np.count_nonzero(mask)    

def get_largest_mask_feet(masks_imgs, label):
    largests_mask_lf = 0
    largests_mask_rf = 0
    lf_index = -1
    rf_index = -1
    # import pdb; pdb.set_trace()
    for idx, masks_img in enumerate(masks_imgs):
        size = get_mask_size(masks_img, label.index('Left-shoe'))
        if size > largests_mask_lf:
            largests_mask_lf = size
            lf_index = idx
        size = get_mask_size(masks_img, label.index('Right-shoe'))
        if size > largests_mask_rf:
            largests_mask_rf = size
            rf_index = idx
    return lf_index, rf_index        
    

def main():
    # videos_pats = "/home/user_219/omer_gil/BoT-SORT/best_images_parts"
    videos_pats = "/home/user_219/omer_gil/BoT-SORT/best_images_parts"
    output_folder = "/home/user_219/omer_gil/BoT-SORT/best_images_parts"
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    
    
    videos = sorted(os.listdir(videos_pats))
    for vid in videos:
        print("Processing video ", vid)
        ids = sorted(os.listdir(os.path.join(videos_pats, vid)))
        for id in tqdm(ids):
            masks = []
            images = []
            names = []
            with torch.no_grad():
                root_dir = os.path.join(videos_pats, vid, id)
                bottom_file_list = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) ])# if "bottom_" in f])
                print(bottom_file_list)
                dataset = SimpleFolderDataset(root = root_dir, file_list = bottom_file_list, input_size=input_size, transform=transform)
                dataloader = DataLoader(dataset)
                for idx, batch in enumerate(dataloader):
                    image, meta = batch
                    img_name = meta['name'][0]
                    print("img_name is ", img_name)
                    c = meta['center'].numpy()[0]
                    s = meta['scale'].numpy()[0]
                    w = meta['width'].numpy()[0]
                    h = meta['height'].numpy()[0]
                    output = model(image.cuda())
                    upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
                    upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                    upsample_output = upsample_output.squeeze()
                    upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

                    logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
                    parsing_result = np.argmax(logits_result, axis=2)
                    print(os.path.basename(img_name) + " LS Size: " + str(get_mask_size(parsing_result, label.index('Left-shoe'))))
                    print(os.path.basename(img_name) + " RS Size: " + str(get_mask_size(parsing_result, label.index('Right-shoe'))))
                    
                    masks.append(parsing_result)
                    # Worst coding ever here:
                    images.append(image)
                    names.append(img_name)
            idx_lf, idx_rf = get_largest_mask_feet(masks, label)
            img_rf_path = None
            img_lf_path = None
            # More bad programming ahead:
            if idx_rf != -1:
                img_rf_path = names[idx_rf]
                image_rf = Image.open(img_rf_path)
                image_rf.save(os.path.join(os.path.dirname(img_rf_path), 'rf_best_image_' + os.path.basename(img_rf_path)))
            if idx_lf != -1:
                img_lf_path = names[idx_lf]
                image_lf = Image.open(img_lf_path)
                image_lf.save(os.path.join(os.path.dirname(img_lf_path), 'lf_best_image_' + os.path.basename(img_lf_path)))
            # cv2.imwrite(os.path.join(output_path, 'rf_best_image_' + names[idx_rf]), image_rf)
            # cv2.imwrite(os.path.join(output_path, 'lf_best_image_' + names[idx_lf]), image_lf)
    return


if __name__ == '__main__':
    
    main()
