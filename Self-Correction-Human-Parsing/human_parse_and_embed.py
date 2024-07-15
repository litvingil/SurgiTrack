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
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

import hubconf
# import dinov2_test.dinov2.hubconf as hubconf

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

class Dino_model:
    def __init__(self) -> None:
        self.dino = hubconf.dinov2_vitb14()
        self.dino = self.dino.cuda()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(560, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(560),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def extract_features(self, images_dict, feature_path):
        # import pdb; pdb.set_trace()
        # Use image tensor to load in a batch of images
        imgs_tensor = torch.zeros(4, 3, 560, 560)

        if isinstance(images_dict["head"], Image.Image):
            imgs_tensor[0] = self.transform(images_dict["head"])[:3]
        if isinstance(images_dict["upper"], Image.Image):
            imgs_tensor[1] = self.transform(images_dict["upper"])[:3]
        if isinstance(images_dict["rf"], Image.Image):
            imgs_tensor[2] = self.transform(images_dict["rf"])[:3]
        if isinstance(images_dict["lf"], Image.Image):
            imgs_tensor[3] = self.transform(images_dict["lf"])[:3]

        # Inference
        with torch.no_grad():
            features_dict = self.dino.forward_features(imgs_tensor.cuda())
            # import pdb; pdb.set_trace()
            features = features_dict['x_norm_patchtokens']

        np.save(feature_path, features.cpu().numpy())
        return


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


def plot_colors(palette, labels, output_path):
    """ Plots the colors from the palette.
    Args:
        palette: The color palette
    """
    # Reshape the palette to have 3 values (R, G, B) per color
    palette = np.array(palette).reshape(-1, 3)
    
    # Create an array of indices for the colors
    indices = np.arange(palette.shape[0])

    # Create a figure and a subplot
    fig, ax = plt.subplots(1, 1, figsize=(15, 2),
                           constrained_layout=True,
                           dpi=80)

    # Set the facecolor of the subplot to the RGB values
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')

    ax.set_xlim(0, palette.shape[0])
    ax.set_ylim(0, 1)
    ax.set_xticks(indices + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha='right')  # Rotate labels for readability
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)
    ax.grid(True, which='major', axis='x', color='white')

    for idx, color in enumerate(palette):
        ax.fill_between([idx, idx+1], 0, 1, color=color/255)

    plt.savefig(output_path)


def crop_non_zero(img, orig):
    """
    Crops an image to the region containing non-zero (non-black) pixels.

    Args:
    img (numpy array): The full image that contains mainly zeros and some non-zero regions.

    Returns:
    PIL.Image: The cropped image containing only the non-zero regions.
    """
    # Convert the image to a NumPy array
    # img_array = np.array(img)
    
    # Check if the image is grayscale or color
    if len(img.shape) == 3:
        # Sum over the color channels to identify non-black (non-zero) pixels
        sum_channels = np.sum(img, axis=2)
        non_zero_indices = np.nonzero(sum_channels)
    else:
        # Directly find non-zero indices for grayscale images
        non_zero_indices = np.nonzero(img)
    
    # Determine the bounding box of non-zero pixels
    y_min, y_max = non_zero_indices[0].min(), non_zero_indices[0].max()
    x_min, x_max = non_zero_indices[1].min(), non_zero_indices[1].max()
    
    img = Image.fromarray(img)

    # Crop the original image using the determined bounding box
    cropped_img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    
    return cropped_img

def apply_mask_to_image(image, mask):
    """
    Applies a 2D boolean mask to a 3D image array (h, w, 3).

    Args:
    image (numpy.ndarray): The input image with shape (h, w, 3).
    mask (numpy.ndarray): A boolean mask with shape (h, w).

    Returns:
    numpy.ndarray: The masked image.
    """
    # Ensure the mask is broadcasted correctly across the color channels
    masked_image = np.copy(image)  # Make a copy of the image to modify
    for c in range(3):  # Assuming the first dimension is the color channels
        masked_image[~mask,c] = 0  # Apply the mask, setting unmasked areas to zero

    return masked_image

def crop_image(masks_img,original_img_path,label,body_part):
    # ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    if body_part == "head":
        mask = ((masks_img == label.index('Hair')) | (masks_img == label.index('Hat')) | (masks_img == label.index('Face')) | (masks_img == label.index('Sunglasses'))| (masks_img == label.index('Scarf')))
    elif body_part == "upper":
        mask = ((masks_img == label.index('Upper-clothes')) | (masks_img == label.index('Dress')) | (masks_img == label.index('Coat')) | (masks_img == label.index('Scarf')) | (masks_img == label.index('Jumpsuits')) | (masks_img == label.index('Right-arm')) | (masks_img == label.index('Left-arm')) | (masks_img == label.index('Glove')))
    elif body_part == "shoes":
        mask = ((masks_img == label.index('Left-shoe')) | (masks_img == label.index('Right-shoe')) | (masks_img == label.index('Socks')))
    elif body_part == "lf":
        mask = masks_img == label.index('Left-shoe')
    elif body_part == "rf":
        mask = masks_img == label.index('Right-shoe')

    if True not in mask:
        return None
    original_img = np.array(Image.open(original_img_path))
    
    img_full = apply_mask_to_image(original_img,mask)

    body_part = crop_non_zero(img_full, original_img)
    return body_part

def human_parse(model,batch, input_size):
    image, meta = batch
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
    
    return parsing_result



def main():
    dino = Dino_model()
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

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
    input_dir = '/home/user_219/omer_gil/BoT-SORT/best_images_parts/'
    output_dir = '/home/user_219/omer_gil/BoT-SORT/best_images_parts_embeddings/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for camera in sorted(os.listdir(input_dir)):
        if not os.path.exists(os.path.join(output_dir, camera)):
                os.makedirs(os.path.join(output_dir, camera))
        for id in sorted(os.listdir(os.path.join(input_dir, camera))):
            embed_dict = {"head": [], "upper": [], "rf": [], "lf": []}
            id_input_dir = os.path.join(os.path.join(input_dir, camera, id))
            file_list = os.listdir(id_input_dir)
            dataset = SimpleFolderDataset(root=id_input_dir, input_size=input_size, transform=transform,file_list = file_list)
            dataloader = DataLoader(dataset)
            id_output_dir = os.path.join(os.path.join(output_dir, camera, id))
            if not os.path.exists(id_output_dir):
                os.makedirs(id_output_dir)

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(dataloader)):
                    
                    img_name = batch[1]['name'][0]

                    if "head" in img_name or "body" in img_name or "rf_best_image_" in img_name or "lf_best_image_" in img_name:
                        parsing_result = human_parse(model,batch, input_size)

                    if "head" in img_name:
                        head = crop_image(parsing_result,os.path.join(id_input_dir, img_name), label, "head")
                        if head is None:
                            continue
                        head_path = os.path.join(id_output_dir, img_name)
                        head.save(head_path)
                        embed_dict["head"] = head.convert("RGB")
                    if "body" in img_name:
                        upper = crop_image(parsing_result,os.path.join(id_input_dir, img_name), label, "upper")
                        if upper is None:
                            continue                    
                        upper_path = os.path.join(id_output_dir, img_name)
                        upper.save(upper_path)
                        embed_dict["upper"] = upper.convert("RGB")
                    if "rf_best_image_" in img_name:
                        rf = crop_image(parsing_result,os.path.join(id_input_dir, img_name), label, "rf")
                        if rf is None:
                            continue                    
                        rf_path = os.path.join(id_output_dir, img_name)
                        rf.save(rf_path)
                        embed_dict["rf"] = rf.convert("RGB")
                    if "lf_best_image_" in img_name:
                        lf = crop_image(parsing_result,os.path.join(id_input_dir, img_name), label, "lf")
                        if lf is None:
                            continue                    
                        lf_path = os.path.join(id_output_dir, img_name)
                        lf.save(lf_path)
                        embed_dict["lf"] = lf.convert("RGB")
                dino.extract_features(embed_dict, os.path.join(id_output_dir, 'features.npy'))
    return


if __name__ == '__main__':
    main()
