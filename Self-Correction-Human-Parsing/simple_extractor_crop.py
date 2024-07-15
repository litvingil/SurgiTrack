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
    parser.add_argument("--model-restore", type=str, default='', help="restore pretrained model parameters.")
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

def crop_image(masks_img,original_img_path,label):
    # import pdb; pdb.set_trace()
    # ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    head_mask =  ((masks_img == label.index('Hair')) | (masks_img == label.index('Hat')) | (masks_img == label.index('Face')) | (masks_img == label.index('Sunglasses'))| (masks_img == label.index('Scarf')))
    upper_mask = ((masks_img == label.index('Upper-clothes')) | (masks_img == label.index('Dress')) | (masks_img == label.index('Coat')) | (masks_img == label.index('Scarf')) | (masks_img == label.index('Jumpsuits')) | (masks_img == label.index('Right-arm')) | (masks_img == label.index('Left-arm')) | (masks_img == label.index('Glove')))
    lower_mask = ((masks_img == label.index('Pants')) | (masks_img == label.index('Skirt')) | (masks_img == label.index('Right-leg')) | (masks_img == label.index('Left-leg')))
    shoes_mask = ((masks_img == label.index('Left-shoe')) | (masks_img == label.index('Right-shoe')) | (masks_img == label.index('Socks')))

    original_img = np.array(Image.open(original_img_path))
    
    head_full =  apply_mask_to_image(original_img,head_mask)
    upper_full = apply_mask_to_image(original_img,upper_mask)
    lower_full = apply_mask_to_image(original_img,lower_mask)
    shoes_full = apply_mask_to_image(original_img,shoes_mask)

    head = crop_non_zero(head_full, original_img)
    upper = crop_non_zero(upper_full, original_img)
    lower = crop_non_zero(lower_full, original_img)
    shoes = crop_non_zero(shoes_full, original_img)

    return head, upper, lower, shoes

def main():
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
    dataset = SimpleFolderDataset(root=args.input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    palette = get_palette(num_classes)
    plot_colors(palette, label, os.path.join(args.output_dir, "palette.png"))

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
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

            head, upper, lower, shoes = crop_image(parsing_result,os.path.join(args.input_dir, img_name), label)

            head_path = os.path.join(args.output_dir, img_name[:-4] + '_head.png')
            upper_path = os.path.join(args.output_dir, img_name[:-4] + '_upper.png')
            lower_path = os.path.join(args.output_dir, img_name[:-4] + '_lower.png')
            shoes_path = os.path.join(args.output_dir, img_name[:-4] + '_shoes.png')

            head.save(head_path)
            upper.save(upper_path)
            lower.save(lower_path)
            shoes.save(shoes_path)

            # parsing_result_path = os.path.join(args.output_dir, img_name[:-4] + '.png')
            # output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            # # print("Parsing result:", parsing_result)
            # output_img.putpalette(palette)
            # output_img.save(parsing_result_path)
            # if args.logits:
            #     logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
            #     np.save(logits_result_path, logits_result)
    return


if __name__ == '__main__':
    main()
