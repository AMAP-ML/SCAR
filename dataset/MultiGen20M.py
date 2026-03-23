import json
import traceback
import glob
import math
import os
import random
from io import BytesIO
import pathlib
import cv2

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, transforms
import os,glob 
import numpy as np
from PIL import Image
import re
import pdb
import logging
import sys
sys.path.insert(0,'../')

from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024 * 1024 

# from dataset_parquet import create_parquet

logging.basicConfig(level=logging.INFO)

def containChinese(character):
    for cha in character:
        if '\u4e00' <= cha <= '\u9fa5':
            return True 
    return False 

def find_consecutive(string):
    pattern = r'(\w)\1{4,}'
    result = re.search(pattern, string)
    if result:
        return True
    else:
        return False 
    

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def get_transform(img_size):
    transform_list=[
        transforms.Resize((img_size,img_size), interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        # transforms.RandomCrop((img_size,img_size)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    return transforms.Compose(transform_list)

class MultiGen20M(torch.utils.data.Dataset):
    """
    A dataset for controllable image generation, where data is defined by a JSON file.

    Each line in the JSON file should be a dictionary containing keys for the prompt,
    the image path, and the path to the control signal (e.g., a segmentation map).
    """

    def __init__(
        self,
        args,
        json_file,
        data_root,
        llm_tokenizer,
        resolution=512,
        control_key='control_seg_path', #control_seg_path就是depth
        classifier_free_training_prob=0.1,
        dataset_name='multigen',
        # random_condition = "none",
    ):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset metadata.
            data_root (str): The root directory where the image and control data are stored.
            resolution (int): The target resolution for the images and control signals.
            control_key (str): The key in the JSON object that holds the path to the control signal.
        """
        self.args = args
        # self.random_condition = random_condition
        self.data_root = data_root
        self.resolution = resolution
        self.control_key = control_key
        self.negative_tag = ""
        self.classifier_free_training_prob = classifier_free_training_prob
        self.dataset_name = dataset_name
        print("classifier-free training prob: ",classifier_free_training_prob)
        with open(json_file, 'r') as f:
            self.data = [json.loads(line) for line in f]
            # self.data = [json.loads(line) for line in f if line.strip()]

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])
        
        self.conditioning_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        item = self.data[index]
        
        image_path = os.path.join(self.data_root, item['image_path'])
        if self.control_key == 'random':
            control_path = os.path.join(self.data_root, item["control_seg_path"])
            cond_type = random.choices(['control_hed', 'control_lineart', 'control_canny', 'control_seg'], [0.25]*4, k=1)[0]
            control_path = control_path.replace('control_seg', cond_type)
        else:
            control_path = os.path.join(self.data_root, item[self.control_key])
        # Load image
        try:
            source_image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            # print(f"Warning: Image file not found at {image_path}. Skipping this item.")
            # Return the next valid item
            return self.__getitem__((index + 1) % len(self))

        # img_name = image_path.split('/')[-1].split('.')[0]
        # print(img_name)
        # Load control signal
        # cond_type = ""
        
        # if self.random_condition == 'random':
        #     cond_type = random.choices(['control_hed', 'control_lineart', 'control_canny', 'control_seg'], [0.25]*4, k=1)[0]
        #     control_path = control_path.replace('control_seg', cond_type)
            # print(control_path)

        try:
            control_image = Image.open(control_path).convert("RGB")
        except FileNotFoundError:
            # print(f"Warning: Control file not found at {control_path}. Skipping this item.")
            # Return the next valid item
            return self.__getitem__((index + 1) % len(self))

        # Get prompt
        prompt = item["prompt"]
        if self.classifier_free_training_prob > 0.0 and len(prompt)>20 and random.random() < self.classifier_free_training_prob:
            prompt = self.negative_tag

        if self.classifier_free_training_prob > 0.0 and random.random() < self.classifier_free_training_prob:
            prompt = self.negative_tag
        
        # print(prompt)
        # Apply transformations
        target_image_tensor = self.transform(source_image)
        control_image_tensor = self.conditioning_transform(control_image)
        edit_text_tokens_and_mask = self.llm_tokenizer(
            prompt,
            max_length=120,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        edit_txt_token = edit_text_tokens_and_mask['input_ids']
        edit_txt_attn_mask = edit_text_tokens_and_mask['attention_mask']
        input_ids = edit_txt_token[0]
        input_ids_attn_mask = edit_txt_attn_mask[0]


        return {
            'edited_img': normalize_01_into_pm1(target_image_tensor),
            'input_img': normalize_01_into_pm1(control_image_tensor),
            'input_ids': input_ids,
            'input_ids_attn_mask': input_ids_attn_mask,
            "index":index,
            'mode': 1,
            "dataset": "imagenet"
        }

