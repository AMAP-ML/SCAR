import glob
import random

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import json
# from pycocotools import mask as mask_utils
import torch
from torch.nn import functional as F
from tqdm import tqdm


def find_classes(directory):
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class ImagenetODataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None, image_size=256, val_cond='depth', debuger=False, args=None, **kwargs):

        self.args = args
        self.transforms = transform
        self.split = split
        self.debuger = debuger
        # self.debuger = True
        # print(debuger)
        self.load_dataset(root)
        if split!="train":
            classes, class_to_idx = find_classes(os.path.join(root, "val"))
        else:
            classes, class_to_idx = find_classes(os.path.join(root, split))
        self.cond = {'hed': self.hed_paths, 'sketch': self.sketch_paths, 'canny': self.canny_paths, 'depth': self.depth_paths, 'normal': self.normal_paths}
        self.cond_idx = {'hed': 0, 'sketch': 1, 'canny': 2, 'depth': 3, 'normal': 4}
        self.class_to_idx = class_to_idx
        print('Use ImageFolder Class to IDX')
        self.image_size = image_size
        print(f'ImagenetC dataset init: total images '
              f'{max(len(self.hed_paths), len(self.sketch_paths), len(self.canny_paths), len(self.depth_paths), len(self.normal_paths))}')
        if self.split == 'val':
            self.val_cond = val_cond
            print(f'Warning: Only use {self.val_cond} during the evaluation')
        if self.split == 'val_sub10k':
            self.val_cond = val_cond
            print(f'Warning: Only use {self.val_cond} during the evaluation')

    def load_dataset(self, root):
        if 'ceph' in root:
            cond_info_path = os.path.join(root, f'{self.split}_cond_info_mpi.json')
        else:
            if self.debuger:
                cond_info_path = os.path.join(root, f'{self.split}_cond_info_debug.json')
            else:
                if self.split == 'val_sub10k':
                    cond_info_path = os.path.join(root, "val_cond_info_subset10k.json")
                else:
                    cond_info_path = os.path.join(root, f'{self.split}_cond_info.json')

        print(cond_info_path)
        if self.debuger:
            condition_root = root.replace('imageNet','imageNet_condition_debug')
        else:
            condition_root = root.replace('imageNet','imageNet_condition')
        if os.path.exists(cond_info_path):
            print('load ImageNetO from json')
            with open(cond_info_path, 'r') as file:
                cond_info = json.load(file)
            self.hed_paths = cond_info['hed']
            self.sketch_paths = cond_info['sketch']
            self.canny_paths = cond_info['canny']
            self.depth_paths = cond_info['depth']
            self.normal_paths = cond_info['normal']
            print('hed, sketch, canny, depth, normal')
            print(len(self.hed_paths), len(self.sketch_paths), len(self.canny_paths), len(self.depth_paths), len(self.normal_paths))
        else:
            print('load ImageNetC from glob', condition_root)
            # self.mask_paths = sorted(glob.glob(os.path.join(root, f"{self.split}_mask/" "*", "*.json")))
            self.hed_paths = sorted(glob.glob(os.path.join(condition_root, f"hed/{self.split}/" "*", "*.JPEG")))
            self.sketch_paths = sorted(glob.glob(os.path.join(condition_root, f"sketch/{self.split}/" "*", "*.JPEG")))
            self.canny_paths = sorted(glob.glob(os.path.join(condition_root, f"canny/{self.split}/" "*", "*.JPEG")))
            self.depth_paths = sorted(glob.glob(os.path.join(condition_root, f"depth/{self.split}/" "*", "*.JPEG")))
            self.normal_paths = sorted(glob.glob(os.path.join(condition_root, f"normal/{self.split}/" "*", "*.JPEG")))
            for paths in [self.hed_paths, self.sketch_paths, self.canny_paths, self.depth_paths, self.normal_paths]:
                with tqdm(total=len(paths)) as pbar:
                    for path in paths:
                        size = os.stat(path).st_size
                        pbar.update(1)
                        if size < 1000:
                            try:
                                mask = Image.open(path)
                            except:
                                print(path)
                                paths.remove(path)
            data = {
                # 'mask': self.mask_paths,
                'hed': self.hed_paths,
                'sketch': self.sketch_paths,
                'canny': self.canny_paths,
                'depth': self.depth_paths,
                'normal': self.normal_paths
            }
            with open(cond_info_path, 'w') as file:
                json.dump(data, file)


    def __len__(self):
        return max(len(self.hed_paths), len(self.sketch_paths), len(self.canny_paths), len(self.depth_paths), len(self.normal_paths))


    def __getitem__(self, index: int):
        max_retry = 10  # 最多尝试 10 次
        for _ in range(max_retry):
            try:
                if self.args.cond_type == 'random':
                    cond_type = random.choices(['hed', 'sketch', 'canny', 'normal', 'depth'], [0.2]*5, k=1)[0]
                else:
                    cond_type = self.args.cond_type

                if self.split == 'val' or self.split == "val_sub10k":
                    cond_type = self.val_cond

                cond_path = self.cond[cond_type][index % len(self.cond[cond_type])]
                if self.debuger:
                    image_path = cond_path.replace('imageNet_condition_debug/' + cond_type, 'imageNet')
                else:
                    image_path = cond_path.replace('imageNet_condition/' + cond_type, 'imageNet')
                # print(image_path)
                cls_ = self.class_to_idx[(image_path.split('/')[-2])]
                # print(cls_,type(cls_),image_path.split('/')[-2],111)
                image = Image.open(image_path).convert('RGB')
                cond = Image.open(cond_path).convert('RGB')
                cond = cond.resize(image.size)

                if self.transforms:
                    image, cond = self.transforms(image, cond)

                return {
                    'edited_img': image,
                    'input_img': cond,
                    'cls': cls_,
                    'type': torch.tensor(self.cond_idx[cond_type]),
                    "index":index,
                    "dataset": "imagenet"
                }

            except Exception as e:
                print(f"Skipping index {index} due to error: {e}")
                index = (index + 1) % self.__len__()  # 换一个样本继续尝试


if __name__ == '__main__':
    root= '/mnt/xmap_nas_alg/xusenyan/data2/imageNet'
    split='train'
    cond_info_path = os.path.join(root, 'cond_info.json')
    if os.path.exists(cond_info_path):
        print('load ImageNetC from json')
        with open(cond_info_path, 'r') as file:
            cond_info = json.load(file)
            mask_paths = cond_info['mask']
            canny_paths = cond_info['canny']
            depth_paths = cond_info['depth']
            normal_paths = cond_info['normal']
    condition_root = root.replace('imageNet','imageNet_condition')
    print('load ImageNetC from glob', condition_root)
            # self.mask_paths = sorted(glob.glob(os.path.join(root, f"{self.split}_mask/" "*", "*.json")))
    hed_paths = sorted(glob.glob(os.path.join(condition_root, f"hed/{split}" "*", "*.JPEG")))
    sketch_paths = sorted(glob.glob(os.path.join(condition_root, f"sketch/{split}" "*", "*.JPEG")))
    canny_paths = sorted(glob.glob(os.path.join(condition_root, f"canny/{split}/" "*", "*.JPEG")))
    depth_paths = sorted(glob.glob(os.path.join(condition_root, f"depth/{split}/" "*", "*.JPEG")))
    normal_paths = sorted(glob.glob(os.path.join(condition_root, f"normal/{split}/" "*", "*.JPEG")))

    from tqdm import tqdm
    with tqdm(total=len(normal_paths)) as pbar:
        for path in normal_paths:
            img = Image.open(path)#.convert('RGB')

            pbar.update(1)
    # for path in fail_list:
    #     os.system(f'rm {path}')
