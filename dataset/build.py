from dataset.Edit_SeedxUnsplash import SeedxUnsplash_Dataset
from dataset.Edit_PIPE import PIPE_Dataset
from dataset.Condition_MultiGen_Depth import MultiGen_Depth_Dataset
from dataset.Condition_MultiGen_Canny import MultiGen_Canny_Dataset
from dataset.Condition_Segmentation import Condition_Segmentation_Dataset

import torch
import random

class Combine_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_args,
                 MultiGen_Depth_Dataset=None,
                 MultiGen_Canny_Dataset=None,
                 Condition_Segmentation_Dataset=None,
                 PIPE_Dataset=None,
                 SeedxUnsplash_Dataset=None,
                 mode='train',
                 ):
        self.dataset_args = dataset_args
        self.mode = mode
        self.total_len = 0

        self.MultiGen_Depth_Dataset = MultiGen_Depth_Dataset
        if MultiGen_Depth_Dataset:
            self.total_len += len(MultiGen_Depth_Dataset)

        self.MultiGen_Canny_Dataset = MultiGen_Canny_Dataset
        if MultiGen_Canny_Dataset:
            self.total_len += len(MultiGen_Canny_Dataset)

        self.Condition_Segmentation_Dataset = Condition_Segmentation_Dataset
        if Condition_Segmentation_Dataset:
            self.total_len += len(Condition_Segmentation_Dataset)

        self.PIPE_Dataset = PIPE_Dataset
        if PIPE_Dataset:
            self.total_len += len(PIPE_Dataset)

        self.SeedxUnsplash_Dataset = SeedxUnsplash_Dataset
        if SeedxUnsplash_Dataset:
            self.total_len += len(SeedxUnsplash_Dataset)

        print("total number of images: ", self.total_len)

    def __getitem__(self, i):
        ratio = random.random()

        if self.MultiGen_Depth_Dataset and ratio < self.dataset_args.multigendepth_prob:
            index = random.randint(0, len(self.MultiGen_Depth_Dataset) - 1)
            if self.mode == 'val':
                index = i
            MultiGen_Depth_Dataset_data = self.MultiGen_Depth_Dataset[index]
            return MultiGen_Depth_Dataset_data
        elif self.MultiGen_Canny_Dataset and ratio < self.dataset_args.multigencanny_prob:
            index = random.randint(0, len(self.MultiGen_Canny_Dataset) - 1)
            if self.mode == 'val':
                index = i
            MultiGen_Canny_Dataset_data = self.MultiGen_Canny_Dataset[index]
            return MultiGen_Canny_Dataset_data
        elif self.Condition_Segmentation_Dataset and ratio < self.dataset_args.conditionsegmentation_prob:
            index = random.randint(0, len(self.Condition_Segmentation_Dataset) - 1)
            if self.mode == 'val':
                index = i
            Condition_Segmentation_Dataset_data = self.Condition_Segmentation_Dataset[index]
            return Condition_Segmentation_Dataset_data
        elif self.PIPE_Dataset and ratio < self.dataset_args.pipe_prob:
            index = random.randint(0, len(self.PIPE_Dataset) - 1)
            if self.mode == 'val':
                index = i
            PIPE_Dataset_data = self.PIPE_Dataset[index]
            return PIPE_Dataset_data
        elif self.SeedxUnsplash_Dataset and ratio < self.dataset_args.seedxunsplash_prob:
            index = random.randint(0, len(self.SeedxUnsplash_Dataset) - 1)
            if self.mode == 'val':
                index = i
            SeedxUnsplash_Dataset_data = self.SeedxUnsplash_Dataset[index]
            return SeedxUnsplash_Dataset_data
        
    def __len__(self):
        return self.total_len

def build_dataset(dataset_args, llm_tokenizer, mode='train'):

    if 'multigendepth' in dataset_args.dataset_list:
        MultiGen_Depth_Dataset_path= dataset_args.multigendepth_path
        MultiGen_Depth_train_dataset = MultiGen_Depth_Dataset(
            args=dataset_args,
            dataset_path=MultiGen_Depth_Dataset_path,
            llm_tokenizer=llm_tokenizer,
            mode='train')
    else:
        MultiGen_Depth_train_dataset = None

    if 'multigencanny' in dataset_args.dataset_list:
        MultiGen_Canny_Dataset_path= dataset_args.multigencanny_path
        MultiGen_Canny_train_dataset = MultiGen_Canny_Dataset(
            args=dataset_args,
            dataset_path=MultiGen_Canny_Dataset_path,
            llm_tokenizer=llm_tokenizer,
            mode='train')
    else:
        MultiGen_Canny_train_dataset = None

    if 'conditionsegmentation' in dataset_args.dataset_list:
        Condition_Segmentation_Dataset_path= dataset_args.conditionsegmentation_path
        Condition_Segmentation_train_dataset = Condition_Segmentation_Dataset(
            args=dataset_args,
            dataset_path=Condition_Segmentation_Dataset_path,
            llm_tokenizer=llm_tokenizer,
            mode='train')
    else:
        Condition_Segmentation_train_dataset = None

    if 'pipe' in dataset_args.dataset_list:
        PIPE_Dataset_path= dataset_args.pipe_path
        PIPE_train_dataset = PIPE_Dataset(
            args=dataset_args,
            dataset_path=PIPE_Dataset_path,
            llm_tokenizer=llm_tokenizer,
            mode='train')
    else:
        PIPE_train_dataset = None

    if 'seedxunsplash' in dataset_args.dataset_list:
        SeedxUnsplash_Dataset_path= dataset_args.seedxunsplash_path
        SeedxUnsplash_train_dataset = SeedxUnsplash_Dataset(
            args=dataset_args,
            dataset_path=SeedxUnsplash_Dataset_path,
            llm_tokenizer=llm_tokenizer,
            mode='train')
    else:
        SeedxUnsplash_train_dataset = None

    train_dataset = Combine_Dataset(
                        dataset_args = dataset_args,
                        MultiGen_Depth_Dataset=MultiGen_Depth_train_dataset,
                        MultiGen_Canny_Dataset=MultiGen_Canny_train_dataset,
                        Condition_Segmentation_Dataset=Condition_Segmentation_train_dataset,
                        PIPE_Dataset = PIPE_train_dataset,
                        SeedxUnsplash_Dataset = SeedxUnsplash_train_dataset,
                        mode=mode,
                        )

    val_dataset = None
    return train_dataset
