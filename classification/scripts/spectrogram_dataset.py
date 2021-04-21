from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from os.path import sep
from loguru import logger


# class Label():
#     def __init__(self, labels):
#         self.full_labels = list(labels)
#         self.labels = range(0, len(labels))
#
#         self.unique_labels = np.unique(labels)
#         self.unique_numerical_labels = np.range(len(self.unique_labels))
#
#         self.labels_dict = {}
#         for full_label, label  in zip(labels, self.unique_labels):
#             self.labels_dict[full_label] = label
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, item):
#         label = self.full_labels[item]
#         return self.labels_dict[label]
#
#     def get_full_label(self, item):

def get_file_list(input_directory, suffix, extension='png'):
    path = os.path.join(input_directory, '*', f'*{suffix}.{extension}')
    return glob(path)


def split_dataset(path, split=0.8, seed=None, max_item_per_category=100000):
    """

    :param path: list of image files
    :param split: set_1 fraction (between 0 and 1)
    :param seed: seed for the random number generator to provide
    reproducibility (Default None)
    :param max_item_per_category: maximum number of item per category
    :return: set_1, set_2
    """

    np.random.seed(seed)
    choices = np.random.choice([False, True], size=len(file_list),
                               p=[1-split, split])
    not_choices = [not choice for choice in choices]
    set_1 = np.array(file_list)[choices]
    set_2 = np.array(file_list)[not_choices]

    return set_1, set_2


def split_filename_label(file_list):
    file_list = [os.path.split(f)[-1] for f in tmp]
    labels = [os.path.split(f)[0].split('/')[-1] for f in tmp]


class SpectrogramDataset(Dataset):

    def __init__(self, path, max_image_sample=100000,
                 unused_category=['unknown'], seed=None):

        self.max_image_sample = int(max_image_sample)
        self.path = Path(path)
        dir_list = self.path.glob('*')

        self.category_list = [str(dr).split(sep)[-1] for dr in dir_list
                              if str(dr).split(sep)[-1] not in unused_category]
        self.labels = np.arange(len(self.category_list))

        self.labels_dict = {}
        self.file_list = []
        self.label_list = []
        np.random.seed(seed)
        for label, category in zip(self.labels, self.category_list):
            self.labels_dict[category] = label

            cat_file_list = [fle for fle in
                             (self.path / category).glob('*.jpg')]

            logger.info(f'the {category} category '
                        f'contains {len(cat_file_list)} images')
            nb_files = self.max_image_sample
            if len(cat_file_list) < nb_files:
                nb_files = len(cat_file_list)

            for f in np.random.choice(cat_file_list,
                                      size=nb_files, replace=False):
                self.file_list.append(f)
                self.label_list.append(label)

        self.file_list = np.array(self.file_list)
        self.label_list = np.array(self.label_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # label = self.df_data.iloc[idx]['label']
        # category = self.category_list[idx]
        label = self.label_list[idx]

        image = torch.from_numpy((np.array(Image.open(
            self.file_list[idx])) / 255).astype(np.float32))

        # return {'data': image, 'label': label}
        return image, label

    @property
    def shape(self):
        px_x, px_y = self[0][0].shape
        return len(self.label_list), px_x, px_y

    @property
    def nb_pixel(self):
        return np.prod(self[0][0].shape)

    @property
    def categories(self):
        return self.labels_dict

    @property
    def nb_categories(self):
        return len(self.category_list)

    # def split_data_set(self, split_fraction=0.8, seed=None):
    #
    #     np.random.seed(seed)
    #     choices = np.random.choice([False, True], size=len(self),
    #                                p=[1 - split_fraction, split_fraction])
    #     not_choices = [not choice for choice in choices]
    #     set_1 = self.file_list[choices]
    #     labels_set_1 = self.label_list[choices]
    #     set_2 = self.file_list[not_choices]
    #     labels_set_2 = self.label_list[not_choices]





