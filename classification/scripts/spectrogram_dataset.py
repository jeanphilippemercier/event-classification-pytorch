from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from os.path import sep


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


def split_dataset(file_list, split=0.8, seed=None):
    """

    :param file_list: list of image files
    :param split: set_1 fraction (between 0 and 1)
    :param seed: seed for the random number generator to provide
    reproducibility (Default None)
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

    def __init__(self, file_list):
        # self.df_data = pd.read_csv(csv_file, skipinitialspace=True)

        self.file_list = file_list

        self.category_list = [str(Path(fle).parent).split(sep)[-1]
                              for fle in file_list]

        self.unique_categories = np.unique(self.category_list)
        self.labels = np.arange(len(self.unique_categories))

        self.labels_dict = {}
        for label, category in zip(self.labels, self.unique_categories):
            self.labels_dict[category] = label


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # label = self.df_data.iloc[idx]['label']
        category = self.category_list[idx]
        label = self.labels_dict[category]

        img_file = os.path.join(self.file_list[idx])

        image = torch.from_numpy(np.array(Image.open(
            self.file_list[idx])).astype(np.float32))

        # return {'data': image, 'label': label}
        return image, label

    @property
    def shape(self):
        px_x, px_y = self[0][0].shape
        return len(self.labels), px_x, px_y

    @property
    def nb_pixel(self):
        return np.prod(self[0][0].shape)

    @property
    def categories(self):
        return self.labels_dict

    @property
    def nb_categories(self):
        return len(self.unique_categories)




