from glob import glob
from importlib import reload
import model
reload(model)
import dataset
reload(dataset)
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pathlib import Path
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


import learning_rate_optimizer
reload(learning_rate_optimizer)

import pickle


####
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import copy
from collections import namedtuple
import os
import random
import shutil
import time


import model
reload(model)

import dataset
reload(dataset)

# model = model.Picker().model


input_files = '/home/munguush/python_projects/scripts/p_s_picker_deep_learning/pick_dataset/*.pickle'

file_list = glob(input_files)

file_list_new=[]

for idx in range(0, len(file_list)):

    with open(file_list[idx], 'rb') as f_in:
        data1 = pickle.load(f_in)

    if len(data1['data'].astype(np.float32))==256:
        file_list_new.append(file_list[idx])


picker_dataset = dataset.PickingDataset(file_list_new)

train_dataset, test_dataset = picker_dataset.split(split_fraction=0.8)



BATCH_SIZE = 64
train_iterator = data.DataLoader(train_dataset, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)




# START_LR = 1e-7



# optimizer = optim.Adam(model.parameters(), lr=START_LR)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# criterion = nn.MSELoss()

# model = model.to(device)
# criterion = criterion.to(device)


import model

from importlib import reload

reload(model)


END_LR = 10
NUM_ITER = 100

# lr_finder = learning_rate_optimizer.LRFinder()
# lrs, losses = lr_finder.range_test(END_LR, NUM_ITER)


lr_finder = model.Picker()
lrs, losses = lr_finder.range_test(END_LR, NUM_ITER)


def plot_lr_finder(lrs, losses, skip_start = 5, skip_end = 5):
    
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    plt.show()



plot_lr_finder(lrs, losses, skip_start = 30, skip_end = 30)