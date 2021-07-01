####
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

from dataset import Dataset, PickingDataset
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
from resnet1d import ResNet1D
from importlib import reload
from torch.utils.data import DataLoader
import model
reload(model)
from loguru import logger
import pickle
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import dataset
reload(dataset)

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



class LRFinder:
    def __init__(self):
        # self.picker = model.Picker()
        gpu = True
        self.gpu = gpu
        if gpu:
            device = torch.device("cuda:0" if
                                  torch.cuda.is_available() else "cpu")
            if device == "cpu":
                logger.warning('GPU is not available, the CPU will be used')
            else:
                logger.info('GPU will be used')
        else:
            device = 'cpu'
            logger.info('The CPU will be used')

        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(),lr=1e-7)
        self.model = ResNet1D(in_channels=1, base_filters=32, kernel_size=15,
                              stride=3, groups=1, n_block=16,
                              n_classes=1).to(self.device)
        self.criterion = nn.MSELoss()

        
        torch.save(model.state_dict(), 'init_params.pt')

    def range_test(self, end_lr = 10, num_iter = 100, 
                   smooth_f = 0.05, diverge_th = 5):
        
        lrs = []
        losses1 = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        
        
        for iteration in range(num_iter):

            self.train(train_dataset, batch_size=1000)
            loss=self.iterate.loss
            #update lr
            lr_scheduler.step()
            
            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
                
            if loss < best_loss:
                best_loss = loss

            losses1.append(loss)
            
            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break
                       
        #reset model to initial parameters
        self.model.load_state_dict(torch.load('init_params.pt'))
                    
        return lrs, losses1


    def train(self, dataset: PickingDataset, batch_size: int):
        dataset = train_dataset
        training_loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True, )

        for inputs, targets in tqdm(training_loader):
            inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1])
            self.iterate(inputs, targets)
            torch.cuda.empty_cache()
            # return self.iterate(inputs, targets)   



    def iterate(self, inputs, targets):

        self.optimizer.zero_grad()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        predictions = self.model(inputs)

        loss = self.criterion(predictions, targets.view(len(targets), -1))
        # print(loss.item())
        loss.backward()
        self.optimizer.step()

        # print(loss.item())
        self.losses.append(loss.cpu().item())
        # print(loss.item())
        # self.losses.append(loss.cpu().item())

        # return loss.cpu().item()
    # def _train_batch(self, iterator):
        
    #     self.train1()
    #     # self.picker.train()
        
    #     self.optimizer.zero_grad()
        
    #     inputs, targets = iterator.get_batch()
        
    #     inputs = inputs.to(self.device)
    #     targets = targets.to(self.device)

    #     predictions = self.model(inputs)

    #     loss = self.criterion(predictions, targets.view(len(targets), -1))
    #     # print(loss.item())
    #     loss.backward()
    #     self.optimizer.step()

    #     # print(loss.item())
    #     self.losses.append(loss.cpu().item())
    #     return loss.item()


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

# class IteratorWrapper:
#     def __init__(self, iterator):
#         self.iterator = iterator
#         self._iterator = iter(iterator)

#     def __next__(self):
#         try:
#             inputs, labels = next(self._iterator)
#         except StopIteration:
#             self._iterator = iter(self.iterator)
#             inputs, labels, *_ = next(self._iterator)

#         return inputs, labels

#     def get_batch(self):
#         return next(self)