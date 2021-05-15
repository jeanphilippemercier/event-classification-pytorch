import pandas as pd
from torch.utils.data import Dataset
import settings
import torch
from uquake.core import Stream
import os
from PIL import Image
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from os.path import sep
from loguru import logger
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from uquake.core.trace import Trace
from abc import ABC
import pickle
from loguru import logger
from spectrograms import generate_spectrogram


class MultiChannelClassificationDataset(Dataset):
    def __init__(self, n_active_event_category=1000, n_channels=30,
                 manual_only=True):
        df = pd.read_csv(settings.catalog_file)
        if manual_only:
            df = df[df['evaluation mode'] == 'manual']

        event_types = df['event type'].unique()

        files = []
        labels = []
        classes = []

        for i, event_type in enumerate(np.sort(event_types)):
            replace = False
            if n_active_event_category > len(df[df['event type'] ==
                                                event_type]):
                logger.warning(f'the requested sample size of ' 
                               f'{n_active_event_category} for {event_type} '
                               f'is larger than the population '
                               f'({len(df[df["event type"]])}).\n'
                               f'the sample will contain duplicate')
                replace = True

            tmp = df[df['event type']
                     == event_type].sample(n_active_event_category,
                                           replace=replace)

            for row in tmp.iterrows():
                files.append(row[1]['file'])
                labels.append(row[1]['event type'])
                classes.append(i)

            self.files = files
            self.labels = labels
            self.classes = classes

            self.n_channels = n_channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        st = self.files[idx]

        stds = []
        for tr in st:
            if tr.stats.channel[1] == 'P':
                tr.differentiate()
            stds.append(np.std(tr.data))

        indices = np.argsort(stds)[-1::-1]
        indices = indices[:self.n_channels]

        specs = []
        trs = [st[i].copy() for i in indices]
        st2 = Stream(traces=trs)

        specs = torch.from_numpy(np.array(generate_spectrogram(st2.copy())))

        return specs, self.classes[idx]











