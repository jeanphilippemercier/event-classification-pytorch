import spectrogram_dataset
from spectrogram_dataset import SpectrogramDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from importlib import reload
reload(spectrogram_dataset)
import os
from glob import glob
from tqdm import tqdm
import model
reload(model)

input_directory = '/Users/jpmercier/google_drive/data'
suffix = '128_128'
extension = 'png'

file_list = glob(os.path.join(input_directory, '*', f'*{suffix}.{extension}'))

training, test = spectrogram_dataset.split_dataset(file_list, split=0.8,
                                                   seed=1)

training_dataset = spectrogram_dataset.SpectrogramDataset(training)
test_dataset = spectrogram_dataset.SpectrogramDataset(test)

nb_pixel = training_dataset.nb_pixel
nb_categories = training_dataset.nb_categories

model = model.CNN(nb_categories)
# model = nn.Sequential(
#     nn.Linear(nb_pixel, 128),
#     nn.ReLU(),
#     nn.Linear(128, nb_categories)
# )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 2048

training_loader = DataLoader(training_dataset, batch_size=batch_size,
                             shuffle=True)

n_epochs = 10

# Stuff to store
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)


for it in range(n_epochs):
    train_loss = []
    for inputs, targets in tqdm(training_loader):
        # bubu
        # inputs = inputs.view(-1, nb_pixel)
        inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1],
                             inputs.size()[2])

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

