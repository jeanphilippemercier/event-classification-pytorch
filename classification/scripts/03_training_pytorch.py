import spectrogram_dataset
from spectrogram_dataset import SpectrogramDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from importlib import reload
from torchvision import models
reload(spectrogram_dataset)
import os
from glob import glob
from tqdm import tqdm
import model
import pickle
import matplotlib.pyplot as plt
reload(model)

input_directory = '/data_1/classification_dataset/'
suffix = ''
extension = 'jpg'

# file_list = glob(os.path.join(input_directory, '*', f'*{suffix}.{extension}'))

# training, test = spectrogram_dataset.split_dataset(input_directory, split=0.8,
#                                                    seed=1)

# training_dataset = spectrogram_dataset.SpectrogramDataset(input_directory,
#                                                           max_image_sample=2e6)
# pickle.dump(training_dataset, open('training_dataset.pickle', 'wb'))
# training_dataset = pickle.load(open('training_dataset.pickle', 'rb'))
training_dataset = pickle.load(open('training_dataset.pickle', 'rb'))

# test_dataset = spectrogram_dataset.SpectrogramDataset(input_directory,
#                                                       max_image_sample=4e5)
# pickle.dump(test_dataset, open('test_dataset.pickle', 'wb'))
# test_dataset = pickle.load(open('test_dataset.pickle', 'rb'))

nb_pixel = training_dataset.nb_pixel
nb_categories = training_dataset.nb_categories

# model = model.CNN(nb_categories)
# VGG16
# model = models.vgg16(pretrained=False, progress=True)
# first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1,
#                                dilation=1, groups=1, bias=True)]
# first_conv_layer.extend(list(model.features))
# model.features = nn.Sequential(*first_conv_layer)

# resnet 50
model = models.resnet18(pretrained=False)
weights = model.conv1.weight.clone()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# model.conv1.weight[:, :1] = weights
# model.conv1.weight[:, 3] = model.conv1.weight[:, 0]


# model = nn.Sequential(
#     nn.Linear(nb_pixel, 128),
#     nn.ReLU(),
#     nn.Linear(128, nb_categories)
# )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adam(model.parameters())

batch_size = 2000

training_loader = DataLoader(training_dataset, batch_size=batch_size,
                             shuffle=True)

n_epochs = 10

# Stuff to store
train_losses = []  # np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)

plt.ion()
fig, ax = plt.subplots()
xlim = 1000
# plt.xlim([0, xlim])
plt.ylim([0, 1])
plt.axhline(0.9, color='k', ls='--')
plt.axhline(1.0, color='k', ls='--')
plt.axhline(0.8, color='k', ls='--')

fig2, ax2 = plt.subplots()
plt.xlim([0, 2300])
plt.ylim([0, 2])
plt.show()

for it in tqdm(range(n_epochs)):
    train_loss = []
    iterations = []
    train_accuracy = []
    i = 0
    for inputs, targets in tqdm(training_loader):
        inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1],
                             inputs.size()[2])
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        tmp_choices = [np.argmax(out.cpu().detach().numpy())
                       for out in outputs]
        total_accuracy = np.sum(tmp_choices ==
                                targets.cpu().detach().numpy()) \
                         / len(tmp_choices)

        train_accuracy.append(total_accuracy)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # print(loss.item())
        train_loss.append(loss.item())
        iterations.append(i)
        if i == 0:
            sc, = ax.plot(iterations, train_loss)
            plt.show()
            sc2, = ax.plot(iterations, train_accuracy)
        else:
            # plt.figure(1)
            # plt.clf()
            # plt.plot(iterations, train_losses)
            # plt.show()
            # ax.plot(iterations, train_losses)
            if np.max(iterations) > xlim:
                xlim += 1000
                plt.xlim([0, xlim])
            sc.set_data(iterations, train_loss)
            sc2.set_data(iterations, train_accuracy)

            ax = plt.gca()

            # recompute the ax.dataLim
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()
            plt.draw()

            fig.canvas.draw_idle()
            # fig2.canvas.draw_idle()
            try:
                fig.canvas.flush_events()
                # fig2.canvas.flush_events()
            except NotImplementedError:
                pass
        i += 1

    # train_losses.append(np.min(train_loss))
    # train_accuracy.append(torch.sum())
    # if it == 0:
    #     sc2, = ax2.plot(it, train_losses)
    #     plt.plot()
    # else:
    #     sc2.set_data(np.arange(len(train_losses)), train_losses)
    #     fig.canvas.draw_idle()
    #     try:
    #         fig.canvas.flush_events()
    #     except NotImplementedError:
    #         pass



