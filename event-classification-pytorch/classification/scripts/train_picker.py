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

# input_files = '/data_1/pick_dataset/*.pickle'


input_files = '/home/munguush/python_projects/scripts/p_s_picker_deep_learning/pick_dataset/*.pickle'

file_list = glob(input_files)

file_list_new=[]

for idx in range(0, len(file_list)):

    with open(file_list[idx], 'rb') as f_in:
        data = pickle.load(f_in)

    if len(data['data'].astype(np.float32))==256:
        file_list_new.append(file_list[idx])



picker_dataset = dataset.PickingDataset(file_list_new)

train_dataset, test_dataset = picker_dataset.split(split_fraction=0.8)

picker = model.Picker()

epoch = 20
train_losses = []
test_losses = []
batch_size = 2000
for e in tqdm(range(0, epoch)):
    picker.train(train_dataset, batch_size=batch_size)
    predictions, test_loss_mean, test_loss_std = picker.validate(
        test_dataset, batch_size=batch_size)
    test_losses.append(test_loss_mean)
    predictions, train_loss_mean, train_loss_std = picker.validate(
        train_dataset, batch_size=batch_size)
    train_losses.append(train_loss_mean)

    plt.figure(1)
    plt.clf()
    plt.plot(train_losses, label='train losses')
    plt.plot(test_losses, label='test losses')
    plt.legend()
    plt.show()
    plt.pause(0.1)
    plt.close()


output_file_path = Path('/home/munguush/jp_project/result2(16filters).pickle')

picker.save(output_file_path)



output_test_dataset_path = Path('/home/munguush/jp_project/test_dataset(16filters).pickle')

test_dataset.save(output_test_dataset_path)

filehandler = open(output_test_dataset_path,"wb")

pickle.dump(test_dataset,filehandler)
filehandler.close()