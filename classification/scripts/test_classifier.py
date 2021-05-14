from uquake.core import read, read_events, Stream, Trace
from importlib import reload
import model
reload(model)
from glob import glob
import matplotlib.pyplot as plt
from create_classification_dataset import event_type_lookup
from loguru import logger
import numpy as np
from tqdm import tqdm
from dataset import spectrogram
from PIL import Image, ImageOps

ec = model.EventClassifier.read('classifier_model_26_epoch.pickle')
ec.model.eval()
# ec.device = 'cpu'
# ec.model.to(ec.device)

type_list = ['blast', 'seismic event', 'noise']

confusion = np.zeros((5, 3))

event_dict = {'blast': 0, 'noise': 2, 'seismic event': 1,
              'uncertain - seismic event or blast': 3,
              'uncertain': 4}

ec2 = model.EventClassifier(3)
ec2.model = ec.model
ec = ec2

threshold = 10
threshold_lower = 4
i = 0
sampling_rate = 6000
sequence_length_second = 2


# def generate_spectrogram(stream: Stream):
#
#     specs = []
#     for tr in tqdm(stream.copy()):
#         # check if tr contains NaN
#         if np.nan in tr.data:
#             continue
#
#         tr = tr.trim(endtime=tr.stats.starttime + 2, pad=True,
#                      fill_value=0)
#
#         tr = tr.detrend('demean').detrend('linear')
#         tr = tr.taper(max_percentage=1, max_length=1e-2)
#
#         spec = spectrogram(tr)
#         spec /= np.max(spec)
#
#         # plt.clf()
#         # plt.imshow(spec)
#         # plt.pause(0.1)
#
#         # prediction = ec.predict_spectrogram(spec)
#         # print(prediction)
#
#         # input()
#
#         specs.append(spec)
#
#     return specs

with open('results_classification_test.csv', 'w') as f_out:
    f_out.write('file name, datetime, event class, predicted event class\n')
    for fle in tqdm(glob('/data_1/ot-reprocessed-data/*.xml')):
        try:
            cat, st = read_events(fle), read(fle.replace('.xml', '.mseed'))
        except Exception as e:
            logger.error(e)
            continue

        predictions = ec2.predict(st)

        n_seismic_events = (predictions[0] == event_dict[
            'seismic event']).sum().item()
        n_blasts = (predictions[0] == event_dict['blast']).sum().item()

        n_e_b = n_seismic_events + n_blasts
        if n_e_b >= threshold:
            if n_seismic_events / n_e_b >= 0.7:
                predicted_event_class = 'seismic event'
            elif n_blasts / n_e_b >= 0.7:
                predicted_event_class = 'blast'
            else:
                predicted_event_class = 'uncertain - seismic event or blast'
        elif n_e_b >= threshold_lower:
            predicted_event_class = 'uncertain'
        else:
            predicted_event_class = 'noise'

        event_class = event_type_lookup[cat[0].event_type]

        print(f'predicted event class: {predicted_event_class}')
        print(f'event class: {event_type_lookup[cat[0].event_type]}')
        predicted_event_class_index = event_dict[predicted_event_class]
        event_class_index = event_dict[event_class]
        confusion[predicted_event_class_index, event_class_index] += 1

        # input()

        if i % 20 == 19:
            print(f'\n{confusion}\n')

        i += 1

        s_out = f'{fle},{cat[0].origins[-1].time},{event_class},' \
                f'{predicted_event_class}\n'
        f_out.write(s_out)