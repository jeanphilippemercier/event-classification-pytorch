import pickle
import pandas as pd
import os
from pathlib import Path
from gcp_storage import RequestEventGCP
from tqdm import tqdm
from loguru import logger

data_directory = Path(os.environ['SEISMICDATADIR'])

events = pickle.load(
    open(data_directory / 'events_list.pickle', 'rb'))
labels = pickle.load(
    open(data_directory / 'labels_list.pickle', 'rb'))

df_labels = pd.read_csv(data_directory / 'labels.csv')

df_labels['label'] = df_labels['label'].replace('resonant/spurious noise',
                                                'structured noise')

df_labels['label'] = df_labels['label'].replace('noise',
                                                'unstructured noise')

seismic_data_bucket = 'seismic-data'
# spectrogram_bucket = 'spectrogram-auto-ml-ot'
spectrogram_bucket = 'event-classification-mel-spectrograms'

# event_gcp = RequestEventGCP(events[0],
# seismic_data_bucket,
# spectrogram_bucket)

with open(data_directory / 'training_input_256_256.csv', 'w') as training_file:
    for event in tqdm(events):
        event_gcp = RequestEventGCP(event,
                                    seismic_data_bucket,
                                    spectrogram_bucket,
                                    spectrogram_height=256,
                                    spectrogram_width=256)
        labels_dict = df_labels[df_labels['event_resource_id'] ==
                                event_gcp.event_resource_id]
        try:
            file_names, spec_labels = event_gcp.write_spectrogram_to_bucket(
                labels_dict, db_scale=True)
        except Exception as e:
            logger.error(e)
            continue

        for file_name, spec_label in zip(file_names, spec_labels):
            training_file.write(f'{file_name}, {spec_label}\n')

with open(data_directory / 'training_input_128_128.csv', 'w') as training_file:
    for event in tqdm(events):
        event_gcp = RequestEventGCP(event,
                                    seismic_data_bucket,
                                    spectrogram_bucket,
                                    spectrogram_height=128,
                                    spectrogram_width=128)
        labels_dict = df_labels[df_labels['event_resource_id'] ==
                                event_gcp.event_resource_id]
        try:
            file_names, spec_labels = event_gcp.write_spectrogram_to_bucket(
                labels_dict, db_scale=True)
        except Exception as e:
            logger.error(e)
            continue

        for file_name, spec_label in zip(file_names, spec_labels):
            training_file.write(f'gs://{spectrogram_bucket}/{file_name}, '
                                f'{spec_label}\n')

with open(data_directory / 'training_input_128_128_linear_scale.csv', 'w') \
        as training_file:
    for event in tqdm(events):
        event_gcp = RequestEventGCP(event,
                                    seismic_data_bucket,
                                    spectrogram_bucket,
                                    spectrogram_height=128,
                                    spectrogram_width=128)
        labels_dict = df_labels[df_labels['event_resource_id'] ==
                                event_gcp.event_resource_id]
        try:
            file_names, spec_labels = event_gcp.write_spectrogram_to_bucket(
                labels_dict, db_scale=False)
        except Exception as e:
            logger.error(e)
            continue

        for file_name, spec_label in zip(file_names, spec_labels):
            training_file.write(f'gs://{spectrogram_bucket}/{file_name}, '
                                f'{spec_label}\n')



# st = st.resample(2000)
# st = st.filter('bandpass', freqmin=10, freqmax=1000)

# def librosa_spectrogram(tr, max_frequency=1000, height=256, width=256):
#     """
#         Using Librosa mel-spectrogram to obtain the spectrogram
#         :param tr: stream trace
#         :param height: image hieght
#         :param width: image width
#         :return: numpy array of spectrogram with height and width dimension
#     """
#     data = get_norm_trace(tr)
#     signal = data * 255
#     # signal = tr.data
#     hl = int(signal.shape[0] // (width * 1.1))  # this will cut away 5% from
#     # start and end
#     # height_2 = height * tr.stats.sampling_rate // max_frequency
#     spec = melspectrogram(signal, n_mels=height,
#                           hop_length=int(hl))
#     img = amplitude_to_db(spec)
#     start = (img.shape[1] - width) // 2
#     print(start)
#
#     return img[: , start:start + width]
#
#
# #############################################
# # Data preparation
# #############################################
# def get_norm_trace(tr, taper=True):
#     """
#     :param tr: microquake.core.Stream.trace
#     :param taper: Boolean
#     :return: normed composite trace
#     """
#
#     # c = tr[0]
#     # c = tr.composite()
#     tr = tr.detrend('demean').detrend('linear').taper(max_percentage=0.05,
#                                                       max_length=0.01)
#     tr.data = tr.data / np.abs(tr.data).max()
#
#     nan_in_trace = np.any(np.isnan(tr.data))
#
#     if nan_in_trace:
#         logger.warning('NaN found in trace. The NaN will be set '
#                        'to 0.\nThis may cause instability')
#         tr.data = np.nan_to_num(tr.data)
#
#     return tr.data
