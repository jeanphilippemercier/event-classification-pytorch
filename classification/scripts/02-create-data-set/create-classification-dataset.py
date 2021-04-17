import pandas
from uquake.core import read, read_events, read_inventory, UTCDateTime
from uquake.core.trace import Trace
from uquake.core.logging import logger
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import pickle
from useis.core.project_manager import ProjectManager
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import matplotlib.pyplot as plt
import torch
from uquake.waveform.pick import calculate_snr
from PIL import Image, ImageOps


def get_waveform(waveform_file, inventory):
    st = read(waveform_file)
    for i, tr in enumerate(st):
        for sensor in inventory.sensors:
            if sensor.alternate_code == tr.stats.station:
                st[i].stats.network = inventory[0].code
                st[i].stats.station = sensor.station.code
                st[i].stats.location = sensor.location_code
                for channel in sensor.channels:
                    if tr.stats.channel in channel.code:
                        st[i].stats.channel = channel.code
                        break
                break
    return st


def get_event_id(filename):
    cat = read_events(filename)
    return str(filename.stem), cat[0].resource_id.id

# with open(data_directory / 'training_input_128_128.csv', 'w') as training_file:
#     for event in tqdm(events):
#         event_gcp = RequestEventGCP(event,
#                                     seismic_data_bucket,
#                                     spectrogram_bucket,
#                                     spectrogram_height=128,
#                                     spectrogram_width=128)
#         labels_dict = df_labels[df_labels['event_resource_id'] ==
#                                 event_gcp.event_resource_id]
#         try:
#             file_names, spec_labels = event_gcp.write_spectrogram_to_bucket(
#                 labels_dict, db_scale=True)
#         except Exception as e:
#             logger.error(e)
#             continue
#
#         for file_name, spec_label in zip(file_names, spec_labels):
#             training_file.write(f'gs://{spectrogram_bucket}/{file_name}, '
#                                 f'{spec_label}\n')


event_type_lookup = {'anthropogenic event': 'noise',
                     'acoustic noise': 'noise',
                     'reservoir loading': 'noise',
                     'road cut': 'noise',
                     'controlled explosion': 'blast',
                     'quarry blast': 'blast',
                     'earthquake': 'seismic event',
                     'sonic boom': 'test pulse',
                     'collapse': 'noise',
                     'other event': 'noise',
                     'thunder': 'noise',
                     'induced or triggered event': 'unknown',
                     'explosion': 'blast',
                     'experimental explosion': 'blast'}


class TrainingClassifier(ProjectManager):
    def __init__(self, path: str, project_name: str, network_code: str,
                 input_data_dir: str, output_data_dir: str,
                 sampling_rate: int = 6000, num_threads: int = -2,
                 replication_level: int = 5, snr_threshold: float = 10,
                 sequence_length_second: float = 2,
                 perturbation_range_second: float = 1, image_width=128,
                 image_height=128, buffer_image_fraction=0.05):
        super().__init__(path, project_name, network_code)
        self.input_data_dir = Path(input_data_dir)
        self.output_data_dir = Path(output_data_dir)
        self.output_data_dir.mkdir(parents=True, exist_ok=True)

        if num_threads < 0:
            self.num_threads = int(np.ceil(cpu_count() + num_threads))
        else:
            self.num_threads = num_threads

        self.waveform_file_list = self.input_data_dir.glob('*.mseed')
        self.num_files = len([f for f in self.waveform_file_list])
        self.sampling_rate = sampling_rate
        self.replication_level = replication_level
        self.snr_threshold = snr_threshold
        self.sequence_length_second = sequence_length_second
        self.perturbation_range = perturbation_range_second

        self.image_width = image_width
        self.image_height = image_height
        self.buffer_image_fraction = buffer_image_fraction

        self.buffer_image_sample = int(self.image_width
                                       * self.buffer_image_fraction)

        hop_length = int(self.sequence_length_second * self.sampling_rate //
                         (self.image_width + 2 * self.buffer_image_sample))

        self.mel_spec = MelSpectrogram(sample_rate=self.sampling_rate,
                                       n_mels=self.image_height,
                                       hop_length=hop_length,
                                       power=1,
                                       pad_mode='reflect',
                                       normalized=True)

        # self.mel_spec = MelSpectrogram(sample_rate=self.sampling_rate,
        #                                n_fft=self.image_height,
        #                                hop_length=hop_length,
        #                                power=1,
        #                                pad_mode='reflect',
        #                                normalized=True)

        self.amplitude_to_db = AmplitudeToDB()

    def spectrogram(self, trace: Trace, perturbation_sample: float = None):

        trace = trace.taper(max_length=0.01, max_percentage=0.05)
        # plt.figure(3)
        # plt.clf()
        # plt.plot(trace.data)

        n_pts = self.sequence_length_second * self.sampling_rate

        target_sequence_length = int(
            2 ** (np.ceil(np.log(n_pts) / np.log(2))))
        n_zeros = target_sequence_length - len(trace.data)

        data = np.hstack((trace.data, np.zeros(n_zeros)))

        data = np.roll(data, perturbation_sample)

        # data = data[:sequence_length_second * sampling_rate]

        torch_data = torch.tensor(data).type(torch.float32)

        spec = (self.mel_spec(torch_data) + 1e-3)[:, self.buffer_image_sample:
                                                     self.buffer_image_sample
                                                     + self.image_width]
        spec_db = self.amplitude_to_db(spec.abs())
        spec_db = spec_db - spec_db.min()
        spec_db = (spec_db / spec_db.max() * 255).type(torch.uint8)
        img = Image.fromarray(np.array(spec_db.tolist()).astype(
            np.uint8))

        return img

    def create_training_image(self, waveform_file):
        waveform_file_path = self.input_data_dir / waveform_file
        # with Pool(self.num_threads) as pool:
        #     file_list = [f for f in self.input_data_dir.glob('*.mseed')]
        #     event_list = list(tqdm(pool.imap(__create_training_images_event__,
        #                                      file_list),
        #                            total=len(file_list)))
        cat = read_events(waveform_file_path.with_suffix('.xml'))
        st = read(waveform_file_path)

        event_time = cat[0].preferred_origin().time.timestamp

        st = st.detrend('demean').detrend(
            'linear').taper(max_percentage=0.1,
                            max_length=0.01).resample(
            sampling_rate=self.sampling_rate)

        trs = []
        snrs = []
        if event_type_lookup[cat[0].event_type] == 'seismic event':
            for arrival in cat[0].preferred_origin().arrivals:
                # if arrival.pick.evaluation_mode == 'automatic':
                #     if arrival.pick.snr is None:
                #         continue
                #     elif arrival.pick.snr < self.snr_threshold:
                #         continue
                site = arrival.pick.site
                for tr in st.select(site=site):
                    snr = calculate_snr(tr, arrival.pick.time,
                                        pre_wl=20e-3,
                                        post_wl=20e-3)
                    if snr < self.snr_threshold:
                        continue
                    tr.trim(starttime=tr.stats.starttime,
                            endtime=tr.stats.starttime +
                                    self.sequence_length_second,
                            pad=True,
                            fill_value=0)
                    trs.append(tr.copy().resample(sampling_rate=
                                                  int(self.sampling_rate)))
                    snrs.append(snr)

        else:
            for tr in st:
                trs = [tr.trim(endtime=tr.stats.starttime +
                                       self.sequence_length_second)]
                snrs.append(0)

        for tr, snr in zip(trs, snrs):
            perturbations = (np.random.rand(self.replication_level)
                             * self.perturbation_range
                             * self.sampling_rate).astype(int)

            for i, perturbation in enumerate(perturbations):
                img = self.spectrogram(tr,
                                       perturbation_sample=perturbation)

                filename = f'{event_time:0.0f}_{tr.stats.site}_' \
                           f'{tr.stats.channel}_{i}.jpg'

                out_dir = (self.output_data_dir /
                           event_type_lookup[cat[0].event_type])

                out_dir.mkdir(parents=True, exist_ok=True)

                out_file_name = out_dir / filename
                ImageOps.grayscale(img).save(out_file_name, format='JPEG')

                    # spec_db = spec_db[:, start_pixel: start_pixel + self.image_width]
        # spec_db = np.log(spec)

        # plt.figure(3)
        # plt.clf()
        # plt.plot(spec_db[:, 1] / spec_db[:, 1].max())
        # plt.plot(spec[:, 1] / spec[:, 1].max())
        #
        # plt.figure(1)
        # plt.clf()
        # img.show()
        # # plt.imshow(spec_db)
        # # plt.colorbar()
        #
        # plt.figure(2)
        # plt.clf()
        # plt.plot(data)
        # plt.show()
        # input('crapout')
        # return spec_db
        # from ipdb import set_trace
        # set_trace
        #         spectrogram /= spectrogram[spectrogram > 0].min()


if __name__ == '__main__':
    name = 'event_classification'
    network = 'OT'
    base_directory = '/data_2/projects/'
    ts = TrainingClassifier(base_directory, name, network,
                            '/data_1/ot-reprocessed-data/',
                            '/data_1/classification_dataset')
    nb_list = len([f for f in ts.input_data_dir.glob('*.mseed')])
    for f in tqdm(ts.input_data_dir.glob('*.mseed'), total=nb_list):
        try:
            ts.create_training_image(f)
        except Exception as e:
            logger.info(e)

    # with Pool(ts.num_threads) as pool:
    #     file_list = [f for f in ts.input_data_dir.glob('*.mseed')]
    #     event_list = list(tqdm(pool.imap(ts.create_training_image,
    #                                      file_list),
    #                            total=len(file_list)))



# class TrainingSet(ProjectManager):
#     def __init__(self, path, project_name, network_code, input_data_dir,
#                  output_data_dir, common_dir='../../common', num_threads=-2,
#                  base_data_augmentation=2, sequence_length=1,
#                  buffer_before=0.02, buffer_after=0.2, number_sample=6000,
#                  image_height=192, image_width=192, sampling_rate=6000):
#         super().__init__(path, project_name, network_code)
#
#         self.input_data_dir = Path(input_data_dir)
#         self.output_data_dir = Path(output_data_dir)
#         self.output_data_dir.mkdir(parents=True, exist_ok=True)
#         self.file_list = self.input_data_dir.glob('*.xml')
#         self.alternative_data_dir = Path(input_data_dir.replace(
#             '-reprocessed', ''))
#         self.common = Path(common_dir)
#         self.inventory = read_inventory(str(self.common / 'inventory.xml'))
#         self.labels = pd.read_csv(self.common / 'labels.csv')
#         self.labels['new_label'] = self.labels['label']
#         self.labels['new_label'][self.labels['label'] ==
#             'resonant/spurious noise'] = 'structured noise'
#         self.labels['new_label'][self.labels['label'] ==
#                                  'noise'] = 'unstructured noise'
#         self.sequence_length = sequence_length
#         self.base_data_augmentation = base_data_augmentation
#         self.buffer_before = buffer_before
#         self.buffer_after = buffer_after
#         self.number_sample = number_sample
#         self.sampling_rate = sampling_rate
#
#         self.image_height = image_height
#         self.image_width = image_width
#
#         self.site_lookup = {}
#         self.buffer_fraction = 0.05  # buffer to add on both side of the image
#         self.buffer_sample = int((self.sequence_length * self.sampling_rate)
#                                  * self.buffer_fraction)
#
#         self.buffer_image_sample = int(self.image_width *
#                                        self.buffer_fraction)
#
#         hl = int(self.sequence_length * self.sampling_rate //
#                  (self.image_width + 2 * self.buffer_image_sample))
#
#         self.mel_spec = MelSpectrogram(sample_rate=self.sampling_rate,
#                                        n_mels=self.image_height,
#                                        hop_length=hl,
#                                        power=2,
#                                        pad_mode='reflect')
#
#         for site in self.inventory.sites:
#             self.site_lookup[site.alternate_code] = site.code
#
#         if num_threads < 0:
#             self.num_threads = int(np.ceil(cpu_count() - num_threads))
#         else:
#             self.num_threads = num_threads
#
#         self.num_files = len([f for f in self.file_list])
#         self.file_list = self.input_data_dir.glob('*.xml')
#
#         self.event_map_file = self.common / 'event_map.pickle'
#
#         if self.event_map_file.exists():
#             self.event_map = pickle.load(open(self.event_map_file, 'rb'))
#         else:
#             with Pool(self.num_threads) as pool:
#
#                 self.event_map = {}
#                 for result in tqdm(pool.imap(get_event_id,
#                                              self.input_data_dir.glob(
#                                                  '*.xml')),
#                                    total=self.num_files):
#                     self.event_map[result[1]] = result[0]
#
#             pickle.dump(self.event_map, open(self.event_map_file, 'wb'))
#
#         self.label_count = {}
#         self.total_labels = 0
#         self.max_nb_labels = 0
#         for label in np.unique(self.labels['new_label']):
#             count = len(self.labels[self.labels['new_label'] == label])
#             self.label_count[label] = count
#             self.total_labels += count
#             if self.label_count[label] > self.max_nb_labels:
#                 self.max_nb_labels = self.total_labels
#
#     def create_dataset(self):
#         with Pool(self.num_threads) as pool:
#             event_ids = np.unique(self.labels['event_resource_id'])
#             pool(self.prepare_data, event_ids)
#
#     def read_data(self, event_id):
#         stem = self.event_map[event_id]
#         filename = self.input_data_dir / stem
#         filename_alternate = self.alternative_data_dir / stem
#         event_file = filename.with_suffix('.xml')
#         waveform_file = filename.with_suffix('.mseed')
#         cat = read_events(event_file)
#         try:
#             st = read(waveform_file)
#         except Exception as e:
#             logger.error(e)
#             waveform_file = filename_alternate.with_suffix('.mseed')
#             st = get_waveform(waveform_file, self.inventory)
#
#         return cat, st
#
#     def prepare_data(self, event_id, ):
#         pass
#
#     def create_spectrogram(self, trace: Trace, window_start_time: UTCDateTime):
#
#         trace = trace.resample(sampling_rate=self.sampling_rate).detrend(
#             'demean').detrend('linear').filter('highpass', freq=15)
#
#         window_start_sample = int((window_start_time - trace.stats.starttime) *
#                                   self.sampling_rate) - self.buffer_sample
#
#         window_end_sample = (window_start_sample + self.sequence_length
#                              * self.sampling_rate + self.buffer_sample)
#
#         if window_start_sample < 0:
#             window_start_sample = 0
#             window_end_sample = (self.sequence_length * self.sampling_rate
#                                  + 2 * self.buffer_sample)
#
#         data = trace.data
#         data /= np.max(data)
#
#         trim_data = torch.tensor(data[window_start_sample:
#                                       window_end_sample]).type(torch.float32)
#
#         plt.figure(2)
#         plt.clf()
#         plt.plot(trim_data)
#         plt.show()
#         # trim_data /= trim_data.max()
#
#         spectrogram = self.mel_spec(trim_data)
#         spectrogram /= spectrogram[spectrogram > 0].min()
#
#         # return spectrogram
#
#         a2db = AmplitudeToDB()
#         db_spectrogram = a2db(spectrogram[:, self.buffer_image_sample:
#                                           self.buffer_image_sample +
#                                           self.image_width])
#
#         db_spectrogram = db_spectrogram / db_spectrogram.max() * 255
#         db_spectrogram = db_spectrogram.type(torch.uint8)
#
#         return db_spectrogram
#
#     def prepare_training_data(self, event_id):
#         labels = self.labels[self.labels['event_resource_id'] == event_id]
#
#         cat, st = self.read_data(event_id)
#         if cat[0].preferred_origin():
#             origin = cat[0].preferred_origin()
#         else:
#             origin = cat[0].origins[-1]
#
#         origin_time = origin.time
#         tts = self.travel_times.travel_time(origin.loc)
#
#         for label in labels.iterrows():
#             label = label[1]
#             duplication_level = int(self.max_nb_labels /
#                                     self.label_count[label['new_label']] *
#                                     self.base_data_augmentation)
#
#             site_code = self.site_lookup[str(label['sensor'])]
#             trs = st.copy().select(site=site_code)
#
#             if label['new_label'] in ['signal', 'blast']:
#                 p_travel_time = tts['P'][site_code]
#                 s_travel_time = tts['S'][site_code]
#
#                 p_predicted_arrival_time = origin_time + p_travel_time
#                 s_predicted_arrival_time = origin_time + s_travel_time
#
#                 maximum_start_time = p_predicted_arrival_time - 0.05
#                 minimum_start_time = s_predicted_arrival_time + 0.1 - \
#                                      self.sequence_length
#
#                 if minimum_start_time < trs[0].stats.starttime:
#                     minimum_start_time = trs[0].stats.starttime
#
#                 if (maximum_start_time > trs[0].stats.endtime -
#                         self.sequence_length):
#
#                     maximum_start_time = trs[0].stats.endtime - \
#                     self.sequence_length
#             else:
#                 minimum_start_time = trs[0].stats.starttime
#                 maximum_start_time = (trs[0].stats.endtime -
#                                       self.sequence_length)
#
#             start_time_span = maximum_start_time - minimum_start_time
#
#             for tr in st.copy().select(site=site_code):
#                 for jitter in np.random.rand(duplication_level):
#                     jitter_second = jitter * start_time_span
#                     start_time = minimum_start_time + jitter_second
#
#                     spectrogram = self.create_spectrogram(tr, start_time)
#                     plt.figure(1)
#                     plt.clf()
#                     plt.imshow(spectrogram)
#                     plt.show()
#                     input(f'{label["new_label"]} {start_time} ')
#


            #
            #
            #
            # else:
            #     min_start_time = start_time
            #     max_start_time = tr.stats.endtime - self.sequence_length
            #
            # for tr in st.copy().select(site=site_code):
            #
            #     span_sample = self.sequence_length * tr.stats.sampling_rate
            #     buffer_frac = 0.05
            #     buffer = int(span_sample * buffer_frac)
            #     buffer_image = int(self.image_width * buffer_frac)
            #
            #
            #     start_time = self.sequence_length
            #
            #     if label['new_label'] == 'signal':
            #         for perturbation_normalized in np.random.rand(
            #                 duplication_level):
            #
            #             perturbation_second = (perturbation_normalized *
            #                                    time_delta)
            #
            #             sample_rate = tr.stats.sampling_rate
            #             start_time = min_start_time + perturbation_second
            #             start_sample = int((start_time - tr.stats.starttime)
            #                                * sample_rate)
            #             if start_sample < buffer:
            #                 start_sample = buffer
            #             end_sample = int(start_sample + self.sequence_length
            #                              * sample_rate)
            #
            #             data = tr.data[start_sample - buffer:
            #                            end_sample + buffer].astype(np.float32)
            #
            #             data /= np.max(np.abs(data))
            #             hl = int(data.shape[0] // (self.image_width +
            #                                        2 * buffer_image))
            #             mel_spec = MelSpectrogram(sample_rate=sample_rate,
            #                                       n_mels=self.image_height,
            #                                       hop_length=hl,
            #                                       power=2,
            #                                       pad_mode='reflect')
            #
            #             a2db = AmplitudeToDB()
            #             spectrogram = a2db(np.abs(
            #                 mel_spec(torch.tensor(data))))
            #
            #             spectrogram = spectrogram[:,
            #                           buffer_image:
            #                           self.image_width+buffer_image]
            #
            #             spectrogram = np.abs(spectrogram)
            #             spectrogram /= spectrogram.max()


                        # start_sample = int((start_time - tr.stats.starttime) *
                        #                    self.sampling_rate)
                        #
                        #
                        # end_sample = int(start_sample + self.sequence_length
                        #                  * self.sampling_rate)
                        # if end_sample > len(tr.data):
                        #     end_sample = len(tr.data)
                        # tr_out = tr.copy().resample(self.sampling_rate)
                        # data_new = tr_out.data[start_sample:end_sample]



        # return st, cat

        # for label in np.unique(self.labels['label']):
        #     new_label = self.convert_label(label)
        #
        # event_ids = np.unique(self.labels['event_resource_id'])
        # with Pool(self.num_threads) as pool:
        #
        # for event_id in :
        #     cat = read_events(filename)
        #     try:
        #         st = read(filename.with_suffix('.mseed'))
        #     except Exception as e:
        #         logger.error(e)
        #         try:
        #             alternate_file = (self.alternative_data_dir
        #                               / filename.stem)
        #             st = get_waveform(alternate_file.with_suffix('.mseed'),
        #                               self.inventory)
        #         except Exception as e:
        #             logger.error(e)
        #             continue
        #
        #     if cat[0].event_type in ['earthquake',
        #                              'induced or triggered event']:
        #         pass
        #
        #
        #     kaboum


# if __name__ == '__main__':
#     name = 'event_classification'
#     network = 'OT'
#     base_directory = '/data_2/projects/'
#     ts = TrainingSet(base_directory, name, network,
#                      '/data_1/ot-reprocessed-data/',
#                      '/data_1/classification_dataset')
#
#     # ts.create_dataset()