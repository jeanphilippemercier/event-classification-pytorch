from google.cloud import storage
# import cloudstorage as gcs
import os
from pathlib import Path
import requests
from io import BytesIO
from loguru import logger
from microquake.clients.api_client import RequestEvent
from microquake.core import read
from spectrogram import librosa_spectrogram
from tqdm import tqdm


home = Path(os.environ['HOME'])
credentials_file = home / '.gcp/uQuake-event-classification-81c15bfe858a.json'

# storage_client = storage.Client.from_service_account_json(
#     json_credentials_path=credentials_file)


class RequestEventGCP(RequestEvent):
    def __init__(self, request_event, seismic_data_bucket,
                 spectrogram_bucket, spectrogram_height=256,
                 spectrogram_width=256, spectrogram_sampling_rate=2000):
        super().__init__(request_event.__dict__)
        self.credentials_json = credentials_file
        self.storage_client = storage.Client.from_service_account_json(
            json_credentials_path=self.credentials_json)
        self.blob_base_name = self.event_file.split('/')[-1].split('.')[0]
        self.seismic_data_bucket = self.storage_client.bucket(
            seismic_data_bucket)
        self.spectrogram_bucket = self.storage_client.bucket(
            spectrogram_bucket)
        self.spectrogram_height=spectrogram_height
        self.spectrogram_width=spectrogram_width
        self.spectrogram_sampling_rate = spectrogram_sampling_rate

    def write_data_to_bucket(self, force=False):
        bucket = self.storage_client.bucket(self.seismic_data_bucket)
        file_base_url = self.event_file[:-4]
        for extension in ['.xml', '.mseed', '.context_mseed',
                          '.variable_mseed']:
            blob = self.seismic_data_bucket.blob(self.blob_base_name +
                                                 extension)
            if (blob.exists()) & (not force):
                logger.info(f'blob {blob} already exist ... skipping!')
                continue
            logger.info(f'writing {self.blob_base_name + extension} to the bucket')
            file_url = file_base_url + extension
            try:
                file_obj = BytesIO(requests.get(file_url).content)
            except Exception as e:
                logger.error(e)

            try:
                blob.upload_from_file(file_obj)
            except Exception as e:
                logger.error(e)

    def delete_spectrograms(self):
        blobs = self.storage_client.list_blobs(self.spectrogram_bucket)
        for blob in tqdm(blobs):
            blob.delete()

    def write_spectrogram_to_bucket(self, label_dict):
        """
        create a spectrogram
        :param label_dict: A label dictionary, containing two keys <sensor>
        and <label>. The dictionary should contain a list of sensors and a list
        of labels
        :return:
        """
        st = self.get_waveform_from_bucket()
        # if self.spectrogram_bucket is None:
        #     logger.error('The bucket where to store the spectrogram needs '
        #                   'to be specified')
        #     return
        # try:
        #     st = self.get_waveform_from_bucket()
        # except Exception as e:
        #     logger.error(e)
        #     try:
        #         st = self.get_waveforms()
        #         if not st:
        #             logger.warning(f'no waveform for event '
        #                            f'{self.event_resource_id}')
        #             return
        #         self.write_data_to_bucket(force=True)
        #     except Exception as e:
        #         logger.error(e)
        #         return
        #
        # if not st:
        #     logger.warning(f'no waveform for event '
        #                    f'{self.event_resource_id}')
        #     return
        st = st.resample(sampling_rate=self.spectrogram_sampling_rate)
        spec_names = []
        labels = []
        for i in tqdm(range(len(label_dict['sensor']))):
            sensor = label_dict['sensor'].iloc[i]
            label = label_dict['label'].iloc[i]
            for tr in st.select(station=str(sensor)):
                channel = tr.stats.channel

                label.replace('/', '_').replace(' ', '_')

                spec_name = f'{label}/{self.blob_base_name}_' \
                    f'{label}_{sensor}_{channel}.png'

                spec_names.append(spec_name)
                labels.append(label)

                blob = self.spectrogram_bucket.blob(spec_name)
                # if blob.exists():
                #     continue
                    # blob.delete()
                try:
                    spec = librosa_spectrogram(tr.copy(),
                                               height=self.spectrogram_height,
                                               width=self.spectrogram_width)
                except Exception as e:
                    logger.error(e)
                    continue

                # spec_file_obj = BytesIO(spec.tobytes())
                spec_file_obj = BytesIO()
                spec_file_obj.seek(0)
                spec.save(spec_file_obj, 'png')

                blob.upload_from_file(spec_file_obj)

        return spec_names, labels

    def get_waveform_from_bucket(self):

        blob = self.seismic_data_bucket.blob(self.blob_base_name + '.mseed')
        if not blob.exists():
            logger.warning(f'blob {self.blob_base_name} does not exist '
                           f'in bucket {self.seismic_data_bucket}')
            return

        st = read(BytesIO(blob.download_as_bytes()))
        return st





