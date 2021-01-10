from google.cloud import storage
# import cloudstorage as gcs
import os
from pathlib import Path
import requests
from io import BytesIO
from loguru import logger
from microquake.clients.api_client import RequestEvent
from microquake.core import read
from spectrogram import librosa_spectrogram(tr,  height=256, width=256):


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


    def write_data_to_bucket(self):
        bucket = self.storage_client.bucket(self.seismic_data_bucket)
        file_base_url = self.event_file[:-4]
        for extension in ['.xml', '.mseed', '.context_mseed',
                          '.variable_mseed']:
            blob = self.seismic_data_bucket.blob(self.blob_base_name +
                                                 extension)
            if blob.exists():
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

    def write_spectrogram_to_bucket(self, ):
        if self.spectrogram_bucket is None:
            logger.error('The bucket where to store the spectrogram needs '
                          'to be specified')

            st = self.get_waveform_from_bucket()
            if not st:
                logger.warning(f'no waveform for event '
                               f'{self.event_resource_id}')
                return
            st = st.resample(sampling_rate=self.spectrogram_sampling_rate)


    def create_spectrogram(self, tr):

        st_blob = input_bucket.blob(self.blob_base_name + '.mseed')


    def load_waveform_from_bucket(self):

        blob = self.seismic_data_bucket.blob(self.blob_base_name + '.mseed')
        if not blob.exists():
            logger.warning(f'blob {self.blob_base_name} does not exist '
                           f'in bucket {self.seismic_data_bucket}')
            return

        self.st = read(BytesIO(blob.download_as_bytes()))





