from google.cloud import storage
import os
from pathlib import Path
import requests
from io import BytesIO
from loguru import logger
from microquake.clients.api_client import RequestEvent


home = Path(os.environ['HOME'])
credentials_file = home / '.gcp/uQuake-event-classification-81c15bfe858a.json'

# storage_client = storage.Client.from_service_account_json(
#     json_credentials_path=credentials_file)


class RequestEventGCP(RequestEvent):
    def __init__(self, request_event):
        self.credentials_json = credentials_file
        self.api_base_url
        super().__init__(request_event.__dict__)

    def write_data_to_bucket(self, bucket_name):
        storage_client = storage.Client.from_service_account_json(
            json_credentials_path=self.credentials_json)
        bucket = storage_client.bucket(bucket_name)
        file_base_url = self.event_file[:-4]
        blob_base_name = self.event_file.split('/')[-1].split('.')[0]
        for extension in ['.xml', '.mseed', '.context_mseed',
                          '.variable_mseed']:
            blob = bucket.blob(blob_base_name + extension)
            if blob.exists():
                logger.info(f'blob {blob} already exist ... skipping!')
                continue
            logger.info(f'writing {blob_base_name + extension} to the bucket')
            file_url = file_base_url + extension
            try:
                file_obj = BytesIO(requests.get(file_url).content)
            except Exception as e:
                logger.error(e)

            try:
                blob.upload_from_file(file_obj)
            except Exception as e:
                logger.error(e)

