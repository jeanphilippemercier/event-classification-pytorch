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
    def __init__(self, request_event, credentials_json, bucket=None):
        self.credentials_json = credentials_json
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
            file_obj = BytesIO(requests.get(file_url).content)

            blob.upload_from_file(file_obj)

    # def get_labels(self):







def write_to_bucket(bucket_name, file_obj, blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if blob.exist():
        logger.info(f'blob {blob_name} already exists... skipping')
    return blob.upload_from_file(file_obj)


def write_from_azure_storage(bucket_name, file_url, blob_name):
    bucket = storage_client.bucket(bucket_name)

    blob_name =

    blob = bucket.blob(blob_name)
    if blob.exist():
        logger.info(f'blob {blob_name} already exists... skipping')
        continue

    file_obj = BytesIO(requests.get(file_url).content)

    return blob.upload_from_file(file_obj)
    # file_obj = BytesIO(requests.get(res[0].event_file).content)

