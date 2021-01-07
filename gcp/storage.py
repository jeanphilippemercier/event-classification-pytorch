from google.cloud import storage
import os
from pathlib import Path
import requests
from io import BytesIO

home = Path(os.environ['HOME'])
credentials_file = home / '.gcp/uQuake-event-classification-81c15bfe858a.json'

storage_client = storage.Client.from_service_account_json(
    json_credentials_path=credentials_file)


def write_to_bucket(bucket_name, file_obj, blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.upload_from_file(file_obj)


def write_from_azure_storage(bucket_name, file_url, blob_name):
    file_obj = BytesIO(requests.get(file_url).content)
    # file_obj = BytesIO(requests.get(res[0].event_file).content)

