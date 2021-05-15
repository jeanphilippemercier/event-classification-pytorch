import os
import uuid
from azure.storage.blob import (BlobServiceClient, BlobClient,
                                ContainerClient, __version__)
from tqdm import tqdm
from pathlib import Path

import pickle

connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(
    'permanentdbfilesblob')

base_url = 'https://sppwaveform.blob.core.windows.net/permanentdbfilesblob/'
output_directory = Path('/data_2/ot-data-all')
output_directory.mkdir(parents=True, exist_ok=True)

blob_list = [blob for blob in container_client.list_blobs(name_starts_with=
                                                          'events/')]
#
# pickle.dump(blob_list)

# blob_list = pickle.load(open('blob_list.pickle', 'rb'))
#
catalog_blobs = [blob.name for blob in blob_list if '.xml' in blob.name]

for blob in tqdm(container_client.list_blobs('events')):
    blob = blob.name
    if '.xml' not in blob:
        continue
    try:
        output_file = output_directory / blob
        if output_file.exists():
            continue
        os.system(f'wget {base_url}{blob} -P {output_file.parent}')
    except KeyboardInterrupt:
        break
