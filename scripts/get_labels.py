import requests
from microquake.clients import api_client
from microquake.core.settings import settings
import json
from tqdm import tqdm
from loguru import logger

api_base_url = settings.API_BASE_URL
username = settings.API_USER
password = settings.API_PASSWORD
sc = api_client.SeismicClient(api_base_url, username, password)

tmp = api_client.get_event_types(api_base_url, username=username,
                                 password=password)

event_types = list(tmp.keys())


def extract_labels(event_resource_id):
    encoded_resource_id = api_client.encode(event_resource_id)
    request_url = f'{api_base_url}events/' \
        f'{encoded_resource_id}/trace_labels'
    response = requests.get(request_url)
    return json.loads(response.content)


out_dict={'event_id': [],
          'sensor_id': [],
          'label_id': [],
          'label': []}

events_list = []
labels_list = []
for event_type in event_types:
    logger.info(f'processing {event_type}')
    status = 'rejected'
    if event_type == 'seismic event':
        status = 'accepted'

    response, events = sc.events_list(evaluation_mode='manual',
                                           status=status, page_size=1000)

    for event in tqdm(events):
        labels = extract_labels(event.event_resource_id)

        for label in labels:
            labels_list.append(label)

# converting list of dictionary to dictionary of list

labels_list = {k: [dic[k] for dic in labels_list] for k in labels_list[0]}
events_list = {k: [dic[k] for dic in events_list] for k in events_list[0]}