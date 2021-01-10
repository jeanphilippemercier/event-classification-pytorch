import requests
from microquake.clients import api_client
from microquake.core.settings import settings
import json
from tqdm import tqdm
from loguru import logger
import pickle
from pathlib import Path
import os
import pandas as pd

api_base_url = settings.API_BASE_URL
username = settings.API_USER
password = settings.API_PASSWORD
sc = api_client.SeismicClient(api_base_url, username, password)

tmp = api_client.get_event_types(api_base_url, username=username,
                                 password=password)

event_types = list(tmp.keys())


def extract_labels(event_resource_id):
    sensors_list = []
    labels_list = []

    encoded_resource_id = api_client.encode(event_resource_id)
    request_url = f'{api_base_url}events/' \
        f'{encoded_resource_id}/trace_labels'
    response = requests.get(request_url)
    labels = json.loads(response.content)
    for label in labels:
        if label['sensor'] is None:
            continue

        sensors_list.append(label['sensor']['code'])
        labels_list.append(label['label']['name'])

    return sensors_list, labels_list


events_list = []
labels_list = []
labels_dict_out = {'event_resource_id': [], 'sensor': [], 'label': []}
for event_type in event_types:
    logger.info(f'processing {event_type}')
    status = 'rejected'
    if event_type == 'seismic event':
        status = 'accepted'

    response, events = sc.events_list(evaluation_mode='manual',
                                      status=status,
                                      event_type=event_type,
                                      page_size=1000)

    for event in tqdm(events):
        # if not hasattr(event, 'trace_labels'):
        #     continue

        sensors, labels = extract_labels(event.event_resource_id)

        if not labels:
            continue

        for sensor, label in zip(sensors, labels):
            labels_dict_out['sensor'].append(sensor)
            labels_dict_out['label'].append(label)
            labels_dict_out['event_resource_id'].append(
                event.event_resource_id)

        events_list.append(event)

        for lbl in labels:
            labels_list.append(lbl)

# converting list of dictionary to dictionary of list

data_directory = Path(os.environ['SEISMICDATADIR'])

events_dict = {}
for key in events_list[0].keys():
    events_dict[key] = []
for event in events_list:
    for key in events_list[0].keys():
        events_dict[key].append(event.__dict__[key])
df_event = pd.DataFrame(events_dict)
df_label = pd.DataFrame(labels_dict_out)

pickle.dump(events_list, open(data_directory / 'events_list.pickle', 'wb'))
pickle.dump(labels_list, open(data_directory / 'labels_list.pickle', 'wb'))

# pickle.dump(df_event, open(data_directory / 'events_dataframe.pickle', 'wb'))
# pickle.dump(df_event, open(data_directory / 'labels.'))

df_event.to_csv(data_directory / 'events.csv')
df_label.to_csv(data_directory / 'labels.csv')

# labels_list = {k: [dic[k] for dic in labels_list] for k in labels_list[0]}
# events_list = {k: [dic[k] for dic in events_list] for k in events_list[0]}