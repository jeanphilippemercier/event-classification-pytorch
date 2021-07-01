import requests
from uquake.clients.old_api_client import api_client
from uquake.core.settings import settings
import json
from tqdm import tqdm
from loguru import logger
import pickle
from pathlib import Path
import os
import pandas as pd
from datetime import datetime

api_base_url = settings.API_BASE_URL
username = settings.API_USER
password = settings.API_PASS

tmp = api_client.get_event_types(api_base_url, username=username,
                                 password=password)

tmp = api_client.get_event_types(api_base_url, username=username,
                                 password=password)

event_types = list(tmp.keys())


def extract_labels(base_url, event_resource_id, user=None, passwd=None):

    if base_url[-1] != '/':
        base_url += '/'

    sensors_list = []
    labels_list = []

    encoded_resource_id = api_client.encode(event_resource_id)
    request_url = f'{base_url}events/' \
        f'{encoded_resource_id}/trace_labels'
    response = requests.get(request_url, auth=(user, passwd))
    labels = json.loads(response.content)
    for label in labels:
        if label['sensor'] is None:
            continue

        sensors_list.append(label['sensor']['code'])
        labels_list.append(label['label']['name'])

    return sensors_list, labels_list


def get_catalogue(base_url, user=None, passwd=None, timeout=200, **kwargs):
    if base_url[-1] != '/':
        base_url += '/'
    request_url = base_url + 'events'

    params = kwargs

    query = True
    events = []
    while query:
        if query == True:
            re = requests.get(request_url, params=params,
                              auth=(user, passwd)).json()
        else:
            re = requests.get(query).json()
        if not re:
            break
        logger.info(f"page {re['current_page']} of "
                    f"{re['total_pages']}")

        query = re['next']

        for event in re['results']:
            events.append(api_client.RequestEvent(event))

    return events

events_list = []
labels_list = []
labels_dict_out = {'event_resource_id': [], 'sensor': [], 'label': []}
for event_type in event_types:
    logger.info(f'processing {event_type}')
    status = 'rejected'
    if event_type == 'seismic event':
        status = 'accepted'

    # response, events = sc.events_list(evaluation_mode='manual',
    #                                   status=status,
    #                                   event_type=event_type,
    #                                   page_size=1000)
    try:
        events = get_catalogue(api_base_url, user=username, passwd=password,
                               status=status, event_type=tmp[event_type],
                               evaluation_mode='manual', page_size=1000)
    except Exception as e:
        logger.error(e)
        try:
            events = get_catalogue(api_base_url, user=username,
                                   passwd=password,
                                   status=status, event_type=tmp[event_type],
                                   evaluation_mode='manual', page_size=1000)
        except Exception as e:
            logger.error(e)
            continue

    for event in tqdm(events):
        # if not hasattr(event, 'trace_labels'):
        #     continue

        try:
            sensors, labels = extract_labels(api_base_url,
                                             event.event_resource_id,
                                             user=username, passwd=password)
        except:
            try:
                sensors, labels = extract_labels(api_base_url,
                                                 event.event_resource_id,
                                                 user=username,
                                                 passwd=password)
            except:
                continue

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

data_directory = Path(os.environ['EVENT_CLASSIFICATION_DATA_DIR'])

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