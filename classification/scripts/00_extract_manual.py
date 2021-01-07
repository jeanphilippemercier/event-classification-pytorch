from microquake.clients.api_client import SeismicClient, get_event_types
from microquake.core.settings import settings
from loguru import logger
from time import time
import os
from pathlib import Path
from gcp_storage import RequestEventGCP
import pickle

api_user = settings.get('api_user')
api_password = settings.get('api_password')
api_url = settings.get('api_base_url')

data_directory = Path(os.environ['SEISMICDATADIR'])

sc = SeismicClient(api_url, api_user, api_password)

event_types = get_event_types(api_url, username=api_user,
                              password=api_password).keys()


result, res = sc.events_list(evaluation_mode='manual', 
                             page_size=100)

logger.info('writing the event list')
pickle.dump(res, open(data_directory / 'event_list.pickle', 'wb'))

for i, re in enumerate(res):
        re_gcp = RequestEventGCP(re)
        re_gcp.write_data_to_bucket('seismic-data')



