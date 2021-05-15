from useis.clients.old_api_client.api_client import get_catalog
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import copy
api_base_url = 'https://api.microquake.org/api/v1/'
output_file_path = Path('data/')
output_file_path.mkdir(parents=True, exist_ok=True)
# df=pd.read_csv('map_events.csv')
seismic_events = get_catalog(api_base_url, event_type='seismic event',
                             evaluation_status='accepted', page_size=1000)
other_events = get_catalog(api_base_url, evaluation_status='rejected',
                           page_size=1000)
events = []
for event in seismic_events:
    events.append(event)
for event in other_events:
    events.append(event)
original_events = copy.deepcopy(events)
df = pd.DataFrame({'events': events})
df.to_csv('map_events.csv', index=False)
downloaded_events = os.listdir('data')
print('Before: ' + str(len(df)))
exclude_index = []
for i in range(0, len(df)):
    if Path(df['events'][i].event_file).name in downloaded_events:
        exclude_index.append(i)
    else:
        pass
# for i in range(0, len(df)):
#     if df['events'][i].split('event_file')[1].split('events/')[1].split('\n')[0] in downloaded_events:
#         exclude_index.append(i)
#     else:
#         pass
df=df.drop(exclude_index)
df.reset_index(drop=True, inplace=True)
print('After: ' + str(len(df)))
output_file_path = Path('/home/munguush/jp_project/data/')
for event in tqdm(df['events']):
    try:
        cat = event.get_event()
        cat.write(output_file_path / Path(event.event_file).name)
        fixed_length = event.get_waveforms()
        fixed_length.write(output_file_path / Path(event.waveform_file).name)
        variable_length = event.get_variable_length_waveforms()
        variable_length.write(output_file_path /
                              Path(event.variable_size_waveform_file).name)
        context = event.get_context_waveforms()
        context.write(output_file_path / Path(event.waveform_context_file).name)
    except Exception as e:
        print(e)
        pass