from useismic.processors import nlloc

from uquake.core import (read_events, read)
# from uquake.core.settings import settings
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
from uquake.core.logging import logger


__cpu_count__ = cpu_count()

input_data_path = Path('/data_1/ot-data')
output_data_path = Path('/data_1/ot-reprocessed-data')

# pm = project_manager.ProjectManager.init_from_settings(settings)
name = 'event_classification'
network = 'OT'
base_directory = '/data_2/projects/'


def get_waveform(waveform_file, inventory):
    st = read(waveform_file)
    for i, tr in enumerate(st):
        for sensor in inventory.sensors:
            if sensor.alternate_code == tr.stats.station:
                st[i].stats.network = inventory[0].code
                st[i].stats.station = sensor.station.code
                st[i].stats.location = sensor.location_code
                for channel in sensor.channels:
                    if tr.stats.channel in channel.code:
                        st[i].stats.channel = channel.code
                        break
                break
    return st


def get_catalog(event_file, inventory):
    cat = read_events(event_file)
    for i, pick in enumerate(cat[0].picks):
        for sensor in inventory.sensors:
            if sensor.alternate_code == pick.waveform_id.station_code:
                cat[0].picks[i].waveform_id.network_code = inventory[0].code
                cat[0].picks[i].waveform_id.station_code = sensor.station.code
                cat[0].picks[i].waveform_id.location_code = \
                    sensor.location_code
                cat[0].picks[i].waveform_id.channel_code = \
                    sensor.channels[0].code
                break
    return cat


def process(input_file):
    nll = nlloc.NLLOC(base_directory, name, network)

    inventory = nll.inventory
    try:
        cat = get_catalog(str(input_file), inventory)
        fixed_length = get_waveform(input_file.replace('xml', 'mseed'),
                                    inventory)
        variable_length = get_waveform(input_file.replace('xml',
                                                          'variable_mseed'),
                                       inventory)
        context_trace = get_waveform(input_file.replace('xml',
                                                        'context_mseed'),
                                     inventory)
        not_located_event_id = None
    except Exception as e:
        logger.error(e)
        not_located_event_ids = input_file
        return

    if len(cat[0].preferred_origin().arrivals) != 0:
        try:
            loc = nll.run_location(event=cat[0], multithreading=False)
            cat_out = loc + cat
        except Exception as e:
            logger.error(e)
            cat_out = cat
            not_located_event_id = input_file
    else:
        cat_out = cat
        not_located_event_id = input_file
    output_file = Path(str(input_file).replace('ot-data',
                                               'ot-reprocessed-data'))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cat_out.write(str(output_file))
    fixed_length.write(str(output_file).replace('xml', 'mseed'))
    variable_length.write(str(output_file).replace('xml', 'variable_mseed'))
    context_trace.write(str(output_file).replace('xml', 'context_mseed'))
    return not_located_event_id


catalogue = dict(id=[], event_time=[], event_type=[], evaluation_mode=[],
                 magnitude=[])

cpu_utilisation = 0.8
num_threads = int(np.ceil(cpu_utilisation * __cpu_count__))

files = [str(cat_file) for cat_file in input_data_path.glob('*.xml')]

with Pool(num_threads) as pool:
    pbar = tqdm(total=len(files))
    results = []
    for result in pool.imap_unordered(process, files):
        if result is not None:
            results.append(result)
        pbar.update(1)

# for fle in files:
#     get_catalog(fle)
#         # print(f'{i} of {len(files)}')

# for event_file in tqdm(input_data_path.glob('*.xml')):
#     event = get_catalog(str(event_file))

