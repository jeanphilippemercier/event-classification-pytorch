from pathlib import Path
from uquake.core import read_inventory
from useis.core.project_manager import ProjectManager

input_data_dir = Path('/data_1/ot-reprocessed-data/')
output_directory = Path('/data_1/multi-channel-event-classification')
output_directory.mkdir(parents=True, exist_ok=True)

inventory_file = Path('/data_2/projects/event_classification/')

pm = ProjectManager('/data_2', 'event_classification', 'OT')
inventory = pm.inventory

catalog_file = output_directory / 'catalog.csv'

event_type_lookup = {'anthropogenic event': 'noise',
                     'acoustic noise': 'noise',
                     'reservoir loading': 'noise',
                     'road cut': 'noise',
                     'controlled explosion': 'blast',
                     'quarry blast': 'blast',
                     'earthquake': 'seismic event',
                     'sonic boom': 'noise',
                     'collapse': 'noise',
                     'other event': 'noise',
                     'thunder': 'noise',
                     'induced or triggered event': 'noise',
                     'explosion': 'blast',
                     'experimental explosion': 'blast'}

n_channels = 30