from uquake.core import read_events
from importlib import reload
import settings
from multiprocessing import Pool
from tqdm import tqdm
reload(settings)


def create_catalog(mseed_file):
    cat = read_events(mseed_file.with_suffix('.xml'))
    if cat[0].preferred_origin() is not None:
        evaluation_mode = cat[0].preferred_origin().evaluation_mode
    else:
        evaluation_mode = cat[0].origins[-1].evaluation_mode
    return (str(mseed_file),
            settings.event_type_lookup[cat[0]['event_type']],
            evaluation_mode)


file_list = [fle for fle in settings.input_data_dir.glob('*.mseed')]

with Pool(44) as p:
    list_tmp = list(tqdm(p.imap(create_catalog, file_list)))

with open(settings.catalog_file, 'w') as f_out:
    f_out.write('file,event type,evaluation mode\n')
    for item in list_tmp:
        f_out.write(f'{item[0]},{item[1]},{item[2]}\n')


