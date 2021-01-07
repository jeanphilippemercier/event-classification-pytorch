from microquake.clients.api_client import SeismicClient, get_event_types
from microquake.core.settings import settings
from loguru import logger
from time import time
from os import path
from pathlib import Path

api_user = settings.get('api_user')
api_password = settings.get('api_password')
api_url = settings.get('api_base_url')

sc = SeismicClient(api_url, api_user, api_password)

event_types = get_event_types(api_url, username=api_user,
                              password=api_password).keys()


res, res = sc.events_list(evaluation_mode='manual', 
                          page_size=1000)

bubu
datadir = os.environ['SEISMICDATADIR']

outdir = Path(datadir)

outdir_type = outdir  # / 'seismic event' 
outdir_type.mkdir(mode=555, parents=True, exist_ok=True)
 

for i, re in enumerate(res):
    try:
        rid = re.event_resource_id.replace('/','_')[10:]
        if rid[-2:] == '.e':
            rid = rid[:-2]
        filename = outdir_type / f'{rid}.xml'
        t0 = time()
        logger.info(f'processing event:{re.event_resource_id} ({i+1}/{len(res)} '
                    f'-- {(i+1) / len(res) * 100:0.0f}%)')

        cat_file = filename.with_suffix('.xml')
        if not cat_file.exists():
            logger.info(f'downloading catalog')
            cat = re.get_event()
            cat.write(cat_file)
        else:
            logger.info(f'catalog file already exists... skipping')

        st_file = filename.with_suffix('.mseed')
        if not st_file.exists():
            logger.info(f'downloading fixed length waveforms')
            st = re.get_waveforms()
            st.write(st_file)
        else:
            logger.info('fixed lenght trace already exists... skipping')

        ctx_file = filename.with_suffix('.context_mseed')
        if not ctx_file.exists():
            logger.info(f'downloading context trace')
            ctx = re.get_context_waveforms()
            ctx.write(ctx_file)
        else:
            logger.info('context trace already exists... skipping')

        vl_file = filename.with_suffix('.variable_mseed')
        if not vl_file.exists():
            logger.info('downloading variable length waveforms')
            vl = re.get_variable_length_waveforms()
            vl.write(vl_file)
        else:
            logger.info('variable length file already exists... skipping')
        t1 = time()
        logger.info(f'done processing the event in {t1 - t0:0.0f} seconds')
    except Exception as e:
        logger.error(e)


