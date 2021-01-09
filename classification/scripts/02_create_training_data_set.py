from librosa.feature import melspectrogram
from librosa import amplitude_to_db
from microquake.core import read
from loguru import logger
import pickle
import numpy as np
from PIL import Image

events = pickle.load(
    open('/home/jpmercier01/data/seismic_data/events_list.pickle', 'rb'))
labels = pickle.load(
    open('/home/jpmercier01/data/seismic_data/labels_list.pickle', 'rb'))



# st = st.resample(2000)
# st = st.filter('bandpass', freqmin=10, freqmax=1000)


def librosa_spectrogram(tr, max_frequency=1000, height=256, width=256):
    """
        Using Librosa mel-spectrogram to obtain the spectrogram
        :param tr: stream trace
        :param height: image hieght
        :param width: image width
        :return: numpy array of spectrogram with height and width dimension
    """
    data = get_norm_trace(tr)
    signal = data * 255
    # signal = tr.data
    hl = int(signal.shape[0] // (width * 1.1))  # this will cut away 5% from
    # start and end
    # height_2 = height * tr.stats.sampling_rate // max_frequency
    spec = melspectrogram(signal, n_mels=height,
                          hop_length=int(hl))
    img = amplitude_to_db(spec)
    start = (img.shape[1] - width) // 2
    print(start)

    return img[: , start:start + width]


#############################################
# Data preparation
#############################################
def get_norm_trace(tr, taper=True):
    """
    :param tr: microquake.core.Stream.trace
    :param taper: Boolean
    :return: normed composite trace
    """

    # c = tr[0]
    # c = tr.composite()
    tr = tr.detrend('demean').detrend('linear').taper(max_percentage=0.05,
                                                      max_length=0.01)
    tr.data = tr.data / np.abs(tr.data).max()

    nan_in_trace = np.any(np.isnan(tr.data))

    if nan_in_trace:
        logger.warning('NaN found in trace. The NaN will be set '
                       'to 0.\nThis may cause instability')
        tr.data = np.nan_to_num(tr.data)

    return tr.data
