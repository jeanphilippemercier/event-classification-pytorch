from uquake.core import Stream, Trace
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np
import torch


def generate_spectrogram(stream: Stream):

    specs = []
    trs = []
    for tr in stream.copy():
        # check if tr contains NaN
        if np.any(np.isnan(tr.data)):
            continue
        trs.append(tr.copy())

    st2 = Stream(traces=trs)
    st2 = st2.trim(endtime=tr.stats.starttime + 2, pad=True,
                   fill_value=0)
    st2 = st2.detrend('demean').detrend('linear')

    st2 = st2.taper(max_percentage=1, max_length=1e-2)

    for tr in st2:
        spec = spectrogram(tr)
        spec /= np.max(spec)

        specs.append(spec)

    return specs

sampling_rate = 6000
# num_threads = int(np.ceil(cpu_count() - 10))
num_threads = 10
replication_level = 5
snr_threshold = 10
sequence_length_second = 2
perturbation_range_second = 1
image_width = 128
image_height = 128
buffer_image_fraction = 0.05

buffer_image_sample = int(image_width * buffer_image_fraction)

hop_length = int(sequence_length_second * sampling_rate //
                 (image_width + 2 * buffer_image_sample))


def spectrogram(trace: Trace):

    trace.resample(sampling_rate)

    mel_spec = MelSpectrogram(sample_rate=sampling_rate,
                              n_mels=image_height,
                              hop_length=hop_length,
                              power=1,
                              pad_mode='reflect',
                              normalized=True)

    amplitude_to_db = AmplitudeToDB()

    trace.data = trace.data - np.mean(trace.data)
    trace = trace.taper(max_length=0.01, max_percentage=0.05)
    trace = trace.trim(starttime=trace.stats.starttime,
                       endtime=trace.stats.starttime + sequence_length_second,
                       pad=True, fill_value=0)
    data = trace.data

    torch_data = torch.tensor(data).type(torch.float32)

    spec = (mel_spec(torch_data))
    spec_db = amplitude_to_db(spec.abs() + 1e-3)
    spec_db = (spec_db - spec_db.min()).numpy()
    # spec_db = (spec_db / spec_db.max()).type(torch.float32)
    return spec_db
