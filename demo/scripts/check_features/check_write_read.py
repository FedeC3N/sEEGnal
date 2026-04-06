"""
Compare the saved features with the estimated features

Federico Ramírez-Toraño
01/04/2026

"""

# imports
import numpy
from sEEGnal.tools.mne_tools import prepare_eeg
from sEEGnal.tools.fc_tools import compute_plv, compute_ciplv
from dev.init.init import init
from sEEGnal.tools.bids_tools import build_BIDS_object, read_plv, read_ciplv, read_relative_power_spectrum
from sEEGnal.tools.psd_tools import multitaper_psd, normalize_psd

# Init the database
config, files, sub, ses, task = init()
config['subsystem'] = 'feature_extraction'

# Go through each subject
current_index = 0

# current info
current_file = files[current_index]
current_sub = sub[current_index]
current_ses = ses[current_index]
current_task = task[current_index]

# Create the subjects following AI-Mind protocol
BIDS = build_BIDS_object(config, current_sub, current_ses, current_task)



# POWER
sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
}

freq_limits_components = [
    config['component_estimation']['low_freq'],
    config['component_estimation']['high_freq']
]

freq_limits_signal = [
    config['feature_extraction']['relative_power_spectrum']['sensor']['freq_limits'][0],
    config['feature_extraction']['relative_power_spectrum']['sensor']['freq_limits'][-1]
]

raw = prepare_eeg(
    config,
    BIDS,
    preload=True,
    channels_to_include=config['global']["channels_to_include"],
    channels_to_exclude=config['global']["channels_to_exclude"],
    freq_limits=freq_limits_components,
    notch_filter=True,
    resample_frequency=config['component_estimation']['resample_frequency'],
    set_annotations=True,
    crop_seconds=config['component_estimation']['crop_seconds'],
    rereference='average'
)

raw = prepare_eeg(
    config,
    BIDS,
    raw=raw,
    apply_sobi=sobi,
    freq_limits=freq_limits_signal,
    metadata_badchannels=True,
    epoch_definition=config['feature_extraction']['relative_power_spectrum']['epoch_definition']
)

sfreq = raw.info["sfreq"]


params = config['feature_extraction']['relative_power_spectrum']['sensor']

# raw is epoched here → shape (n_epochs, n_channels, n_times)
data = raw.get_data()

psd, freqs = multitaper_psd(
    data=data,
    sfreq=raw.info['sfreq'],
    fmin=freq_limits_signal[0],
    fmax=freq_limits_signal[-1],
    bandwidth=params['bandwidth'],
    adaptive=params['adaptive'],
    average_epochs=True,
    dtype=numpy.float32
)

relative_power_spectrum = normalize_psd(psd, mode="relative")

relative_power_spectrum_read,freqs,metadata = read_relative_power_spectrum(config, BIDS, space='sensor')

print(f"Power differences: {numpy.mean(relative_power_spectrum.flatten() - relative_power_spectrum_read.flatten())}")

# PLV
sobi = {
        'desc': 'sobi',
        'components_to_include': ['brain', 'other'],
        'components_to_exclude': []
}

freq_limits_components = [
    config['component_estimation']['low_freq'],
    config['component_estimation']['high_freq']
]

freq_limits_signal = [
    config['feature_extraction']['plv']['freq_limits'][0],
    config['feature_extraction']['plv']['freq_limits'][-1]
]

raw = prepare_eeg(
    config,
    BIDS,
    preload=True,
    channels_to_include=config['global']["channels_to_include"],
    channels_to_exclude=config['global']["channels_to_exclude"],
    freq_limits=freq_limits_components,
    notch_filter=True,
    resample_frequency=config['component_estimation']['resample_frequency'],
    set_annotations=True,
    crop_seconds=config['component_estimation']['crop_seconds'],
    rereference='average'
)

raw = prepare_eeg(
    config,
    BIDS,
    raw=raw,
    apply_sobi=sobi,
    freq_limits=freq_limits_signal,
    metadata_badchannels=True,
    epoch_definition=config['feature_extraction']['plv']['epoch_definition']
)


params = config['feature_extraction']['plv']['sensor']

# Save memory
nepochs, nchannels, nsamples = raw.get_data().shape

iband = 0
current_band = params['freq_bands_name'][iband]

# Filter the data
banddata = raw.copy().load_data()
banddata.filter(
    params['freq_bands_limits'][iband][0],
    params['freq_bands_limits'][iband][1]
)

# plv and metadata
plv_vector = compute_plv(
    data=banddata.get_data(),
    average_epochs=True
)


# read
plv_read, metadata_read = read_plv(config, BIDS, space='sensor',band_name=current_band)
print(f"PLV differences: {numpy.mean(plv_vector - plv_read)}")

# ciPLV
ciplv_vector = compute_ciplv(
    data=banddata.get_data(),
    average_epochs=True
)


# read
ciplv_read, metadata_read = read_ciplv(config, BIDS, space='sensor',band_name=current_band)
print(f"ciPLV differences: {numpy.mean(ciplv_vector - ciplv_read)}")