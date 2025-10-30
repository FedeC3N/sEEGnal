# sEEGnal

Package for automatic preprocessing of EEG recordings.

This repository comprises three high-level blocks, namely, standardize, badchannel detection and artifact detection.

## Standardize

Converts the original data structure to a BIDS-compliance data structure.

For more information regarding BIDS, consult [BIDS official website](https://bids-specification.readthedocs.io/en/stable/index.html).

##### Inputs
```
def standardize(config,current_file,bids_path):
```
config (dict): Configuration parameters (paths, parameters, etc).  
current_file (str): Name of the file to process.  
bids_path (BIDSpath): Associated bids_path.  


##### Outputs
results (dict): Main information of the process.

In addition, the module creates new BIDS-folders in the path specified in config.


## Badchannel detection

Marks badchannels in EEG recordings based on different criteria.

For EEG:

- Channels with impedances above a certain threshold.
- Channels with impossible low or high amplitudes.
- Channels with significantly higher energy in 45-55 Hz range compared to the rest of channels.
- Channels with gel bridge.
- Channels with significantly higher standard deviation of amplitude compared to the rest of channels.

##### Inputs
```
def badchannel_detection(config,bids_path):
```
config (dict): Configuration parameters (paths, parameters, etc).  
bids_path (BIDSpath): Associated bids_path.


##### Outputs
results (dict): Main information of the process.

In addition, the module creates new BIDS-derivatives files in the corresponding path. The files are:  
- *_channels.tsv: File with all channels classified as good or bad and the reason of "bad channel". Also include impedance information.
- *_desc-sobi-badchannels_*: Files with information about the Independent Component Analysis (mixing, unmixing, classification,...)



## Artifact detection

First, performs an independent component analysis (ICA) and then label the ICs using [MNE-ICALabel](https://mne.tools/mne-icalabel/stable/index.html).

Then looks for EOG arrtifacts, muscle artifacts, sensor artifacts, and "other" artifacts.

- EOG. Filter the recording in low frequencies. Compare frontal channels vs rest of the channels and look for high amplitude peaks significantly different.
- Muscle. Get the time series reconstructed using "muscle" components. Filter between 110-145 Hz and look for high amplitude bursts.
- Sensor. Filter the data in low frequencies. Look for high amplitude peaks.
- Other. Look for bursts with impossible amplitudes.

##### Inputs
```
def artifact_detection(config,bids_path):
```
config (dict): Configuration parameters (paths, parameters, etc).  
bids_path (BIDSpath): Associated bids_path.


##### Outputs
results (dict): Main information of the process.

In addition, the module creates new BIDS-derivatives files in the corresponding path. The files are:
- *_artifacts_annotations: Files with information about the artifacts: onset (seconds), duration (seconds), label (type of artifact).
- \*_desc-sobi-artifacts: Files with information about the Independent Component Analysis in the artifact detection module (mixing, unmixing, classification,...)

## Use

An example folder structure for preprocessing a single EEG recording is provided in "demo" folder.

```
demo
├───data
│   └───sourcedata
│       └───eeg
│               HIQ_001_0_2EC_2024-07-16_13-19-47.eeg
│               HIQ_001_0_2EC_2024-07-16_13-19-47.vhdr
│               HIQ_001_0_2EC_2024-07-16_13-19-47.vmrk
│
└───scripts
        config.json
        init.py
        run_sEEGnal.py
		
		
#### Files
- run_sEEGnal provides a script for preprocessing automatically any number of files.  
- confi.json is a mandatory file with all the configuration parameters needed for running sEEGnal: frequencies, channels, filenames, paths, ...  
- init.py simplifies run_sEEGnal.py by looking for files and defining some parameters.   

```

After execution, the folder structure would look like these:  
```
demo
├───data
│   │   dataset_description.json
│   │   participants.json
│   │   participants.tsv
│   │   README
│   │
│   ├───derivatives
│   │   └───sEEGnal
│   │       └───clean
│   │           └───sub-001
│   │               └───ses-0
│   │                   └───eeg
│   │                           sub-001_ses-0_task-2EC_channels.tsv
│   │                           sub-001_ses-0_task-2EC_desc-artifacts_annotations.json
│   │                           sub-001_ses-0_task-2EC_desc-artifacts_annotations.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi-badchannels_annotations.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi-badchannels_annotations.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi-badchannels_mixing.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi-badchannels_mixing.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi-badchannels_prediction_scores.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi-badchannels_prediction_scores.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi-badchannels_unmixing.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi-badchannels_unmixing.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi_annotations.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi_annotations.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi_artifacts_annotations.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi_artifacts_annotations.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi_artifacts_mixing.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi_artifacts_mixing.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi_artifacts_prediction_scores.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi_artifacts_prediction_scores.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi_artifacts_unmixing.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi_artifacts_unmixing.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi_mixing.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi_mixing.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi_prediction_scores.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi_prediction_scores.tsv
│   │                           sub-001_ses-0_task-2EC_desc-sobi_unmixing.json
│   │                           sub-001_ses-0_task-2EC_desc-sobi_unmixing.tsv
│   │
│   ├───sourcedata
│   │   └───eeg
│   │           HIQ_001_0_2EC_2024-07-16_13-19-47.eeg
│   │           HIQ_001_0_2EC_2024-07-16_13-19-47.vhdr
│   │           HIQ_001_0_2EC_2024-07-16_13-19-47.vmrk
│   │
│   └───sub-001
│       └───ses-0
│           │   sub-001_ses-0_scans.tsv
│           │
│           └───eeg
│                   sub-001_ses-0_task-2EC_channels.tsv
│                   sub-001_ses-0_task-2EC_eeg.eeg
│                   sub-001_ses-0_task-2EC_eeg.json
│                   sub-001_ses-0_task-2EC_eeg.vhdr
│                   sub-001_ses-0_task-2EC_eeg.vmrk
│
└───scripts
       config.json
       init.py
       run_sEEGnal.py

```



## Contributors
[Federico Ramírez Toraño](https://github.com/FedeC3N)  
[Ricardo Bruña](https://github.com/rbruna) 