# sEEGnal

Package for automatic preprocessing of EEG recordings.

This repository comprises three high-level blocks: **standardize**, **bad channel detection**, and **artifact detection**.

---

## Installation  

The files included in the repository allow you to build the sEEGnal package wheel.

1. Download the repository and extract it into a folder  
2. Open a terminal in that folder  
3. Run:

```
python -m pip install setuptools wheel build
python -m build --wheel
python -m pip install ./dist/*.whl
```

---

## Use

An example folder structure and scripts needed for preprocessing a single EEG recording are provided in the `demo` folder.

```
demo
├── data
│   └── sourcedata
│       └── eeg
│           HIQ_001_0_2EC_2024-07-16_13-19-47.eeg
│           HIQ_001_0_2EC_2024-07-16_13-19-47.vhdr
│           HIQ_001_0_2EC_2024-07-16_13-19-47.vmrk
│
└── scripts
    config.json
    init.py
    run_sEEGnal.py
```

### Files

- `run_sEEGnal.py`: script for preprocessing any number of files automatically  
- `config.json`: mandatory configuration file (frequencies, channels, paths, etc.)  
- `init.py`: helper script to locate files and define parameters  

### Steps to run a first test

1. Copy EEG files in BrainVision format (`.eeg`, `.vhdr`, `.vmrk`)  
2. Define parameters in `config.json`  
3. Define paths and file patterns in `init.py`  
4. Run `run_sEEGnal.py` from the top folder (`demo`)  

---

## Standardize

Converts the original data structure into a BIDS-compliant dataset.

For more information, see the [BIDS specification](https://bids-specification.readthedocs.io/en/stable/index.html).

### Usage

```python
from sEEGnal.standardize.standardize import standardize

results = standardize(config, current_file, BIDS)
```

### Outputs

In addition to returning a results dictionary, the module creates BIDS-compliant files:

- `*_eeg.*` — EEG recording in BIDS format  
- `*_channels.tsv` — updated channel information (including impedance)  
- `*_channels.json` — channel metadata  
- `*_eeg.json` — recording metadata  
- `*_coordsystem.json` / `*_electrodes.tsv` — electrode positions (if available)  

---

## Bad Channel Detection

Automated detection of bad EEG channels using complementary criteria:

1. Impossible amplitude detection  
2. Component-based anomaly detection (SOBI)  
3. Gel bridge detection (correlation + spatial proximity)  
4. High deviation detection (robust statistics)  

### Usage

```python
from sEEGnal.preprocess.badchannel_detection import badchannel_detection

results = badchannel_detection(config, BIDS)
```

### Outputs

In addition to returning a results dictionary, the module creates BIDS derivative files:

- `*_channels.tsv`  
  - Channel classification (good/bad)  
  - Reason(s) for rejection  
  - Impedance information (if available)  

- `*_desc-sobi-badchannels_*`  
  - Mixing/unmixing matrices  
  - Component classification  
  - Metadata for reproducibility  

---

## Artifact Detection

Automated detection of EEG artifacts using an iterative SOBI-based workflow.

### Artifact types

- **EOG**: low-frequency frontal activity with large amplitude peaks  
- **Muscle**: high-frequency bursts (110–145 Hz) from muscle components  
- **Sensor (jumps)**: abrupt signal discontinuities  
- **Other**: segments with implausible amplitudes  

### Workflow

1. Estimate SOBI components (`desc-sobi_artifacts`)  
2. Detect muscle, sensor, and other artifacts  
3. Save preliminary annotations  
4. Re-estimate SOBI excluding detected artifacts (`desc-sobi`)  
5. Detect muscle, sensor, EOG, and other artifacts  
6. Merge and save final annotations  

### Usage

```python
from sEEGnal.preprocess.artifact_detection import artifact_detection

results = artifact_detection(config, BIDS)
```

### Outputs

In addition to returning a results dictionary, the module creates:

- `*_artifacts_annotations`  
  - Onset (seconds)  
  - Duration (seconds)  
  - Label (e.g., `bad_muscle`, `bad_EOG`, `bad_jump`, `bad_other`)  

- `*_desc-sobi_artifacts_*`  
  - Initial SOBI decomposition  
  - Mixing/unmixing matrices  
  - Component classification  

- `*_desc-sobi_*`  
  - Final SOBI decomposition  
  - Updated component classification  
  - Metadata for reproducibility  

---

## Contributors

[Federico Ramírez Toraño](https://github.com/FedeC3N)  
[Ricardo Bruña](https://github.com/rbruna)  
