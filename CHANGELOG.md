# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.3.0
Artifacts are correctly removed before second SOBI in artifacts detection  
Now you can apply new parameters to a previous loaded eeg pasing the raw as 
input to prepare_eeg()  
Artifact SOBI is estimated in epochs  
We are using median and MAD to detect bad channels (high deviation and power 
spectrum criteria)  
We are using median and MAD to detect artifacts (EOG, sensor)  
Muscle detection is based on the absolute value of each component.


## v1.2.0
Remove impedances as bad channel criterion  
prepare_raw is now prepare_eeg()  
You can apply SOBI with prepare_eeg  
New order for the steps loading a EEG.

## v1.1.1
Use YAPF for code formatting
Use variance instead of Welch to identify power spectrum bad channels.
prepare_raw() is now prepare_eeg()
Bad channels are dropped inside prepare_eeg()
Unnecessary demean step when identifying bad channels and artifacts

## v1.1.0 - 30/10/2025

Now it can be used in other computers and Python versions.

When looking for bad channels with any criteria, the channels previously marked as "bad" are removed.

New functionallity in prepare_raw(): re-reference using the average or the median.

When looking for artifacts, the channels marked as "bad" are interpolated.

In the standardization process, only the included channels in config are copied. If the EEG recording has 128 channels and I define 10, the standardize EEG will have 10 channels.



## v1.0.0 - 30/10/2025

First release of sEEGnal.  

It fully runs but with many constraints.


