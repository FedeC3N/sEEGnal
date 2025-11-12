# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## v1.1.0 - 30/10/2025

Now it can be used in other computers and Python versions.

When looking for bad channels with any criteria, the channels previously marked as "bad" are removed.

New functionallity in prepare_raw(): re-reference using the average or the median.

When looking for artifacts, the channels marked as "bad" are interpolated.

In the standardization process, only the included channels in config are copied. If the EEG recording has 128 channels and I define 10, the standardize EEG will have 10 channels.



## v1.0.0 - 30/10/2025

First release of sEEGnal.  

It fully runs but with many constraints.


