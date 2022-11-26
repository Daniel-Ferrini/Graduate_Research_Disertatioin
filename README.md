# Graduate_Research_Disertatioin

---
## Overview

This repository contains all the documents and files pertaining to the M.Sc Space Science and Engineering dissertation on the topic of The Identification and Classification of Magnetic Switchbacks Using Statistical and Machine Learning Methods.

The project involved the interfacing and extracting of real data taken from the NASA Parker Solar Probe, the respective code for the extraction and processing of this data can be found in the following files:

```
cdf_loader.py

cdf_fetcher.py

data_model.py
```

Following the data extraction and processing, the statistical and machine learning models were developed and tested, the respective code for the implementation of these models can be found in the following files:

```
utils.py
```

In order to visualise and display the project statistics various plots and methods were developed, the respective code for this data visualisation can be found in the following repositories:


```
evaluation_tools.py

model_evaluation.py
```

---
## Repository Contents

The repository and its file contents can be depicted in the third order breakdown-tree as follows:

|--> Dissertation<\br>
|--> Code
	|--> cdf_loader.py
	|--> cdf_fetcher.py
	|--> data_model.py
	|--> evaluation_tools.py
	|--> model_evaluation.py
	|--> utils.py
	|--> data_files
		|--> distance.dat
		|--> parker_mag_data.cdf
		|--> parker_wind_data.cdf
		|--> raw_data
		|--> solar_data
	|--> meta
		|--> second
		|--> third
		|--> fourth
	|--> model_files
		|--> dbscan.pkl
		|--> gaussian_mixture.pkl
		|--> hierarchical.pkl
		|--> k_means.pkl

---
## Acknowledgements

This project was developed and presented alongside the supervision of the following:

Dr Daniel Verscharen
Dr Andy Smith

All acknowledgements and gratitude are to be awarded to these individuals for their support and contributions.

---
