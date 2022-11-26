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

|--> Dissertation<br>
|--> Code<br>
<&emsp>	|--> cdf_loader.py<br>
<&emsp>	|--> cdf_fetcher.py<br>
<&emsp>	|--> data_model.py<br>
<&emsp>	|--> evaluation_tools.py<br>
<&emsp>	|--> model_evaluation.py<br>
<&emsp>	|--> utils.py<br>
<&emsp>	|--> data_files<br>
<&emsp><&emsp>		|--> distance.dat<br>
<&emsp><&emsp>		|--> parker_mag_data.cdf<br>
<&emsp><&emsp>		|--> parker_wind_data.cdf<br>
<&emsp><&emsp>		|--> raw_data<br>
<&emsp><&emsp>		|--> solar_data<br>
<&emsp>	|--> meta<br>
<&emsp><&emsp>		|--> second<br>
<&emsp><&emsp>		|--> third<br>
<&emsp><&emsp>		|--> fourth<br>
<&emsp>	|--> model_files<br>
<&emsp><&emsp>		|--> dbscan.pkl<br>
<&emsp><&emsp>		|--> gaussian_mixture.pkl<br>
<&emsp><&emsp>		|--> hierarchical.pkl<br>
<&emsp><&emsp>		|--> k_means.pkl<br>

---
## Acknowledgements

This project was developed and presented alongside the supervision of the following:

Dr Daniel Verscharen

Dr Andy Smith

All acknowledgements and gratitude are to be awarded to these individuals for their support and contributions.

---
