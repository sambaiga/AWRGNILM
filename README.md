# Adaptive Weighted Recurrence Graph for Appliance Recognition in Non-Intrusive Load Monitoring
This repository is the official implementation of [Adaptive Weighted Recurrence Graph for Appliance Recognition in Non-Intrusive Load Monitoring](). 
<img src="Adaptive-RP.png" width="80%" height="50%">

In this paper, we propose Adaptive weighted Recurrence Graph for Appliance classification in NILM. The AWDRG  treat  recurrence hyper-parameters  as a learn-able parameters like ordinary neural network weights.
This package contains a Python implementation of Adaptive Recurrence Graph for Appliance classification in NILM. 

## Requirements
- python
- numpy
- pandas
- matplotlib
- tqdm
- torch
- sklearn
- seaborn
- nptdms 





## Usage

1. Preprocess the data for a specific dataset. Note: the data directory provided includes preprocessed data for the two datasets LILAC and PLAID.
2. To replicate experiment results you can run the `run_experiments.py` code in the src directory. 
3. The script used to analyse results and produce visualisation presented in this paper can be found in notebook directory
    - Results Analysis notebook provide scripts for results and error analysis.
    - Visualisation paper notebook provide scripts for reproducing most of the figure used in this paper.



