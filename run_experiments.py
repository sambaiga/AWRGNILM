import sys
sys.path.append("../src/")
from src.experiment_one import  embending_size_parameter_experiments, parameters_initilisation_experiments
from src.experiment_two import  comparison_with_baseline_experiments
from datetime import date

def start_logging(filename):
    f = open('logs/experiment-{}.txt'.format(filename), 'w')
    sys.stdout = f
    print(f"Starting {filename} experiment")
    return f

def stop_logging(f):
    f.close()
    sys.stdout = sys.__stdout__

#DEFINE EXPERIMENTAL VARIABLES 

datasets = ["lilac", "plaid"]
feature = "adaptive"
n_epochs  =  100
batch_size = 16
width=50
#RUN EXPERIMENT ONE
experiment_name = 'ADRP-NILM-experiment_one_param_initilization:{}'.format(date.today().strftime('%m-%d-%H-%M'))
f = start_logging(experiment_name)
for dataset in datasets:
    parameters_initilisation_experiments(dataset=dataset,
                                image_type = "adaptive",
                                width  = width,
                                epochs = n_epochs,
                                batch_size = batch_size)
stop_logging(f)
exit()