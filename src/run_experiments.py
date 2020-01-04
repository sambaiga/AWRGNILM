import sys
from experiment_one import  embending_size_parameter_experiments, parameters_initilisation_experiments
from experiment_two import  comparison_with_baseline_experiments
from datetime import date

def start_logging(filename):
    f = open('../logs/experiment-{}.txt'.format(filename), 'w')
    sys.stdout = f
    return f

def stop_logging(f):
    f.close()
    sys.stdout = sys.__stdout__

#DEFINE EXPERIMENTAL VARIABLES for EXP ONE 

datasets = ["plaid", "lilac"]
feature = "adaptive"
n_epochs  =  10
batch_size = 16
width=50
"""
#RUN EXPERIMENT ONE
experiment_name = 'ADRP-NILM-experiment_one_param_initilization:{}'.format(date.today().strftime('%m-%d-%H-%M'))
f = start_logging(experiment_name)
print(f"Starting {experiment_name} experiment")

for dataset in datasets:
    parameters_initilisation_experiments(dataset=dataset,
                                image_type = feature ,
                                width  = width,
                                epochs = n_epochs,
                                batch_size = batch_size)
stop_logging(f)
exit()
"""


#RUN EXPERIMENT ONE_EMBENDING
experiment_name = 'ADRP-NILM-experiment_one_emb_size_parameter:{}'.format(date.today().strftime('%m-%d-%H-%M'))
f = start_logging(experiment_name)
print(f"Starting {experiment_name} experiment")

for dataset in datasets:
    embending_size_parameter_experiments(dataset=dataset,
                                image_type = feature,
                                epochs = n_epochs,
                                batch_size = batch_size)
stop_logging(f)
exit()


#DEFINE EXPERIMENTAL VARIABLES for EXP ONE 
datasets = ["lilac", "plaid"]
features = ["vi","adaptive"]
n_epochs  =  300
batch_size = 16
width=50

#RUN EXPERIMENT TWO
experiment_name = 'ADRP-NILM-experiment_two_comparison_with_baseline:{}'.format(date.today().strftime('%m-%d-%H-%M'))
f = start_logging(experiment_name)
print(f"Starting {experiment_name} experiment")
for dataset in datasets:
    for feature in features:
        comparison_with_baseline_experiments(dataset=dataset,
                                    image_type = feature ,
                                    width  = width,
                                    epochs = n_epochs,
                                    batch_size = batch_size)
stop_logging(f)
exit()