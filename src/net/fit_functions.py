import torch
import torch.nn as nn
import numpy as np
import os
import csv


def get_accuracy(y_pred:torch.Tensor, y_true:torch.Tensor, softmax:bool=True):
    "Compute multi class accuracy when `y_pred` and `y_true` are the same size."
    if softmax: y_pred = torch.softmax(y_pred, 1)
    _, y_pred = torch.max(y_pred.data, 1)
    return (y_pred == y_true.data).float()


class Checkpoint(object):
    
    def __init__(self,saved_model_path: str=None, patience: int =20, checkpoint: bool = False, score_mode: str="max",
                 min_delta: float=1e-4, save_final_model: bool = False):

        
        self.saved_model_path = saved_model_path
        self.checkpoint = checkpoint
        self.save_final_model = save_final_model
        self.patience = patience
        self.min_delta = min_delta
        self.score_mode = score_mode
        self.num_bad_epochs = 0
        self.is_better = None
        self.best = None
        self._init_is_better(score_mode, min_delta)
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

        

    def _init_is_better(self, score_mode, min_delta):
        if score_mode not in {'min', 'max'}:
            raise ValueError('mode ' + score_mode + ' is unknown!')

        elif score_mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta

        elif score_mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

    def early_stopping(self, metric, states):

        if self.best is None:
            self.best = metric

        if self.is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
            states['best_score'] = self.best

            if self.checkpoint:
                self.save_checkpoint(states)

        else:
            self.num_bad_epochs += 1

        if (self.num_bad_epochs >= self.patience) or np.isnan(metric):
            terminate_flag = True

        else:
            terminate_flag = False

        return terminate_flag

    def save_checkpoint(self, state):
        """
        Save best models
        arg:
           state: model states
           is_best: boolen flag to indicate whether the model is the best model or not
           saved_model_path: path to save the best model.
        """
        print("save best model")
        torch.save(state['state_dict'], self.saved_model_path)

        #torch.save(state, self.saved_model_path)

    def load_saved_model(self, model):
        saved_model_path = self.saved_model_path

        if os.path.isfile(saved_model_path):
            model.load_state_dict(torch.load(saved_model_path))
        else:
            print("=> no checkpoint found at '{}'".format(saved_model_path))
            
        return model


class CSVLogger():
    def __init__(self, filename, fieldnames=['epoch']):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        """
        writer = csv.writer(self.csv_file)
        
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])
        """

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

