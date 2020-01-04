import random
import torch
import numpy as np
from functools import partial
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, zero_one_loss
from net.model import Conv2DAdaptiveRecurrence, Conv2D
from net.fit import perform_training
from net.fit_functions import get_accuracy, CSVLogger, Checkpoint
from net.init_net import weight_init
from utils.data_generator import get_loaders
from utils.visual_functions import *
from utils.feature_representation  import generate_input_feature


seed = 4783957
print("set seed")
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(seed)




