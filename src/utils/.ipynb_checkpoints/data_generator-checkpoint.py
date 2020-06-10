import torch
import numpy as np


def get_correct_labels_lilac(labels):
    correct_1_phase_motor = [920,923,956, 959, 961, 962, 1188]
    correct_hair = [922, 921, 957, 958,  960, 963, 1181, 1314]
    correct_bulb = [1316]
    
    correct_labels = []
    for idx, l in enumerate(labels):
        if idx in correct_1_phase_motor:
            correct_labels.append('1-phase-async-motor')
        elif idx in correct_hair:
            correct_labels.append('Hair-dryer')
        elif idx in correct_bulb:
            correct_labels.append('Bulb')
        else:
            correct_labels.append(l)
    correct_labels = np.hstack(correct_labels)
    return correct_labels


def get_data(submetered=True, data_type="lilac", isc=False):
    

    if submetered:
        path_sub = f"../data/{data_type}/submetered/"
        print(f"Load {data_type} submetered data from {path_sub}")
        current = np.load(path_sub+"current.npy")
        voltage = np.load(path_sub+"voltage.npy")
        label = np.load(path_sub+"labels.npy")
       
    else:
        path_sub = f"../data/{data_type}/aggregated_isc/" if isc else f"../data/{data_type}/aggregated/"
        print(f"Load {data_type} aggregated data from {path_sub}")
        current = np.load(path_sub+"current.npy")
        voltage = np.load(path_sub+"voltage.npy")
        label = np.load(path_sub+"labels.npy")

    
    return current, voltage, label


class Dataset(torch.utils.data.Dataset):
    

    def __init__(self, feature,  label, width=50):
       
        self.feature   = feature
        self.label    = label
        

        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
       
        feature = self.feature[index]
        label =  self.label[index]
        
        return feature, label
        
        
def get_loaders(input_tra, input_val, label_tra, label_val,
                batch_size=64):
   
    tra_data = Dataset(input_tra, label_tra)
    val_data = Dataset(input_val, label_val)
    
    tra_loader=torch.utils.data.DataLoader(tra_data, batch_size, shuffle=True, num_workers=4,drop_last=False)
    val_loader=torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    loaders = {'train':tra_loader, 'val':val_loader}
    
    return loaders  
