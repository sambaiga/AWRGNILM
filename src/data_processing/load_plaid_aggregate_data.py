import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 20})



def read_events_labels(file1, file2, data_path='/home/ibcn079/data/Data/'):
    a = pd.read_csv(data_path+file1,header=None)
    a = a.values

    b = pd.read_csv(data_path+file2,header=None)
    b = b.values

    events = {}
    labels = {}
    for index in range(len(a)):
        line = a[index]
        ev = [int(i) for i in line[0].strip().split(" ")]
        events[index] = ev

        line = b[index]
        ev = [int(i) for i in line[0].strip().split(" ")]
        labels[index] = ev
    return events, labels


def select_data_appliance_type(folder, appliance_id, data_path):
    file1 = folder + '_events'
    file2 = folder + '_labels'
    
    events, labels = read_events_labels(file1, file2, data_path)
    data_appliance_id = []
    files_appliance_id = []
    for index, e, l in zip(events.keys(), events.values(), labels.values()):
        if appliance_id not in l:
            continue
            
        ids_appl = np.where(np.array(l) == appliance_id)[0]
        events_appliance_id = np.array(e)[ids_appl]
        
        # Read the file.
        f = data_path+folder+'/'+str(index+1)
        a = pd.read_csv(f,names=['current','voltage'])
        
        data_appliance_id += [a[i-1000:i+1000].values for i in events_appliance_id]
        files_appliance_id += [index+1 for i in events_appliance_id]
        
    return data_appliance_id, files_appliance_id

def select_zc(voltage):
    zero_crossing = np.where(np.diff(np.sign(voltage)))[0]
    
    if voltage[zero_crossing[0]+1] > 0:
        zero_crossing = zero_crossing[0:]
    else:
        zero_crossing = zero_crossing[1:]
        
    if len(zero_crossing) % 2 == 1:
        zero_crossing = zero_crossing[:-1]
        
    if zero_crossing[-1] + 250 >= len(voltage):
        zero_crossing = zero_crossing[:-2]
        
    return zero_crossing


def transform(data, on_event):
    c = np.empty((0,500))
    v = np.empty((0,500))
    
    for i, j in zip(data, on_event):
        current = i[:,0]
        voltage = i[:,1]

        zc = select_zc(voltage)

        before_event = np.concatenate([current[zc[0]:zc[0]+250],current[zc[1]:zc[1]+250]])
        after_event = np.concatenate([current[zc[-2]:zc[-2]+250],current[zc[-1]:zc[-1]+250]])

        if j:
            diff = after_event - before_event
            vtemp = np.concatenate([voltage[zc[-2]:zc[-2]+250],voltage[zc[-1]:zc[-1]+250]])
            
        else:
            diff = [before_event - after_event]
            vtemp = np.concatenate([voltage[zc[0]:zc[0]+250],voltage[zc[1]:zc[1]+250]])
            
        v = np.vstack((v,vtemp))
        c = np.vstack((c,diff))
            
    return c, v

def get_agggregate_data(folder, data_path='/home/ibcn079/data/Data/'):
    
    data = []
    labels = []
    on_event = []
    print("Load data")
    with tqdm(total=14) as pbar:
        for i in range(1,14):
            t, f = select_data_appliance_type(folder, i, data_path)
            data += t
            labels += [i] * len(t)
            if i != 9:
                on_event += [1, 0] * int(len(t) / 2)
            else:
                on_event += [1, 1, 0] * int(len(t) / 3)
            pbar.set_description('processed: %d' % (1 + i))
            pbar.update(1)
        pbar.close()
                
    current, voltage = transform(data, on_event)
    print(f"currents size:{len(current)}")  
    print(f"labels size:{len(labels)}")
    print(f"voltage:{len(voltage)}")  
    assert len(current)==len(voltage)==len(labels)
        
    return np.array(current), np.array(voltage), np.array(labels), np.array(on_event)
    
    
    


def select_submetered_appliance_data(folder, appliance_id, data_path):
    file1 = folder + '_events'
    file2 = folder + '_labels'
    
    events, labels = read_events_labels(file1, file2, data_path)
    data_appliance_id = []
    files_appliance_id = []
    for index, e, l in zip(events.keys(), events.values(), labels.values()):
        if appliance_id not in l:
            continue
            
        ids_appl = np.where(np.array(l) == appliance_id)[0]
        events_appliance_id = np.array(e)[ids_appl]
        
        # Read the file.
        f =  data_path+folder+'/'+str(index+1)
        a = pd.read_csv(f,names=['current','voltage'])
        
        data_appliance_id += [a[i-1000:i+1000].values for i in events_appliance_id]
        files_appliance_id += [index+1 for i in events_appliance_id]
        
    return data_appliance_id, files_appliance_id
    
def get_submetered_data(data_path='/home/ibcn079/data/Data/'):
    data = []
    labels = []
    on_event = []
    progress_bar = tqdm(range(1,14))
    for i in progress_bar:
        progress_bar.set_description('appliance ' + str(i))
        t, f = select_submetered_appliance_data('FINALSUBMETERED', i, data_path)
        data += t
        labels += [i] * len(t)
        if i != 9:
            on_event += [1, 0] * int(len(t) / 2)
        else:
            on_event += [1, 1, 0] * int(len(t) / 3)
            
        progress_bar.set_description('processed: %d' % (1 + i))
            
    current, voltage = transform(data, on_event)
    print(f"currents size:{len(current)}")  
    print(f"labels size:{len(labels)}")
    print(f"states:{len(on_event)}") 
    print(f"voltage:{len(voltage)}")  
    assert len(current)==len(voltage)==len(labels)
    
    return np.array(current), np.array(voltage), np.array(labels), np.array(on_event)

def get_data(submetered=False):

    names = ['CFL','ILB','Waterkettle','Fan','AC','HairIron','LaptopCharger','SolderingIron','Fridge','Vacuum','CoffeeMaker','FridgeDefroster']
    if submetered:
        current, voltage, label,on_events = get_submetered_data(data_path="../../data/Data/")
    else:
        current, voltage, label,on_events = get_agggregate_data('FINALAGGREGATED', "../../data/Data/")
    
    return current, voltage, label, on_events

if __name__ == "__main__":
    submetered = False
    save_path="../data/plaid/submetered/" if submetered else "../data/plaid/aggregated/"
    current, voltage, labels, on_events = get_data(submetered)
    np.save(save_path+"current.npy", current)
    np.save(save_path+"voltage.npy", voltage)
    np.save(save_path+"labels.npy", labels)
    np.save(save_path+"on_events.npy", on_events)

