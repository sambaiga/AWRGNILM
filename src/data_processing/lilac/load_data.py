import numpy as np
import pandas as pd
import os
from processing_functions import *
from visualizer_functions import *
from transform_functions import *

import random

def seed(seed=4783957):

    print("set seed")
    np.random.seed(seed)
    random.seed(seed)
    
    
def filter_zeros(x, normal=True, eps=6e-2):
    c = x
    if len(c)>0:
        if c.max() <= eps and abs(c.min()) <= eps:
            c[abs(c) <= eps] = 0
    return c
    
def get_isc_components(v, i):
    
    i_f = isc_transform(i)
    v_f = isc_transform(v)

    return v_f, i_f

def get_balanced_components(v, i):
    i_f = isc_transform(i)
    v_f = isc_transform(v)

    i_b = balance_component(i_f[:,0])
    v_b = balance_component(v_f[:,0])

    return v_b, i_b

def get_ubalanced_components(v, i):
    i_f = isc_transform(i)
    v_f = isc_transform(v)

    i_ub, _ = unbalance_component(i_f[:,1], i_f[:,2])
    v_ub, _ = unbalance_component(v_f[:,1], v_f[:,2])

    return v_ub, i_ub

def get_zero_components(v, i):
    i_f = isc_transform(i)
    v_f = isc_transform(v)

    _, i_zeo  = unbalance_component(i_f[:,1], i_f[:,2])
    _, v_zero = unbalance_component(v_f[:,1], v_f[:,0])

    return v_zero, i_zeo
def zero_crossings(y_axis, x_axis = None):
    """
    Algorithm to find zero crossings. Smoothens the curve and finds the
    zero-crossings by looking for a sign change.
    
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find zero-crossings
    x_axis -- A x-axis whose values correspond to the 'y_axis' list and is used
        in the return to specify the postion of the zero-crossings. If omitted
        then the indice of the y_axis is used. (default: None)
    return -- the x_axis value or the indice for each zero-crossing
    """
   
    length = len(y_axis)
    if x_axis == None:
        x_axis = range(length)
    
    x_axis = np.asarray(x_axis)
    
    zero_crossings = np.where(np.diff(np.sign(y_axis)))[0]
    times = [x_axis[indice] for indice in zero_crossings]
    
    #check if zero-crossings are valid
    diff = np.diff(times)
    #if diff.std() / diff.mean() > 0.1:
        #raise AssertionError ("false zero-crossings found")
    
    return times




def get_zero_crossing(voltage, NN=1000):
  
    zero_crossing = zero_crossings(y_axis=voltage)
    if len(zero_crossing)>0:
        if voltage[zero_crossing[0]+1] > 0:
            zero_crossing = zero_crossing[0:]
        else:
            zero_crossing = zero_crossing[1:]
        if len(zero_crossing) % 2 == 1:
            zero_crossing = zero_crossing[:-1]

        if zero_crossing[-1] + NN >= len(voltage):
            zero_crossing = zero_crossing[:-2]
    else:
        zero_crossing = None
        
    return zero_crossing

def VI_transform(current, voltage, on_event):
    

    zc = get_zero_crossing(voltage)

    before_event = np.concatenate([current[zc[0]:zc[0]+500],current[zc[1]:zc[1]+500]])
    after_event = np.concatenate([current[zc[-2]:zc[-2]+500],current[zc[-1]:zc[-1]+500]])

    if on_event:
        c = after_event - before_event
        v = np.concatenate([voltage[zc[-2]:zc[-2]+500],voltage[zc[-1]:zc[-1]+500]])
        
    else:
        c = before_event - after_event
        v = np.concatenate([voltage[zc[0]:zc[0]+500],voltage[zc[1]:zc[1]+500]])
        
    return c, v
        
  
def select_zc(voltage):
    zero_crossing = np.where(np.diff(np.sign(voltage)))[0]
    
    if voltage[zero_crossing[0]+1] > 0:
        zero_crossing = zero_crossing[0:]
    else:
        zero_crossing = zero_crossing[1:]
        
    if len(zero_crossing) % 2 == 1:
        zero_crossing = zero_crossing[:-1]
        
    if zero_crossing[-1] + 1000 >= len(voltage):
        zero_crossing = zero_crossing[:-2]
        
    return zero_crossing


def transform(current, voltage, on_event):
    

    zc = select_zc(voltage)

    before_event = np.concatenate([current[zc[0]:zc[0]+500],current[zc[1]:zc[1]+500]])
    after_event = np.concatenate([current[zc[-2]:zc[-2]+500],current[zc[-1]:zc[-1]+500]])

    if on_event:
        c = after_event - before_event
        v = np.concatenate([voltage[zc[-2]:zc[-2]+500],voltage[zc[-1]:zc[-1]+500]])
        
    else:
        c = before_event - after_event
        v = np.concatenate([voltage[zc[0]:zc[0]+500],voltage[zc[1]:zc[1]+500]])
        
    return c, v


def get_path(path="/home/ibcn079/data/LILAC/", measurement=3, test=False):

    if measurement == 1:
        data_folder = "test_data_single" if test else "Single/"
    elif measurement == 2:
        data_folder = "test_data_double" if test else "Double/"
    elif measurement == 3:
        data_folder = "test_data_triple" if test else"Triple/"

    path = path+data_folder
    return path





def get_voltage_current(data):
    on_event = data[data["on-events"] == 1].index.values
    off_event = data[data["on-events"] == -1].index.values
    assert (on_event[1]-on_event[0]) == (off_event[1]-off_event[0])

    df = data[["I1", "I2", "I3", "V1", "V2", "V3"]]
    voltages = ["V1", "V2", "V3"]
    currents = ["I1", "I2", "I3"]
    i = df[currents].values
    v = df[voltages].values

    return v, i, on_event, off_event




def steady_state_value(data, on_pattern, appliances, ilim=10, vlim=350, vis=True, num_cycles=2):
    
    
    
    plt.figure(figsize=(18, 8))
    for k in range(1, 4):
        plt.subplot(1,3, k)
        I = data['I{}'.format(k)].values
        V = data['V{}'.format(k)].values
        
        #pq=calculatePower(I, V, NN=1000)
        #event_visualizer(pq[:,0], appliances, on_pattern, phase=k)
        #plt.tight_layout()
    #plt.show()
    on_event = data[data["on-events"] == 1].index.values
    off_event = data[data["on-events"] == -1].index.values
    assert (on_event[1]-on_event[0]) == (off_event[1]-off_event[0])

    data = data[["I1", "I2", "I3", "V1", "V2", "V3"]]
    voltages = ["V1", "V2", "V3"]
    currents = ["I1", "I2", "I3"]
    i_b = data[currents].values
    v_b = data[voltages].values

    
    vb, ib=get_balanced_components(v_b, i_b)
    
    
    
    
    periods = int(50000//50)
    window_size = int(periods*num_cycles)
    
    
    idx=1
    for j, event_id in enumerate(on_event):
        current = []
        voltages = []
        for k in range(1,4):
            c=i_b[:,k-1]
            v=v_b[:,k-1]
    
            v_=v[event_id-window_size:event_id+window_size]
            i_=c[event_id-window_size:event_id+window_size]
            i_, v_=transform(i_, v_, 1)
            current.append(i_)
            voltages.append(v_)
            plt.subplot(1,3,idx)
            plt.plot(i_)
            plt.title(f"phase:{k} {appliances[j]}",  fontsize=6)
            plt.tight_layout()
        plt.show()    
        ct = CT_transform(current)
        for k in range(1,4):
            c=ct[:,k-1]
            v=voltages[k-1]
            plt.subplot(1,3,idx)
            plt.plot(c)
            plt.title(f"phase:{k} {appliances[j]}",  fontsize=6)
            plt.tight_layout()
        plt.show() 
        idx+=1
           
    
   
   
    idx=1
    for j, event_id in enumerate(on_event):
    
        for k in range(1,4):
            c=ib[:,k-1]
            v=vb[:,k-1]
        
            v_=v[event_id-window_size:event_id+window_size]
            i_=c[event_id-window_size:event_id+window_size]
            i_, v_=transform(i_, v_, 1)
            plt.subplot(1,3,idx)
            plt.plot(v_)
            plt.title(f"phase:{k} {appliances[j]}",  fontsize=6)
            plt.tight_layout()
        idx+=1
        
    plt.show()
    """
   
    for j, event_id in enumerate(on_event):
        v_=v[event_id-window_size:event_id+window_size]
        i_=c[event_id-window_size:event_id+window_size]
        i_, v_=transform(i_, v_, 1)
        plt.subplot(1,3,j+1)
        plt.plot(v_,i_)
        #plt.title(appliances[j])
        plt.tight_layout()
    plt.show()
    
    for j, event_id in enumerate(off_event):
        v_=v[event_id-window_size:event_id+window_size]
        i_=c[event_id-window_size:event_id+window_size]
        i_, v_=transform(i_, v_, 0)
        plt.subplot(1,3,j+1)
        plt.plot(v_,i_)
        #plt.title(appliances[j])
        plt.tight_layout()
    plt.show()
    """
    
    


if __name__ == "__main__":
    path = get_path(path="/home/ibcn079/data/LILAC/")
    print(path)
    measurement = 3
    for root, k, fnames in sorted(os.walk(path)):
        folder_id = root.strip().split("/")[-1]
        

        if folder_id:
            if int(folder_id)>1:
                break
            for j, fname in enumerate(sorted(fnames)):
                
                df, appliances,  on_pattern = get_data(
                    root, fname, measurement)
                v, i, on_event, off_event = get_voltage_current(df)
                #v,i=get_balanced_components(v, i)
                steady_state_value(df, on_pattern, appliances, ilim=10, vlim=350, vis=True, num_cycles=2)
                print(len(on_event))
                
                
                    
