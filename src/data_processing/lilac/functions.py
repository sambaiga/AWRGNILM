import numpy as np 
from load_data import VI_transform, transform
from transform_functions import *
import pandas as pd
import os 
import shutil
import matplotlib.pyplot as plt

appliance_names = {'1-phase-async-motor':1, '3-phase-async-motor':2, 'Bulb':3, 
                   'Coffee-machine':4, 'Drilling-machine':5,'Dumper-machine':6, 
                   'Freq-conv-squirrel-3-2x':7 ,'Kettle':8, 'Fluorescent-lamp':9, 
                   'Raclette':10, 'Refrigerator':11,'Resistor':12, 'Squirrel-3-async':13,
                  'Squirrel-3-async-2x':14 ,'Vacuum-cleaner':15, 'Hair-dryer':16}

def appliances_check(appliances):
    for idx, ls in enumerate(appliances):
        if ls in ["Halogen", "Krypton"]:
            appliances[idx]="Bulb"
        if ls=="RC106":
             appliances[idx]="Raclette"
        
        if ls =="3-phase-asyn-motor":
            appliances[idx] = "3-phase-async-motor"
             
        if ls in ["Nistra", "1-phase-asyn-motor"]:
            appliances[idx]="1-phase-async-motor"
            
        if ls in ['Vacuum-cleaner', 'Vacuum-clearner']:
            appliances[idx]="Vacuum-cleaner"
            
        if ls in ["Lumilux", 'Lumilux']:
            appliances[idx]="Fluorescent-lamp"
    
    #appliances = [appliance_names[i] for i in temp] 
    return appliances

def get_transformed_three_phase_VI(i_st, v_st, event_id):
    
    Ie = []
    Ve = []
    for k in range(0,3):
        if k==2:
            ce,ve = transform(get_real_value(i_st[:,k]), get_real_value(v_st[:,0]), event_id)
            #im_0=(max(ce)- min(ce))/2
            im_0 = np.mean(ce**2)**0.5
        else:
            ce,ve =transform(get_real_value(i_st[:,k]), get_real_value(v_st[:,k]), event_id)
            if k==0:
                #im_1=(max(ce)- min(ce))/2
                im_1=np.mean(ce**2)**0.5
            else:
                #im_2 =(max(ce)- min(ce))/2
                im_2= np.mean(ce**2)**0.5
        Ie.append(ce)
        Ve.append(ve)
    imb_0 = round(im_0/im_1, 2)
    imb_neg = round(im_2/im_1, 2)
    return Ie, Ve, [imb_0, imb_neg]


def get_three_phase_VI(i_st, v_st, event_id):
    
    Ie = []
    Ve = []
    for k in range(0,3):
        ce,ve = transform(i_st[:,k], v_st[:,k], event_id)
        Ie.append(ce)
        Ve.append(ve)
    
    return Ie, Ve

def get_max_current_from_list(I):
    i_max =[]
    for k in range(len(I)):
        i_max.append(I[k].max())
        
    return i_max

def check_phase(I):
    Irms = np.round(get_max_current_from_list(I),1)
    phase_ratio=np.rint(Irms/max(Irms)).astype(np.int)
    phase_ratio_idx = np.where(phase_ratio==1)[0]
    return phase_ratio_idx

def get_transform(c, v, event_id, transform_type):
    if transform_type == "ct":
        # print(transform_type)
        i_ct = CT_transform(c)
        v_ct = CT_transform(v)

        Ie, Ve, imb = get_transformed_three_phase_VI(
            i_ct, v_ct, event_id)

    elif transform_type == "isc":
        # print(transform_type)
        i_st = isc_transform(c)
        v_st = isc_transform(v)
        Ie, Ve, imb = get_transformed_three_phase_VI(
            i_st, v_st, event_id)
    return Ie, Ve, imb
    
def get_adaptive_transform(c, v, event_id, transform_type):
    
    I, V = get_three_phase_VI(c, v, event_id)
    phase_ratio_id = check_phase(I)
    if len(phase_ratio_id)>=2:
        Ie, Ve, _ = get_transform(c,v, event_id, transform_type)
        """
        phase_ratio_id = check_phase(Ie).tolist()
        
        for i in range(0,3):
            if i in phase_ratio_id:
                continue
            else:
                Ie[i]=np.zeros_like(Ie[0])
        """
       
    else:
        """
        for i in range(0,3):
            if i in phase_ratio_id.tolist():
               continue
            else:
                I[i]=np.zeros_like(I[0])
        """
        Ve = V
        Ie = I
    return Ie, Ve
       
        
    

def get_three_phase_power(df):
    power = {}
    for k in range(1, 4):
    
        I = df['I{}'.format(k)].values
        V = df['V{}'.format(k)].values
        pq=calculatePower(I, V, NN=1000)
        power[f'P{k}']=pq[:,0]
        
    power=pd.DataFrame.from_dict(power)

    return power


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles  


def get_Irms_from_list(C):
    Irms = []
    for i in range(len(C)):
        Irms.append(np.mean(C[i]**2)**0.5)
    return Irms
def crete_folder(data_filter,  trans, transform_type,  exp_id="01", result_path='results/'):
    
    filter_var="with_filter" if data_filter else "without_filter"
    if trans is None:
        trans_var = "combined"
    elif trans==0:
        trans_var = "transform_after"
    elif trans==1:
        trans_var = "transform_before"
    elif trans==2:
        trans_var = "adaptive_transform_before"
        
    file_name = trans_var+"_"+filter_var+"_"+transform_type+"_"+exp_id if transform_type else trans_var+"_"+filter_var+"_"+exp_id
    save_path=os.path.join(result_path,file_name)
           
    if  os.path.exists(save_path):
        shutil.rmtree(save_path)
        print("Delete {}".format(save_path))
        os.makedirs(save_path)
        print("Create {}".format(save_path))
    else:
        os.makedirs(save_path)
        print("Create {}".format(save_path))    
    
    return save_path


