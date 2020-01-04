from nptdms import TdmsFile
import numpy as np

def read_tdms(file_name):
    tdms_file = TdmsFile(file_name)
   
    readings={}
    for i in range(1,4):
        readings['V{}'.format(i)]=tdms_file.object('data', 'U{}'.format(i)).data
        readings['I{}'.format(i)]=tdms_file.object('data', 'I{}'.format(i)).data
    return readings




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
    if diff.std() / diff.mean() > 0.1:
        raise AssertionError ("false zero-crossings found")
    
    return times



def get_zero_crossing(voltage):
    '''[summary]
    the zero-crossing is the instantaneous point at which there is no voltage present.
    In a sine wave or other simple waveform, this normally occurs twice during each cycle. 
    It is a device for detecting the point where the voltage crosses zero in either direction

    Arguments:
        voltage {[type]} -- [description]

    Returns:
        [type] -- [description]
    '''

    zero_crossing = zero_crossings(y_axis=voltage)
    if len(zero_crossing)>0:
        if voltage[zero_crossing[0]+1] > 0:
            zero_crossing = zero_crossing[0:]
        else:
            zero_crossing = zero_crossing[1:]
        if len(zero_crossing) % 2 == 1:
            zero_crossing = zero_crossing[:-1]

        if zero_crossing[-1] + 500 >= len(voltage):
            zero_crossing = zero_crossing[:-2]
    else:
        zero_crossing = None
        
    return zero_crossing



def get_iv_trajectory(v_a, v_b, i_a, i_b, event=1):
    '''[summary]

    Arguments:
        v_a {[type]} -- [description]
        v_b {[type]} -- [description]
        i_a {[type]} -- [description]
        i_b {[type]} -- [description]

    Keyword Arguments:
        event {int} -- [description] (default: {1})

    Returns:
        [type] -- [description]
    '''

    if event == 1:
        zc = get_zero_crossing(v_a)
    else:
        zc = get_zero_crossing(v_b)

    if zc is not None:

        before_event = np.concatenate([i_b[zc[0]:zc[0]+500], i_b[zc[1]:zc[1]+500]])
        after_event = np.concatenate(
            [i_a[zc[-2]:zc[-2]+500], i_a[zc[-1]:zc[-1]+500]])

        if event == 1:
            diff = after_event - before_event
            vtemp = np.concatenate(
                [v_a[zc[-2]:zc[-2]+500], v_a[zc[-1]:zc[-1]+500]])
        else:

            diff = before_event - after_event
            vtemp = np.concatenate(
                [v_b[zc[-2]:zc[-2]+500], v_b[zc[-1]:zc[-1]+500]])
    else:
        raise AssertionError("No zero crossing")


    return vtemp, diff

