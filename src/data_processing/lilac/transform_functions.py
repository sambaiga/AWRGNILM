import numpy as np
import math
import scipy.signal as signal
import warnings
warnings.filterwarnings("ignore")


alpha = np.exp(1j*2*math.pi/3)

def isc_transform(s_t):
    F     = np.array([[1, alpha, alpha**2],
                  [1, alpha**2, alpha],
                  [1, 1, 1]
                 ])*2/3
    out = np.dot(F, s_t.T)

    return get_real_value(out.T)

def balance_component(s_postive):
    s_postive = s_postive.reshape(-1,1)
    A = np.array([1, alpha**2, alpha]).reshape(-1,1)
    return get_real_value(A.dot(s_postive.T).T)

def unbalance_component(s_negative, s_zero):
    s_negative= s_negative.reshape(-1,1)
    s_zero= s_zero.reshape(-1,1)
    B = np.array([1, alpha, alpha**2]).reshape(-1,1)
    return get_real_value(B.dot(s_negative.T).T), s_zero


def get_real_value(s_f):
    s=(s_f + np.conjugate(s_f))*0.5
    return s.real

def butter_lowpass(cutoff=50, fs=50e3, order=4, nyq=0.05):
    nyq = nyq * fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, 'low')
    return b, a


def butter_lowpass_filter(data, cutoff=50, fs=50e3, order=4, nyq=0.05):
    b, a = butter_lowpass(cutoff, fs, order=order, nyq=nyq)
    y = signal.filtfilt(b, a, data.flatten())
    return y


def filter_zeros(x, eps=4e-1):
    c = x
    if c.max() <= eps and abs(c.min()) <= eps:
        c[abs(c) <= eps] = 0
    return c


def CT_transform(s_t):
    
    
    """
    CM = np.array([[math.sqrt(2)/math.sqrt(3), -1.0/math.sqrt(6), -1.0/math.sqrt(6)],
                   [0, 1.0/math.sqrt(2), -1.0/math.sqrt(2)],
                   [1/2, 1/2, 1/2]])
    """

    CM = np.array([[1, -1.0/2, -1.0/2],
                   [0, math.sqrt(3)/2, -1.0*math.sqrt(3)/2],
                   [1, 1, 1]])

    CM = CM*2/3

    return np.dot(CM, s_t.T).T
