import torch
import numpy as np
from tqdm import tqdm


def center(X,w):
    minX = np.amin(X)
    maxX = np.amax(X)
    dist = max(abs(minX),maxX)
    X[X<-dist] = -dist
    X[X>dist] = dist
    d = (maxX-minX)/(w-1)
    return (X,d)

def get_distance_measure(x:torch.Tensor, p:int=1):
    """Given input Nxd input compute  NxN  distance matrix, where dist[i,j]
        is the square norm between x[i,:] and x[j,:]
        such that dist[i,j] = ||x[i,:]-x[j,:]||^p]]
    
    Arguments:
        x {torch.Tensor} -- [description]
    
    Keyword Arguments:
        p {int} -- [description] (default: {1})
    
    Returns:
        [dist] -- [NxN  distance matrix]
    """
    
    N, D =  x.size()
    dist=torch.repeat_interleave(x, N, dim=1)  
    dist.permute(1,0)
    dist = torch.pow(torch.abs(dist - dist.permute(1,0))**p,1/p)

    return  dist

def get_img_from_VI(V, I, width,hard_threshold=False,para=.5):
    '''Get images from VI, hard_threshold, set para as threshold to cut off,5-10
    soft_threshold, set para to .1-.5 to shrink the intensity'''
    # center the current and voltage, get the size resolution of mesh given width
    d = V.shape[0]
    # doing interploation if number of points is less than width*2
    if d<2* width:
        newI = np.hstack([V, V[0]])
        newV = np.hstack([I, I[0]])
        oldt = np.linspace(0,d,d+1)
        newt = np.linspace(0,d,2*width)
        I = np.interp(newt,oldt,newI)
        V = np.interp(newt,oldt,newV)
        
    (I,d_c)  = center(I,width)
    (V,d_v)  = center(V,width)
    
    #  find the index where the VI goes through in current-voltage axis
    ind_c = np.ceil((I-np.amin(I))/d_c)
    ind_v = np.ceil((V-np.amin(V))/d_v)
    ind_c[ind_c==width] = width-1
    ind_v[ind_v==width] = width-1
    
    Img = np.zeros((width,width))
    
    for i in range(len(I)):
        Img[int(ind_c[i]),int(width-ind_v[i]-1)] += 1
    
    if hard_threshold:
        Img[Img<para] = 0
        Img[Img!=0] = 1
        return Img
    else:
        return (Img/np.max(Img))**para




def paa(series:np.array, emb_size:int, scaler=None):
    """
    Piecewise Aggregate Approximation (PAA)  a dimensionality reduction 
      method for time series signal based on saxpy.
      https://github.com/seninp/saxpy/blob/master/saxpy/paa.py
    
    Arguments:
        series {np.array} -- [NX1 input series]
        emb_size {int} -- [embedding dimension]
    
    Returns:
        [series] -- [emb_size x 1]
    """
    
    series_len = len(series)
    if scaler:
        series = series/scaler

    # check for the trivial case
    if (series_len == emb_size):
        return np.copy(series)
    else:
        res = np.zeros(emb_size)
        # check when we are even
        if (series_len % emb_size == 0):
            inc = series_len // emb_size
            for i in range(0, series_len):
                idx = i // inc
                np.add.at(res, idx, series[i])
                # res[idx] = res[idx] + series[i]
            return res / inc
        # and process when we are odd
        else:
            for i in range(0, emb_size * series_len):
                idx = i // series_len
                pos = i // emb_size
                np.add.at(res, idx, series[pos])
                # res[idx] = res[idx] + series[pos]
            return res / series_len

def multi_dimension_paa(series:np.array, emb_size:int):
    """Multidimensional PAA reduce  series input from N x d to emb_size x d
    
    Arguments:
        series {np.array} -- [Nxd]
        emb_size {int} -- [embedding dimension]
    
    Returns:
        [2D array emb_sizexd] -- [description]
    """
    paa_out = np.zeros((series.shape[0], emb_size))
    
    for k in range(series.shape[0]):
        paa_out[k] = paa(series[k].flatten(), emb_size)
    return paa_out




def create_distance_similarity_matrix(series:np.array, emb_size:int, p:int):
    """[summary]
    
    Arguments:
        series {np.array} -- [description]
        emb_size {int} -- [description]
    
    Keyword Arguments:
        scale {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """

    if series.ndim==1:
        series = series[:,None]
    if series.shape[1]< series.shape[0]:
        series = series.T

    series = multi_dimension_paa(series, emb_size)
    series = series.T 
    series = torch.tensor(series).float()
    dist  = get_distance_measure(series, p)
    return dist.unsqueeze(0)


def create_N_distance_similarity_matrix(series:np.array, emb_size:int, p:int):
    """[summary]
    
    Arguments:
        series {np.array} -- [description]
        emb_size {int} -- [description]
    
    Keyword Arguments:
        scale {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """
    
    d, T =  series.shape
    dist = torch.empty(d, emb_size, emb_size)
    for k in range(d):
         dist[k] = create_distance_similarity_matrix(series[k,:], emb_size, p)

    return dist


def create_voltage_current_image(current, voltage, width):

    n = len(current)
    Imgs = np.empty((n,width,width), dtype=np.float64)
    
    with tqdm(n) as pbar:
        for i in range(n):
            Imgs[i,:,:] = get_img_from_VI(current[i,],voltage[i,], width,True,1)
            pbar.set_description('processed: %d' % (1 + i))
            pbar.update(1)
        pbar.close() 
    vi = np.reshape(Imgs,(n,1, width, width))
    vi = torch.tensor(vi).float()
    return  vi

def create_N_voltage_current_image(current, voltage, width):
    n, d, T=current.shape
    vi = []
    for k in range(d):
        vi += [create_voltage_current_image(current[:,k,:], voltage[:,k,:], width)]
    vi = torch.cat(vi, 1)

    return  vi



def generate_input_feature(current, voltage, image_type, width=50, multi_dimension=True, p=1):
        
    if image_type=="vi":
        if  current.ndim==3:
            inputs = create_N_voltage_current_image(current, voltage, width)
        else:
            inputs = create_voltage_current_image(current, voltage, width)
    else:
        inputs = []
        if  multi_dimension and current.ndim==3:
            with tqdm(len(current)) as pbar:
                for i in range(len(current)):
                    inputs+=[create_N_distance_similarity_matrix(current[i], width, p)] 
                    pbar.set_description('processed: %d' % (1 + i))
                    pbar.update(1)
                pbar.close() 
        else:
            with tqdm(len(current)) as pbar:
                for i in range(len(current)):
                    inputs+= [create_distance_similarity_matrix(current[i], width, p)] 
                    pbar.set_description('processed: %d' % (1 + i)  )
                    pbar.update(1)
                pbar.close()    
        inputs = torch.stack(inputs)
    return inputs
        