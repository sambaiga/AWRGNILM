import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 20})

def set_figure_size(fig_width=None, fig_height=None, columns=2):
    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES
    return (fig_width, fig_height)


def format_axes(ax):
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)
    
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)
    return ax

def figure(fig_width=None, fig_height=None, columns=2):
    """
    Returns a figure with an appropriate size and tight layout.
    """
    fig_width, fig_height =set_figure_size(fig_width, fig_height, columns)
    fig = plt.figure(figsize=(fig_width, fig_height))
    return fig

def subplots(fig_width=None, fig_height=None, *args, **kwargs):
    """
    Returns subplots with an appropriate figure size and tight layout.
    """
    fig_width, fig_height = get_width_height(fig_width, fig_height, columns=2)
    fig, axes = plt.subplots(figsize=(fig_width, fig_height), *args, **kwargs)
    return fig, axes

def legend(ax, ncol=3, loc=9, pos=(0.5, -0.1)):
    leg=ax.legend(loc=loc, bbox_to_anchor=pos, ncol=ncol)
    return leg

def savefig(filename, leg=None, format='.pdf', *args, **kwargs):
    """
    Save in PDF file with the given filename.
    """
    if leg:
        art=[leg]
        plt.savefig(filename + format, additional_artists=art, bbox_inches="tight", *args, **kwargs)
    else:
        plt.savefig(filename + format,  bbox_inches="tight", *args, **kwargs)
    plt.close()
    

def event_visualizer(pa, appliances, on_pattern,phase):
    font = {'family': 'serif',
        'color':  'red'
        }
    ilim=pa.max()
    t=np.arange(len(pa))*2e-2      
    plt.title("Power-Signal")
    
    plt.plot(t, pa)
    plt.ylim(0, ilim)
    if len(on_pattern)>1:
        plt.text(10-1,pa.max()/2,appliances[on_pattern[0]-1]+" on",  rotation=90, fontdict=font,  horizontalalignment='center', verticalalignment='center')
        plt.text(15-1,pa.max()/2,appliances[on_pattern[1]-1]+" on", rotation=90, fontdict=font,  horizontalalignment='center', verticalalignment='center')
        plt.text(40,pa.max()/2,appliances[on_pattern[0]-1]+" off",  rotation=90, fontdict=font,  horizontalalignment='center', verticalalignment='center')
        plt.text(45,pa.max()/2,appliances[on_pattern[1]-1]+" off", rotation=90, fontdict=font,  horizontalalignment='center', verticalalignment='center')
    if len(on_pattern)>2:
        plt.text(50,pa.max()/2,appliances[on_pattern[2]-1]+" off", rotation=90, fontdict=font,  horizontalalignment='center', verticalalignment='center')
        plt.text(20-1,pa.max()/2,appliances[on_pattern[2]-1]+" on", rotation=90, fontdict=font,  horizontalalignment='center', verticalalignment='center')
   
    
    plt.xlabel("$T$ $S$")
    plt.ylabel("$i(t)$ $(A)$")
    plt.title(f'phase:{phase}')



def visualize_event(data, on_pattern, ilim=100):
    for k in range(1, 4):
        plt.subplot(1,3, k)
        I = data['I{}'.format(k)].values
        V = data['V{}'.format(k)].values

        
        event_visualizer(I, appliances, on_pattern,ilim=ilim, phase=k)
        plt.tight_layout()


def plot_event(i_on_b, i_on_a, v_on_b, v_on_a,i_on_event, v_on_event,titles,apps, ilim=5, vlim=350):
    plt.subplot(3,3,1)
    plt.title(titles[0])
    plt.plot(i_on_b)
    plt.ylabel("$i(t) A$")
    plt.ylim(-ilim, ilim)
    plt.subplot(3,3,2)
    plt.title(titles[0])
    plt.plot(v_on_b)
    plt.ylabel("$v(t)$")
    plt.ylim(-vlim, vlim)
    plt.subplot(3,3,3)
    plt.title(f"{titles[0]} : {apps[0]} ")
    plt.plot(v_on_b, i_on_b )
    plt.ylabel("$i(t)A$")
    plt.xlabel("$v(t)V$")
    plt.ylim(-ilim, ilim)
    plt.xlim(-vlim, vlim)
    plt.subplot(3,3,4)
    plt.title(titles[1])
    plt.plot(i_on_a)
    plt.ylabel("$i(t)A$")
    plt.ylim(-ilim, ilim)
    plt.subplot(3,3,5)  
    plt.plot(v_on_a)
    plt.ylabel("$v(t)V$")
    plt.ylim(-vlim, vlim)
    plt.title(titles[1])
    plt.subplot(3,3,6)  
    plt.plot(v_on_a,i_on_a)
    plt.ylabel("$i(t)A$")
    plt.xlabel("$v(t)V$")
    plt.ylim(-ilim, ilim)
    plt.xlim(-vlim, vlim)
    plt.title(f"Event: {apps[1]} on")
    plt.subplot(3,3,7)
    plt.title(titles[2])
    plt.plot(i_on_event)
    plt.ylabel("$i(t) A$")
    plt.ylim(-ilim, ilim)
    plt.subplot(3,3,8)
    plt.title(titles[2])
    plt.plot(v_on_a)
    plt.ylabel("$v(t)$")
    plt.ylim(-vlim, vlim)
    plt.subplot(3,3,9)
    plt.title(f"{titles[2]} : {apps[1]} on")
    plt.plot(v_on_event, i_on_event )
    plt.ylabel("$i(t)A$")
    plt.xlabel("$v(t)V$")
    plt.ylim(-ilim, ilim)
    plt.xlim(-vlim, vlim)
    plt.tight_layout()


def plot_confusion_matrix(cm, classes, name,
                          normalize=False,
                          title='Confusion matrix',save = True,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmNorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(9,9))
    plt.imshow(cmNorm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize = 20)
    plt.yticks(tick_marks, classes, fontsize = 20)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cmNorm[i, j] > thresh else "black",fontsize=20) #10

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 25)
    plt.xlabel('Predicted label', fontsize = 25)
    if title:
        plt.title(title)
    
    if save:
        plt.savefig(name, format='pdf', bbox_inches='tight')
    #plt.savefig(name, bbox_inches='tight')
    #plt.show()
    
def plot_Fmeasure(cm, n, name,save=True, title=None):
    av = 0
    p = []
    for i in range(len(n)):
        teller = 2 * cm[i,i]
        noemer = sum(cm[:,i]) + sum(cm[i,:])
        F = float(teller) / float(noemer)
        av += F
        #print('{0} {1:.2f}'.format(names[i],F*100))
        p.append(F*100)

    av = av/len(n)*100
    
    
    volgorde = np.argsort(p)
    p = np.array(p)
    names = np.array(n)
    plt.figure(figsize=(10,10))
    print(len(names))
    print(len(p))
    
    plt.barh(np.arange(len(names)), p[volgorde],  align='center', alpha=0.8)
    plt.yticks(np.arange(len(names)), names[volgorde])
    if title:
        plt.title(title)
    plt.xlabel('F-measure')
    plt.xlim(0,100)
    plt.axvline(x=av,color='orange', linewidth=1.0, linestyle="--")
    a = '{0:0.02f}'.format(av)
    b = '$Fmacro =\ $'+a
    if av > 75:
        plt.text(av-27,0.1,b,color='darkorange')
    else:
        plt.text(av+2,0.1,b,color='darkorange')
    if save:
        savefig(name)
    
    
    
    

 
    
    
    
    
def analyse_mistakes(y_true, y_pred, active,f):
    active_during_faults = {i:[] for i in np.unique(y_true)}
    faults = {i:[] for i in np.unique(y_true)}
    files = {i:[] for i in np.unique(y_true)}
    
    for i in np.unique(y_true):
        ind = np.where(y_true == i)[0]
        correct = y_true[ind] == y_pred[ind]
        active_appl = active[ind]
        selected_y = y_pred[ind]
        selected_files = f[ind]
        
        # select wrong appliances
        ind = np.where(correct == 0)[0]
        active_during_faults[i] = active_appl[ind]
        faults[i] = selected_y[ind]
        files[i] = selected_files[ind]
        
    return active_during_faults, faults, files

def hist_faults(faults, indices):
    res = []
    for j in indices:
        for i in faults[j]:
            res += i
            
    for j in np.unique(res):
        print(j + ' ' + str(res.count(j)))
       
    
    pd.Series(res).value_counts().plot('bar')
        
    return res

def plot_current_from_list(C, V, ilim):
    i=1
    plt.subplot(1,2,1)
    for c,v in  zip(C,V):
        plt.plot(c, label=f'{i}')
        plt.ylabel("Current: $A$")
        plt.ylim(-ilim, ilim)
        plt.tight_layout()
        plt.legend()
        i+=1
        
    i=1
    plt.subplot(1,2,2)
    for c,v in  zip(C,V):
        plt.plot(v, label=f'{i}')
        plt.ylabel("Voltage: $V$")
        plt.tight_layout()
        i+=1

def plot_VI_from_list(C, V, ilim):
    
    fig, axs = plt.subplots(1,len(C), sharey=True) 
    fig.subplots_adjust(hspace = 0.25)
    axs = axs.ravel()

    idx=0
    for c,v in  zip(C,V):
        axs[idx].plot(v,c)
        axs[idx].set_ylabel("Current:$A$")
        axs[idx].set_xlabel("Volatge:$V$")
        axs[idx].set_ylim(-ilim, ilim)
        idx+=1
    fig.tight_layout()  


        
def plot_VI_from_array(C, V):

    for k in  range(0,3):
        plt.subplot(1,3,k+1)
        plt.plot(V[:,k],C[:,k])
        plt.tight_layout()
        
    
        
