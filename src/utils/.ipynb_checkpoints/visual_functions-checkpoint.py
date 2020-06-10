import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import seaborn as sns
#sns.color_palette('husl', n_colors=20)
from sklearn.metrics import confusion_matrix, f1_score
import itertools
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, zero_one_loss
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble" : [r'\usepackage{amsmath}',r'\usepackage{amssymb}'],
        "font.family": "serif",
        # Always save as 'tight'
        "savefig.bbox" : "tight",
        "savefig.pad_inches" : 0.05,
        "xtick.direction" : "in",
        "xtick.major.size" : 3,
        "xtick.major.width" : 0.5,
        "xtick.minor.size" : 1.5,
        "xtick.minor.width" : 0.5,
        "xtick.minor.visible" : False,
        "xtick.top" : True,
        "ytick.direction" : "in",
        "ytick.major.size" : 3,
        "ytick.major.width" : 0.5,
        "ytick.minor.size" : 1.5,
        "ytick.minor.width" : 0.5,
        "ytick.minor.visible" : False,
        "ytick.right" : True,
        "figure.dpi" : 600,
        "font.serif" : "Times New Roman",
        "mathtext.fontset" : "dejavuserif",
        "axes.labelsize": 10,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # Set line widths
        "axes.linewidth" : 0.5,
        "grid.linewidth" : 0.5,
        "lines.linewidth" : 1.,
        # Remove legend frame
        "legend.frameon" : False
}
matplotlib.rcParams.update(nice_fonts)
SPINE_COLOR="gray"
colors =[plt.cm.Blues(0.6), plt.cm.Reds(0.4), plt.cm.Greens(0.6), '#ffcc99', plt.cm.Greys(0.6)]
lilac_names=['1-phase-motor', '3-phase-motor', 'Bulb',
       'CoffeeMaker', 'Drilling', 'Dumper',
       'FluorescentLamp', 'Freq-conv-squirrel-3', 'HairDryer',
       'Kettle', 'Raclette', 'Refrigerator', 'Resistor',
       'Squirrel-3', 'Squirrel-3-2x', 'Vacuum']

plaid_names = ['CFL','ILB','Waterkettle','Fan','AC','HairIron','LaptopCharger','SolderingIron','Fridge','Vacuum','CoffeeMaker','FridgeDefroster']
lilac_labels={'1-phase-async-motor':"1P-Motor", '3-phase-async-motor':"3P-Motor", 'Bulb':"ILB",
       'Coffee-machine':"CM", 'Drilling-machine':"DRL", 'Dumper-machine':"3P-DPM",
       'Fluorescent-lamp':"CFL", 'Freq-conv-squirrel-3-2x':"3P-FCS-2x", 'Hair-dryer':"Dryer",
       'Kettle':"KT", 'Raclette':"RC", 'Refrigerator':"Fridge", 'Resistor':"Resistor",
       'Squirrel-3-async':"3P-SQL", 'Squirrel-3-async-2x':"3P-SQL-2x", 'Vacuum-cleaner':"Vacuum"}

plaid_labels = {"Compact fluorescent lamp":'CFL',
               'Bulb':"ILB",'Kettle':"KT",'Fan':"Fan",'AC':'AC',
               'HairIron':"HairIron",'LaptopCharger':"Laptop",
               'SolderingIron':"SLD",'Fridge':"Fridge",'Vacuum':"Vacuum",'CoffeeMaker':"CM",'FridgeDefroster':"FRZ"}



def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], markeredgecolor=color)

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

def savefig(filename, leg=None, format='.eps', *args, **kwargs):
    """
    Save in PDF file with the given filename.
    """
    if leg:
        art=[leg]
        plt.savefig(filename + format, additional_artists=art, bbox_inches="tight", *args, **kwargs)
    else:
        plt.savefig(filename + format,  bbox_inches="tight", *args, **kwargs)
    plt.close()


def plot_learning_curve(tra_loss_list, tra_f1_list, val_loss_list, val_f1_list):
    
    def line_plot(y_train, y_val, early_stoping, y_label="Loss", y_min=None, y_max=None, best_score=None):
        iterations = range(1,len(y_train)+1)
        if y_min is None:
            y_min = min(min(y_train), min(y_val))
            y_min = max(0, (y_min - y_min*0.01))
        if y_max is None:
            y_max = max(max(y_train), max(y_val))
            y_max = min(1, (y_max + 0.1*y_max))

       
        plt.plot(iterations, y_train, label="training " )
        plt.plot(iterations, y_val, label="validation ")

        if best_score:
            
            plt.title(r"\textbf{Learning curve}"  f": best score: {best_score}",  fontsize=8)
            #plt.axvline(early_stoping, linestyle='--', color='r',label='Early Stopping')
       
        else:
            plt.title(r'\textbf{Learning curve}')
           

        plt.ylabel(y_label)
        #plt.ylim(y_min, y_max)
        plt.xlabel(r"Iterations")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
       
        plt.legend(loc="best")
        ax = plt.gca()
        ax.patch.set_alpha(0.0)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))  
        format_axes(ax)
    

   

    min_val_loss_poss = val_loss_list.index(min(val_loss_list))+1 
    min_val_score_poss = val_f1_list.index(max(val_f1_list))+1 
    
    

    fig = figure(fig_width=8)
    plt.subplot(1,2,1)
    line_plot(tra_loss_list, val_loss_list, min_val_loss_poss, y_label="Loss", y_min=0)
   
    
    plt.subplot(1,2,2)
    
    line_plot(tra_f1_list, val_f1_list, min_val_score_poss, y_label="Accuracy", y_min=None, y_max=1, best_score=np.max(val_f1_list))
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(pad=1.0)
    
   
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',save = True,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmNorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    plt.imshow(cmNorm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]),fontsize=8,
                 horizontalalignment="center",
                 color="white" if cmNorm[i, j] > thresh else "black") #10

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax = plt.gca()
    #ax.tick_params(axis="both", which="both", bottom=False, 
               #top=False, labelbottom=True, left=False, right=False, labelleft=True)
    #plt.yticks([])
    #plt.xticks([])
    if title:
        plt.title(title)
     
    
        
def plot_Fmeasure(cm, n, title="Fmacro"):
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
    p = np.array(p)
    
    volgorde = np.argsort(p)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_color_codes("pastel")
    sns.barplot(x=p[volgorde], 
            y=np.array(n)[volgorde], color='b')
    plt.axvline(x=av,color='orange', linewidth=1.0, linestyle="--")
    a = '{0:0.02f}'.format(av)
    b = '$Fmacro =\ $'+a
    if av > 75:
        plt.text(av-27,0.1,b,color='darkorange')
    else:
        plt.text(av+2,0.1,b,color='darkorange')
    ax.set_xlabel("$Fmacro$",fontsize=18)
    ax.set_ylabel("",fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set(xlim=(0, 100))
    if title:
        plt.title(title, fontsize=20)
    #sns.despine(left=True, bottom=True)
    
    
    
def get_Fmeasure(cm, n):
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
    return av
    
    
def vis_results(true, pred, dataset, fig_path):
    
    cm = confusion_matrix(true, pred)
    plot_Fmeasure(cm, apps[dataset], title=None)
    savefig(f"{fig_path}_fm",format=".pdf")
    
    if dataset=="whited":
         fig, ax = plt.subplots(figsize=(12, 10))
    else:
         fig, ax = plt.subplots(figsize=(10, 8))
    plot_confusion_matrix(cm, apps[dataset], title=None)
   
    
    
def get_fscore(true, pred, dataset):
    cm = confusion_matrix(true, pred)
    f1 = get_Fmeasure(cm, apps[dataset])
    return f1



def get_fscore(cm, names):
    av = 0
    p = []
    for i in range(len(names)):
        teller = 2 * cm[i,i]
        noemer = sum(cm[:,i]) + sum(cm[i,:])
        F = float(teller) / float(noemer)
        av += F
        #print('{0} {1:.2f}'.format(names[i],F*100))
        p.append(F*100)

    av = av/len(names)*100

    p = np.array(p)
    return p, av

def plot_multiple_fscore(names, cm_vi,cm_rp, labels=["baseline", "adaptive RP"]):
    width = 0.4
    #sns.set_color_codes("pastel")
    f1_vi,av_vi = get_fscore(cm_vi, names)
    f1_rp,av_rp = get_fscore(cm_rp, names)
    av = max(av_vi, av_rp)
    width=0.4
    plt.barh(np.arange(len(f1_vi)), f1_vi, width, align='center', color=colors[0], label=labels[0])
    plt.barh(np.arange(len(f1_rp))+ width, f1_rp, width, align='center',color=colors[1], label=labels[1])
    ax = plt.gca()
    ax.set(yticks=np.arange(len(names)) + width, yticklabels=names)
    ax.set_xlabel("$F_1$ macro (\%)'")

    ax.axvline(x=av,color='orange', linewidth=1.0, linestyle="--")
    a = '{0:0.2f}'.format(av)
    b = '$ $'+a
    if av > 75:
        OFFSET = -0.7
        plt.text(av-5,OFFSET,b,color='darkorange')
    else:
        OFFSET = 0
        plt.text(av,OFFSET,b,color='darkorange')
    ax.set_ylabel("")
    ax.tick_params(axis='both', which='major')
    leg=legend(ax,ncol=2, pos=(0.5, -0.2))
    return leg



def get_model_results(dataset="plaid", model_name='CNN', width=50,run_id=1, cv=10, fig_path=None):
    
    multi_dimension=False
    isc=False
    names=list(lilac_labels.values())   if dataset=="lilac"   else list(plaid_labels.values())  
    dataset = dataset+"_isc" if isc else dataset
    baseline = False
    results_cm = {}
    results_f1={}
    results_mcc={}
    results_zl={}
    preds = {}
    trues = {}
    currents = {}
    results = pd.DataFrame( index=[ "vi" ], columns=["F1", "MCC", "ZL"])
    for image_type in ["vi", "adaptive" ]:
        if baseline:
            file_name=f"{dataset}_{image_type}_{str(width)}_{model_name}_baseline_{str(run_id)}_exp_two"

        else:
            file_name=f"{dataset}_{image_type}_{str(width)}_{model_name}_{str(run_id)}_exp_two" 
        if dataset=="lilac" and multi_dimension==True :
            file_name = file_name+"_multi-dimension-norm"

        pred = np.load("../results/"+file_name+"_pred.npy")
        true = np.load("../results/"+file_name+"_true.npy")
        #img = np.load("../results/"+file_name+"_images.npy")
        preds[image_type]=pred
        trues[image_type]=true

        mcc  = matthews_corrcoef(true, pred)
        zl   = zero_one_loss(true, pred)*100
        cm   = confusion_matrix(true, pred)
        f1   = get_Fmeasure(cm, names)
        results_cm[image_type]=cm
        results_f1[image_type]=f1
        results_mcc[image_type]=mcc
        results_zl[image_type]=zl
        f1  = np.load("../results/"+file_name+"_f1.npy")
        zl  = np.load("../results/"+file_name+"_z_one.npy")
        mcc  = np.load("../results/"+file_name+"_mcc.npy")
        print(f'results for {image_type} image type with {dataset} dataset')
        print(f"mcc:{round(mcc.mean(), 2)}:{round(mcc.std(), 2)}")
        print(f"f1:{round(f1.mean()*100, 2)}:{round(f1.std()*100, 2)}")
        print(f"zl:{round(zl.mean(), 2)}:{round(zl.std(), 2)}")
        print('')
        fig=figure(columns=2)
        plot_confusion_matrix(results_cm[image_type], names, title=None)
        ax = plt.gca()
        ax.tick_params(axis="both", which="minor", bottom=False, 
               top=False, labelbottom=True, left=False, right=False, labelleft=True)

        savefig(fig_path+f"cm_{image_type}_{dataset}_agg", format=".pdf")

    fig=figure(columns=2)
    plot_multiple_fscore(names, results_cm["vi"],results_cm['adaptive'], labels=["VI", "RP"])
    ax = plt.gca()
    ax.tick_params(axis="both", which="minor", bottom=False, 
               top=False, labelbottom=True, left=False, right=False, labelleft=True)
    format_axes(ax)
    savefig(fig_path+f"fm_{dataset}_agg", format=".pdf")



    results["F1"] = pd.Series(results_f1)
    results["MCC"] = pd.Series(results_mcc)
    results["ZL"] = pd.Series(results_zl)
    results=results.round(2)
    print(f"results for {dataset}")
    print(results)
    print("")

    return trues, preds