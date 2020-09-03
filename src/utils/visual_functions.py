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
    plt.barh(np.arange(len(f1_rp))+ width, f1_rp, width, align='center',color='darkorange', alpha=0.8, label=labels[1])
    ax = plt.gca()
    ax.set(yticks=np.arange(len(names)) + width, yticklabels=names)
    ax.set_xlabel("$F_1$ macro (\%)", fontsize=12)
    ax.set_ylabel("", fontsize=12)

    ax.axvline(x=av,color='darkorange', linewidth=1.0, linestyle="--")
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

    fig=figure(fig_width=4, fig_height=4) if dataset =="plaid" else figure(fig_width=4, fig_height=4.2)
    plot_multiple_fscore(names, results_cm["vi"],results_cm['adaptive'], labels=["VI", "AWRG"])
    ax = plt.gca()
    ax.tick_params(axis="both", which="minor", bottom=False, 
               top=False, labelbottom=True, left=False, right=False, labelleft=True)
    #format_axes(ax)
    savefig(fig_path+f"fm_{dataset}_agg", format=".pdf")



    results["F1"] = pd.Series(results_f1)
    results["MCC"] = pd.Series(results_mcc)
    results["ZL"] = pd.Series(results_zl)
    results=results.round(2)
    print(f"results for {dataset}")
    print(results)
    print("")

    return trues, preds
