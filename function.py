from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
import seaborn as sns
import matplotlib 

def cross_val_report(model_name,model,X,Y):
    #creating a function to return the cross-val scores of the different mretrics
    model_dict={}
    model_dict['model_name']=model_name
    model_dict['mean_accuracy']=np.mean(cross_val_score(model,X,Y))
    model_dict['std_accuracy']=np.std(cross_val_score(model,X,Y))
    
    return model_dict

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def displays(model, x, y):
    sns.set_context('talk')
    fig,ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay.from_estimator(model,x,y ,colorbar=False,ax=ax)
    fig.set_facecolor('#00000000')
    ax.set_xlabel('\nPredicted Label',fontsize=22,color='b');
    ax.set_ylabel('True Label',fontsize=22,color='b');
    # ax.set_xticklabels(['Stay','Churn'],color='b',fontsize=18)
    # ax.set_yticklabels(['Stay','Churn'],color='b',fontsize=18)
    plt.grid(False)
    plt.show()
