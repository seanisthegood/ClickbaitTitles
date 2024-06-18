from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer

import matplotlib.pyplot as plt
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import num2words
import re
import string 
import contractions
class cross_val_scores:
    
    def __init__(self, random_state=42):
        #Initiates a DataFrame with the metrics I will focus on
        self.history = pd.DataFrame(columns=['Name','mean_accuracy','std_accuracy','mean_precision','std_precision','mean_recall','std_recall','mean_f1','std_f1','notes'])
            
    def cross_val_report(self,model, name,X,Y,notes=''):
            #Cross Val Report culls various metrics and returns a 5-Fold Cross Validation
        mean_accuracy = np.mean(cross_val_score(model   ,X,Y,scoring='accuracy'))
        std_accuracy = np.std(cross_val_score(model,X,Y,scoring='accuracy'))
        mean_precision = np.mean(cross_val_score(model,X,Y,scoring='precision'))
        std_precision = np.std(cross_val_score(model,X,Y,scoring='precision'))
        mean_recall = np.mean(cross_val_score(model,X,Y,scoring='recall'))
        std_recall = np.std(cross_val_score(model,X,Y,scoring='recall'))
        mean_f1 = np.mean(cross_val_score(model,X,Y,scoring='f1'))
        std_f1 = np.std(cross_val_score(model,X,Y,scoring='f1'))
        #Adds the metrics to a dataframe
        frame = pd.DataFrame([[name,mean_accuracy,std_accuracy,mean_precision,std_precision,mean_recall,std_recall,mean_f1,std_f1,notes]],columns=['Name','mean_accuracy','std_accuracy','mean_precision','std_precision','mean_recall','std_recall','mean_f1','std_f1','notes'])
        #appends the dataframe to the existing dataframe obect
        self.history = self.history.append(frame)
        #Resets the dataframe
        self.history = self.history.reset_index(drop=True)
        self.history = self.history.sort_values('mean_recall')
        
        #Returns an initial look at the metrics
        return [mean_accuracy, mean_recall,mean_precision]

class model_scores:
    
    def __init__(self, random_state=42):
        #Initiates a DataFrame with the metrics I will focus on
        self.history = pd.DataFrame(columns=['Name','accuracy','precision','recall','f1','notes'])
            
    def report(self,model, name,X,Y,notes=''):
        preds = model.predict(X)
        accuracy = accuracy_score(Y,preds)
        precision = precision_score(Y,preds)
        recall = recall_score(Y,preds)
        f1 = f1_score(Y,preds)

        #Adds the metrics to a dataframe
        frame = pd.DataFrame([[name,accuracy,precision,recall,f1,notes]],columns=['Name','accuracy','precision','recall','f1','notes'])
        #appends the dataframe to the existing dataframe obect
        self.history = self.history.append(frame)
        #Resets the dataframe
        self.history = self.history.reset_index(drop=True)
        self.history = self.history.sort_values('recall')
        #Returns an initial look at the metrics
        return [accuracy, recall,precision]
        

def no_nums(text):
    return re.sub(r"(\d+)", lambda x: f' {num2words.num2words(int(x.group(0)))} ', text)

def Process_Tweet(text,remove_HTML=False,replace_moji_bake=False,remove_mentions=False,remove_hashtags=False,contraction_fix=False, 
                 strip_links=False,no_leading=False,lower=False,remove_numerals=False):
    
    if remove_HTML is True:
        text=text.map(lambda text: BeautifulSoup(text, 'lxml').get_text()).copy()
    
    if replace_moji_bake is True:
    
        text=text.str.replace('۝','')
    
    if remove_mentions==True:
        text= text.str.replace(r'[@]\w+','')
        
    if remove_hashtags==True:
        text=text.str.replace(r'\B#\w*[a-zA-Z]+\w*', '')
    
    if contraction_fix==True:
        text=text.map(lambda text: contractions.fix(text))
        
    if strip_links==True:
        text= text.str.replace('{link}',"")
        text = text.str.replace(r'(http://[^"\s]+)|(https://[^"\s]+)|(http:/)', ' ')
        
    if no_leading==True:
        text= text.apply(lambda x: x.lstrip(string.punctuation))
        
    if lower==True:
        text=text.apply(lambda x: x.lower())
    
    if remove_numerals==True:
        text=text.map(no_nums)
    
    return text

def reporting(model,X_train,y_train):
    plt.style.use('fivethirtyeight')
    ConfusionMatrixDisplay.from_estimator(model,X_train,y_train,display_labels=['Positive/Neutral','Negative'])
    plt.grid(False)
    plt.show()

    preds = model.predict(X_train)
    print (classification_report(y_train,preds))
#Grabbing Stopwords from NLTK
sw = stopwords.words('english')
nltk.download('averaged_perceptron_tagger')

#Borrowed Function to attatch parts of speech for lemmonization
def get_wordnet_pos(treebank_tag):
    '''
    Translate nltk POS to wordnet tags
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def doc_preparer(doc, stop_words=sw):
    '''
    
    :param doc: a document from a corpus
    :return: a document string with words which have been 
            lemmatized, 
            parsed for stopwords, 
            made lowercase,
            and stripped of punctuation and numbers.
    '''
    
    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    doc = regex_token.tokenize(doc)
    doc = [word.lower() for word in doc]
    doc = [word for word in doc if word not in sw]
    doc = pos_tag(doc)
    doc = [(word[0], get_wordnet_pos(word[1])) for word in doc]
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    doc = [lemmatizer.lemmatize(word[0], word[1]) for word in doc ]
    return ' '.join(doc)
   
    


