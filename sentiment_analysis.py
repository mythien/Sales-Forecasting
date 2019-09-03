import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import nltk.classify.util
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.classify import NaiveBayesClassifier
import numpy as np
import re
import string
import nltk
temp = pd.read_csv('E:\Reviews.csv')
temp.head()
permanent = temp[['rating' , 'text' , 'title' , 'division name']]
print(permanent.isnull().sum()) #Checking for null values
permanent.head()
check =  permanent[permanent["rating"].isnull()]
check.head()
senti= permanent[permanent["rating"].notnull()]
permanent.head()
senti["senti"] = senti["rating"]>=4
senti["senti"] = senti["senti"].replace([True , False] , ["pos" , "neg"])
senti["senti"].value_counts().plot.bar()
