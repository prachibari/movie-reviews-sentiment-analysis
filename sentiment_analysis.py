# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 18:03:40 2018

@author: Prachi

# Naive Bayes, Decision Trees and Random Forest are used for NLP.
"""
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

train = pd.read_csv('labeledTrainData.tsv',delimiter='\t',quoting=3)

train.head()
train.shape
train.columns.values
train['review'][0]

corpus=[]


#Data Cleaning and Text Preprocessing
#--------------------------------------------------
# approx 60-70 mins for the for loop
for i in range(0,len(train)):
    #Removing html tags
    review = BeautifulSoup(train['review'][i])
    review = review.get_text()   
    #Removing special characters and numbers
    review = re.sub("[^a-zA-Z]"," ",train['review'][i])
    #converting to lower case and splitting into individual words
    review = review.lower()
    review = review.split()
    #removing stop words and applying steming
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english')) and len(word) > 1]
    #join the words to form a string again, review is a list
    review = ' '.join(review)
    corpus.append(review) 

#Questions
#1. Corpus has to be converted to array before splitting it?
#2. Is CountVectorizer better or PCA for dimension reduction?
#3. Difference between transform and fit_transform
#4. How to choose max features ?


#Splitting into train and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(corpus, train["sentiment"], test_size = 0.20, random_state = 0)

#Creating bag of words model using Count Vectorizer and Naive Bayes
#---------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=4000)
X_train = cv.fit_transform(X_train).toarray()
X_test = cv.transform(X_test).toarray()

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting Test Results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy_naive = (cm[0,0]+cm[1,1])/5000
error_naive = (cm[0,1]+cm[1,0])/5000

#Conclusion
#1. max features = 4000 - Accuracy 0.73
#2. max features = 4500 - Accuracy 0.73 , error rate = 0.26
#3. max features = 5000 - Accuracy 0.71,  error rate = 0.28
#4. max features = 6000 - Accuracy 0.72, error rate = 0.27
#5. max features = 7000 - Accuracy 0.51
# GaussianNB takes less times to compute

#Creating bag of words model using Count Vectorizer and Random Forest
#--------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(X_train,y_train)

#Predicting Results
f_pred = forest.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,f_pred)
accuracy_forest = (cm1[0,0]+cm1[1,1])/5000
error = (cm1[0,1]+cm1[1,0])/5000

#Random forest takes comparitively higher time (like 2 minutes) to execute
# Estimators = 100, Accuracy 0.84 , error rate = 0.15
# Estimators = 150, Accuracy 0.85 , error rate = 0.14
# Estimators = 80, Accuracy 0.84 , error rate = 0.15
# Estimators = 50, Accuracy 0.83 , error rate = 0.16
# Estimators = 70, Accuracy 0.83 , error rate = 0.15

# Chose estimators = 100 since it had the lowest type 2 error ie false negative.