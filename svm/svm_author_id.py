#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
from operator import xor
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

# com kernel linear e todo o conjunto de dados, acuracia de 98,4%
# clf = svm.SVC(kernel='linear')
clf = svm.SVC(C=10000)

""" features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)] """

training_time = time()
test_data = clf.fit(features_train, labels_train)
print("Training time for SVC with rbf kernel: ", round(time() - training_time, 3), "s")

predict_time = time()
predict_data = clf.predict(features_test)
print("Predict time for SVC with rbf kernel: ", round(time() - predict_time, 3), "s")

cris_data = 0
sarah_data = 0

for data in predict_data:
    if data == 1:
        cris_data = cris_data + 1
    else:
        sarah_data = sarah_data + 1

print("Cris e-mails: ", cris_data)
print("Sarah e-mails: ", sarah_data)

predict_accuracy = accuracy_score(predict_data, labels_test)
print("My SVC accuracy was: ", predict_accuracy)

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
