#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("Number of features in my training data : ", len(features_train[0]))

clf = DecisionTreeClassifier(min_samples_split=40)

training_time = time()
test_data = clf.fit(features_train, labels_train)
print("Training time for Decision Tree: ", round(time() - training_time, 3), "s")

predict_time = time()
predict_data = clf.predict(features_test)
print("Predict time for Decision Tree: ", round(time() - predict_time, 3), "s")

predict_accuracy = accuracy_score(predict_data, labels_test)
print("My Decision Tree accuracy: ", predict_accuracy)