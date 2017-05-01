# -*- coding: utf-8 -*-
"""
Created on Mon May  1 00:30:22 2017

@author: Raja2
"""
#import classes and libraries
from Data_access_layer.Loaddata import Getdata
from Classification.Classifers import Classifiers
from Train.Train import Train
from Data_preprocessing.Spilt_data.Split import Split
from Evaluation.Evaluate import Performance_metrix

#load dataset from text
text_database=Getdata()
data=text_database.Getdata_from_textfile('Data\MaleFemale.txt')

#set only the output column
#remove the text values from the dataset
#Normalize data
#Required Automation
source=data[:,1:4]
target=data[:,:1]

#Split data for training and testing
split_data=Split()
s_train,s_test,t_train,t_test=split_data.general_split(source,target)

#Choose classifer and build trained model
clf=Classifiers(log=True)
train=Train()
trained_model_nb=train.train_model(clf.navie_bayes(),s_train,t_train)

#Predict
predicted_output=trained_model_nb.predict(s_test)
print('Input:')
print(s_test)
print('Output:')
print(predicted_output)

#Check performance
nb_performance=Performance_metrix()
nb_performance.measure_performance(source,target,trained_model_nb)
