# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:25:54 2017

@author: Raja2
"""
from sklearn import metrics
class Performance_metrix:
    def measure_performance(self,source,target,trained_model):
        predicted_values=trained_model.predict(source)
        print()
        print("Classifer: {0}".format(trained_model.__class__.__name__))
        print("Accuracy: {0:.3f}".format(metrics.accuracy_score(target,predicted_values)))
        print("Classification report:")
        print(metrics.classification_report(target,predicted_values))
        print("Confusion matrix:")
        print(metrics.confusion_matrix(target,predicted_values))

