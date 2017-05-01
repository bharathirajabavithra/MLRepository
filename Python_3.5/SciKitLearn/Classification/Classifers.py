# -*- coding: utf-8 -*-
"""
Created on Mon May  1 00:22:30 2017

@author: Raja2
"""
class Classifiers:
    def __init__(self,log):
        self.console_log=log
        if(self.console_log):
            print('Classifer is being created..')
    def navie_bayes(self):
        if(self.console_log):
            print('Invoked Navie bayes --> GaussianNB..')
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
        