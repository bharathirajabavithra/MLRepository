# -*- coding: utf-8 -*-
"""
Created on Mon May  1 01:38:33 2017

@author: Raja2
"""
class Split:
    def general_split(self,source,target):
        from sklearn.cross_validation import train_test_split
        return train_test_split(source,target,test_size=0.25,random_state=42)
        