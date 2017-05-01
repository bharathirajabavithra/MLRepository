# -*- coding: utf-8 -*-
"""
Created on Mon May  1 00:54:40 2017

@author: Raja2
"""

class Train:
    def train_model(self,machine_learing_model,source,target):
        return machine_learing_model.fit(source,target.ravel())
        