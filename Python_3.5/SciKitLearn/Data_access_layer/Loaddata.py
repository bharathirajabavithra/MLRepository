# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:14:25 2017

@author: Raja2
"""

import numpy as np
class Getdata:
    def Getdata_from_textfile(self,filename):
        return np.loadtxt(filename)