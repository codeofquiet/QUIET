# -*- coding: utf-8 -*-


import os
import numpy as np
from .data_reader import DataReader

def setup(opt):
    
    reader = DataReader(opt)  
    #####################
    print("experiment/qnn/dataset/qa/__init__.py def setup -----opt")
    res = opt.get_parameter_list()
    for item in res:
        print(item)
    print('\n')
    #####################
    return reader




