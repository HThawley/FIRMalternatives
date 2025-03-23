# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:26:52 2025

@author: u6942852
"""

import pandas as pd 
from Input import *

def read_history():
    df = pd.read_csv(f'Results/History{scenario}.csv', header=None)
    df 

