# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 12:11:25 2025

@author: u6942852
"""

import pandas as pd 

from Input import * 
from ParameterSweep import calculate_costs

df = pd.read_csv('Results/History11.csv', header=None)

df.columns = (['Energy', 'Penalties', 'Gas GWh p.a.', 'Flex GWh p.a.',
               'PHES GWh p.a.', 'Spillage', 'Trans']+
              [f'CDC{n}' for n in range(len(DCloss))]+
              [f'PV{n}' for n in range(pzones)] + 
              [f'W{n}' for n in range(wzones)] + 
              [f'G{n}' for n in range(nodes)] + 
              [f'CPHP{n}' for n in range(nodes)] + 
              ['CPHE'])

Lcoes = calculate_costs(df.to_numpy(), costs)

