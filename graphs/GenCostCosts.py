# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:01:25 2024

@author: u6942852
"""

#GenCost WindSolar graph

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_theme()

fig, ax = plt.subplots(dpi=1200, figsize=(8,4))

genCost = pd.DataFrame(
    data=[
        [2023.5, 'Solar', 'low',  47],
        [2023.5, 'Solar', 'high', 79],
        [2030, 'Solar', 'low',  37],
        [2030, 'Solar', 'high', 63],
        [2040, 'Solar', 'low',  28],
        [2040, 'Solar', 'high', 55],
        [2050, 'Solar', 'low',  22],
        [2050, 'Solar', 'high', 46],
        [2023.5, 'Wind',  'low',  66],
        [2023.5, 'Wind',  'high', 109],
        [2030, 'Wind',  'low',  52],
        [2030, 'Wind',  'high', 88],
        [2040, 'Wind',  'low',  43],
        [2040, 'Wind',  'high', 74],
        [2050, 'Wind',  'low',  41],
        [2050, 'Wind',  'high', 73]
        ],
    columns=['Year', 'Legend', 'assump', 'Cost'])


ax.set_title("Australian Cost Estimates for Onshore Wind and Solar${^3}$")

# fig, ax = plt.subplots(dpi=1200, figsize=(10,5))

sns.lineplot(
    data=genCost, 
    x='Year', 
    y='Cost',
    hue='Legend',
    hue_order=['Wind', 'Solar'],
    )

ax.set_ylabel("Levelised Cost of Electricity\n(2023 $ AUD / MWh)")
ax.grid(axis='y')
ax.set_ylim(0,None)

#%%
fig, ax = plt.subplots(1, figsize=(8,4), dpi=1800)
ax.axvline(x=2023.25,
           color=[0,0,0,0.5], 
           linewidth=0.75)
irena = pd.DataFrame(
    data=[
    [2010,	468,	'Solar'],
    [2011,	476,	'Solar'],
    [2012,	288,	'Solar'],
    [2013,	165,	'Solar'],
    [2014,	136,	'Solar'],
    [2015,	117,	'Solar'],
    [2016,	89, 	'Solar'],
    [2017,	98, 	'Solar'],
    [2018,	80, 	'Solar'],
    [2019,	77, 	'Solar'],
    [2020,	58, 	'Solar'],
    [2021,	45, 	'Solar'],
    [2022,	41, 	'Solar'],
    [2023,	38, 	'Solar'],
    [2010,	141,	'Wind'],
    [2011,	121,	'Wind'],
    [2012,	118,	'Wind'],
    [2013,	95, 	'Wind'],
    [2014,	92, 	'Wind'],
    [2015,	78, 	'Wind'],
    [2016,	75, 	'Wind'],
    [2017,	60, 	'Wind'],
    [2018,	48, 	'Wind'],
    [2019,	45, 	'Wind'],
    [2020,	51, 	'Wind'],
    [2021,	34, 	'Wind'],
    [2022,	33, 	'Wind'],
    [2023,	42,  	'Wind'],
    ],
    columns = ['Year', 'Cost', 'Legend']
    )
irena['Cost'] /= 0.67

plotData = pd.concat((
    genCost[['Year', 'Cost', 'Legend']],
    irena))

sns.lineplot(
    data=irena[irena['Year']>=2013], 
    x='Year', 
    y='Cost',
    hue='Legend',
    hue_order=['Wind', 'Solar'],
    )

sns.lineplot(
    data=genCost, 
    x='Year', 
    y='Cost',
    hue='Legend',
    hue_order=['Wind', 'Solar'],
    legend=False
    )

ax.grid(axis='y')
ax.set_title('Historical and forecasted LCOE of Australian developments')
ax.set_ylabel('LCOE (2023 $ AUD/ MWh)')
ax.set_xlabel('')



