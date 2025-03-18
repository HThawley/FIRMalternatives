# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:41:11 2024

@author: u6942852
"""

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox

import graphutils as gu

dpi=1200
costConstraint=1.02

os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))

#%%
data = pd.read_excel('Data/additions.xlsx', sheet_name='Additions')

data['Other non-renewable energy'] += data['Fossil fuels n.e.s.']

data['Other renewable'] = data[['Bioenergy', 'Geothermal energy', 'Marine energy', 'Solar thermal energy']].sum(axis=1)
data['Wind'] = data[['Offshore Wind', 'Onshore Wind']].sum(axis=1)

data = data.drop(columns=['Fossil fuels n.e.s.','Bioenergy', 'Geothermal energy', 
                          'Marine energy', 'Solar thermal energy',
                          'Offshore Wind', 'Onshore Wind', 'Pumped storage'])
data = data.rename(columns={'Coal and peat':'Coal',
                            'Solar photovoltaic':'Solar PV',
                            'Hydropower (excl. Pumped Storage)':'Hydropower',
                            'Other non-renewable energy':'Other non-renewable'
                            })

data2 = data.copy()
data2['Renewable'] = data2[['Solar PV', 'Wind', 'Other renewable', 'Hydropower']].sum(axis=1)
data2['Fossil'] = data2[['Coal', 'Natural gas', 'Oil', 'Other non-renewable']].sum(axis=1)
data2=data2[['Year', 'Renewable', 'Fossil', 'Nuclear']]

data=data.drop(columns=['Other non-renewable'])


IRENAdata = data.melt(
    id_vars='Year',
    value_vars=data.columns,
    var_name='source', 
    value_name='Capacity (MW)',
    )

IRENAdata2 = data2.melt(
    id_vars='Year',
    value_vars=data2.columns,
    var_name='source', 
    value_name='Capacity (MW)',
    )

data = pd.read_excel('Data/additions.xlsx', sheet_name='AusAdditions')

data['Other non-renewable energy'] += data['Fossil fuels n.e.s.']

data['Other renewable'] = data[['Bioenergy', 'Geothermal energy', 'Marine energy', 'Solar thermal energy']].sum(axis=1)
data['Wind'] = data[['Offshore Wind', 'Onshore Wind']].sum(axis=1)

data = data.drop(columns=['Fossil fuels n.e.s.','Bioenergy', 'Geothermal energy', 
                          'Marine energy', 'Solar thermal energy',
                          'Offshore Wind', 'Onshore Wind', 'Pumped storage'])
data = data.rename(columns={'Coal and peat':'Coal',
                            'Solar photovoltaic':'Solar PV',
                            'Hydropower (excl. Pumped Storage)':'Hydropower',
                            'Other non-renewable energy':'Other non-renewable'
                            })

data2 = data.copy()
data2['Renewable'] = data2[['Solar PV', 'Wind', 'Other renewable', 'Hydropower']].sum(axis=1)
data2['Fossil'] = data2[['Coal', 'Natural gas', 'Oil', 'Other non-renewable']].sum(axis=1)
data2=data2[['Year', 'Renewable', 'Fossil', 'Nuclear']]

data=data.drop(columns=['Other non-renewable'])

Ausdata = data.melt(
    id_vars='Year',
    value_vars=data.columns,
    var_name='source', 
    value_name='Capacity (MW)',
    )

Ausdata2 = data2.melt(
    id_vars='Year',
    value_vars=data2.columns,
    var_name='source', 
    value_name='Capacity (MW)',
    )

IRENAdata['Global Capacity (GW)'] = IRENAdata['Capacity (MW)']/1000.
Ausdata['Australia Capacity (GW)'] = Ausdata['Capacity (MW)']/1000.

IRENAdata['source'] = IRENAdata['source'].str.replace('Wind', 'Global Wind')
IRENAdata['source'] = IRENAdata['source'].str.replace('Solar PV', 'Global Solar PV')
Ausdata['source'] = Ausdata['source'].str.replace('Wind', 'Aus. Wind')
Ausdata['source'] = Ausdata['source'].str.replace('Solar PV', 'Aus. Solar PV')

# Ausdata = pd.read_csv(r"C:\Users\u6942852\OneDrive - Australian National University\Desktop\CleanEnergyCouncil.csv", 
#                       thousands=',', decimal='.', quotechar='"')
# Ausdata[['Solar', 'Wind']] /= 1000
# Ausdata=Ausdata.rename(columns={'Solar':'Aus. Solar PV', 'Wind':'Aus. Wind'})
# Ausdata=Ausdata.melt(
#     id_vars='Year', 
#     value_vars=['Aus. Solar PV', 'Aus. Wind'], 
#     var_name='source', 
#     value_name='Australia Capacity (GW)')

# fig, axs = plt.subplots(2, figsize=(10,3), dpi=1800, sharex=True)
# sns.lineplot(
#     IRENAdata[IRENAdata['Year']>=2013],
#     x='Year', 
#     y='Global Capacity (GW)', 
#     hue='source', 
#     hue_order=['Global Wind','Global Solar PV'],
#     ax=axs[0],
#     )

# sns.lineplot(
#     Ausdata[Ausdata['Year']>=2013],
#     x='Year', 
#     y='Australia Capacity (GW)', 
#     hue='source', 
#     hue_order=['Aus. Wind','Aus. Solar PV'],
#     ax=axs[1],
#     palette='dark',
#     )

# axs[0].set_title('Global net new capacity by year (2013-2023)$^2$')
# axs[1].set_xticks(list(range(2013, 2024, 2)))
# axs[0].set_ylabel('Global New\nCapacity (GW)', fontsize=10)
# axs[0].set_yticks(list(range(0, 500, 100)))
# axs[1].set_ylabel('Australia New\nCapacity (GW)', fontsize=10)
# axs[1].set_yticks(list(range(6)))
# plt.show()

#%%
from Input import *

file =fr"Results\DerlabResults\History{scenario}-resolved.csv"
data = pd.read_csv(file, header=None)

solarCols=[f'pv-{n}' for n in coverage]
windCols=[f'w-{n}' for n in coverage]
phpCols=[f'sp-{n}' for n in coverage]
phsCols=['se-Total']

varCols=solarCols+windCols+phpCols+phsCols

data.columns = ['objective', 'generation', 'cuts', 'LCOE', 'LCOG', 'LCOBS', 'LCOBT', 'LCOBL']+varCols

data['penalties'] = (data['objective']-data['LCOE']).round(4)
data = data[data['penalties'] <0.1]

mincost = data['LCOE'].min()
resolved = data['cuts'].max()
data = data.loc[data['cuts'] == resolved,:]

data = data.drop(columns=['generation', 'cuts'])
data = data[data['LCOE'] < costConstraint*mincost]

data['solar'] = data[solarCols].sum(axis=1)
data['wind'] = data[windCols].sum(axis=1)
data['php'] = data[phpCols].sum(axis=1)
data['phs'] = data[phsCols].sum(axis=1)

data['s/w'] = data['solar']/data['wind']
data['gen'] = data['solar'] + data['wind']
data['phhrs'] = data['phs']/data['php']

varCols.remove('pv-TAS')
data=data.drop(columns='pv-TAS')
data = data.reset_index(drop=True)

data = data.round(4)

#%%
fig, axs = plt.subplots(1, 2, dpi=dpi, sharey=True, figsize=(8,4))

fig.subplots_adjust(hspace=0.2, wspace=0.05)

xscale, yscale = 0.95, -0.25

plotdata = data[varCols]
assert len(plotdata)>0
melted_data = plotdata.melt(
    id_vars=[],
    value_vars=plotdata.columns,
    var_name='vars',
    value_name='capacity',
    )
melted_data['source']=melted_data['vars'].str.extract(r'([a-zA-Z]+)')
melted_data['source']=melted_data['source'].apply(
    lambda x: {'pv':'Solar (GW)', 
                'w':'Wind (GW)', 
                'sp':'Storage (GW)', 
                'se':'Storage (GWh)'}.get(x,x))
melted_data['vars']=melted_data['vars'].apply(lambda x: x.split('-')[1])

sns.boxenplot(
    melted_data[melted_data.loc[:,'source'] != 'Storage (GWh)'],
    x='vars',
    y='capacity',
    hue='source',
    hue_order=['Wind (GW)', 'Solar (GW)', 'Storage (GW)', 'Storage (GWh)'],
    ax=axs[0],
    k_depth="full",
    width_method='exponential',
    )
ax02=axs[0].twinx()
sns.boxenplot(
    melted_data[melted_data.loc[:,'source'] == 'Storage (GWh)'],
    x='vars',
    y='capacity',
    hue='source',
    hue_order=['Wind (GW)', 'Solar (GW)', 'Storage (GW)', 'Storage (GWh)'],
    ax=ax02,
    legend=False,
    k_depth="full",
    width_method='exponential',
    )

axs[0].set_ylim(0,None)
ax02.set_ylim(0,None)

axs[0].tick_params(axis='x', size=8)
plt.setp(axs[0].get_xticklabels(), rotation=-90, ha='right')
axs[0].set_title("a) All near-optimal")
axs[0].set_ylabel("Installed Capacity (GW)")
# axs[0].set_ylabel("")
# ax02.set_ylabel("Installed Capacity (GWh)")
ax02.set_ylabel("")
axs[0].set_xlabel("Zone")

axs[0].legend([], [])
ax02.legend([],[])
axs[0].get_legend().get_frame().set_alpha(0)
ax02.get_legend().get_frame().set_alpha(0)

axs[0].grid(True, which='major', axis='y', linewidth=0.5)

ax02.set_yticks(list(range(0,600,100)), labels=[])
# gu.adjust_legend([axs[0], ax02], xscale,yscale) 

plotdata = data.loc[data['s/w'] >= data['s/w'].quantile(0.9), varCols]
assert len(plotdata)>0
melted_data = plotdata.melt(
    id_vars=[],
    value_vars=plotdata.columns,
    var_name='vars',
    value_name='capacity',
    )
melted_data['source'] = melted_data['vars'].str.extract(r'([a-zA-Z]+)')
melted_data['source']=melted_data['source'].apply(
    lambda x: {'pv':'Solar (GW)', 
                'w':'Wind (GW)', 
                'sp':'Storage (GW)', 
                'se':'Storage (GWh)'}.get(x,x))
melted_data['vars']=melted_data['vars'].apply(lambda x: x.split('-')[1])


sns.boxenplot(
    melted_data[melted_data.loc[:,'source'] != 'Storage (GWh)'],
    x='vars',
    y='capacity',
    hue='source',
    hue_order=['Wind (GW)', 'Solar (GW)', 'Storage (GW)', 'Storage (GWh)'],
    ax=axs[1],
    k_depth="full",
    width_method='exponential',
    )
ax12=axs[1].twinx()
sns.boxenplot(
    melted_data[melted_data.loc[:,'source'] == 'Storage (GWh)'],
    x='vars',
    y='capacity',
    hue='source',
    hue_order=['Wind (GW)', 'Solar (GW)', 'Storage (GW)', 'Storage (GWh)'],
    ax=ax12,
    legend=False,
    k_depth="full",
    width_method='exponential',
    )

axs[1].set_ylim(0,None)
ax12.set_ylim(0,None)

axs[1].tick_params(axis='x', size=8)
plt.setp(axs[1].get_xticklabels(), rotation=-90, ha='right')

axs[1].set_title("b) Top 10% solar to wind ratio")

# axs[1].set_ylabel("Installed Capacity (GW)")
axs[1].set_ylabel("")
ax12.set_ylabel("Installed Capacity (GWh)")
axs[1].set_xlabel("Zone")

axs[1].grid(True, which='major', axis='y', linewidth=0.5)



gu.adjust_legend([axs[1], ax12], xscale, yscale, ncol=4) 

ylim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
axs[0].set_ylim(0, ylim)
axs[1].set_ylim(0, ylim)

ylim2 = 600+600*(ylim-30)/30
ax02.set_ylim(0, ylim2)
ax12.set_ylim(0, ylim2)

ax12.set_yticks(list(range(0,700,100)))
ax02.set_yticks(list(range(0,700,100)), labels=[])

ax0_bbox = axs[0].get_position()
ax1_bbox = axs[1].get_position()

width_diff = (ax0_bbox.xmax-ax0_bbox.xmin) - (ax1_bbox.xmax-ax1_bbox.xmin)
half_wd = width_diff /2 

axs[0].set_position(Bbox(((ax0_bbox.xmin, ax0_bbox.ymin), (ax0_bbox.xmax - half_wd, ax0_bbox.ymax))))
axs[1].set_position(Bbox(((ax1_bbox.xmin-half_wd, ax1_bbox.ymin), (ax1_bbox.xmax, ax1_bbox.ymax))))

axs[0].set_xlabel("")
axs[1].set_xlabel("")

# axs[0].set_position(Bbox(((ax0_bbox.xmin, ax0_bbox.ymin), (ax1_bbox.xmax, ax0_bbox.ymax))))

#%%

fig, axs = plt.subplots(1, 3, figsize=(10,5), sharey=True, dpi=1600)

melted_data = pd.melt(
    data, 
    id_vars=[col for col in data.columns if not col.isupper()],
    value_vars=['LCOBS','LCOBT','LCOBL'],
    var_name='Cost type',
    value_name='Cost ($/MWh)'
    )
melted_data['Cost type'] = melted_data['Cost type'].apply(lambda x:
    {'LCOBS':'Storage', 'LCOBT':'Transmission', 'LCOBL':'Spillage'}.get(x,x))

sns.boxenplot(
    melted_data,
    x = 'Cost type', 
    y = 'Cost ($/MWh)',
    showfliers=False, 
    ax=axs[0],
    )
axs[0].set_title("a) All near optimal")
axs[0].set_xlabel("")
axs[0].tick_params(axis='x', size=7)

melted_data = pd.melt(
    data.loc[data['s/w'] >= data['s/w'].quantile(0.9), :], 
    id_vars=[col for col in data.columns if not col.isupper()],
    value_vars=['LCOBS','LCOBT','LCOBL'],
    var_name='Cost type',
    value_name='Cost ($/MWh)'
    )
melted_data['Cost type'] = melted_data['Cost type'].apply(lambda x:
    {'LCOBS':'Storage', 'LCOBT':'Transmission', 'LCOBL':'Spillage'}.get(x,x))


sns.boxenplot(
    melted_data,
    x = 'Cost type', 
    y = 'Cost ($/MWh)',
    showfliers=False, 
    ax=axs[1]
    )
axs[1].set_title("b) Top 10% solar to\nwind energy ratio")
axs[1].tick_params(axis='x', size=7)


melted_data = pd.melt(
    data.loc[data['s/w'] <= data['s/w'].quantile(0.1), :], 
    id_vars=[col for col in data.columns if not col.isupper()],
    value_vars=['LCOBS','LCOBT','LCOBL'],
    var_name='Cost type',
    value_name='Cost ($/MWh)'
    )
melted_data['Cost type'] = melted_data['Cost type'].apply(lambda x:
    {'LCOBS':'Storage', 'LCOBT':'Transmission', 'LCOBL':'Spillage'}.get(x,x))


sns.boxenplot(
    melted_data,
    x = 'Cost type', 
    y = 'Cost ($/MWh)',
    showfliers=False, 
    ax=axs[2]
    )
axs[2].set_title("c) Bottom 10% solar to\nwind energy ratio")
axs[2].set_xlabel("")
axs[2].tick_params(axis='x', size=7)

axs[0].set_ylabel("Levelised Cost ($/MWh)")

for ax in axs:
    ax.grid(True, which='major', axis='y', linewidth=0.5)

# fig.suptitle("Breakdown of levelised cost of balancing", y=1.08)

#%%

#%%
fig, axs = plt.subplots(1, 3, dpi=dpi, sharey=True, figsize=(12,4))

fig.subplots_adjust(hspace=0.2, wspace=0.05)

xscale, yscale = 0.7, -0.25

plotdata = data[varCols]
assert len(plotdata)>0
melted_data = plotdata.melt(
    id_vars=[],
    value_vars=plotdata.columns,
    var_name='vars',
    value_name='capacity',
    )
melted_data['source']=melted_data['vars'].str.extract(r'([a-zA-Z]+)')
melted_data['source']=melted_data['source'].apply(
    lambda x: {'pv':'Solar (GW)', 
                'w':'Wind (GW)', 
                'sp':'Storage (GW)', 
                'se':'Storage (GWh)'}.get(x,x))
melted_data['vars']=melted_data['vars'].apply(lambda x: x.split('-')[1])

sns.boxenplot(
    melted_data[melted_data.loc[:,'source'] != 'Storage (GWh)'],
    x='vars',
    y='capacity',
    hue='source',
    hue_order=['Wind (GW)', 'Solar (GW)', 'Storage (GW)', 'Storage (GWh)'],
    ax=axs[0],
    k_depth="full",
    width_method='exponential',
    )
ax02=axs[0].twinx()
sns.boxenplot(
    melted_data[melted_data.loc[:,'source'] == 'Storage (GWh)'],
    x='vars',
    y='capacity',
    hue='source',
    hue_order=['Wind (GW)', 'Solar (GW)', 'Storage (GW)', 'Storage (GWh)'],
    ax=ax02,
    legend=False,
    k_depth="full",
    width_method='exponential',
    )

axs[0].set_ylim(0,None)
ax02.set_ylim(0,None)

axs[0].tick_params(axis='x', size=8)
plt.setp(axs[0].get_xticklabels(), rotation=-90, ha='right')
axs[0].set_title("a) All near-optimal")
axs[0].set_ylabel("Installed Capacity (GW)")
# axs[0].set_ylabel("")
# ax02.set_ylabel("Installed Capacity (GWh)")
ax02.set_ylabel("")
axs[0].set_xlabel("Zone")

axs[0].legend([], [])
ax02.legend([],[])
axs[0].get_legend().get_frame().set_alpha(0)
ax02.get_legend().get_frame().set_alpha(0)

axs[0].grid(True, which='major', axis='y', linewidth=0.5)

axs[0].set_yticks(list(range(0, 35, 5)))
ax02.set_yticks(list(range(0,700,100)), labels=[])
# gu.adjust_legend([axs[0], ax02], xscale,yscale) 

plotdata = data.loc[data['s/w'] >= data['s/w'].quantile(0.9), varCols]
assert len(plotdata)>0
melted_data = plotdata.melt(
    id_vars=[],
    value_vars=plotdata.columns,
    var_name='vars',
    value_name='capacity',
    )
melted_data['source'] = melted_data['vars'].str.extract(r'([a-zA-Z]+)')
melted_data['source']=melted_data['source'].apply(
    lambda x: {'pv':'Solar (GW)', 
                'w':'Wind (GW)', 
                'sp':'Storage (GW)', 
                'se':'Storage (GWh)'}.get(x,x))
melted_data['vars']=melted_data['vars'].apply(lambda x: x.split('-')[1])


sns.boxenplot(
    melted_data[melted_data.loc[:,'source'] != 'Storage (GWh)'],
    x='vars',
    y='capacity',
    hue='source',
    hue_order=['Wind (GW)', 'Solar (GW)', 'Storage (GW)', 'Storage (GWh)'],
    ax=axs[1],
    k_depth="full",
    width_method='exponential',
    )
ax12=axs[1].twinx()
sns.boxenplot(
    melted_data[melted_data.loc[:,'source'] == 'Storage (GWh)'],
    x='vars',
    y='capacity',
    hue='source',
    hue_order=['Wind (GW)', 'Solar (GW)', 'Storage (GW)', 'Storage (GWh)'],
    ax=ax12,
    legend=False,
    k_depth="full",
    width_method='exponential',
    )

axs[1].set_ylim(0,None)
ax12.set_ylim(0,None)

axs[1].legend([], [])
ax12.legend([],[])

axs[1].tick_params(axis='x', size=8)
plt.setp(axs[1].get_xticklabels(), rotation=-90, ha='right')

axs[1].set_title("b) Top 10% solar to wind ratio")

# axs[1].set_ylabel("Installed Capacity (GW)")
axs[1].set_ylabel("")
ax12.set_ylabel('')
# ax12.set_ylabel("Installed Capacity (GWh)")
# axs[1].set_xlabel("Zone")
ax12.set_yticks(list(range(0,700,100)), labels=[])
axs[1].set_yticks(list(range(0,35,5)), labels=[])

axs[1].grid(True, which='major', axis='y', linewidth=0.5)

plotdata = data.loc[data['s/w'] <= data['s/w'].quantile(0.1), varCols]
assert len(plotdata)>0
melted_data = plotdata.melt(
    id_vars=[],
    value_vars=plotdata.columns,
    var_name='vars',
    value_name='capacity',
    )
melted_data['source'] = melted_data['vars'].str.extract(r'([a-zA-Z]+)')
melted_data['source']=melted_data['source'].apply(
    lambda x: {'pv':'Solar (GW)', 
                'w':'Wind (GW)', 
                'sp':'Storage (GW)', 
                'se':'Storage (GWh)'}.get(x,x))
melted_data['vars']=melted_data['vars'].apply(lambda x: x.split('-')[1])


sns.boxenplot(
    melted_data[melted_data.loc[:,'source'] != 'Storage (GWh)'],
    x='vars',
    y='capacity',
    hue='source',
    hue_order=['Wind (GW)', 'Solar (GW)', 'Storage (GW)', 'Storage (GWh)'],
    ax=axs[2],
    k_depth="full",
    width_method='exponential',
    )
ax22=axs[2].twinx()
sns.boxenplot(
    melted_data[melted_data.loc[:,'source'] == 'Storage (GWh)'],
    x='vars',
    y='capacity',
    hue='source',
    hue_order=['Wind (GW)', 'Solar (GW)', 'Storage (GW)', 'Storage (GWh)'],
    ax=ax22,
    legend=False,
    k_depth="full",
    width_method='exponential',
    )

axs[2].set_ylim(0,None)
ax22.set_ylim(0,None)

axs[2].tick_params(axis='x', size=8)
plt.setp(axs[2].get_xticklabels(), rotation=-90, ha='right')

axs[2].set_title("c) Bottom 10% solar to wind ratio")
axs[2].set_ylabel("")
ax22.set_ylabel("Installed Capacity (GWh)")
axs[2].set_xlabel("Zone")

axs[2].grid(True, which='major', axis='y', linewidth=0.5)

axs[2].legend([], [])
ax22.legend([],[])

gu.adjust_legend([axs[2], ax22], xscale, yscale, ncol=4) 





ylim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1], axs[2].get_ylim()[1])
axs[0].set_ylim(0, ylim)
axs[1].set_ylim(0, ylim)
axs[2].set_ylim(0, ylim)

ylim2 = 600+600*(ylim-30)/30
axs[0].set_yticks(list(range(0,35,5)), labels=list(range(0,35,5)))

# ylim2 = max(ax02.get_ylim()[1], ax12.get_ylim()[1], ax22.get_ylim()[1])
ax02.set_ylim(0, ylim2)
ax12.set_ylim(0, ylim2)
ax22.set_ylim(0, ylim2)

ax0_bbox = axs[0].get_position()
ax1_bbox = axs[1].get_position()
ax2_bbox = axs[1].get_position()

ave_width = ((ax0_bbox.xmax-ax0_bbox.xmin) +  (ax1_bbox.xmax-ax1_bbox.xmin) + (ax2_bbox.xmax-ax2_bbox.xmin))/3
ave_space = ((ax1_bbox.xmin - ax0_bbox.xmax) + (ax2_bbox.xmin - ax1_bbox.xmax))/2

axs[0].set_position(Bbox(((ax0_bbox.xmin, ax0_bbox.ymin), (ax0_bbox.xmin + ave_width, ax0_bbox.ymax))))
axs[1].set_position(Bbox(((ax0_bbox.xmin + ave_width + ave_space, ax1_bbox.ymin), (ax0_bbox.xmin + ave_space + ave_width*2, ax1_bbox.ymax))))
axs[1].set_position(Bbox(((ax0_bbox.xmin + ave_space*2 + ave_width*2, ax2_bbox.ymin), (ax0_bbox.xmin + ave_space*2 + ave_width*3, ax2_bbox.ymax))))

axs[0].set_xlabel("")
axs[1].set_xlabel("")
axs[2].set_xlabel("")

#%%
costOptim = data.loc[data['LCOE'].argmin(),varCols]

distances = data[varCols].apply(lambda row: sum((row-costOptim)**2), axis=1)

distances.argmax()

