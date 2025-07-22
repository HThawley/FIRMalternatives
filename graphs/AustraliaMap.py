# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:32:22 2024

@author: u6942852
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

cp = sns.color_palette()

if os.getcwd().split('\\')[-1] == 'graphs':
    os.chdir('..')
from Input import *


CHYDRO = CHydro.copy()

# graphs = 'energy'
graphs = 'power'
# graphs = 'both'

states = gpd.read_file('Data/states.geojson')
states = states.loc[states['STATE_NAME'] != 'Western Australia', :]
states = states.loc[states['STATE_NAME'] != 'Australian Capital Territory', :]
states = states.loc[states['STATE_NAME'] != 'Northern Territory', :]


rekey = {'1':'NSW', '2':'VIC', '3':'QLD', '4':'SA', '5':'WA','6':'TAS', '7':'NT', '8':'ACT'}
states_ = {rekey[c]:x for c, x in zip(states['STATE_CODE'], states['geometry'])}

# nsw_xy = states_['NSW'].centroid.x, states_['NSW'].centroid.y
# vic_xy = states_['VIC'].centroid.x, states_['VIC'].centroid.y
# qld_xy = states_['QLD'].centroid.x, states_['QLD'].centroid.y
# sa_xy  = states_['SA'].centroid.x, states_['SA'].centroid.y
# tas_xy = states_['TAS'].centroid.x, states_['TAS'].centroid.y

nsw_xy = (151.2, -33.9)
vic_xy = (144.9, -37.8)
qld_xy = (153.0, -27.5)
sa_xy = (138.6, -34.9)
tas_xy = (147.3, -42.9)

xys = {'NSW':nsw_xy, 
       'VIC':vic_xy,
       'QLD':qld_xy,
       'SA':sa_xy, 
       'TAS':tas_xy,
       }



def marker_size_to_data_radius(ax, size):
    """Convert marker size (pts^2) to approximate data-unit radius"""
    fig = ax.get_figure()
    radius_pts = np.sqrt(size / np.pi)
    radius_in = radius_pts / 72

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_in, height_in = bbox.width, bbox.height
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    dx_per_in = (xlim[1] - xlim[0]) / width_in
    dy_per_in = (ylim[1] - ylim[0]) / height_in

    return radius_in * dx_per_in, radius_in * dy_per_in

def draw_pie(dist, xpos, ypos, size, ax, fig, labels, colors):
    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
    pie = [0] + cumsum.tolist()

    for i, r in enumerate(zip(pie[:-1], pie[1:])):
        r1, r2 = r
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])
    
        ax.scatter(
            [xpos], 
            [ypos], 
            marker=xy, 
            s=size, 
            zorder=100, 
            # label=labels[i], 
            color=colors[i]
            )
        ax.scatter(
            [xpos], 
            [ypos], 
            marker='o',
            s=size,
            zorder=101,
            facecolor=[1,1,1,0],
            edgecolor=[0,0,0,1],
            linewidth=0.5
            )

    return ax

def _draw_arrow(ax, xy1, xy2, r1, r2, direction, width):
    arrow = FancyArrowPatch(
        posA = xy1 + direction * r1, 
        posB = xy2 - direction * r2,
        arrowstyle=f'Simple,head_length=4,head_width={width*1.5},tail_width=0.05',
        color='red', 
        zorder = 1000,
        )
    ax.add_patch(arrow)
    return ax

def plot_arrows(ax, node_pair, sizes, lpgm):
    n1, n2 = node_pair.split('-')
    
    forward=np.maximum(0, lpgm[node_pair]).sum()*resolution/years*pow(10,-6)
    reverse=-np.minimum(0, lpgm[node_pair]).sum()*resolution/years*pow(10,-6)
        
    xy1, xy2 = np.array(xys[n1]), np.array(xys[n2])

    vec = xy2 - xy1
    dist = np.linalg.norm(vec)
    direction = vec/dist
    
    r1x, r1y = marker_size_to_data_radius(ax, sizes[n1])
    r2x, r2y = marker_size_to_data_radius(ax, sizes[n2])
    r1, r2 = np.mean([r1x, r1y]), np.mean([r2x, r2y])
    
    ax = _draw_arrow(ax, xy1, xy2, r1, r2,  direction, forward)
    ax = _draw_arrow(ax, xy2, xy1, r2, r1, -direction, reverse)
    return ax
    

# labels=['Solar PV', 'Wind', 'Gas', 'Hydro', 'Storage']
# colors=[cp[1], cp[5], cp[7], cp[0], cp[2]]

# if graphs == 'both':
#     fig, axs = plt.subplots(1, 2, figsize = (7,6), dpi=1600, sharex=True, sharey=True)
#     fig.subplots_adjust(wspace=0.1)
    
#     axs[0].set_xticks([])
#     axs[0].set_yticks([])
#     axs[1].set_xticks([])
#     axs[1].set_yticks([])

#     states.plot(ax=axs[0], edgecolor='black', facecolor='grey', zorder=10)
#     states.plot(ax=axs[1], edgecolor='black', facecolor='grey', zorder=10)
    

# else: 
#     fig, ax = plt.subplots(1, figsize = (3,6), dpi=1600)

#     ax.set_xticks([])
#     ax.set_yticks([])

#     states.plot(ax=ax, edgecolor='black', facecolor='grey', zorder=10)

#     [ax.spines[pos].set_visible(False) for pos in ('left', 'right', 'top', 'bottom')]
#     ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1]+0.5)


# # =============================================================================
# # Energy Graph
# # =============================================================================
# if graphs == 'both':
#     ax = axs[0]
    
# if graphs == 'both' or graphs == 'energy':
#     # labels=['Solar PV', 'Wind', 'Hydro']
#     # colors=[cp[1], cp[5], cp[0]]
    
#     sizes = {}
#     # plot energy mix 
#     for s in ('NSW', 'VIC', 'SA', 'QLD',): #'TAS'):
#         lpgm = pd.read_csv(f'Results/S{scenario}{s}.csv')
        
#         GPV, GWind, GHydro, GGas, GPHES = (lpgm[col].sum().sum()*resolution/years/pow(10, 6) 
#                                       for col in ('Solar photovoltaics', 'Wind', 
#                                                   ['Hydropower', 'Biomass'], 
#                                                   'Pumped hydro energy storage'))
#         sf= 12
#         draw_pie(np.array([GPV, GWind, GGas, GHydro]), 
#                  *xys[s], 
#                  size = sum([GPV, GWind, GGas, GHydro]) *sf,
#                  ax=ax,
#                  fig=fig,
#                  labels=labels, 
#                  colors=colors,
#                  )
#         draw_pie(np.array([GPHES]), 
#                  *xys[s], 
#                  size = GPHES *sf ,
#                  ax=ax,
#                  fig=fig,
#                  labels=['PHES'], 
#                  colors=[cp[2]],
#                  )
    
#         sizes[s] = sum([GPV, GWind, GHydro])*8
    
    
    
#     lpgm = pd.read_csv(f'Results/S{scenario}.csv')
    
#     # hvdc lines
#     for node_pair in ('NSW-VIC', 'NSW-QLD', 'NSW-SA'):#, 'TAS-VIC'):
#         ax = plot_arrows(ax, node_pair, sizes, lpgm)
    
#     ax.set_title('Energy Production and Transmission')

# =============================================================================
# Capacity graph
# =============================================================================
    
# if graphs == 'both':
#     ax = axs[1]
# if graphs == 'both' or graphs == 'power':    
#     # plot energy mix 
#     if isinstance(scenario, int):
#         x = pd.read_csv(f'Results/Optimisation_resultx{scenario}.csv', header=None).to_numpy().flatten()
#     elif isinstance(scenario, str):
#         x = pd.read_csv(f'Results/{scenario}.csv', header=None).to_numpy().flatten()
#     for i, s in enumerate(('NSW', 'VIC', 'SA', 'QLD')):# , 'TAS')):
#         n = np.where(s==Nodel)[0][0]
#         CPV = x[n]
#         CWind = x[pidx+n]
#         CGas = x[widx+n]
#         CPHES = x[gidx+n]
#         CHydro = CHYDRO[n]
        
#         draw_pie(np.array([CPV, CWind, CPHES, CHydro]), 
#                  *xys[s], 
#                  size = sum([CPV, CWind, CPHES, CHydro]) * 28,
#                  ax=ax,
#                  fig=fig,
#                  labels=labels, 
#                  colors=colors,
#                  )
    
#     lpgm = pd.read_csv(f'Results/S{scenario}.csv')
    
#     scalefactor=1100
#     # hvdc lines
#     for node_pair in ('NSW-VIC', 'NSW-QLD', 'NSW-SA'):#, 'TAS-VIC'):
#         ax.plot(*zip(xys[node_pair[:3]], xys[node_pair[4:]]), color='red', linewidth = np.abs(lpgm[node_pair]).max()/scalefactor, zorder=11)
    
#     ax.set_title('Power Capacity')
    
# if graphs=='both':
#     d = [axs[1].scatter([xys[s][0]], [xys[s][1]], label=labels[i], color=colors[i], marker='s', zorder=0) for i in range(4)]
#     pairs = dict(zip(labels, d))
#     fig.legend(pairs.values(), pairs.keys(), bbox_to_anchor=(0.8, 0.18), ncols=4)
# else:
#     d = [ax.scatter([xys[s][0]], [xys[s][1]], label=labels[i], color=colors[i], marker='s', zorder=0) for i in range(4)]
#     pairs = dict(zip(labels, d))
#     fig.legend(pairs.values(), pairs.keys(), bbox_to_anchor=(0.9, 0.2), ncols=2)
   
if __name__ == '__main__':
    lowgascapacity=np.array([ 21.42803117,   1.40183624,   0.12018559,   0.01779563,
             2.01955567,   5.58814092,   0.02828325,   2.47975086,
             8.38612371,   0.00302648,   0.99253116,   0.01113547,
             0.01356152,   1.41740269,   0.03617752,   2.63585979,
             0.00316804,   0.0082134 ,   0.00001758,   0.00160694,
             0.00022174,   0.00245383,   0.00020735,   7.30845123,
             0.01150658,   8.00627964,   0.20291166,   0.86044789,
             0.48551354,   0.0024191 ,   0.00324891,   0.01180405,
             0.0129977 ,   0.07151923,   0.00134996,   0.00168322,
             0.11670649,   0.00126999,   0.00040563,   0.003323  ,
             0.00107764,   6.72902172,   0.08899083,   0.62973447,
             0.202387  ,   0.00146315,   0.17195984,  12.34541065,
             3.35049099,   1.03135162,   0.08918037,   1.69410816,
           244.61971408])
    highgascapacity=np.array([ 13.44324245,   0.03042661,   0.08320981,   0.0598027 ,
             1.90798041,   3.89798665,   0.01152195,   2.70899114,
             4.0133036 ,   0.00224389,   0.36788718,   0.11075945,
             0.0072957 ,   0.69931175,   0.0089296 ,   2.04028627,
             0.00356553,   0.00995071,   0.00028822,   0.00160694,
             0.00372717,   0.00656985,   0.00015329,   9.83095885,
             0.03697448,  11.30498009,   0.04228254,   1.82514965,
             0.06123881,   0.00367905,   0.00272798,   0.02638362,
             0.00631141,   0.0044457 ,   0.0095844 ,   0.00704485,
             0.33867731,   0.06879694,   0.00341164,   0.00704986,
             0.00065947,   7.09710289,   2.58637258,   2.51068855,
             0.14314435,   0.00121209,   0.30149506,   8.41491665,
             3.350491  ,   0.7224728 ,   0.05978194,   2.07984333,
           257.78343373])
    lowgasusage=np.array([ 13.50336735,   0.00060421,   0.05853668,   3.61242394,
         1.61708409,   3.6427831 ,   0.02445633,   2.93756128,
         3.97477267,   0.00626936,   0.56337142,   0.04176995,
         0.00463043,   0.8201085 ,   0.01376098,   2.11199281,
         0.00160463,   0.0038544 ,   0.00310365,   0.00160694,
         0.00029992,   0.00459554,   0.00145964,  10.32844457,
         0.01854611,  10.62901277,   0.03626164,   1.84179358,
         0.04127208,   0.00964215,   0.00001135,   0.01287566,
         0.00538284,   0.02395853,   0.00111519,   0.00537911,
         0.2437329 ,   0.01654752,   0.00173596,   0.00293564,
         0.00229959,   6.75115671,   0.12778108,   1.62873469,
         0.86762034,   0.00185114,   0.42202793,   8.76776794,
         3.350491  ,   0.7613084 ,   0.03170424,   2.0848817 ,
       340.73674744])
    highgasusage=np.array([ 16.95387497,   0.05520747,   0.10531417,   0.02581193,
             1.95819007,   1.01366738,   0.04299945,   1.27552072,
             4.31433846,   0.0367402 ,   0.57404307,   0.01339207,
             0.00422573,   0.77133715,   0.00824403,   1.79444074,
             0.00106434,   0.01020748,   0.00315906,   0.00160694,
             0.00391424,   0.00292583,   0.00191759,   8.76338156,
             0.02717635,  10.66392994,   0.02346818,   1.64493052,
             0.01524079,   0.00313615,   0.00397588,   0.01265225,
             0.01162516,   0.02679458,   0.00000979,   0.00340842,
             0.25273242,   0.02873085,   0.00317978,   0.01349676,
             0.00036672,   6.02964778,   0.06657464,   2.36039377,
             0.53542515,   0.00362552,   0.60618491,   8.39942208,
             3.35049099,   0.81493729,   0.02469671,   2.32992666,
           215.50867709])
    slgc = Solution(lowgascapacity)
    shgc = Solution(highgascapacity)
    slgu = Solution(lowgasusage)
    shgu = Solution(highgasusage)

    slgc._evaluate(costs)
    shgc._evaluate(costs)
    slgu._evaluate(costs)
    shgu._evaluate(costs)

    labels=['Solar PV', 'Wind', 'Gas', 'Hydro', 'Storage']
    colors=[cp[1], cp[5], cp[7], cp[0], cp[2]]


    network= np.array(['FNQ-QLD', 'NSW-QLD', 'NSW-SA', 'NSW-VIC', 'NT-SA', 'SA-WA', 'TAS-VIC'])
#%%
    for S in (slgc, shgc, slgu, shgu):
        if graphs == 'both':
            fig, axs = plt.subplots(1, 2, figsize = (7,6), dpi=1600, sharex=True, sharey=True)
            fig.subplots_adjust(wspace=0.1)
            
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            states.plot(ax=axs[0], edgecolor='black', facecolor='lightgrey', zorder=10)
            states.plot(ax=axs[1], edgecolor='black', facecolor='lightgrey', zorder=10)
            

        else: 
            fig, ax = plt.subplots(1, figsize = (3,6), dpi=1600)

            ax.set_xticks([])
            ax.set_yticks([])

            states.plot(ax=ax, edgecolor='black', facecolor='lightgrey', zorder=10)

            [ax.spines[pos].set_visible(False) for pos in ('left', 'right', 'top', 'bottom')]
            ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1]+0.5)
        
        for i, s in enumerate(('NSW', 'VIC', 'SA', 'QLD', 'TAS')):
            n = np.where(s==Nodel)[0][0]
            
            CPV = S.CPV[np.where(s==PVl)[0]].sum()
            CWind = S.COnsW[np.where(s==OnsWl)[0]].sum()
            CGas = S.CGas[n]
            CPHES = S.CPHP[n]
            CHydro = S.CHydro[n]
            
            draw_pie(np.array([CPV, CWind, CGas, CHydro, CPHES]), 
                     *xys[s], 
                     size = sum([CPV, CWind, CGas, CHydro, CPHES]) * 28,
                     ax=ax,
                     fig=fig,
                     labels=labels, 
                     colors=colors,
                     )
        
        scalefactor=1
        # hvdc lines
        for node_pair in ('NSW-VIC', 'NSW-QLD', 'NSW-SA', 'TAS-VIC'):
            ax.plot(
                *zip(xys[node_pair[:3]], 
                     xys[node_pair[4:]]), 
                color='red', 
                linewidth = S.CDC[np.where(node_pair==network)[0]]/scalefactor, 
                zorder=11
                )
        
        ax.set_title('Power Capacity')

        d = [ax.scatter([xys[s][0]], [xys[s][1]], label=labels[i], color=colors[i], marker='s', zorder=0) for i in range(5)]
        pairs = dict(zip(labels, d))
        fig.legend(pairs.values(), pairs.keys(), bbox_to_anchor=(0.9, 0.2), ncols=2)