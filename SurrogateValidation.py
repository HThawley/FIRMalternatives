# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 14:24:00 2025

@author: u6942852
"""

import numpy as np
import pandas as pd
from numba import prange
import matplotlib.pyplot as plt
import matplotlib.colors as plc
from scipy.stats import spearmanr, binned_statistic_2d
import seaborn as sns

from torchmlp import MLPmodel as tMLPmodel
from mlp import MLPmodel
from Input import *

@njit(parallel=True)
def ObjWrapper(xn, costs):
    retarr = np.empty(xn.shape[0], dtype=np.float64)
    for i in prange(xn.shape[0]):
        retarr[i] = Obj(xn[i], costs)
    return retarr

@njit
def Obj(x, costs):
    S = Solution(x)
    S._evaluate(costs)
    return S.LCOE + S.Penalties

def plot_1D_slice(p1, p2, true_func, surrogate_func, n_steps=100, t_args=()):
    t = np.linspace(0, 1, n_steps)
    line_points = np.array([p1 * (1 - step) + p2 * step for step in t])

    multiple_surrogates=False
    n_funcs = 1
    if hasattr(surrogate_func, "__iter__"):
        surrogate_values = [sf(line_points) for sf in surrogate_func]
        n_funcs = len(surrogate_func)
        multiple_surrogates=True

    # Expensive operation: calling the true function multiple times
    true_values = true_func(line_points, *t_args)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('1D Landscape Slice Comparison', fontsize=16)

    # Plot both lines for a direct comparison
    ax.plot(t, true_values, label='True Function', color='C0', zorder=2)
    for n in range(n_funcs):
        ax.plot(t, surrogate_values[n], label=f'Surrogate {n}', color=f'C{n+1}', linestyle='--', zorder=3+n)

    ax.set_title('Surrogate vs. True Function Landscape')
    ax.set_xlabel('Interpolation (0 -> p1, 1 -> p2)')
    ax.set_ylabel('Objective Value')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_1D_slice_from_points(p1, p2, all_points, y_true, surrogate_func, threshold_dist=1):
    line_vec = p2 - p1
    line_len_sq = np.sum(line_vec**2)
    
    # Find points near the line segment
    projections = np.dot(all_points - p1, line_vec) / line_len_sq
    distances = np.linalg.norm(all_points - (p1 + projections[:, np.newaxis] * line_vec), axis=1)
    
    # Filter points that are between p1 and p2 and within the distance threshold
    mask = (projections >= 0) & (projections <= 1) & (distances < threshold_dist)
    nearby_points_projections = projections[mask]
    nearby_points_true_values = y_true[mask]
    
    fig, ax = plt.subplots()
    ax.scatter(
        nearby_points_projections, 
        nearby_points_true_values, 
        label=f"True values from {mask.sum()} nearby points",
        color="red", 
        alpha=0.6, 
        zorder=2,
        )
    t_surrogate = np.linspace(0, 1, 200)
    line_points_surrogate = np.array([p1 * (1 - step) + p2 * step for step in t_surrogate])
    surrogate_values_line = surrogate_func(line_points_surrogate)
    ax.plot(t_surrogate, surrogate_values_line, label='Surrogate', color='blue', zorder=3)

    ax.set_title('True Function (scatter) vs. Surrogate (line)')
    ax.set_xlabel('Interpolation along random line')
    ax.set_ylabel('Costs + Penalties')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_2D_scatter_slice(X, y_true, y_pred, dim1, dim2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True, sharex=True)
    
    # Normalize color maps to the same scale for fair comparison
    norm = plc.Normalize(
        vmin=min(y_true.min(), y_pred.min()),
        vmax=min(y_true.max(), y_pred.max()),
        )
    
    # Scatter plot for Surrogate
    sc1 = ax1.scatter(
        X[:, dim1], 
        X[:, dim2], 
        c=y_pred, 
        norm=norm, 
        cmap='viridis', 
        s=10, 
        )
    fig.colorbar(sc1, ax=ax1)
    ax1.set_title('Surrogate Landscape')
    ax1.set_xlabel(f'Dimension {dim1}')
    ax1.set_ylabel(f'Dimension {dim2}')
    
    # Scatter plot for True Function
    sc2 = ax2.scatter(
        X[:, dim1], 
        X[:, dim2], 
        c=y_true, 
        norm=norm, 
        cmap='viridis', 
        s=10, 
        )
    fig.colorbar(sc2, ax=ax2)
    ax2.set_title('True Landscape')
    ax2.set_xlabel(f'Dimension {dim1}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def plot_2D_convolved_slice(X, y_true, y_pred, dim1, dim2, grid_size=50):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True, sharex=True)
    fig.suptitle(f'2D Binned Average Slice (Dimensions {dim1} vs {dim2})', fontsize=16)
    
    x_coords = X[:, dim1]
    y_coords = X[:, dim2]
    
    # Use binned_statistic_2d to compute the average value in each grid cell
    # This is effectively a 2D histogram where the cell value is the mean, not the count.
    pred_stat, x_edge, y_edge, _ = binned_statistic_2d(x_coords, y_coords, y_pred, statistic='mean', bins=grid_size)
    true_stat, _, _, _ = binned_statistic_2d(x_coords, y_coords, y_true, statistic='mean', bins=grid_size)
    
    # Normalize color maps to the same scale for fair comparison
    norm = plc.Normalize(
        vmin=min(np.nanmin(pred_stat), np.nanmin(true_stat)),
        vmax=max(np.nanmax(pred_stat), np.nanmax(true_stat)),
        )
    
    # Plot for Surrogate using pcolormesh
    # We transpose the statistic matrix because pcolormesh expects (Y, X) indexing.
    im1 = ax1.pcolormesh(x_edge, y_edge, pred_stat.T, cmap='viridis', norm=norm, shading='auto')
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Surrogate Landscape (Binned Average)')
    ax1.set_xlabel(f'Dimension {dim1}')
    ax1.set_ylabel(f'Dimension {dim2}')
    
    # Plot for True Function using pcolormesh
    im2 = ax2.pcolormesh(x_edge, y_edge, true_stat.T, cmap='viridis', norm=norm, shading='auto')
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('True Landscape (Binned Average)')
    ax2.set_xlabel(f'Dimension {dim1}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def rank_correlation_plot(y_true, y_pred, axmax=None):
    fig, ax = plt.subplots()

    y_max = max(y_true.max(), y_pred.max())
    axmax = y_max if axmax is None else axmax
    y_min = min(y_true.min(), y_pred.min())

    mask = (y_true < axmax) & (y_pred < axmax)
    sns.histplot(
        x=y_true[mask], 
        y=y_pred[mask],
        )
    
    # ax.scatter(
    #     x=y_true, 
    #     y=y_pred,
    #     # size=0.1,
    #     )
    ax.plot(
        [0, axmax], 
        [0, axmax], 
        color = [1, 0, 0, 0.5], 
        linestyle="--",
        linewidth = 0.5,
        zorder=np.inf,
        )
    ax.set_xlabel("FIRM result")
    ax.set_ylabel("MLP result")
    ax.set_title("Comparing true results to prediction")
    
    if axmax is not None:
        ax.set_xlim(y_min, min(ax.get_xlim()[1], axmax))
        ax.set_ylim(y_min, min(ax.set_ylim()[1], axmax))
    
def spearman_rank_coefficient(y_true, y_pred):
    return spearmanr(y_pred, y_true)


if __name__ == "__main__":
    
    data = pd.read_csv(
        "Results/firmpoints.csv", 
        skiprows=2_000_000,
        nrows=4_000_000,
        header=None,
        ).to_numpy()[0::2]

    y_true = data[:, 0] + data[:, 2] # lcoe + penalties
    X = data[:, 16:]
    
    MODEL_FILE_PATH = "MLP_models/mlp-pt-s1-s0.pt"
    tmlp = tMLPmodel(MODEL_FILE_PATH)
    
    MODEL_FILE_PATH = "MLP_models/mlp-full-s1-s0.json"
    mlp = MLPmodel(MODEL_FILE_PATH)

    y_pred = mlp.predict(X).sum(axis=1)
    ty_pred = tmlp.predict(X).sum(axis=1)
    
    # rank_correlation_plot(y_true, y_pred, axmax=300)
    stat, pvalue = spearman_rank_coefficient(y_true, y_pred)
    print(f"""rank correlation {stat:.4f} / 1.0. pvalue: {pvalue}.""")
    stat, pvalue = spearman_rank_coefficient(y_true, ty_pred)
    print(f"""rank correlation {stat:.4f} / 1.0. pvalue: {pvalue}.""")
    
    rank_correlation_plot(y_true[0::1], y_pred[0::1], 400)
    rank_correlation_plot(y_true[0::1], ty_pred[0::1], 400)
    
    def surrogate_func_wrapper1(points):
        return tmlp.predict(points).sum(axis=1)
    
    def surrogate_func_wrapper2(points):
        return mlp.predict(points).sum(axis=1)
    
    rng = np.random.default_rng(1)
    for _ in range(10):
        p1, p2 = X[rng.integers(0, len(data))], X[rng.integers(0, len(data))]
        # plot_1D_slice_from_points(p1, p2, X, y_true, surrogate_func_wrapper, 4)
        plot_1D_slice(p1, p2, ObjWrapper, (surrogate_func_wrapper1, surrogate_func_wrapper2), 50, t_args=(costs,))
    
    plt.show()
    
    raise KeyboardInterrupt
    for dim1 in range(1):
        for dim2 in range(53):
            if dim1==dim2: 
                continue
            # plot_2D_scatter_slice(X, y_true, y_pred, dim1, dim2)
            plot_2D_convolved_slice(X, y_true, y_pred, dim1, dim2, grid_size=50)
    