# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:22:27 2025

@author: u6942852
"""

import numpy as np
import pandas as pd
import chaospy as cp
# from scipy.stats import gaussian_kde # For Kernel Density Estimation
import os
import pickle

from Input import (scenario, DClengths, undersea_mask, network_mask, Raw_Costs, lb, ub)
from Costs import Raw_Costs 
from Optimisation import Optimise, Objective
from ParameterSweep import calculate_costs
from Timekeeper import keeptime, PrintTimekeeper, timekeeper

#%%


raw_costs = Raw_Costs(scenario, DClengths, undersea_mask, network_mask)
costs = raw_costs.CostFactors()

CSV_FILE_PATH = "Results/firmpoints.csv"
PCE_MODEL_FILENAME = 'pce_surrogate_model.pkl'


def save_pce_model(pce_model_to_save, filename):
    """
    Saves the trained Chaospy PCE model to a file using pickle.
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(pce_model_to_save, f)
        print(f"PCE model successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving PCE model to {filename}: {e}")

def load_pce_model(filename):
    """
    Loads a Chaospy PCE model from a file using pickle.
    """
    try:
        with open(filename, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"PCE model successfully loaded from {filename}")
        return loaded_model
    except FileNotFoundError:
        print(f"Error: Model file not found at {filename}. Please ensure it exists.")
        return None
    except Exception as e:
        print(f"Error loading PCE model from {filename}: {e}")
        return None

input_data = pd.read_csv(CSV_FILE_PATH, nrows = 20000, header=None).to_numpy()
    
rng = np.random.default_rng(seed=1)
rng.shuffle(input_data)

lcoes = calculate_costs(input_data, costs)
input_data = input_data[:, 15:]

cutoff = int(0.8*len(lcoes))

test_qoi_data = lcoes[cutoff:]
test_input_data = input_data[cutoff:, :]

qoi_data = lcoes[:cutoff]
input_data = input_data[:cutoff, :]

joint_distribution = cp.J(*[cp.Uniform(l, u) for l, u in zip(lb, ub)])

print("\nCalculating weights based on input data density (KDE)...")

POLYNOMIAL_ORDER = 2 # You might need to experiment with this value

pce_model = None
if os.path.exists(PCE_MODEL_FILENAME):
    pce_model = load_pce_model(PCE_MODEL_FILENAME)
else: 
    print("Model file not found. Proceeding with training.")
    
    # 1. Generate the orthogonal polynomials for the given order and distribution
    # This creates the basis functions for the PCE.
    print("checkpoint1")
    polynomial_basis = cp.expansion.stieltjes(POLYNOMIAL_ORDER, joint_distribution)

    # 2. Fit the PCE model using `chaospy.fit_regression` with LARS method.
    # This method internally handles the basis evaluation and sparse coefficient selection,
    # avoiding the explicit construction of a large dense design matrix and the
    # memory issues associated with `cp.sum` of a full basis.
    print("checkpoint2")
    print("Fitting PCE model using cp.fit_regression(method='LARS')...")
    pce_model = cp.fit_regression(
        polynomials=polynomial_basis,
        abscissas=input_data.T, # raw_data expects (n_features, n_samples)
        evals=qoi_data,
        # model='LARS' # Use Least Angle Regression for sparse fitting
    )
    print("PCE model built successfully using Sparse Regression (LARS).")
    print(f"Number of terms in PCE: {len(pce_model.coefficients)}")
    # Note: residuals from `fit_regression` are not directly available like `np.linalg.lstsq`

    # --- Save the trained model ---
    print("checkpoint3")
    save_pce_model(pce_model, PCE_MODEL_FILENAME)
    
    print("checkpoint4")

predicted_outputs = pce_model(*test_input_data.T)

def RMSE(arr1, arr2):
    return np.mean((arr1-arr2)**2)**0.5

rmse = RMSE(predicted_outputs,test_qoi_data)
    
print(f"RMSE: {rmse}")



print("checkpoint5")


print("\nCalculating Sobol Indices for variance attribution...")
# raise KeyboardInterrupt()
    # Calculate first-order Sobol indices
sobol_first_order = cp.Sens_t(pce_model, joint_distribution)
print(f"\nFirst-order Sobol Indices (S1):\n{sobol_first_order}")

# Calculate total-order Sobol indices
sobol_total_order = cp.Sens_t_squared(pce_model, joint_distribution)
print(f"\nTotal-order Sobol Indices (ST):\n{sobol_total_order}")

# You can also calculate higher-order interaction indices if needed:
# sobol_second_order = cp.Sens_t(pce_model, joint_distribution, order=2)
# print(f"\nSecond-order Sobol Indices (S2):\n{sobol_second_order}")

# Interpretation:
# S1[i] represents the proportion of the output variance explained by the i-th input alone.
# ST[i] represents the proportion of the output variance explained by the i-th input
# and all its interactions with other inputs.
# If ST[i] >> S1[i], it indicates strong interactions involving input i.

# Sum of S1 should be <= 1. Sum of ST can be > 1 if there are strong interactions.
print(f"\nSum of First-order Sobol Indices: {np.sum(sobol_first_order):.4f}")
print(f"Sum of Total-order Sobol Indices: {np.sum(sobol_total_order):.4f}")
print("Sobol indices:", sobol_first_order)
    
print("\n--- Outline Complete ---")
print("This version uses weighted least squares to prioritize regions with higher data density.")
