# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:22:27 2025

@author: u6942852
"""

import numpy as np
import pandas as pd
import chaospy as cp
from scipy.stats import gaussian_kde # For Kernel Density Estimation

from Input import * 
from Costs import Raw_Costs 
from Optimisation import Optimise, Objective
from ParameterSweep import calculate_costs
from Timekeeper import keeptime, PrintTimekeeper, timekeeper

#%%


raw_costs = Raw_Costs(scenario, DClengths, undersea_mask, network_mask)
costs = raw_costs.CostFactors()

CSV_FILE_PATH = "Results/firmpoints.csv"
NUM_INPUTS = 50
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
try:
    input_data = pd.read_csv(CSV_FILE_PATH, nrows = 20000, header=None).to_numpy()
except FileNotFoundError as e:
    import os
    print(os.getcwd())
    raise e
    
rng = np.random.default_rng()
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


# Use Kernel Density Estimation (KDE) to estimate the density of input data.
# The 'bw_method' can be 'scott', 'silverman', or a scalar. Experiment if needed.
# input_data needs to be (n_features, n_samples) for KDE.
kde = gaussian_kde(input_data.T)

# Evaluate KDE at each training data point to get its density
densities = kde(input_data.T)

# Use densities as weights. Normalize them for better numerical stability.
# A common normalization is to make them sum to the number of samples.
weights = densities / np.sum(densities) * len(densities)


# ---Build the PCE Surrogate Model using Weighted Least Squares ---
# Choose the polynomial order. Higher order captures more non-linearity but increases complexity.
# For 50 inputs, a low order (e.g., 1 or 2) is often a good starting point,
# especially with sparse PCE.
POLYNOMIAL_ORDER = 2 # You might need to experiment with this value

print(f"\nBuilding PCE model with polynomial order {POLYNOMIAL_ORDER} using Weighted Least Squares...")

# 1. Generate the orthogonal polynomials for the given order and distribution
# This creates the basis functions for the PCE.
print("line72")
polynomial_basis = cp.expansion.stieltjes(POLYNOMIAL_ORDER, joint_distribution)

# 2. Evaluate the polynomial basis at the input_data points to form the design matrix (Vandermonde matrix)
# The design matrix 'A' will have shape (n_samples, n_terms)
# where n_terms is the number of polynomials in the basis.
# input_data is (n_samples, n_features), but `polynomial_basis` expects (n_features, n_samples)
print("line79")
design_matrix = polynomial_basis(*input_data.T).T # Transpose input_data for evaluation, then transpose result

# 3. Perform Weighted Least Squares (WLS)
# We use the square root of weights for the WLS transformation.
# This transforms the problem from min ||Ax - b||^2 to min ||WAx - Wb||^2
# where W is a diagonal matrix with sqrt(weights) on the diagonal.
sqrt_weights = np.sqrt(weights)
weighted_design_matrix = design_matrix * sqrt_weights[:, np.newaxis] # Apply weights row-wise
weighted_qoi_data = qoi_data * sqrt_weights # Apply weights to output data

# Solve for the coefficients using numpy's least squares solver
# `rcond=None` is used to suppress a future warning about default value changes.
pce_coefficients, residuals, rank, s = np.linalg.lstsq(weighted_design_matrix, weighted_qoi_data, rcond=None)

# 4. Construct the chaospy polynomial from the calculated coefficients
pce_model = cp.sum(polynomial_basis * pce_coefficients[:, np.newaxis])
# Note: cp.sum(polynomials * coefficients) is the way to create the final polynomial.
# The `[:, np.newaxis]` is important if coefficients is a 1D array to enable broadcasting.

print("PCE model built successfully using Weighted Least Squares.")
print(f"Number of terms in PCE: {len(pce_coefficients)}")
print(f"Residuals from WLS: {residuals}") # Lower residuals indicate a better fit (weighted)


# --- Step 4: Use the Surrogate Model for Prediction ---
print("\nUsing the surrogate model for prediction...")

# Generate new input points for prediction (e.g., a few random samples)

# Predict the output using the PCE model
predicted_outputs = pce_model(*test_input_data)

print(f"New input samples (transposed for display):\n{new_input_samples.T}")
print(f"Predicted outputs:\n{predicted_outputs}")

# --- Step 5: Calculate Variance Attributable to Each Input (Sobol Indices) ---
# This directly addresses your requirement to maintain the variance attributable to each input.
# First-order Sobol indices (S1) quantify the main effect of each input.
# Total-order Sobol indices (ST) quantify the main effect plus all interactions involving that input.
print("\nCalculating Sobol Indices for variance attribution...")
raise KeyboardInterrupt()
try:
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

except Exception as e:
    print(f"An error occurred while calculating Sobol indices: {e}")
    print("Ensure the PCE model was built correctly and distributions are well-defined.")

print("\n--- Outline Complete ---")
print("This version uses weighted least squares to prioritize regions with higher data density.")
