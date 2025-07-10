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
from time import perf_counter
#%%
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

input_data = pd.read_csv(CSV_FILE_PATH, skiprows = 1000000, nrows = 200000, header=None).to_numpy()
    
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

POLYNOMIAL_ORDER = 2 # You might need to experiment with this value

pce_model = None
# if os.path.exists(PCE_MODEL_FILENAME):
    # pce_model = load_pce_model(PCE_MODEL_FILENAME)
if False: 
    pass
else: 
    print("Model file not found. Proceeding with training.")
    s = perf_counter()
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
    e = perf_counter()
    print("took", e-s, "seconds to train on", len(input_data), "points")
    # --- Save the trained model ---
    print("checkpoint3")
    save_pce_model(pce_model, PCE_MODEL_FILENAME)
    
    print("checkpoint4")

predicted_test_outputs = pce_model(*test_input_data.T)

s = perf_counter()
predicted_outputs = pce_model(*input_data.T)
e = perf_counter()
print((e-s)/len(input_data))

def RMSE(arr1, arr2):
    return np.mean((arr1-arr2)**2)**0.5

rmse = RMSE(predicted_test_outputs,test_qoi_data)
    
print(f"RMSE: {rmse}")


print("Starting SOBOL")

# slice points out for fit
num_points = input_data.shape[0] - (input_data.shape[0] % (input_data.shape[1]+2))
print(num_points)
sobol_data = input_data[:num_points, :]
sobol_lcoes = lcoes[:num_points]


# from numba import njit

# @njit
def _calculate_v_ei_numerator(M0, Y_A, Y_AB_i, E_Y):
    """
    Helper function to calculate the numerator of V(E(Y|X_i)) for a single parameter.
    Args:
        M0 (int): The base number of samples (M / (N + 2)).
        Y_A (np.ndarray): The output quantities corresponding to matrix A.
        Y_AB_i (np.ndarray): The output quantities corresponding to matrix A_B^(i).
        E_Y (float): The overall mean of all output quantities.
    Returns:
        float: The estimated V(E(Y|X_i)) for the given parameter.
    """
    # The estimator for V(E(Y|X_i)) in Saltelli's method is:
    # V(E(Y|X_i)) = (1/M0) * sum(Y_A * Y_AB_i) - E_Y^2
    return (1 / M0) * np.sum(Y_A * Y_AB_i) - E_Y**2

from tqdm import tqdm

def calculate_first_order_sobol(points, quantities):
    """
    Calculates the first-order Sobol indices for a given set of points and quantities.

    This implementation manually calculates the indices based on the Saltelli sampling
    scheme. It assumes that the input 'points' and 'quantities' are pre-generated
    and ordered according to this scheme.

    Args:
        points (np.ndarray): An (M x N) array of input points, where M is the
                             total number of samples and N is the number of parameters.
        quantities (np.ndarray): An (M) array of corresponding output quantities.

    Returns:
        np.ndarray: A 1D array of first-order Sobol indices for each parameter.

    Raises:
        ValueError: If input dimensions are incorrect or if M is not compatible
                    with the Saltelli sampling structure (M % (N + 2) != 0).
        ZeroDivisionError: If the total variance of quantities is zero,
                           making Sobol indices undefined.
    """
    # --- Input Validation ---
    if not isinstance(points, np.ndarray) or points.ndim != 2:
        raise ValueError("Input 'points' must be a 2D NumPy array.")
    if not isinstance(quantities, np.ndarray) or quantities.ndim != 1:
        raise ValueError("Input 'quantities' must be a 1D NumPy array.")
    if points.shape[0] != quantities.shape[0]:
        raise ValueError("Number of rows in 'points' must match the length of 'quantities'.")

    M, N = points.shape
    if N == 0:
        # If there are no parameters, there are no Sobol indices to calculate.
        return np.array([])

    # --- Determine M0 (base sample size) ---
    # For Saltelli sampling, M must be a multiple of (N + 2).
    # M = M0 * (N + 2)
    # M0 = M / (N + 2)
    if M % (N + 2) != 0:
        raise ValueError(
            f"Total number of samples M ({M}) must be a multiple of (N + 2) "
            f"({N + 2}) for the Saltelli sampling method. "
            "Please ensure your input data conforms to this structure."
        )
    M0 = M // (N + 2)

    # --- Slice Quantities Array ---
    # Extract the output quantities corresponding to the A, B, and A_B^(i) matrices.
    # Y_A: Quantities for the base matrix A
    Y_A = quantities[0:M0]
    # Y_B: Quantities for the base matrix B
    Y_B = quantities[M0:2*M0]
    # Y_AB_list: List of quantities for each A_B^(i) matrix (A with i-th column from B)
    Y_AB_list = [quantities[(2 + i) * M0 : (2 + i + 1) * M0] for i in range(N)]

    # --- Calculate Total Variance of Y ---
    V_Y = np.var(quantities)
    if V_Y == 0:
        # If the output quantity is constant, its variance is zero, and Sobol indices are undefined.
        raise ZeroDivisionError("Total variance of quantities is zero. Sobol indices are undefined.")

    # --- Calculate Mean of Y (E_Y) ---
    E_Y = np.mean(quantities)

    # --- Calculate V(E(Y|X_i)) for each parameter in parallel ---
    # This part can be parallelized as the calculation for each parameter is independent.
    first_order_variances = np.zeros(N)
    
    for i in tqdm(range(N)):
        first_order_variances[i] = _calculate_v_ei_numerator(M0, Y_A, Y_AB_list[i], E_Y)
    
    # --- Calculate First-Order Sobol Indices ---
    # S_i = V(E(Y|X_i)) / V(Y)
    sobol_indices = first_order_variances / V_Y

    return sobol_indices

sobol1 = calculate_first_order_sobol(sobol_data, sobol_lcoes)
print(sobol1)

num_points = predicted_test_outputs.shape[0] - (predicted_test_outputs.shape[0] % (predicted_test_outputs.shape[1]+2))
print(num_points)
sobol_data = predicted_test_outputs[:num_points, :]
sobol_lcoes = predicted_test_outputs[:num_points]

sobol2 = calculate_first_order_sobol(sobol_data, sobol_lcoes)
print(sobol2)

print(sobol2/sobol1)
