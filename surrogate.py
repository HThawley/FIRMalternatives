# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:22:27 2025

@author: u6942852
"""

import numpy as np
import pandas as pd
import chaospy as cp
from datetime import datetime as dt
from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from numba import njit
import json
from numpoly import ndpoly
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

np.set_printoptions(suppress=True)
raw_costs = Raw_Costs(scenario, DClengths, undersea_mask, network_mask)
costs = raw_costs.CostFactors()

class PCEmodel:
    def __init__(
            self, 
            metadata_path: str=None,
            ):
        self._is_trained = False
        
        if metadata_path is not None:
            self.load_model(metadata_path)
            
        else: 
            self.coefficients = None
            self.polynomial_order = None
            self.method = None
            self.scaler_mean = None
            self.scaler_scale = None

            self.num_inputs = 0
            
    def _data_assertions(self, input, output):
        assert isinstance(input, np.ndarray), "input should be 2d numpy array"
        assert input.ndim == 2, "input should be 2d numpy array"
        assert isinstance(output, np.ndarray), "output should be 1d numpy array"
        assert output.ndim == 1, "output should be 1d numpy array"
        assert input.shape[1] == self.num_inputs
        assert input.shape[0] == output.shape[0], "input (N, M) and output (N,) shapes should match"
        assert ((input - self.ub) < 0.001).all(), "input does not obey supplied bounds"
        assert ((self.lb - input) < 0.001).all(), "input does not obey supplied bounds"
    
    def _create_scaler(
            self, 
            input,
            ):
        scaler = StandardScaler()
        if self.scaler_mean is not None and self.scaler_scale is not None:
            scaler.mean_ = self.scaler_mean
            scaler.scale_ = self.scaler_scale
            scaler.n_features_in_ = len(self.scaler_mean)
        else: 
            scaler.fit(input)
            self.scaler_mean = scaler.mean_ 
            self.scaler_scale = scaler.scale_ 
        return scaler
    
    def preprocess(
            self, 
            input, 
            ):
        # input = normalize(input, self.lb, self.ub)
        scaler = self._create_scaler(input)
        # input = scaler.transform(input)
        return input
        
    def train(
            self, 
            input, 
            output, 
            bounds, 
            polynomial_order=2, 
            verbose=True, 
            method="lars",
            ):
        self.lb, self.ub = bounds
        assert len(self.lb) == len(self.ub)
        self.num_inputs = len(self.lb)
        self._data_assertions(input, output)
        
        assert isinstance(polynomial_order, int)
        assert polynomial_order > 1
        
        self.method = method
        self.polynomial_order = polynomial_order
        
        if self.method == "lars":
            method = lm.Lars(fit_intercept=False)
        
        start = dt.now()
        if verbose:
            print("Starting training:", start)
            print("Preprocessing training data... | Time:", dt.now())
        input = self.preprocess(input)
        joint_distribution = cp.J(*[cp.Uniform(0,1) for _ in range(self.num_inputs)])
        if verbose: 
            print("Creating polynomial basis...   | Time:", dt.now())
        polynomial_basis = cp.expansion.stieltjes(self.polynomial_order, joint_distribution)
        if verbose: 
            print("Fitting Model...               | Time:", dt.now())
        self.model = cp.fit_regression(
            polynomials=polynomial_basis,
            abscissas=input.T, 
            evals=output,
            model=lm.Lars(fit_intercept=False), 
            )
        self._is_trained = True
        if verbose: 
            print("Finished Succesfully.          | Time:", dt.now())
            print("Took:", dt.now() - start)

    def predict(
            self, 
            input,
            ):
        self.preprocess(input)
        return self.model(*input.T)
        
    def score(
            self, 
            true_output,
            predicted_output,
            metric="mean_poisson_deviance",
            ):
        assert metric in ("mean_poisson_deviance", "mean_squared_error")
        if metric == "mean_poisson_deviance":
            return mean_poisson_deviance(true_output, predicted_output)
        else: # metric == "mean_squared_error"
            return mean_squared_error(true_output, predicted_output)
        
    def save_model(
            self,
            filepath:str,
            overwrite = False,
            ):
        assert self._is_trained, "Cannot save an untrained model"
        metadata = {
            "polynomial_order" : self.polynomial_order,
            "method" : self.method,
            "num_inputs" : self.num_inputs,
            "scaler_mean" : self.scaler_mean.tolist(),
            "scaler_scale" : self.scaler_scale.tolist(),
            "lb" : self.lb.tolist(),
            "ub" : self.ub.tolist(),
            "exponents" : self.model.exponents.tolist(),
            "coefficients" : self.model.coefficients,
            "names" : self.model.names,
            }
        for k, v in metadata.items():
            assert v is not None, f"Cannot save an untrained model. ({k} is None)"
        if overwrite is False:
            if os.path.exists(filepath):
                os.mkdir("tmp_model_save")
                with open("tmp.json", "w") as f:
                    json.dump(metadata, f, indent=4)
                raise Exception(
"""Cannot overwrite existing saved model. Pass "`overwrite = True` or 
remove existing file. Current model saved as "tmp.json" """)
        with open(filepath+".json", "w") as f:
            json.dump(metadata, f, indent=4)

    
    def load_model(
            self, 
            filepath,
            verbose=True,
            ):
        start = dt.now()
        if verbose: 
            print("Reading save file... | Time:", dt.now())
        with open(filepath+".json", "r") as f:
            metadata = json.load(f)
    
        self.polynomial_order = metadata.get("polynomial_order")
        self.method = metadata.get("method")
        self.num_inputs = metadata.get("num_inputs")
        self.scaler_mean = np.array(metadata.get("scaler_mean"))
        self.scaler_scale = np.array(metadata.get("scaler_scale"))
        self.lb = np.array(metadata.get("lb"))
        self.ub = np.array(metadata.get("ub"))
        
        exponents = np.array(metadata.get("exponents"))
        coefficients = np.array(metadata.get("coefficients"))
        names = tuple(metadata.get("names"))
        
        if verbose: 
            print("Creating Model...     | Time:", dt.now())
            
        self.model = cp.polynomial_from_attributes(
            exponents = exponents,
            coefficients = coefficients,
            names = names,
            )
        self._is_trained=True
        if verbose: 
            print("Finished Succesfully. | Time:", dt.now())
            print("Took:", dt.now() - start)
    
@njit
def normalize(data, lb, ub):
    """ data.shape[0] == len(lb) == len(ub) """
    data = (data - lb) / (ub - lb)
    return data
    
@njit
def rmse(arr1, arr2):
    return np.mean((arr1-arr2)**2)**0.5

if __name__=="__main__":
    CSV_FILE_PATH = "Results/firmpoints.csv"
    
    input_data = pd.read_csv(
        CSV_FILE_PATH, 
        skiprows = 4_000_000,
        nrows=20_000, 
        header=None
        ).to_numpy()
        
    rng = np.random.default_rng(seed=1)
    rng.shuffle(input_data)
    
    lcoes = calculate_costs(input_data, costs)
    input_data = input_data[:, 15:]
    
    cutoff = int(0.8*len(lcoes))
    
    test_output = lcoes[cutoff:]
    test_input = input_data[cutoff:, :]
    
    train_output = lcoes[:cutoff]
    train_input = input_data[:cutoff, :]
    
    if os.path.exists("pce.json"):
        model = PCEmodel("pce")
    else:
        model = PCEmodel()
        model.train(train_input, train_output, (lb, ub), 3)
        
    model.save_model("pce")

    pred_train_output = model.predict(train_input)
    try: 
        train_score = model.score(train_output, pred_train_output)
    except: 
        train_score = 0.0
    train_mse = model.score(train_output, pred_train_output, "mean_squared_error")
    train_rmse = rmse(train_output, pred_train_output)
    
    pred_test_output = model.predict(test_input)
    try: 
        test_score = model.score(test_output, pred_test_output)
    except: 
        test_score = 0.0
    test_mse = model.score(test_output, pred_test_output, "mean_squared_error")
    test_rmse = rmse(train_output, pred_train_output)
    
    print(f"""
    Poisson deviation:
        score on training dataset: {train_score} / 1.0
        score on testing dataset: {test_score} / 1.0
    mean squared error:
        score on training dataset: {train_mse} / 1.0
        score on testing dataset: {test_mse} / 1.0
    raw rmse: 
        score on training dataset: {train_rmse}
        score on testing dataset: {test_rmse}
        mean of (test + train) outputs: {np.mean(lcoes)}
        """)
        
    raise KeyboardInterrupt()
    print("Starting sobol")
    
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
    
    # slice points out for fit
    num_points = input_data.shape[0] - (input_data.shape[0] % (input_data.shape[1]+2))
    print(num_points)
    sobol_data = input_data[:num_points, :]
    sobol_lcoes = lcoes[:num_points]
    
    sobol1 = calculate_first_order_sobol(sobol_data, sobol_lcoes)
    print(sobol1)
    
    num_points = test_input.shape[0] - (test_input.shape[0] % (test_input.shape[1]+2))
    print(num_points)
    sobol_data = test_input[:num_points, :]
    sobol_lcoes = test_predicted[:num_points]
    
    sobol2 = calculate_first_order_sobol(sobol_data, sobol_lcoes)
    print(sobol2)
    
    print(sobol2/sobol1)
