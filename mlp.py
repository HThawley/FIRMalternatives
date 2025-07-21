# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 19:41:39 2025

@author: u6942852
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime as dt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
from numba import njit
from time import perf_counter

#%%
from Input import (scenario, DClengths, undersea_mask, network_mask, Raw_Costs, lb, ub)
from Costs import Raw_Costs 
from Optimisation import Optimise, Objective
from ParameterSweep import calculate_costs, deduplicate_history
from Timekeeper import keeptime, PrintTimekeeper, timekeeper

np.set_printoptions(suppress=True)
raw_costs = Raw_Costs(scenario, DClengths, undersea_mask, network_mask)
costs = raw_costs.CostFactors()

#%%
@njit
def rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error between true and predicted values.
    This function is JIT-compiled with Numba for performance.
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))

class MLPmodel:
    """
    A wrapper class for a scikit-learn MLPRegressor to create a surrogate model.

    This class handles model training, prediction, evaluation, and persistence
    (saving/loading)
    """
    def __init__(self, model_path: str = None):
        """
        Initializes the MLPmodel.

        Args:
            model_path (str, optional): If provided, loads a pre-trained model
                                        from this path. Defaults to None.
        """
        self._is_trained = False
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.num_inputs = 0

        if model_path is not None:
            self.load_model(model_path)

    def _data_assertions(self, X, y):
        """
        Validates the shape and type of input and output data.
        """
        assert isinstance(X, np.ndarray), "Input (X) should be a 2D numpy array"
        assert X.ndim == 2, "Input (X) should be a 2D numpy array"
        assert isinstance(y, np.ndarray), "Output (y) should be a 1D numpy array"
        assert y.ndim == 1, "Output (y) should be a 1D numpy array"
        assert X.shape[1] == self.num_inputs, f"Input features should be {self.num_inputs}"
        assert X.shape[0] == y.shape[0], "Input (N, M) and output (N,) shapes should match"
    
    def _create_scaler(self, X):
        scaler = StandardScaler()
        if self.scaler_mean is not None and self.scaler_scale is not None:
            scaler.mean_ = self.scaler_mean
            scaler.scale_ = self.scaler_scale
            scaler.n_features_in_ = len(self.scaler_mean)
        else: 
            scaler.fit(X)
            self.scaler_mean = scaler.mean_ 
            self.scaler_scale = scaler.scale_ 
        return scaler
    
    def preprocess(self, X):
        scaler = self._create_scaler(X)
        X_scaled = scaler.transform(X)
        return X_scaled

    def train(
            self, 
            X, 
            y,
            verbose=True, 
            **mlp_params, 
            ):
        """
        Trains the MLP surrogate model.

        Args:
            X_train (np.ndarray): The (N, M) array of training input features.
            y_train (np.ndarray): The (N,) array of training output values.
            bounds (tuple): A tuple containing the lower (lb) and upper (ub)
                            bounds of the input features.
            mlp_params (dict, optional): Dictionary of parameters to pass to
                                         MLPRegressor. Defaults to a standard configuration.
            verbose (bool, optional): If True, prints training progress. Defaults to True.
        """
        self.num_inputs = X.shape[1]
        self._data_assertions(X, y)

        start = dt.now()
        if verbose:
            print(f"Starting training at: {start}")

        # 1. Preprocessing: Scale the input data
        if verbose:
            print("Fitting scaler and transforming training data...")
        X_scaled = self.preprocess(X)

        self.mlp_params = mlp_params
        self.model = MLPRegressor(**mlp_params)

        # 3. Model Training
        if verbose:
            print("Fitting MLP Regressor...")
        self.model.fit(X_scaled, y)

        self._is_trained = True
        if verbose:
            print(f"Finished training successfully. Time taken: {dt.now() - start}")
            print(f"Final loss: {self.model.loss_}")
            print(f"Number of iterations: {self.model.n_iter_}")

    def predict(self, X):
        """
        Makes predictions on new data using the trained model.

        Args:
            X (np.ndarray): The (N, M) array of input features for prediction.

        Returns:
            np.ndarray: The (N,) array of predicted values.
        """
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet. Call .train() first.")
        
        X_scaled = self.preprocess(X)        
        return self.model.predict(X_scaled)

    def score(self, y_true, y_pred, metric="r2_score"):
        """
        Evaluates the model's performance.

        Args:
            y_true (np.ndarray): The true output values.
            y_pred (np.ndarray): The predicted output values.
            metric (str, optional): The metric to use. Can be "r2_score" or
                                    "mean_squared_error". Defaults to "r2_score".

        Returns:
            float: The calculated score.
        """
        if metric == "r2_score":
            return r2_score(y_true, y_pred)
        elif metric == "mean_squared_error":
            return mean_squared_error(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def save_model(self, filepath: str, overwrite=False):
        """
        Saves the trained model and scaler to a file using json.

        Args:
            filepath (str): The path to save the model file.
            overwrite (bool, optional): If False, raises an error if the file
                                        already exists. Defaults to False.
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        if not overwrite and os.path.exists(filepath):
            raise FileExistsError(
                f"File '{filepath}' already exists. Pass overwrite=True to replace it."
            )

        metadata = {}
        metadata["num_inputs"] = self.num_inputs
        metadata["scaler_mean"] = self.scaler_mean.tolist()
        metadata["scaler_scale"] = self.scaler_scale.tolist()
        
        for k, v in metadata.items():
            assert v is not None, f"Cannot save an untrained model. ({k} is None)"
        
        for k, v in self.mlp_params.items():
            metadata[k] = v
        
        try: 
            layers = [layer.shape for layer in self.model.intercepts_][:-1]
            metadata["layers"] = layers
            for i in range(len(layers)+1):
                metadata[f"intercepts_{i}"] = self.model.intercepts_[i].tolist()
                metadata[f"coefs_{i}"] = self.model.coefs_[i].tolist()
        except Exception as E:
            raise E("Model not trained properly. Cannot save")

        with open(filepath+".json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Model saved successfully to {filepath}")

    def load_model(self, filepath: str, verbose=True):
        """
        Loads a model and scaler from a file.

        Args:
            filepath (str): The path to the model file.
            verbose (bool, optional): If True, prints loading status. Defaults to True.
        """
        start = dt.now()
        if verbose:
            print(f"Loading model from {filepath}...")

        with open(filepath+".json", "r") as f:
            metadata = json.load(f)

        self.num_inputs = metadata["num_inputs"] 
        self.scaler_mean = np.array(metadata["scaler_mean"])
        self.scaler_scale = np.array(metadata["scaler_scale"])

        layers = [tuple(layer) for layer in metadata["layers"]]
        
        mlp_params = {k: v for k, v in metadata.items() if 
                      (k not in ("num_inputs", "scaler_mean", "scaler_scale", "layers", "hidden_layer_sizes"))
                      and ("coefs_" not in k) and ("intercepts_" not in k)}
        
        self.model = MLPRegressor(
            hidden_layer_sizes = tuple(layers), 
            **mlp_params,
            )
        
        self.model.coefs_ = [np.array(metadata[f"coefs_{i}"]) for i in range(len(layers) + 1)]
        self.model.intercepts_ = [np.array(metadata[f"intercepts_{i}"]) for i in range(len(layers) + 1)]

        self._is_trained = True
        
        if verbose:
            print(f"Model loaded successfully. Time taken: {dt.now() - start}")


if __name__ == "__main__":
    STEP = 1
    START = 0
    PREC = 2
    MODEL_FILE_PATH = f"mlp-full-s{STEP}-s{START}-p{PREC}"

    # CSV_FILE_PATH = "Results/Firmpoints-dedup{PREC}.csv"
    CSV_FILE_PATH = "Results/firmpoints.csv"


    input_data = pd.read_csv(
        CSV_FILE_PATH, 
        # skiprows = 4_000_000,
        # nrows=20_000, 
        header=None,
        )
    
    # input_data = deduplicate_history(input_data, commit=False, precision=2, subset=list(range(15, input_data.shape[1])))
    # input_data.to_csv("Results/Firmpoints-dedup2.csv", header=False, index=False)
    input_data= input_data.to_numpy()
    
    input_data = input_data[START::STEP, :]
    print(input_data.shape)
    og_shape = input_data.shape
    rng = np.random.default_rng(seed=1)
    rng.shuffle(input_data)
    
    lcoes = calculate_costs(input_data, costs)
    input_data = input_data[:, 15:]
    
    cutoff = int(0.90*len(lcoes))
    
    Y_test = lcoes[cutoff:]
    X_test = input_data[cutoff:, :]
    
    Y_train = lcoes[:cutoff]
    X_train = input_data[:cutoff, :]

    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")


    # --- Model Training or Loading ---
    model = MLPmodel()
    if False: # os.path.exists(MODEL_FILE_PATH):
        print("Found existing model. Loading it.")
        model.load_model(MODEL_FILE_PATH)
    else:
        print("No existing model found. Training a new one.")
        # These parameters are a good starting point but may need tuning
        mlp_hyperparams = {
            "loss": "poisson",
            'hidden_layer_sizes': (128, 64),
            'activation': 'tanh',
            'solver': 'adam',
            'alpha': 0.0001,
            'max_iter': 500,
            'early_stopping': True,
            'n_iter_no_change': 50, # Stop if validation score doesn't improve
            'verbose': True,
        }
        model.train(X_train, Y_train, mlp_params=mlp_hyperparams)
        model.save_model(MODEL_FILE_PATH, overwrite=True)

    # --- Evaluation ---
    print("\n--- Model Evaluation ---")
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    # Evaluate on the training set
    start = perf_counter()
    pred_train = model.predict(X_train)
    end = perf_counter()
    print(f"Time to evaluate {X_train.shape[0]} solutions: {(1000*(end-start)):.4f} ms")
    print(f"    ({(1_000_000*(end-start)/X_train.shape[0])} micro_s solution)")
    train_r2 = model.score(Y_train, pred_train, "r2_score")
    train_mse = model.score(Y_train, pred_train, "mean_squared_error")
    train_rmse = rmse(Y_train, pred_train)

    # Evaluate on the testing set
    start = perf_counter()
    pred_test = model.predict(X_test)
    end = perf_counter()
    print(f"Time to evaluate {X_test.shape[0]} solutions: {(1000*(end-start)):.4f}. ms")
    print(f"    ({(1_000_000*(end-start)/X_test.shape[0]):.4f} micro_s per solution)")
    test_r2 = model.score(Y_test, pred_test, "r2_score")
    test_mse = model.score(Y_test, pred_test, "mean_squared_error")
    test_rmse = rmse(Y_test, pred_test)

    print(f"""
    R-squared (R²):
        Training set: {train_r2:.6f}
        Testing set:  {test_r2:.6f}

    Mean Squared Error (MSE):
        Training set: {train_mse:.6f}
        Testing set:  {test_mse:.6f}

    Root Mean Squared Error (RMSE):
        Training set: {train_rmse:.6f}
        Testing set:  {test_rmse:.6f}
    
    Statistics of the target variable ('lcoe'):
        Mean:     {np.mean(lcoes):.4f}
        Std Dev:  {np.std(lcoes):.4f}
    """)


