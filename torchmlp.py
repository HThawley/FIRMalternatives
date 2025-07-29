# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 15:41:00 2025

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
from pathlib import Path
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
    For multi-output, this computes the overall RMSE across all values.
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))

# --- PyTorch Model Definition ---
class _Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer_sizes, alpha):
        super(_Net, self).__init__()
        layers = []
        input_size = num_inputs
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU(alpha)) # Using Leaky ReLU 
            input_size = hidden_size
        layers.append(nn.Linear(input_size, num_outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLPmodel:
    """
    A PyTorch-based wrapper class to create a surrogate model.
    This class handles model training, prediction, evaluation, and persistence.
    """
    def __init__(self, model_path: str = None):
        self._is_trained = False
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.num_inputs = None
        self.num_outputs = None
        self.hidden_layer_sizes = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if model_path is not None:
            self.load_model(model_path)

    def _data_assertions(self, X, y):
        assert isinstance(X, np.ndarray), "Input (X) should be a 2D numpy array"
        assert X.ndim == 2, "Input (X) should be a 2D numpy array"
        assert isinstance(y, np.ndarray), "Output (y) should be a 2D numpy array"
        assert y.ndim == 2, "Output (y) must be a 2D numpy array."
        assert X.shape[1] == self.num_inputs, "Input feature count mismatch"
        assert y.shape[1] == self.num_outputs, "Output feature count mismatch"
        assert X.shape[0] == y.shape[0], "Input and output must have the same number of samples"

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
        return scaler.transform(X)

    def train(self, X, y, validation_split=0.1, verbose=True, **train_params):
        self.num_inputs = X.shape[1]
        self.num_outputs = y.shape[1]
        
        try: 
            train_params["hidden_layer_sizes"] = tuple(
                (int(self.num_inputs * abs(multiple)) if multiple < 0 else multiple) 
                for multiple in train_params["hidden_layer_sizes"])
        except KeyError:
            pass
        
        # Extract training parameters with defaults
        epochs = train_params.get('epochs', 300)
        batch_size = train_params.get('batch_size', 256)
        learning_rate = train_params.get('learning_rate', 0.001)
        self.hidden_layer_sizes = train_params.get('hidden_layer_sizes', (128, 64))
        patience = train_params.get('patience', 50) # For early stopping
        self.alpha = train_params.get('alpha', 50) # For early stopping

        # --- Data Preparation ---
        X_scaled = self.preprocess(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Create validation set
        dataset = TensorDataset(X_tensor, y_tensor)
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # --- Model Initialization ---
        self.model = _Net(self.num_inputs, self.num_outputs, self.hidden_layer_sizes, self.alpha).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # --- Training Loop ---
        start = dt.now()
        if verbose: print(f"Starting training on {self.device} at: {start}")
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model state
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose: print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load the best model state found during training
        self.model.load_state_dict(best_model_state)
        self._is_trained = True
        if verbose: print(f"Finished training. Time taken: {dt.now() - start}")

    def predict(self, X):
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet. Call .train() first.")
        
        X_scaled = self.preprocess(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()

    def score(self, y_true, y_pred, metric="r2_score"):
        if metric == "r2_score":
            return r2_score(y_true, y_pred, multioutput='uniform_average')
        elif metric == "mean_squared_error":
            return mean_squared_error(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def save_model(self, filepath: str, overwrite=False):
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        
        filepath_with_ext = filepath if filepath.endswith(".pt") else f"{filepath}.pt"
        path = Path(filepath_with_ext)

        if not overwrite and path.exists():
            raise FileExistsError(f"File '{path}' already exists. Pass overwrite=True.")
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_scale': self.scaler_scale,
            'num_inputs': self.num_inputs,
            'num_outputs': self.num_outputs,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'alpha':self.alpha,
        }
        torch.save(state, path)
        print(f"Model saved successfully to {path}")

    def load_model(self, filepath: str, verbose=True):
        filepath_with_ext = filepath if filepath.endswith(".pt") else f"{filepath}.pt"
        if not os.path.exists(filepath_with_ext):
            raise FileNotFoundError(f"Model file not found at {filepath_with_ext}")

        start = dt.now()
        if verbose: print(f"Loading model from {filepath_with_ext}...")

        state = torch.load(filepath_with_ext, map_location=self.device, weights_only=False)
        
        self.num_inputs = state['num_inputs']
        self.num_outputs = state['num_outputs']
        self.hidden_layer_sizes = state['hidden_layer_sizes']
        self.alpha = state['alpha']
        self.scaler_mean = state['scaler_mean']
        self.scaler_scale = state['scaler_scale']

        self.model = _Net(self.num_inputs, self.num_outputs, self.hidden_layer_sizes).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        
        self._is_trained = True
        if verbose: print(f"Model loaded successfully. Time taken: {dt.now() - start}")
#%%

if __name__ == "__main__":
    STEP = 1
    START = 0
    MODEL_FILE_PATH = f"MLP_models/mlp-pt-s{STEP}-s{START}"

    CSV_FILE_PATH = "Results/firmpoints.csv"

    input_data = pd.read_csv(
        CSV_FILE_PATH, 
        # skiprows = 4_000_000,
        # nrows=20_000, 
        header=None,
        )
    
    input_data= input_data.to_numpy()
    
    input_data = input_data[START::STEP, :]
    print(input_data.shape)
    og_shape = input_data.shape
    rng = np.random.default_rng(seed=1)
    rng.shuffle(input_data)
    
    output_data = input_data[:, np.array([0, 2])]
    input_data = input_data[:, 16:]
    
    cutoff = int(0.90*len(input_data))
    
    Y_test = output_data[cutoff:, :]
    X_test = input_data[cutoff:, :]
    
    Y_train = output_data[:cutoff, :]
    X_train = input_data[:cutoff, :]
    del input_data
    
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- Model Training or Loading ---
    model = MLPmodel()
    if os.path.exists(MODEL_FILE_PATH + '.pt'):
        print("Found existing model. Loading it.")
        model.load_model(MODEL_FILE_PATH)
    else:
        print("No existing model found. Training a new one.")
        # PyTorch-specific hyperparameters
        pytorch_params = {
            'hidden_layer_sizes': (-2, -2),
            'epochs': 1000,
            'batch_size': 512,
            'learning_rate': 0.001,
            'alpha':0.001,
            'patience': 75, # For early stopping
        }
        model.train(X_train, Y_train, **pytorch_params)
        model.save_model(MODEL_FILE_PATH, overwrite=True)
#%%
        
    # --- Evaluation ---
    print("\n--- Model Evaluation ---")
    
    # Evaluate on the training set
    start = perf_counter()
    pred_train = model.predict(X_train)
    end = perf_counter()
    print(f"Time to evaluate {X_train.shape[0]} solutions: {(1000*(end-start)):.4f} ms")
    print(f"    ({(1_000_000*(end-start)/X_train.shape[0]):.4f} micro_s per solution)")
    train_r2 = model.score(Y_train, pred_train, "r2_score")
    train_mse_cost = model.score(Y_train[:, 0], pred_train[:, 0], "mean_squared_error")
    train_mse_pen  = model.score(Y_train[:, 1], pred_train[:, 1], "mean_squared_error")
    train_rmse_cost = rmse(Y_train[:, 0], pred_train[:, 0])
    train_rmse_pen  = rmse(Y_train[:, 1], pred_train[:, 1])

    # Evaluate on the testing set
    start = perf_counter()
    pred_test = model.predict(X_test)
    end = perf_counter()
    print(f"Time to evaluate {X_test.shape[0]} solutions: {(1000*(end-start)):.4f} ms")
    print(f"    ({(1_000_000*(end-start)/X_test.shape[0]):.4f} micro_s per solution)")
    test_r2 = model.score(Y_test, pred_test, "r2_score")
    test_mse_cost = model.score(Y_test[:, 0], pred_test[:, 0], "mean_squared_error")
    test_mse_pen  = model.score(Y_test[:, 1], pred_test[:, 1], "mean_squared_error")
    test_rmse_cost = rmse(Y_test[:, 0], pred_test[:, 0])
    test_rmse_pen = rmse(Y_test[:, 1], pred_test[:, 1])


    print(f"""
    R-squared (R²):
        Training set: {train_r2:.6f}
        Testing set:  {test_r2:.6f}

    Mean Squared Error Cost (MSE): 
        Training set: {train_mse_cost:.6f}
        Testing set:  {test_mse_cost:.6f}
        
    Mean Squared Error Penalties (MSE):
        Training set: {train_mse_pen:.6f}
        Testing set:  {test_mse_pen:.6f}

    Root Mean Squared Error Cost (RMSE):
        Training set: {train_rmse_cost:.6f}
        Testing set:  {test_rmse_cost:.6f}
    
    Root Mean Squared Error Penalties (RMSE):
        Training set: {train_rmse_pen:.6f}
        Testing set:  {test_rmse_pen:.6f}
    
    Statistics of Cost:
        Mean:     {np.mean(Y_train[:, 0]):.4f}
        Std Dev:  {np.std(Y_train[:, 0]):.4f}
        
    Statistics of Penalties:
        Mean:     {np.mean(Y_train[:, 1]):.4f}
        Std Dev:  {np.std(Y_train[:, 1]):.4f}
    """)
