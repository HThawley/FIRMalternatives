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
from scipy.stats import spearmanr
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
def rmse_score(y_true, y_pred):
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
        weight_decay = train_params.get('weight_decay', 1e-5) # L2 Regularization
        self.alpha = train_params.get('alpha', 0.001) # L2 Regularization
        
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

        self.model = _Net(self.num_inputs, self.num_outputs, self.hidden_layer_sizes, self.alpha).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        
        self._is_trained = True
        if verbose: print(f"Model loaded successfully. Time taken: {dt.now() - start}")
#%%

if __name__ == "__main__":
    MODEL_FILE_PATH = "MLP_models/mlp-nopt2"

    CSV_FILE_PATH = "Results/firmpoints.csv"

    input_data = pd.read_csv(
        CSV_FILE_PATH, 
        header=None,
        )
    
    input_data= input_data.to_numpy()

    ### Justification of training data trimming.
    # Divide FIRM solutions into 3 regions: 
    #     * Highly overbuilt: very large capex, 
    #                         very low opex, 
    #                         almost always technically feasible, 
    #                         penalties = 0,
    #           -> FIRM is approximately linear
    #     * Near optimal: (usually) large capex, 
    #                     (usually) low opex, 
    #                     technical feasibility is highly variable, 
    #                     penalties is 0 or relatively low
    #           -> FIRM is highly non-linear and peaky
    #     * Highly underbuilt: very low capex, 
    #                          very high or very low opex, 
    #                          almost always infeasible, 
    #                          penalties is large
    #           -> FIRM is largely monotonic
    # 
    # We want the mlp to have similar topology to the firm function overall. 
    # Building off these conceptions:
    #    * A low degree of surrogate complexity is needed to model the 
    #             overbuilt and underbuilt regions. 
    #    * A very high degree of surrogate complexity is needed to model the 
    #             near-optimal region
    # 
    #  -> we should train the model on many many many more near-optimal points than under/overbuilt

    # arbitrarily pick cost slack
    upper_cost_slack = 0.30
    upper_cost_slack += 1
    lower_cost_slack = 0.15
    lower_cost_slack += 1
    objective = input_data[:, 0] + input_data[:, 2] # cost + penalties
    optimum = objective.min()
    upper = optimum * upper_cost_slack
    lower = optimum / lower_cost_slack
    
    preds = [
        "cost", 
        "penalties", 
        ]
    # Update below based on preds
    output_data = input_data[:, np.array([0, 2])] 
    # output_data = input_data[:, 0] # cost only
    # output_data = input_data[:, 2] # penalties only
    
    input_data = input_data[:, 16:] # trim excess statistics
    og_shape = input_data.shape
    
    sort_cost = np.argsort(objective)
    near_optimal_idx = np.where(objective[sort_cost] < upper and 
                                objective[sort_cost] > lower)[0] # should be cost 
    
    print("full input data:", input_data.shape)
    near_optimal_input = input_data[sort_cost[:near_optimal_idx], :]
    non_optimal_input = input_data[sort_cost[near_optimal_idx:], :]
    del input_data
    near_optimal_output = output_data[sort_cost[:near_optimal_idx], :]
    non_optimal_output = output_data[sort_cost[near_optimal_idx:], :]

    print("Training & validating on near-optimal data. Testing on all data")
    rng = np.random.default_rng(seed=1)
    shuffleidx = np.arange(len(near_optimal_input))
    rng.shuffle(shuffleidx)
    
    near_optimal_input = near_optimal_input[shuffleidx]
    near_optimal_output = near_optimal_output[shuffleidx]
    del shuffleidx
    
    cutoff = int(0.90*len(near_optimal_input))
    
    Y_test = near_optimal_output[cutoff:, :]
    X_test = near_optimal_input[cutoff:, :]
    
    Y_train = near_optimal_output[:cutoff, :]
    X_train = near_optimal_input[:cutoff, :]
    
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- Model Training or Loading ---
    
    pytorch_params = {
        'hidden_layer_sizes': (-2, -2),
        'epochs': 1000,
        'batch_size': 512,
        'learning_rate': 0.001,
        'alpha':0.001,
        'patience': 75, # For early stopping
        'weight_decay':1e-5, # L2 regularization
    }
    
#%%
        
    # --- Evaluation ---
    print("\n--- Model Evaluation ---")
    
    printstr=f"""
full input data: {og_shape}
near-optimal +{int(100*(upper_cost_slack-1)):.0f}%/-{int(100*(lower_cost_slack)):.0f}% data: {near_optimal_input.shape}
non-optimal +{int(100*(upper_cost_slack-1)):.0f}%/-{int(100*(lower_cost_slack)):.0f}% data: {non_optimal_input.shape}

Train set size: {X_train.shape[0]}
Test set size: {X_test.shape[0]}
Non-optimal set size: {non_optimal_input.shape[0]}

Training & validating on near-optimal data. Testing on all data
"""
    
   
    def evaluate_and_score(model, n, y_true, X):
        start = perf_counter()
        y_pred = model.predict(X).flatten() # one value only
        end = perf_counter()
        mse = model.score(y_true[:, n], y_pred, "mean_squared_error")
        rmse = rmse_score(y_true[:, n], y_pred)
        spea = spearmanr(y_true[:, n], y_pred)
        r2 = model.score(y_true[:, n], y_pred, "r2_score")
        return end-start, mse, rmse, spea, r2
    
    for n, pred in enumerate(preds):
        if os.path.exists(f"{MODEL_FILE_PATH}-{pred}.pt"):
            print("Found existing cost model. Loading it.")
            model = MLPmodel()
            model.load_model(f"{MODEL_FILE_PATH}-{pred}")
        else: 
            print(f"No existing {pred} model found. Training a new one.")
            model = MLPmodel()
            model.train(X_train, np.atleast_2d(Y_train[:, n]).T, **pytorch_params) 
            model.save_model(f"{MODEL_FILE_PATH}-{pred}", overwrite=True)
    
        train_stats = evaluate_and_score(model, n, Y_train, X_train)
        test_stats = evaluate_and_score(model, n, Y_test, X_test)
        nonopt_stats = evaluate_and_score(model, n, non_optimal_output, non_optimal_input)
    
        printstr += f"""
Evaluation time on {pred}:
    Training: {1000*train_stats[0]:.2f} ms  | {1_000_000*train_stats[0]/X_train.shape[0]:.2f} micro sec per 1
    Testing:  {1000*test_stats[0]:.2f} ms  | {1_000_000*test_stats[0]/X_test.shape[0]:.2f} micro sec per 1
    Non-opt:  {1000*nonopt_stats[0]:.2f} ms  | {1_000_000*nonopt_stats[0]/non_optimal_input.shape[0]:.2f} micro sec per 1
    
Statistics of {pred}:
    near-optimal: 
            Mean:     {np.mean(near_optimal_output[:, n]):.4f}
            Std Dev:  {np.std(near_optimal_output[:, n]):.4f}
            Sparsity: {np.isclose(near_optimal_output[:, n], 0).sum()} / {near_optimal_output.shape[0]} zeros
        training:
            Mean:     {np.mean(Y_train[:, n]):.4f}
            Std Dev:  {np.std(Y_train[:, n]):.4f}
            Sparsity: {np.isclose(Y_train[:, n], 0).sum()} / {Y_train.shape[0]} zeros
        testing:
            Mean:     {np.mean(Y_test[:, n]):.4f}
            Std Dev:  {np.std(Y_test[:, n]):.4f}
            Sparsity: {np.isclose(Y_test[:, n], 0).sum()} / {Y_test.shape[0]} zeros
    non-optimal: 
        Mean:     {np.mean(non_optimal_output[:, n]):.4f}
        Std Dev:  {np.std(non_optimal_output[:, n]):.4f}
        Sparsity: {np.isclose(non_optimal_output[:, n], 0).sum()} / {non_optimal_output.shape[0]} zeros
    
{pred} - Mean Squared Error (MSE): 
    Training set: {train_stats[1]:.6f}  ({100*train_stats[1]/np.mean(Y_train[:, n]):.4f}% | true_mean={np.mean(Y_train[:, n]):.4f})
    Testing set:  {test_stats[1]:.6f}  ({100*test_stats[1]/np.mean(Y_test[:, n]):.4f}% | true_mean={np.mean(Y_test[:, n]):.4f})
    non-optimal:  {nonopt_stats[1]:.6f}  ({100*nonopt_stats[1]/np.mean(non_optimal_output[:, n]):.4f}% | true_mean={np.mean(non_optimal_output[:, n]):.4f})

{pred} - Root Mean Squared Error Cost (RMSE):
    Training set: {train_stats[2]:.6f}  ({100*train_stats[2]/np.mean(Y_train[:, n]):.4f}% | true_mean={np.mean(Y_train[:, n]):.4f})
    Testing set:  {test_stats[2]:.6f}  ({100*test_stats[2]/np.mean(Y_test[:, n]):.4f}% | true_mean={np.mean(Y_test[:, n]):.4f})
    non-optimal:  {nonopt_stats[2]:.6f}  ({100*nonopt_stats[2]/np.mean(non_optimal_output[:, n]):.4f}% | true_mean={np.mean(non_optimal_output[:, n]):.4f})

{pred} - spearman rank correlation:
    Training set: {train_stats[3][0]:.6f} (pvalue: {train_stats[3][1]})
    Testing set:  {test_stats[3][0]:.6f} (pvalue: {test_stats[3][1]})
    non-optimal:  {nonopt_stats[3][0]:.6f} (pvalue: {nonopt_stats[3][1]})

{pred} - R-squared (R²):
    Training set: {train_stats[4]:.6f} 
    Testing set:  {test_stats[4]:.6f}
    non-optimal:  {nonopt_stats[4]:.6f}
"""
        

        
        
    print(printstr)

    with open(MODEL_FILE_PATH+"-stats.txt", "w") as file:
        print(printstr, file=file)