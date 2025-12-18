#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import json
#from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['text.usetex'] = True

import seaborn as sns
from scipy.stats import t
from typing import Callable, Dict, Tuple, List
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

class DataHandler:
    """
    Class to handle data loading, splitting, and preprocessing.

    This module is part of the research presented in the paper:
    "DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".
    author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de
    """

    def __init__(self, file_path: str):
        """
        Initialize the DataHandler with the path to the data file.

        Args:
            file_path (str): Path to the CSV file containing the dataset.
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        # Extract design category from 'Experiment' column
        self.data['Design_Category'] = self.data['Experiment'].apply(lambda x: x.split('_')[0])

    def get_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Get datasets split by design category.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of datasets.
        """
        datasets = {
            'Fastback_F': self.data[self.data['Design_Category'] == 'F'],
            'Combined_All': self.data
        }
        return datasets


# -------------------------------
# Geometry Encoder
# -------------------------------
class GeometryAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(16, latent_dim)
        self.logvar_layer = nn.Linear(16, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)      # Cd or another scalar property
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode input into latent distribution parameters
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        y = self.regressor(z)

        # Return all components for loss computation
        return y, z, x_hat, mu, logvar
    
def get_trained_surrogate_model(input_dim: int, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
    
    model = GeometryAE(input_dim=input_dim, latent_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1000):  # few epochs are enough for this assignment
        model.train()
        optimizer.zero_grad()

        y_pred, z, x_hat, mu, logvar = model(X_train)
        loss_reg = criterion(y_pred, y_train)
        loss_recon = criterion(x_hat, X_train)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = loss_reg + 0.001*loss_recon + 0.001*kl_loss

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch:02d}, Loss = {loss.item():.4f}")
    
    return model

def evaluate_surrogate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    
    model.eval()
    with torch.no_grad():
        y_pred, _, _, _, _ = model(X_test)
        mae = torch.mean(torch.abs(y_pred - y_test))
        print("MAE:", mae.item())
    
    plot_true_vs_pred(y_true=y_test.numpy(), y_pred=y_pred.numpy())
    

def visualise_latent_with_tsne(model, X_test_norm: np.ndarray, y_test: List[str], X_test: np.ndarray, exp: List[str]) -> None:
    
    model.eval()

    with torch.no_grad():
        # Encode only the test set
        Z_test = model.encoder(torch.tensor(X_test_norm, dtype=torch.float32))
        Z_test = Z_test.numpy()   # convert to NumPy for sklearn/TSNE

    tsne = TSNE(
        n_components=2,
        perplexity=20,
        learning_rate='auto',
        init='random',
        random_state=42
    )

    Z_tsne = tsne.fit_transform(Z_test)

    ysort = np.argsort(y_test)
    ymin1 = exp[X_test.index[ysort[1]]]
    ymax1 = exp[X_test.index[ysort[-2]]]
    ymin2 = exp[X_test.index[ysort[3]]]
    ymax2 = exp[X_test.index[ysort[-4]]]

    plt.figure(figsize=(5,4))
    plt.scatter(Z_tsne[:,0], Z_tsne[:,1],c=y_test, cmap='coolwarm', s=40)
    plt.colorbar(label="$c_d$")
    plt.title("t-SNE Projection of Latent Space")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    file_path = Path(__file__).resolve().parent
    plt.savefig(file_path/"latent_cluster.pdf", format="pdf", dpi=1200, bbox_inches="tight", pad_inches=0, transparent=False)


    plt.figure(figsize=(5,4))
    plt.scatter(Z_tsne[:,0], Z_tsne[:,1],c=y_test, cmap='coolwarm', s=40)
    plt.colorbar(label="$c_d$")
    plt.title("t-SNE Projection of Latent Space")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.scatter(Z_tsne[ysort[1],0], Z_tsne[ysort[1],1],
            marker='o', color='yellow', edgecolors='k', alpha=1,
            s=100, zorder=10)
    plt.scatter(Z_tsne[ysort[-2],0], Z_tsne[ysort[-2],1],
            marker='o', color='yellow', edgecolors='k', alpha=1,
            s=100, zorder=10)
    plt.scatter(Z_tsne[ysort[2],0], Z_tsne[ysort[2],1],
            marker='o', color='yellow', edgecolors='k', alpha=1,
            s=100, zorder=10)
    plt.scatter(Z_tsne[ysort[-3],0], Z_tsne[ysort[-3],1],
            marker='o', color='yellow', edgecolors='k', alpha=1,
            s=100, zorder=10)
    plt.tight_layout()
    plt.savefig(file_path/"latent_cluster_highlighted.pdf", format="pdf", dpi=1200, bbox_inches="tight", pad_inches=0, transparent=False)

def plot_true_vs_pred(y_true, y_pred, title=None,
                      xlabel="$c_d$ (True)", ylabel="$c_d$ (Predicted)",
                      annotate_metrics=True, log_scale=False,
                      dpi=1200, marker='o', color='blue'):
    """
    Plot y_true vs y_pred scatter, 1:1 line, and optional metrics.
    Accepts numpy arrays or torch tensors.
    """
    # Convert to numpy
    try:
        import torch
        if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    except Exception:
        pass

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"

    # metrics
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    # R^2 (simple)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    fig, ax = plt.subplots(figsize=(4,4))
    # scatter + 1:1 line
    ax.scatter(y_true, y_pred, marker=marker, color=color, alpha=0.3)
    vmin = min(np.nanmin(y_true), np.nanmin(y_pred))
    vmax = max(np.nanmax(y_true), np.nanmax(y_pred))
    pad = 0.02 * (vmax - vmin) if vmax > vmin else 0.1
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color='gray', linestyle='--', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if annotate_metrics:
        txt = f"RMSE: {rmse:.3g}\nMAE: {mae:.3g}\n$R^2$: {r2:.3g}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.9))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    plt.tight_layout()

    # Save before show; high-res PDF
    file_path = Path(__file__).resolve().parent
    plt.savefig(file_path/"true_vs_pred_scatter.pdf", format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0)

def main(file_path: str) -> None:
    """
    Main function to run the entire workflow.

    Args:
        file_path (str): Path to the input data file.
    """
    # Initialize handlers
    data_handler = DataHandler(file_path)
    datasets = data_handler.get_datasets()

    dataset = datasets['Combined_All']

    # Split the dataset into training and test sets
    X = dataset.drop(columns=['Experiment', 'Average Cd', 'Std Cd', 'Design_Category'])
    y = dataset['Average Cd']
    exp = dataset['Experiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    X_train_norm = scaler.transform(X_train)
    X_test_norm  = scaler.transform(X_test)
        
    model = get_trained_surrogate_model(
        input_dim=X_train_norm.shape[1], 
        X_train=torch.tensor(X_train_norm, dtype=torch.float32), 
        y_train=torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        )
    
    evaluate_surrogate_model(
        model, 
        X_test=torch.tensor(X_test_norm, dtype=torch.float32), 
        y_test=torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        )   
    
    visualise_latent_with_tsne(
        model, 
        X_test_norm=X_test_norm, 
        y_test=y_test.values,
        X_test = X_test, 
        exp = exp
        )   



file_path = Path(__file__).resolve().parent
file_path = file_path / 'DrivAerNet_ParametricData.csv'
main(file_path)
