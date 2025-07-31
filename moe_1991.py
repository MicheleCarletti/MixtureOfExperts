"""
@author Michele Carletti
Reproduction of original MoE paper from Jacobs et al. (1991)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset. Synthetic data 4-class F1/F2 vowel
def generate_syn_vowels(n_per_class=100):
    np.random.seed(0)
    vowels = {
        '/i/': [300, 2500],
        '/I/': [400, 2000],
        '/a/': [700, 1200],
        '/A/': [800, 1000]        
    }
    X, y = [], []
    for label, (f1_mean, f2_mean) in vowels.items():
        f1 = np.random.normal(f1_mean, 40, n_per_class)
        f2 = np.random.normal(f2_mean, 100, n_per_class)
        X.extend(np.stack([f1, f2], axis=1))
        y.extend([label]*n_per_class)
    return np.array(X), np.array(y)

# --- Expert ---
class Expert(nn.Module):
    def __init__(self, in_dim=2, hidden=5, out_dim=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.fc(x)

# --- Gating Network ---
class GatingNet(nn.Module):
    def __init__(self, in_dim=2, hidden=5, num_experts=7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_experts)
        )
    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

# --- Mixture of Experts ---
class MoE(nn.Module):
    def __init__(self, n_experts=7):
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(n_experts)])
        self.gate = GatingNet(num_experts=n_experts)

    def forward(self, x):
        gate_weights = self.gate(x)     # (B, E)
        experts_out = torch.stack([expert(x) for expert in self.experts], dim=1)    # (B, E, C)
        gated_out = torch.sum(gate_weights.unsqueeze(-1) * experts_out, dim=1)  # (B, C)
        return gated_out, gate_weights

# Training loop
def train(X_train, y_train, model, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        logits, gates = model(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = (logits.argmax(dim=1) == y_train).float().mean().item()
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

# Main
def main():
    # Create dataset
    X, y = generate_syn_vowels()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))

    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.long),
        test_size=0.2, random_state=42
    )

    # Train the model
    model = MoE(n_experts=7)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    train(X_train, y_train, model, opt, nn.CrossEntropyLoss())

    # Visualize expert usage
    model.eval()
    with torch.no_grad():
        grid_x, grid_y = np.meshgrid(np.linspace(-2.5, 4, 200), np.linspace(-2.5, 4, 200))
        grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        _, gate_weights = model(grid_tensor)
        dominant_expert = gate_weights.argmax(dim=1).numpy().reshape(grid_x.shape)

        plt.contourf(grid_x, grid_y, dominant_expert, alpha=0.3, levels=np.arange(8)-0.5, cmap='tab10')
        plt.scatter(X[:, 0], X[:, 1], c=y_encoded, cmap='tab10', s=10, edgecolors='k')
        plt.title("Dominant Expert Regions (Gating Network)")
        plt.xlabel("F1 (normalized)")
        plt.ylabel("F2 (normalized)")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()