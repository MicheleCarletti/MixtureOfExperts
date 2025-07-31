"""
@author Michele Carletti
A simple Mixture of Experts-based classifier for MNIST
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# --- Expert ---
class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Gating Network ---
class GatingNetwork(nn.Module):
    def __init__(self, num_experts, temp=1.0, noise_std=1.0, use_noise=True):
        super().__init__()
        self._use_noise = use_noise
        self._noise_std = noise_std
        self._temp = temp

        self.fc = nn.Sequential(
            nn.Linear(28*28, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        if self.training and self._use_noise:
            noise = torch.rand_like(logits) * self._noise_std # Gaussian noise
            logits = logits + noise
            
        return F.softmax(logits / self._temp, dim=1)

# --- MoE Classificatore ---
class MoE(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
        self.gating = GatingNetwork(num_experts, temp=10.0, noise_std=0.5)

    def forward(self, x):
        gating_weights = self.gating(x)  # (B, E)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, E, C)
        # Weighted sum: sum over experts
        #output = torch.bmm(expert_outputs, gating_weights.unsqueeze(2)).squeeze(2)  # (B, C)
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)  # (B, C)
        return output

# --- Load balance ---
def entropy_loss(gating_weights):
    # Gating weights: (B, E)
    return -torch.mean(torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=1))

# --- Training ---
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss_ce = criterion(out, y)
        loss_en = entropy_loss(model.gating(x))
        loss = loss_ce + 0.1 * loss_en     # Loss with load balancing among the experts
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

# --- Test ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

# --- Visualize routing path ---
def visualize_gating(model, dataloader, device, num_samples=20, cols=5):
    model.eval()
    x_batch, y_batch = next(iter(dataloader))
    x_batch = x_batch[:num_samples].to(device)

    with torch.no_grad():
        gating_weights = model.gating(x_batch).cpu().numpy()

    rows = math.ceil(num_samples / cols)
    fig, axs = plt.subplots(rows * 2, cols, figsize=(cols * 2.5, rows * 2.5))
    axs = axs.reshape(rows * 2, cols)

    for idx in range(num_samples):
        row = (idx // cols) * 2
        col = idx % cols

        # Immagine
        axs[row, col].imshow(x_batch[idx].cpu().squeeze(), cmap='gray')
        axs[row, col].axis('off')
        axs[row, col].set_title(f"Label: {y_batch[idx].item()}")

        # Gating weights
        axs[row + 1, col].bar(range(gating_weights.shape[1]), gating_weights[idx])
        axs[row + 1, col].set_ylim(0, 1)
        axs[row + 1, col].set_xticks(range(gating_weights.shape[1]))
        axs[row + 1, col].set_ylabel("Weight")
        axs[row + 1, col].set_xlabel("Expert")

    plt.tight_layout()
    plt.show()

# --- Main ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    train_only = False
    if train_only:
        model = MoE(num_experts=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(20):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, Test Acc = {test_acc:.4f}, Test Loss = {test_loss:.4f}")
        torch.save(model.state_dict(), './data/moe_trained.pth')
        print("Checkpoint saved!")
    else:
        ev_model = MoE(num_experts=3)
        ev_model.load_state_dict(torch.load("./data/moe_trained.pth", map_location=torch.device('cpu')))
        ev_model.to(device)
        visualize_gating(ev_model, test_loader, device)

if __name__ == "__main__":
    main()
