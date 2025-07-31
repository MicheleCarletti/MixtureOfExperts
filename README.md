# Mixture of Experts (MoE) Models

This repository provides two implementations of Mixture of Experts (MoE) models:

1. A deep MoE model trained on the MNIST dataset using PyTorch.
2. A reproduction of the original MoE paper by Jacobs et al. (1991), using synthetic F1/F2 vowel classification.

---

## 1. MoE for MNIST Classification

**File**: `moe.py`

A modular neural MoE architecture for digit classification using the MNIST dataset. It includes:
- Multiple feedforward experts (MLPs)
- A gating network with optional noise and temperature control
- Entropy-based load balancing for expert utilization
- Visualization of input images and their expert routing weights

### Features
- Adjustable number of experts
- Temperature-scaled softmax gating
- Optional noise during training for exploration
- Visualization of expert usage across inputs

### Usage
Set `train_only = True` to train and save the model, or `False` to load a saved checkpoint and visualize results.

# 2. Original MoE Reproduction (Jacobs et al., 1991)

**File:** `moe_1991.py`

---

## Description

This implementation reproduces the Mixture of Experts (MoE) model from the original paper by Jacobs et al. (1991). The model is applied to a synthetic 2D vowel classification task, where vowels are represented by their first two formants (F1 and F2).

---

## Features

- **2D input data** with 4 vowel classes: `/i/`, `/I/`, `/a/`, `/A/`.
- **7 experts**, each a shallow neural network.
- **Tanh activations**, consistent with the original paper.
- **Gating network** produces soft assignments of inputs to experts.
- **Visualization** of gating decisions, showing dominant expert regions in the input space.

---

## Output

The script generates a 2D plot showing:
- The dominant expert selected for each region of the input space.
- Scatter plot of the synthetic vowel data colored by class.

---
