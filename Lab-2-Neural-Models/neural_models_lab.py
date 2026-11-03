"""Neural Models lab: XOR, gradients, symmetry, activations, and 3-class output.

Run with: python neural_models_lab.py
Dependencies: torch, numpy (matplotlib is optional and only needed for plotting).
"""
from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn

SEED = 0
STEPS = 5000
LR = 0.03


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_BINARY = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


class BinaryXOR(nn.Module):
    """Exactly 2 inputs -> 2 hidden units -> 1 logit."""
    def __init__(self, activation: str = "tanh") -> None:
        super().__init__()
        activations = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.fc1 = nn.Linear(2, 2)
        self.activation = activations[activation]()
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


def train_binary(activation: str = "tanh", seed: int = SEED, steps: int = STEPS):
    set_seed(seed)
    model = BinaryXOR(activation)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    with torch.no_grad():
        initial_loss = criterion(model(X), Y_BINARY).item()
    first_layer_grad = None
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y_BINARY)
        loss.backward()
        if step == 0:
            first_layer_grad = model.fc1.weight.grad.detach().clone()
        optimizer.step()
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits)
        final_loss = criterion(logits, Y_BINARY).item()
        predictions = (probabilities >= 0.5).float()
    return model, initial_loss, final_loss, probabilities, predictions, first_layer_grad


def symmetry_experiment():
    """Zero all parameters and show that hidden units remain indistinguishable."""
    model = BinaryXOR("tanh")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    loss = nn.BCEWithLogitsLoss()(model(X), Y_BINARY)
    loss.backward()
    return model.fc1.weight.grad.detach().clone(), model.fc1.bias.grad.detach().clone()


# Class 0=(0,0), class 1=disagreement, class 2=(1,1).
Y_MULTI = torch.tensor([0, 1, 1, 2], dtype=torch.long)


class ThreeClassModel(nn.Module):
    """2 inputs -> 2 hidden units -> 3 unnormalized class logits."""
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


def train_multiclass(seed: int = SEED, steps: int = STEPS):
    set_seed(seed)
    model = ThreeClassModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    with torch.no_grad():
        initial_loss = criterion(model(X), Y_MULTI).item()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(X), Y_MULTI)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        final_loss = criterion(logits, Y_MULTI).item()
        accuracy = (logits.argmax(dim=1) == Y_MULTI).float().mean().item()
        shifted_probabilities = torch.softmax(logits + 100.0, dim=1)
    return model, initial_loss, final_loss, logits, probabilities, accuracy, shifted_probabilities


def main() -> None:
    print("=== Binary XOR: tanh hidden layer + BCEWithLogitsLoss ===")
    model, initial, final, probs, preds, grad = train_binary("tanh")
    print(f"Initial loss: {initial:.6f}")
    print(f"Final loss:   {final:.6f}")
    print("Target labels:", Y_BINARY.squeeze(1).int().tolist())
    print("Predictions: ", preds.squeeze(1).int().tolist())
    print("Probabilities:\n", probs.squeeze(1).numpy())
    print(f"Accuracy: {(preds == Y_BINARY).float().mean().item():.0%}")
    print("First-layer weight gradient at first backward pass:\n", grad.numpy())
    print(f"First-layer gradient norm: {grad.norm().item():.6f}")

    print("\n=== Zero-initialization symmetry check ===")
    weight_grad, bias_grad = symmetry_experiment()
    print("Hidden-layer weight gradients:\n", weight_grad.numpy())
    print("Hidden-layer bias gradients:", bias_grad.numpy())
    print("Identical hidden-unit weight gradients:", torch.allclose(weight_grad[0], weight_grad[1]))
    print("Identical hidden-unit bias gradients:", torch.allclose(bias_grad[0], bias_grad[1]))

    print("\n=== Activation comparison (same seed and training setup) ===")
    for activation in ("sigmoid", "tanh", "relu"):
        _, init_loss, final_loss, p, pred, g = train_binary(activation)
        accuracy = (pred == Y_BINARY).float().mean().item()
        print(f"{activation:7s} initial_loss={init_loss:.6f} final_loss={final_loss:.6f} "
              f"accuracy={accuracy:.0%} first_layer_grad_norm={g.norm().item():.6f}")

    print("\n=== Three-class task ===")
    multi, init_loss, final_loss, logits, probabilities, accuracy, shifted = train_multiclass()
    print("Target classes:", Y_MULTI.tolist())
    print(f"Initial loss: {init_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print("Output-layer weight shape:", tuple(multi.fc2.weight.shape))
    print("Logits:\n", logits.numpy())
    print("Softmax probabilities:\n", probabilities.numpy())
    print("Probability row sums:", probabilities.sum(dim=1).numpy())
    print(f"Accuracy: {accuracy:.0%}")
    print("Softmax unchanged after adding 100 to every logit:",
          torch.allclose(probabilities, shifted, atol=1e-6))


if __name__ == "__main__":
    main()
