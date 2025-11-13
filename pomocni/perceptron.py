import torch
from torch import nn


# =========================
# Tri varijante perceptrona (1 izlaz, sigmoid aktivacija)
# =========================

class PerceptronV1(nn.Module):
    """V1: Rucni izracun: for-petlja preko značajki (bez sum/elementwise i bez matmul-a) + sigmoid."""

    def __init__(self, d):
        super().__init__()
        # Parametri kao tenzori s gradijentom
        self.w = nn.Parameter(torch.randn(d) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # x: (N, D)
        # Izračun skalarno-produkta w·x eksplicitnom for-petljom
        N, D = x.shape
        z = torch.zeros((N, 1), dtype=x.dtype, device=x.device)
        # Zbrajamo doprinos svake značajke: z += x[:, i] * w[i]
        for i in range(self.w.shape[0]):
            z += x[:, i:i+1] * self.w[i]
        z = z + self.b  # dodaj pomak (bias)
        return torch.sigmoid(z)


class PerceptronV2(nn.Module):
    """V2: Eksplicitna matr. mnozenja: X @ W^T + b, sa sigmoidom."""

    def __init__(self, d):
        super().__init__()
        # W: (1, D) da bismo mogli raditi X @ W.T
        self.W = nn.Parameter(torch.randn(1, d) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # x: (N, D)
        z = x @ self.W.t() + self.b  # (N,1)
        return torch.sigmoid(z)


class PerceptronV3(nn.Module):
    """V3: Klasican nn.Linear(D,1) + sigmoid."""

    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 1)

    def forward(self, x):
        A = torch.sigmoid(self.fc1(x))
        return A