# Minimalan primjer treniranja jednostavnog perceptrona u PyTorch-u
# Bez nepotrebne apstrakcije: tri verzije modela i dvije funkcije gubitka.
#
# Pretpostavka: postoje dvije CSV datoteke — jedna za trening, druga za validaciju.
# Svaki red: label (0/1), zatim niz ulaza (npr. 6*N vrijednosti).
#
# Pokretanje (primjer):
#   python3 treniranje_perceptron.py
#
# Napomena: potrebno je imati instaliran torch: pip3 install --break-system-packages torch

import csv
import os
import time

import torch
import torch.nn as nn


# =========================
# PODESIVE VARIJABLE (na vrhu)
# =========================

# Putanje do CSV-a (train i val)
TRAIN_CSV = "train_dataset.csv"
VAL_CSV = "val_dataset.csv"

# Hiperparametri
EPOCHS = 200
LR = 0.001

# Jednostavan minibatch (veličina batcha)
BATCH_SIZE = 4


# =========================
# Tri varijante perceptrona (1 izlaz, sigmoid)
# =========================

class PerceptronV1(nn.Module):
    """V1: Rucni izracun: for-petlja preko značajki (bez sum/elementwise i bez matmul-a) + sigmoid."""

    def __init__(self, d):
        super().__init__()
        # Parametri kao tenzori s gradijentom
        self.w = nn.Parameter(torch.randn(d) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # x: (N, D)
        # Izračun skalarno-produkta w·x eksplicitnom for-petljom (bez .sum i bez matmul-a)
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
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


# =========================
# Dvije funkcije gubitka
# =========================

def loss_mse(pred, target):
    # Least squares: mean( (pred - target)^2 )
    return ((pred - target) ** 2).mean()


def loss_bce(pred, target):
    # Binary cross-entropy: -mean( y*log(p) + (1-y)*log(1-p) )
    eps = 1e-7
    p = pred.clamp(eps, 1.0 - eps)
    return (-(target * torch.log(p) + (1 - target) * torch.log(1 - p))).mean()


def izracun_tocnosti(pred, target):
    # Prag 0.5 -> 0/1
    klas = (pred >= 0.5).float()
    return (klas.eq(target).float().mean()).item()


# =========================
# Ucitavanje podataka iz CSV-a (bez DataLoadera)
# =========================

def ucitaj_csv_putem_csv_modula(putanja):
    X = []
    y = []
    with open(putanja, "r") as f:
        rdr = csv.reader(f)
        for row in rdr:
            vals = [float(x) for x in row]
            y.append([vals[0]])  # prvi stupac je label (0/1)
            X.append(vals[1:])  # Ostali stupci su mjerenja sa senzora

    # Pretvori u tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y


# =========================
# Generator mini-batcheva (bez pristranosti)
# =========================

def napravi_minibatcheve(N, batch_size, X, y):
    """
        napravi slučajnu permutaciju i reži u uzastopne batch-eve.
    """
    perm = torch.randperm(N)
    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]


def main():
    print("\n=== Minimalno treniranje perceptrona (PyTorch) ===")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Val   CSV: {VAL_CSV}")
    print(f"Epohe: {EPOCHS} | LR: {LR} | Batch: {BATCH_SIZE}")

    Xtr, ytr = ucitaj_csv_putem_csv_modula(TRAIN_CSV)
    Xva, yva = ucitaj_csv_putem_csv_modula(VAL_CSV)
    Ntr, D = Xtr.shape
    Nva = Xva.shape[0]
    assert Xva.shape[1] == D, "Train i Val moraju imati isti broj ulaznih značajki"

    print(f"Dimenzije: Train N={Ntr}, D={D} | Val N={Nva}")

    model = PerceptronV3(D)
    criterion = loss_mse

    # Jednostavan trening petlja: rucno azuriranje parametara bez optimizera
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Trening kroz (mini)batch-eve
        model.train()
        kum_loss = 0.0
        kum_acc = 0.0
        brojac = 0

        # Iteracija preko generatora minibatcheva (slučajna permutacija po epohi)
        for xb, yb in napravi_minibatcheve(Ntr, BATCH_SIZE, Xtr, ytr):

            # Forward prop
            pred = model(xb)
            loss = criterion(pred, yb)

            # Back prop
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward()

            # Rucni SGD korak
            with torch.no_grad():
                for p in model.parameters():
                    p -= LR * p.grad

            bs = xb.shape[0]
            kum_loss += loss.item() * bs
            kum_acc += izracun_tocnosti(pred.detach(), yb) * bs
            brojac += bs

        train_loss = kum_loss / brojac
        train_acc = kum_acc / brojac

        # Validacija
        model.eval()
        with torch.no_grad():
            pred_val = model(Xva)
            val_loss = criterion(pred_val, yva).item()
            val_acc = izracun_tocnosti(pred_val, yva)

        dt = time.time() - t0
        print(f"Epoch {epoch:3d} | train loss: {train_loss:.4f} acc: {train_acc:.3f} | val loss: {val_loss:.4f} acc: {val_acc:.3f} | {dt:.2f}s")

        # Spremi kompletan model nakon svake epohe
        try:
            path = os.path.join('modeli', f"perceptron.pt")
            torch.save(model, path)  # spremamo cijeli model (za lako ucitavanje kasnije)
            print(f"[Spremanje] Model spremljen: {path}")
        except Exception as e:
            print(f"[Upozorenje] Model nije spremljen: {e}")

    print("Gotovo.")


if __name__ == "__main__":
    main()
