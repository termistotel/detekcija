# Minimalan primjer treniranja jednostavne neuralne mreze u PyTorch-u
#
# Pretpostavka: postoje dvije CSV datoteke - jedna za trening (train_dataset.csv)
#                                          - druga za validaciju (val_dataset.csv)
#
# Pokretanje (primjer):
#   python3 treniranje_perceptron.py
#
# Napomena: potrebno je imati instaliran torch: pip3 install --break-system-packages torch

import csv
import os
import time

import torch

from pomocni.perceptron import PerceptronV3

# =========================
# PODESIVE VARIJABLE
# =========================

# Putanje do CSV-a (train i val)
TRAIN_CSV = "train_dataset.csv"
VAL_CSV = "val_dataset.csv"

# Hiperparametri
EPOCHS = 20000  # Broj epoha ucenja
LR = 0.5        # Brzina ucenja (Learning Rate


# =========================
# Ucitavanje podataka iz CSV-a
# =========================

def ucitaj_csv_putem_csv_modula(putanja):
    X = []
    y = []
    with open(putanja, "r") as f:
        rdr = csv.reader(f)
        for row in rdr:
            vals = [float(x) for x in row]
            y.append([vals[0]])   # prvi stupac je label (0/1)
            X.append(vals[1:])  # Ostali stupci su mjerenja sa senzora

    # Pretvori u pytorch tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y


# =======================================
# Funkcija gubitka - binary cross-entropy
# =======================================

def loss_bce(pred, target):
    # Binary cross-entropy: -mean( y*log(p) + (1-y)*log(1-p) )
    eps = 1e-7
    p = pred.clamp(eps, 1.0 - eps)
    return (-(target * torch.log(p) + (1 - target) * torch.log(1 - p))).mean()


# =============================
# Izracun tocnosti (Accuracy)
# =============================

def izracun_tocnosti(pred, target):
    # Prag 0.5 -> 0/1
    klas = (pred >= 0.5).float()
    tocnost = (klas == target).float()
    srednja_tocnost = torch.mean(tocnost).item()
    tocnost_postoci = srednja_tocnost * 100.0
    return tocnost_postoci


if __name__ == "__main__":
    print("\n=== Treniranje neuronske mreze (PyTorch) ===")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Val   CSV: {VAL_CSV}")
    print(f"Epohe: {EPOCHS} | Brzina ucenja: {LR}")

    Xtr, ytr = ucitaj_csv_putem_csv_modula(TRAIN_CSV)
    Xva, yva = ucitaj_csv_putem_csv_modula(VAL_CSV)
    Ntr, D = Xtr.shape
    Nva = Xva.shape[0]

    print(f"Dimenzije: Train N={Ntr}, D={D} | Val N={Nva}")

    model = PerceptronV3(D)

    # Jednostavan trening petlja: rucno azuriranje parametara bez optimizera
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()

        # Forward prop
        pred = model(Xtr)
        loss = loss_bce(pred, ytr)

        # Back prop
        model.zero_grad()
        loss.backward()

        # Rucni gradijentni spust
        with torch.no_grad():
            for p in model.parameters():
                p -= LR * p.grad

        train_loss = loss.item()
        train_acc = izracun_tocnosti(pred.detach(), ytr)

        # Validacija
        model.eval()
        with torch.no_grad():
            pred_val = model(Xva)
            val_loss = loss_bce(pred_val, yva).item()
            val_acc = izracun_tocnosti(pred_val.detach(), yva)

        dt = time.time() - t0

        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d} | train loss: {train_loss:.4f} acc: {train_acc:.1f}% | val loss: {val_loss:.4f} acc: {val_acc:.1f}% | {dt:.2f}s", end='\r')

            # Spremi tezine (state_dict) nakon svake epohe
            path = os.path.join('modeli', f"perceptron.pth")
            torch.save(model.state_dict(), path)\

    print("Gotovo.")
