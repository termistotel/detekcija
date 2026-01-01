# Dokumentacija: korak‑po‑korak vodič (od nule do inferencije)

Ovaj vodič vodi vas kroz cijeli proces izrade projekta od nule na Raspberry Pi‑ju: fizičko spajanje BMI160 senzora, provjera veze i I2C adrese, brzo ispisivanje i grafička vizualizacija podataka, prikupljanje podataka, treniranje jednostavnog modela te pokretanje inferencije s mini web sučeljem. Sve je napisano jednostavno, za učenike srednjih škola.

Napomena: primjeri pretpostavljaju da ste u mapi projekta:
```
cd /path/to/rpi-detekcija
```

---

## 1) Oprema i osnovna instalacija

- Raspberry Pi (s aktiviranim I2C)
- Senzor BMI160 (I2C varijanta)
- Spojni vodiči (jumperi)
- Internet za instalaciju paketa

Omogućite I2C na Raspberry Pi‑ju:
```
sudo raspi-config
# Interfacing Options → I2C → Enable
sudo reboot
```

Instalirajte osnovne pakete:
```
sudo apt-get update
sudo apt-get install -y i2c-tools python3-pip python3-tk
pip3 install --break-system-packages torch flask
```

---

## 2) Spajanje senzora (I2C)

BMI160 spojite ovako (I2C):
- BMI160 VCC → 3.3V na RPi
- BMI160 GND → GND na RPi
- BMI160 SCL → RPi SCL (tipično pin 5, GPIO3)
- BMI160 SDA → RPi SDA (tipično pin 3, GPIO2)

Provjerite da nisu zamijenjeni SDA/SCL i da koristite 3.3V (ne 5V!).

---

## 3) Provjera veze i pronalazak I2C adrese

Alatom `i2cdetect` možete skenirati I2C sabirnicu (najčešće `-y 1`):
```
sudo i2cdetect -y 1
```
Trebali biste vidjeti neku adresu (npr. `0x68` ili `0x69`). Ako ništa ne vidite:
- Provjerite kablove i napajanje.
- Potvrdite da je I2C omogućen u `raspi-config`.
- Provjerite da je BMI160 stvarno I2C modul.

Mali Python isječak za čitanje jednog registra preko `smbus2` (opcionalno):
```python
# Provjera "WHO_AM_I" (CHIP_ID) registra BMI160 (0x00 očekuje ~0xD1)
from smbus2 import SMBus

I2C_BUS = 1
ADDR = 0x68  # promijenite ako je i2cdetect pokazao 0x69
CHIP_ID_REG = 0x00

with SMBus(I2C_BUS) as bus:
    chip_id = bus.read_byte_data(ADDR, CHIP_ID_REG)
print(hex(chip_id))
```

U ovom projektu već postoji pomoćni modul za senzor: `pomocni/bmi160_senzor.py` s funkcijom `citaj_podatke()`, pa vam taj isječak nije nužan za daljnji rad.

---

## 4) Brza provjera podataka (print i grafika)

Najbrže: pokrenite kratku vizualizaciju u Tkinteru koja već dolazi u projektu:
```
python3 prikaz_vektora.py
```
Time dobijete tri 2D projekcije akceleracije (top/front/side) i tri bar‑a za gyro.

Ako želite samo ispis u terminalu, možete privremeno dodati ovaj mini isječak u novu datoteku npr. `brzi_print.py`:
```python
from time import sleep
from pomocni.bmi160_senzor import citaj_podatke

while True:
    ax, ay, az, gx, gy, gz = citaj_podatke()
    print(f"acc=({ax:.1f},{ay:.1f},{az:.1f})  gyro=({gx:.1f},{gy:.1f},{gz:.1f})")
    sleep(0.1)
```

---

## 5) Prikupljanje podataka (dataset)

Za prikupljanje primjera (za treniranje) koristimo skriptu:
```
python3 prikupljanje_podataka.py
```
Na vrhu datoteke su važni parametri (izmijenite po potrebi):
- `MODE` — "running" ili "resting" (sprema se kao oznaka 1 ili 0)
- `AGGREGATE_N` — koliko uzastopnih očitanja čini jedan podatkovni uzorak
- `TARGET_HZ` — ciljano učestalost čitanja senzora
- `OUTPUT_FILE` — CSV u koji se APPEND‑a (ne briše se staro)

Format jednog retka u CSV‑u:
```
label, ax_1, ay_1, az_1, gx_1, gy_1, gz_1, ..., ax_N, ay_N, az_N, gx_N, gy_N, gz_N
```
Savjeti:
- Snimajte ravnotežno: približno jednako primjera za `mirovanje (0)` i `trčanje (1)`.
- Držite `AGGREGATE_N` i `TARGET_HZ` konstantnim kroz sve snimke.
- Snimajte više kraćih sesija u različitim uvjetima (na ruci, u džepu, sporije/brže hodanje/trčanje).

Podatke podijelite na `train.csv` i `val.csv` (npr. 80%/20%). To možete ručno (kopiraj/izbaci retke) ili jednostavnim skriptom.

---

## 6) Jednostavan model i treniranje (PyTorch)

Treniranje pokrećete:
```
python3 treniranje_perceptron.py
```
Na vrhu su podesivi parametri: putanje do `TRAIN_CSV`, `VAL_CSV`, broj epoha `EPOCHS`, stopa učenja `LR`, veličina batcha `BATCH_SIZE`, odabir modela `MODEL_VERSION` (1/2/3) i gubitka `LOSS` ("mse" ili "bce").

Tri varijante perceptrona:
1. V1: ručna for‑petlja koja računa `w·x + b` bez `matmul`‑a
2. V2: eksplicitna matrica: `X @ W^T + b`
3. V3: ugrađeni `nn.Linear(D,1)`

Gubitci (ručno definirani):
- Least squares (MSE)
- Binary cross‑entropy (BCE)

L2 regularizacija (smanjuje preučenje) je podržana parametrom `L2_LAMBDA` na vrhu skripte. Primjer kako se L2 dodaje u gubitak:
```python
# poslje baznog loss‑a
if L2_LAMBDA > 0:
    l2 = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and 'bias' not in name:
            l2 = l2 + (p ** 2).sum()
    loss = loss + L2_LAMBDA * l2
```

Mini‑batch petlja koristi generator koji svaku epohu prođe kroz podatke u nasumičnom poretku:
```python
for xb, yb in napravi_minibatcheve(Ntr, BATCH_SIZE, Xtr, ytr):
    pred = model(xb)
    loss = criterion(pred, yb)
    # backward i jednostavni SGD update
```

Skripta nakon svake epohe sprema TEŽINE (state_dict) u mapu `modeli/`, npr. `perceptron_v3_last.pth`.

---

## 7) Inferencija (učitavanje težina + mini web server)

Za inferenciju koristimo skriptu koja učitava težine, čita senzor i prikazuje rezultat lokalno preko Flask‑a:
```
python3 inferencija_server.py
```
Podesivi parametri (na vrhu skripte):
- `AGGREGATE_N` — mora odgovarati onome iz treninga (duljina ulaza `D=2*N` za (a,g) iznose)
- `TARGET_HZ` — ciljano čitanje senzora
- `THRESHOLD` — prag odluke 0/1 (mirovanje/trčanje)

Unutar skripte model se instancira i učitavaju se težine (state_dict):
```python
D = 2 * AGGREGATE_N
model = PerceptronV3(D)
sd = torch.load("modeli/perceptron.pth", map_location="cpu")
model.load_state_dict(sd)
model.eval()
```

Skripta kontinuirano puni prozor od `N` očitanja (koristi iznose `a` i `g`), radi predikciju i servira jednostavnu web stranicu na `http://<IP>:5000/` koja se sama osvježava (mali JavaScript). Prikaz je uvećan, a detekcija `trcanje` se boja plavom.

---

## 8) Kratki primjer: kako sve ide redom

1. Spoji BMI160 prema poglavlju 2.
2. Provjeri I2C adresu `i2cdetect -y 1`.
3. Pokreni brzu vizualizaciju: `python3 prikaz_vektora.py`.
4. Snimi podatke: `python3 prikupljanje_podataka.py` (više sesija, 0 i 1).
5. Pripremi `train.csv` i `val.csv` (iste dimenzije `6*N`).
6. Treniraj: `python3 treniranje_perceptron.py` (po želji `L2_LAMBDA=1e-4`).
7. Pokreni inferenciju: `python3 inferencija_server.py` i otvori `http://<IP>:5000/`.

---

## 9) Rješavanje problema (kratko)

- `i2cdetect` ne vidi adresu: provjeri kablove, 3.3V, omogućeni I2C, ispravan senzor.
- `Tkinter` greška: instaliraj `python3-tk`.
- PyTorch greška/instalacija: `pip3 install --break-system-packages torch`.
- Dimenzije se ne slažu u inferenciji: uskladi `AGGREGATE_N` s treningom.
- Slab rezultat modela: snimi više podataka, uravnoteži klase, smanji LR, uključi `L2_LAMBDA`.

Sretno s eksperimentiranjem!
