# detekcija
detekcija trcanja

# Omoguci i2c
sudo zraspi-config nonint do_i2c 0

# Provjera
sudo apt install i2c-tools
i2cdetect -y 1

# Paketi
# (https://circuitpython-bmi160.readthedocs.io/en/latest/examples.html)
pip3 install --break-system-packages circuitpython-bmi160

## Prikupljanje podataka (za treniranje)
Minimalni skript za prikupljanje podataka s BMI160 nalazi se u `prikupljanje_podataka.py`. Skript cita akcelerometar i ziroskop te periodicki sprema agregirane prozore ocitanja u CSV datoteku (redovi se NADODAJU, ne prepisuju).

Pokretanje:
```bash
python3 prikupljanje_podataka.py
```

Podesivi parametri su na vrhu datoteke:
- `MODE` — "running" (kretanje) ili "resting" (mirovanje); sprema se kao oznaka 1 ili 0.
- `AGGREGATE_N` — broj uzastopnih ocitanja po jednoj spremljenoj tocki (NN ulaz = 6 * N).
- `TARGET_HZ` — ciljna frekvencija citanja.
- `OUTPUT_FILE` — putanja do CSV-a (nadodavanje).
- `PRINT_EVERY` — koliko cesto ispisati postignute frekvencije.

Format retka u CSV-u:
`label, ax_1, ay_1, az_1, gx_1, gy_1, gz_1, ..., ax_N, ay_N, az_N, gx_N, gy_N, gz_N`

## Inferencija (ucitavanje modelskih tezina + web prikaz)
Minimalni skript za inferenciju nalazi se u `inferencija_server.py`. Skripta instancira `PerceptronV3(D)` gdje je `D = 6 * N`, ucitava SAMO tezine modela iz `.pth` datoteke (npr. `modeli/perceptron_v3_last.pth`), cita BMI160 u stvarnom vremenu, agregira N ocitanja te ispisuje rezultat u terminal i preko jednostavnog Flask servera.

Pokretanje:
```bash
python3 inferencija_server.py
```

Podesivo na vrhu skripte:
- `MODEL_PATH` — putanja do spremljenih tezina (state_dict), npr. `modeli/perceptron_v3_last.pth`
- `AGGREGATE_N` — broj uzastopnih ocitanja (ulaz dimenzije `6*N`)
- `TARGET_HZ` — ciljana frekvencija citanja senzora
- `THRESHOLD` — prag za binarnu odluku 0/1
- `HOST`, `PORT` — Flask server

Web sucelje:
- Otvori `http://localhost:5000/` (ili IP Raspberryja) za jednostavan pregled.
- JSON API na `http://localhost:5000/api`.

### Treniranje i spremanje tezina
Skripta `treniranje_perceptron.py` nakon svake epohe sprema SAMO tezine modela (PyTorch `state_dict`) u mapu `modeli/`:
- `modeli/perceptron_v3_ep{E}.pth` — tezine nakon epohe E
- `modeli/perceptron_v3_last.pth` — zadnje tezine (prepisuju se svaku epohu)

Inferencija ucitava `perceptron_v3_last.pth`. Za starije checkpointove koji spremaju cijeli model (`.pt`) postoji pokusaj unatragne kompatibilnosti (fallback) u skripti.
