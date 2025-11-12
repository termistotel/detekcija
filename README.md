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
