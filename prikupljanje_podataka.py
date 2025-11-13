# Minimalna skripta za prikupljanje podataka s BMI160 (CSV se nadodaje)
# Pokretanje: python3 prikupljanje_podataka.py
#
# Sto radi
# - Cita BMI160 (ax, ay, az, gx, gy, gz) podesivom frekvencijom
# - Agregira N uzastopnih ocitanja u jednu tocku za treniranje
# - Nadodaje redove u CSV kako se prethodni podaci ne bi prepisali
# - Oznacava svaki red kao kretanje (running) ili mirovanje (resting) za buduce treniranje
# - Periodicki ispisuje stvarno postignutu frekvenciju citanja (Hz)

import csv
import os
import time

from pomocni.bmi160_senzor import citaj_podatke


# =========================
# Podesivi parametri
# =========================

# Nacin aktivnosti
MODE = 0  # 1 (kretanje) ili 0 (mirovanje)

# Gdje spremati CSV (pri svakom pokretanju se nadodaje)
OUTPUT_FILE = "train_dataset.csv"

# Koliko uzastopnih ocitanja agregirati u jednu spremljenu tocku
# Ulaz u neuronsku mrezu bit ce 6 * AGGREGATE_N
AGGREGATE_N = 5

# Ciljana frekvencija citanja u Hz (ocitanja u sekundi)
TARGET_HZ = 10.0

# Koliko cesto ispisati stvarnu frekvenciju (u sekundama)
PRINT_EVERY = 5.0


def _flatten_window(window):
    flat = []
    for vals in window:
        flat.extend(vals)
    return flat


def main():
    label = MODE

    print("=== BMI160 Data Gatherer ===")
    print(f"label={label}")
    print(f"Broj agregiranih ocitanja N: {AGGREGATE_N} (row inputs = 6 * N = {6 * AGGREGATE_N})")
    print(f"Brzina ocitanja: {TARGET_HZ:.2f} Hz")
    print(f"Izlazni CSV: {OUTPUT_FILE}")

    # Mjerenje vremena i brojaci
    period = 1.0 / float(TARGET_HZ) if TARGET_HZ > 0 else 0.0
    last_report = time.perf_counter()
    read_count = 0
    saved_count = 0

    window = []

    # Otvori jednom u modu nadodavanja; nakon svakog upisa napravi flush
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        while True:
            t0 = time.perf_counter()

            # Citanje senzora
            ax, ay, az, gx, gy, gz = citaj_podatke()

            # Racunanje iznosa vektora
            a = (ax ** 2 + ay ** 2 + az ** 2) ** 0.5
            g = (gx ** 2 + gy ** 2 + gz ** 2) ** 0.5

            # Dodaj iznose vektora
            window.append((a/10, g/100))
            read_count += 1

            # Ako smo dosegnuli N ocitanja, spremi jednu tocku
            if len(window) >= AGGREGATE_N:
                row = [label] + _flatten_window(window[:AGGREGATE_N])
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno()) if hasattr(os, "fsync") else None
                saved_count += 1
                # Makni prvi red i nastavi dalje
                window = window[1:]

            # Periodicki izvjestaj o postignutim frekvencijama
            now = time.perf_counter()
            if now - last_report >= PRINT_EVERY:
                elapsed = now - last_report
                read_hz = read_count / elapsed
                save_hz = saved_count / elapsed
                print(f"Read: {read_hz:6.2f} Hz | Saved datapoints: {save_hz:6.2f} Hz")
                last_report = now
                read_count = 0
                saved_count = 0

            # Cekaj kako bi odrzao zadanu frekvenciju petlje
            if period > 0:
                dt = time.perf_counter() - t0
                to_sleep = period - dt
                if to_sleep > 0:
                    time.sleep(to_sleep)


if __name__ == "__main__":
    main()
