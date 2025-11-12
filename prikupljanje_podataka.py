# Minimalan skript za prikupljanje podataka s BMI160 (CSV se nadodaje)
# Pokretanje: python3 prikupljanje_podataka.py
#
# Sto radi
# - Cita BMI160 (ax, ay, az, gx, gy, gz) podesivom frekvencijom
# - Agregira N uzastopnih ocitanja u jednu tocku za treniranje
# - Nadodaje redove u CSV kako se prethodni podaci ne bi prepisali
# - Oznacava svaki red kao kretanje (running) ili mirovanje (resting) za buduce treniranje
# - Periodicki ispisuje stvarno postignutu frekvenciju citanja (Hz)

from __future__ import annotations

import csv
import os
import sys
import time
from typing import List, Tuple

from bmi160_senzor import citaj_podatke


# =========================
# Podesivi parametri
# =========================

# Nacin aktivnosti: odaberi jedno od {"running", "resting"}
MODE = "running"  # "running" (kretanje) ili "resting" (mirovanje)

# Koliko uzastopnih ocitanja agregirati u jednu spremljenu tocku
# Ulaz u neuronsku mrezu bit ce 6 * AGGREGATE_N
AGGREGATE_N = 10

# Ciljana frekvencija citanja u Hz (ocitanja u sekundi)
TARGET_HZ = 5.0

# Gdje spremati CSV (pri svakom pokretanju se nadodaje)
OUTPUT_FILE = "dataset.csv"

# Koliko cesto ispisati stvarnu frekvenciju (u sekundama)
PRINT_EVERY = 5.0


# Kao label koristimo 1 za kretanje, 0 za mirovanje
def _label_from_mode(mode: str) -> int:
    m = mode.strip().lower()
    if m in ("running", "moving", "run", "move"):
        return 1
    if m in ("rest", "resting", "idle", "still"):
        return 0
    return 0


def _flatten_window(window: List[Tuple[float, float, float, float, float, float]]) -> List[float]:
    flat: List[float] = []
    for (ax, ay, az, gx, gy, gz) in window:
        flat.extend([ax, ay, az, gx, gy, gz])
    return flat


def main() -> int:
    label = _label_from_mode(MODE)

    print("=== BMI160 Data Gatherer ===")
    print(f"Mode: {MODE} -> label={label}")
    print(f"Aggregate N: {AGGREGATE_N} (row inputs = 6 * N = {6 * AGGREGATE_N})")
    print(f"Target read rate: {TARGET_HZ:.2f} Hz")
    print(f"Output: {OUTPUT_FILE} (append)")
    print(f"Print actual rates every: {PRINT_EVERY:.1f}s")
    print("Press Ctrl+C to stop.\n")

    # Mjerenje vremena i brojaci
    period = 1.0 / float(TARGET_HZ) if TARGET_HZ > 0 else 0.0
    last_report = time.perf_counter()
    read_count = 0
    saved_count = 0

    window: List[Tuple[float, float, float, float, float, float]] = []

    try:
        # Otvori jednom u modu nadodavanja; nakon svakog upisa napravi flush
        with open(OUTPUT_FILE, "a", newline="") as f:
            writer = csv.writer(f)

            while True:
                t0 = time.perf_counter()

                # Citanje senzora
                ax, ay, az, gx, gy, gz = citaj_podatke()
                window.append((ax, ay, az, gx, gy, gz))
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
                    read_hz = read_count / elapsed if elapsed > 0 else 0.0
                    save_hz = saved_count / elapsed if elapsed > 0 else 0.0
                    print(f"Read: {read_hz:6.2f} Hz | Saved datapoints: {save_hz:6.2f} Hz | Buffer: {len(window)}/{AGGREGATE_N}")
                    last_report = now
                    read_count = 0
                    saved_count = 0

                # Spavaj kako bi odrzao zadanu frekvenciju petlje
                if period > 0:
                    dt = time.perf_counter() - t0
                    to_sleep = period - dt
                    if to_sleep > 0:
                        time.sleep(to_sleep)

    except KeyboardInterrupt:
        print("\nZaustavljanje...")
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
