"""
Minimalna inferencija: ucitaj spremljeni model, citaj BMI160, agregiraj N ocitanja,
izracunaj vjerojatnost (sigmoid izlaz) i posluzi jednostavnu web stranicu (Flask).

Pokretanje: python3 inferencija_server.py

Napomena: treba imati instaliran torch i flask
  pip3 install --break-system-packages torch flask
"""

import time
import threading
from collections import deque

import torch
from flask import Flask, jsonify

from pomocni.bmi160_senzor import citaj_podatke
from pomocni.perceptron import PerceptronV3


# =========================
# Podesivi parametri
# =========================
AGGREGATE_N = 5                      # koliko uzastopnih ocitanja ulazi u model (D = 6*N)
TARGET_HZ = 10.0                      # ciljana frekvencija citanja senzora

THRESHOLD = 0.5                       # prag za 0/1 odluku
HOST, PORT = "0.0.0.0", 5000            # Flask server IP adresa i port


# Globalno stanje koje prikazujemo na / i /api
state = {
    "p": None,          # vjerojatnost (0..1)
    "y": None,          # klasa 0/1 po THRESHOLD
    "last": None,       # zadnje ocitanje (ax,ay,az,gx,gy,gz)
    "window": [],       # posljednjih N ocitanja
    "hz": TARGET_HZ,    # jednostavna procjena Hz (inicijalno cilj)
}


def _flatten_window(window):
    flat = []
    for vals in window:
        flat.extend(vals)
    return flat


def senzor_inferenca(model):
    """Jednostavna dretva: citaj senzor TARGET_HZ, puni prozor N i radi inferenciju."""
    period = 1.0 / TARGET_HZ if TARGET_HZ > 0 else 0.0
    win = []
    reads = 0

    while True:
        t0 = time.perf_counter()

        ax, ay, az, gx, gy, gz = citaj_podatke()

        # Racunanje iznosa vektora
        a = (ax ** 2 + ay ** 2 + az ** 2) ** 0.5
        g = (gx ** 2 + gy ** 2 + gz ** 2) ** 0.5

        win.append((a/10, g/100))
        reads += 1

        state["last"] = (a, g)
        state["window"] = list(win)

        # Kad skupimo N uzoraka, slozi input i pozovi model
        if len(win) == AGGREGATE_N:
            flat = _flatten_window(win)

            # Ucitaj spremljene podatke u tensor
            x = torch.tensor(flat, dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                p = model(x).view(-1).item()  # ocekivano: sigmoid izlaz [0..1]

            if p >= THRESHOLD:
                y = 1
            else:
                y = 0

            state["p"] = p
            state["y"] = y

            win = win[1:]

        if state["p"] is not None:
            print(f"p={state['p']:.3f}  y={state['y']}", end='\r')

        # Odspavaj do sljedece iteracije
        if period > 0:
            dt = time.perf_counter() - t0
            sp = period - dt
            if sp > 0:
                time.sleep(sp)


def napravi_app():
    app = Flask(__name__)

    # Jednostavan HTML prikaz na index pageu
    @app.get("/")
    def index():
        p = state["p"]
        y = state["y"]
        last = state["last"]
        hz = state["hz"]
        last_txt = "n/a" if last is None else f"a={last[0]:.1f} g={last[1]:.1f}"
        p_txt = "n/a" if p is None else f"{p:.3f}"
        y_txt = "n/a" if y is None else str(y)
        return (
            "<html><body style='font-family:monospace'>"
            f"<h3>Inferencija (N={AGGREGATE_N}, Hz~{hz:.1f})</h3>"
            f"<div>Zadnje: {last_txt}</div>"
            f"<div>p (vjerojatnost): <b>{p_txt}</b> | klasa: <b>{y_txt}</b> (prag {THRESHOLD})</div>"
            "<div><small>Osvjezi stranicu za najnovije stanje.</small></div>"
            "</body></html>"
        )

    @app.get("/api")
    def api():
        return jsonify({
            "p": state["p"],
            "y": state["y"],
            "last": state["last"],
            "window": state["window"],
            "hz": state["hz"],
        })

    return app


def main():
    MODEL_PATH = "modeli/perceptron.pth"

    print("=== Inferencija BMI160 + Flask ===")
    print(f"Model (tezine): {MODEL_PATH}")
    print(f"N={AGGREGATE_N}, TARGET_HZ={TARGET_HZ} Hz, THRESHOLD={THRESHOLD}")

    # Dimenzija ulaza: 2 vrijednosti po ocitanju * N uzoraka
    D = 2 * AGGREGATE_N
    model = PerceptronV3(D)
    # Ucitaj SAMO tezine (state_dict)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    # Pokreni dretvu za citanje senzora i inferenciju
    t = threading.Thread(target=senzor_inferenca, args=(model,), daemon=True)
    t.start()

    # Flask server (glavna dretva)
    app = napravi_app()
    app.run(host=HOST, port=PORT)


if __name__ == "__main__":
    main()
