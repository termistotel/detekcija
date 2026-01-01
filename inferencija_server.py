"""
Minimalna inferencija: ucitaj spremljene TEZINE modela (state_dict), citaj BMI160,
agregiraj N ocitanja, izracunaj vjerojatnost i posluzi vrlo jednostavnu
web stranicu koja se dinamicki osvjezava pomocu malog javascripta-a.

Pokretanje: python3 inferencija_server.py

Napomena: treba imati instaliran torch i flask
  pip3 install --break-system-packages torch flask
"""

import time
import threading

import torch
from flask import Flask, jsonify

from pomocni.bmi160_senzor import citaj_podatke
from pomocni.perceptron import PerceptronV3


# =========================
# Podesivi parametri
# =========================
AGGREGATE_N = 5                      # koliko uzastopnih ocitanja ulazi u model (D = 2*N jer koristimo iznose a,g)
TARGET_HZ = 10.0                     # ciljana frekvencija citanja senzora

THRESHOLD = 0.5                       # prag za 0/1 odluku
HOST, PORT = "0.0.0.0", 5000          # Flask server IP adresa i port


# Globalno stanje koje prikazujemo na / i /api
state = {
    "p": None,          # vjerojatnost (0..1)
    "y": None,          # klasa 0/1 po THRESHOLD
    "label": None,      # naziv klase: "mirovanje" ili "trcanje"
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
    last_rate_t = time.perf_counter()

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
            state["label"] = "trcanje" if y == 1 else "mirovanje"

            win = win[1:]

        if state["p"] is not None:
            lbl = state.get("label") or ("trcanje" if state.get("y") == 1 else "mirovanje")
            print(f"p={state['p']:.3f}  klasa={lbl}    ", end='\r')

        # Jednostavna procjena postignutog Hz (svake ~1s)
        now = time.perf_counter()
        if now - last_rate_t >= 1.0:
            elapsed = now - last_rate_t
            state["hz"] = reads / elapsed
            reads = 0
            last_rate_t = now

        # Odspavaj do sljedece iteracije
        if period > 0:
            dt = time.perf_counter() - t0
            sp = period - dt
            if sp > 0:
                time.sleep(sp)


def napravi_app():
    app = Flask(__name__)

    # Jednostavan HTML prikaz na index pageu (dinamicki refresh preko fetch + setInterval)
    @app.get("/")
    def index():
        return (
            "<html>"
            "<head>"
            "<meta charset='utf-8'>"
            "<style>body{font-family:monospace;background:#fff;color:#111;margin:20px;}"
            ".big{font-size:32px;line-height:1.3} .mid{font-size:18px} .blue{color:#06f}</style>"
            "</head>"
            "<body>"
            f"<div class='big'><b>Inferencija</b> (N={AGGREGATE_N})</div>"
            "<div class='mid' id='hz'>Hz: --.-</div>"
            "<div class='mid' id='last'>Zadnje: n/a</div>"
            f"<div class='big'>p: <span id='p'>n/a</span> | klasa: <b id='y'>n/a</b> (prag {THRESHOLD})</div>"
            "<script>\n"
            "async function refresh(){\n"
            "  try{\n"
            "    const r = await fetch('/api');\n"
            "    const j = await r.json();\n"
            "    document.getElementById('hz').textContent = 'Hz: ' + (j.hz? j.hz.toFixed(1):'--.-');\n"
            "    if(j.last){ document.getElementById('last').textContent = 'Zadnje: a=' + j.last[0].toFixed(1) + ' g=' + j.last[1].toFixed(1); }\n"
            "    if(j.p!=null){ document.getElementById('p').textContent = j.p.toFixed(3);}\n"
            "    const y = document.getElementById('y');\n"
            "    if(j.label!=null){ y.textContent = j.label; y.className = (j.label==='trcanje')? 'blue' : ''; }\n"
            "  }catch(e){}\n"
            "}\n"
            "setInterval(refresh, 500);\n"
            "refresh();\n"
            "</script>"
            "</body></html>"
        )

    @app.get("/api")
    def api():
        return jsonify({
            "p": state["p"],
            "y": state["y"],
            "label": state["label"],
            "last": state["last"],
            "window": state["window"],
            "hz": state["hz"],
        })

    return app


def main():
    MODEL_PATH = "modeli/perceptron.pth"  # tezine spremljene iz treniranja (state_dict)

    print("=== Inferencija BMI160 + Flask ===")
    print(f"Model (tezine): {MODEL_PATH}")
    print(f"N={AGGREGATE_N}, TARGET_HZ={TARGET_HZ} Hz, THRESHOLD={THRESHOLD}")

    # Dimenzija ulaza: 2 vrijednosti po ocitanju (a,g) * N uzoraka
    D = 2 * AGGREGATE_N
    model = PerceptronV3(D)

    # Ucitaj tezine (state_dict)
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
