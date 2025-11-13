"""
Prikaz BMI160 podataka u Tkinteru

• Akcelerometar: tri projekcije vektora (Top XY, Front XZ, Side YZ)
• Gyro: 3 trake (gx, gy, gz)

Pokretanje: python3 prikaz_vektora.py
"""

import tkinter as tk
from pomocni.bmi160_senzor import citaj_podatke

# Velicina prozora i skale (po potrebi prilagodite)
W, H = 960, 600
SA = 10.0   # skala za crte vektora akcelerometra
SG = 2.0    # skala za gyro trake (barovi)
BAR_MAX = 180  # max duljina trake u pikselima (svaka strana)

root = tk.Tk()
root.title("BMI160 — Akcelerometar (3 projekcije) + Gyro trake")
cv = tk.Canvas(root, width=W, height=H, bg="black", highlightthickness=0)
cv.pack()


# Pomoćne funkcije za crtanje
def crtaj_kriz_i_vektor(x0, y0, w, h, a, b, boja, oznaka_a, oznaka_b):
    """Panel s koordinatnim križem i vektorom iz sredine.
    a ide po x (vodoravno), b ide po y (okomito, pozitivno prema gore).
    """
    cx, cy = x0 + w // 2, y0 + h // 2
    cv.create_rectangle(x0, y0, x0 + w, y0 + h, outline="#233")
    cv.create_line(x0 + 8, cy, x0 + w - 8, cy, fill="#234")  # os x
    cv.create_line(cx, y0 + 8, cx, y0 + h - 8, fill="#234")  # os y
    # vektor: y ide gore pa minus
    x_end = cx + int(a * SA)
    y_end = cy - int(b * SA)
    cv.create_line(cx, cy, x_end, y_end, fill=boja, width=3, capstyle=tk.ROUND)
    cv.create_text(x0 + 10, y0 + 12, anchor="w", fill="#9cf", text=f"{oznaka_a}/{oznaka_b}")


def crtaj_gyro_trake(x0, y0, w, h, gx, gy, gz):
    """Tri jednostavne trake za gx, gy, gz (0 u sredini, +/- lijevo/desno)."""
    pad = 10
    bw = w - 2 * pad
    bh = (h - 4 * pad) // 3
    cv.create_rectangle(x0, y0, x0 + w, y0 + h, outline="#233")

    def traka(r, val, boja, tag):
        ty = y0 + pad + r * (bh + pad)
        cx = x0 + pad + bw // 2
        cv.create_line(cx, ty, cx, ty + bh, fill="#345")
        pix = int(max(-BAR_MAX, min(BAR_MAX, val * SG)))
        if pix >= 0:
            cv.create_rectangle(cx, ty + 4, cx + pix, ty + bh - 4, fill=boja, outline="")
        else:
            cv.create_rectangle(cx + pix, ty + 4, cx, ty + bh - 4, fill=boja, outline="")
        cv.create_text(x0 + 8, ty + bh // 2, anchor="w", fill="#faa", text=tag)

    traka(0, gx, "#f55", "gx")
    traka(1, gy, "#f77", "gy")
    traka(2, gz, "#faa", "gz")


def crtaj():
    # Ocitaj BMI160: akcelerometar (ax,ay,az) i gyro (gx,gy,gz)
    ax, ay, az, gx, gy, gz = citaj_podatke()

    cv.delete("all")

    # Raspored: 2x2 panela
    pw, ph = W // 2 - 20, H // 2 - 20
    x1, y1 = 10, 10
    x2, y2 = W // 2 + 10, 10
    x3, y3 = 10, H // 2 + 10
    x4, y4 = W // 2 + 10, H // 2 + 10

    # Akcelerometar projekcije
    # Top (XY): x vodoravno, y okomito
    crtaj_kriz_i_vektor(x1, y1, pw, ph, ax, ay, "#6f6", "X", "Y")
    # Front (XZ): x vodoravno, z okomito
    crtaj_kriz_i_vektor(x2, y2, pw, ph, ax, az, "#6f6", "X", "Z")
    # Side (YZ): y vodoravno, z okomito
    crtaj_kriz_i_vektor(x3, y3, pw, ph, ay, az, "#6f6", "Y", "Z")

    # Gyro trake (dolje desno)
    crtaj_gyro_trake(x4, y4, pw, ph, gx, gy, gz)

    # Kratke brojke (gore lijevo i desno)
    cv.create_text(12, H - ph - 8, anchor="sw", fill="#9cf",
                   text=f"ax:{ax:5.1f} ay:{ay:5.1f} az:{az:5.1f}")
    cv.create_text(W - 12, H - ph - 8, anchor="se", fill="#f99",
                   text=f"gx:{gx:5.1f} gy:{gy:5.1f} gz:{gz:5.1f}")

    # ~30-60 FPS (ovdje ~30ms)
    root.after(30, crtaj)


crtaj()
root.mainloop()
