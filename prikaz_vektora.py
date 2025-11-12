# Jednostavan realtime prikaz 3D vektora s BMI160 (lagano i kratko)
# Pokretanje: python3 prikaz_vektora.py

import tkinter as tk
from bmi160_senzor import citaj_podatke

# Prosireni prozor
W, H = 1024, 720        # velicina platna
CX, CY = W // 3, H // 2  # centar za akcelerometar (lijeva 1/3)
OX, OY = W - W // 3, H // 2  # "desni" centar za gyro (desna 1/3)

# Skale (po potrebi prilagodite)
S = 10    # skala za akcelerometar
SG = 4    # skala za gyro vektor
SGBAR = 1.5  # skala za gyro trake (bar graf)
BAR_MAX = 160  # maksimalna duzina trake u pikselima (svaka strana)

# Perspektiva: debljina linije vs Y dubina
def debljina_iz_y(y):
    return max(1, min(8, int(6 - 0.04 * y)))

root = tk.Tk()
root.title("3D vektori — akcelerometar + gyro")
cv = tk.Canvas(root, width=W, height=H, bg="black", highlightthickness=0)
cv.pack()

# jednostavna projekcija 3D -> 2D (x, y, z) -> (u, v)
# koristimo mali "pseudo-3D": u = x + 0.5*y, v = -z + 0.5*y

def proj(origin_x, origin_y, x, y, z):
    u = origin_x + (x + 0.5 * y)
    v = origin_y - (z + 0.5 * y)
    return u, v


def crtaj_gyro_barove(x0, y0, w, h, gx, gy, gz):
    """Vrlo lagan 3-osni bar-graf za gyro, s oznakama X/Y/Z.
    x0, y0: gornji lijevi kut; w,h: dimenzije panela.
    """
    pad = 8
    bw = w - 2 * pad
    bh = (h - 4 * pad) // 3  # po red

    # okvir panela
    cv.create_rectangle(x0, y0, x0 + w, y0 + h, outline="#233", fill="", width=1)

    def jedna_traka(row, val, boja, oznaka):
        ty = y0 + pad + row * (bh + pad)
        cx = x0 + pad + bw // 2
        # os sredine
        cv.create_line(cx, ty, cx, ty + bh, fill="#345", width=1)
        # vrijednost
        pix = int(max(-BAR_MAX, min(BAR_MAX, val * SGBAR)))
        if pix >= 0:
            cv.create_rectangle(cx, ty + 4, cx + pix, ty + bh - 4, fill=boja, outline="")
        else:
            cv.create_rectangle(cx + pix, ty + 4, cx, ty + bh - 4, fill=boja, outline="")
        # oznaka
        cv.create_text(x0 + 6, ty + bh // 2, anchor="w", fill="#9cf", text=oznaka)

    jedna_traka(0, gx, "#f55", "gx")
    jedna_traka(1, gy, "#f77", "gy")
    jedna_traka(2, gz, "#faa", "gz")


def crtaj():
    ax, ay, az, gx, gy, gz = citaj_podatke()

    # Akcelerometar (zeleni vektor)
    u_a, v_a = proj(CX, CY, ax * S, ay * S, az * S)
    w_a = debljina_iz_y(ay * S)

    # Gyro (crveni vektor)
    u_g, v_g = proj(OX, OY, gx * SG, gy * SG, gz * SG)
    w_g = debljina_iz_y(gy * SG)

    cv.delete("all")

    # osi i pomocne crte
    # srednja horizontalna
    cv.create_line(10, CY, W - 10, CY, fill="#223", width=1)
    # vertikalne kroz centre
    cv.create_line(CX, 10, CX, H - 10, fill="#223", width=1)
    cv.create_line(OX, 10, OX, H - 10, fill="#223", width=1)
    # blaga Z dijagonala (oko lijevog centra)
    cv.create_line(CX - 60, CY + 60, CX + 60, CY - 60, fill="#223", width=1)
    # blaga Z dijagonala (oko desnog centra)
    cv.create_line(OX - 60, OY + 60, OX + 60, OY - 60, fill="#223", width=1)

    # vektori
    cv.create_line(CX, CY, u_a, v_a, fill="lime", width=w_a, capstyle=tk.ROUND)
    cv.create_oval(CX - 4, CY - 4, CX + 4, CY + 4, fill="white", outline="")

    cv.create_line(OX, OY, u_g, v_g, fill="#f33", width=w_g, capstyle=tk.ROUND)
    cv.create_oval(OX - 4, OY - 4, OX + 4, OY + 4, fill="white", outline="")

    # tekstualni info (gore lijevo/desno)
    cv.create_text(8, 8, anchor="nw", fill="#9cf",
                   text=f"ax:{ax:5.1f} ay:{ay:5.1f} az:{az:5.1f}")
    cv.create_text(W - 8, 8, anchor="ne", fill="#f99",
                   text=f"gx:{gx:5.1f} gy:{gy:5.1f} gz:{gz:5.1f}")

    # mali gyro bar-panel na krajnjoj desnoj strani
    panel_w = 240
    panel_h = 160
    panel_x = W - panel_w - 10
    panel_y = H - panel_h - 10
    crtaj_gyro_barove(panel_x, panel_y, panel_w, panel_h, gx, gy, gz)

    # ~60 FPS (16 ms)
    root.after(16, crtaj)


crtaj()
root.mainloop()
