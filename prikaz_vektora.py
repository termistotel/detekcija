# Jednostavan realtime prikaz 3D vektora s BMI160 (lagano i kratko)
# Pokretanje: python3 prikaz_vektora.py

import tkinter as tk
from bmi160_senzor import citaj_podatke

W, H = 360, 300  # malo platno = brze i lakse na RPi
CX, CY = W // 2, H // 2
S = 10  # faktor skaliranja vektora

root = tk.Tk()
root.title("3D vektor")
root.resizable(False, False)
cv = tk.Canvas(root, width=W, height=H, bg="black", highlightthickness=0)
cv.pack()

# jednostavna projekcija 3D -> 2D (x, y, z) -> (u, v)
# u = x + 0.5*z, v = -y + 0.5*z (mali "pseudo-3D")

def crtaj():
    ax, ay, az, *_ = citaj_podatke()
    x, y, z = ax * S, ay * S, az * S
    u = CX + (x + 0.5 * z)
    v = CY + (-y + 0.5 * z)

    cv.delete("all")

    # osi (radi orijentacije)
    cv.create_line(10, CY, W - 10, CY, fill="#223", width=1)  # X
    cv.create_line(CX, 10, CX, H - 10, fill="#223", width=1)  # Y
    cv.create_line(CX - 60, CY + 60, CX + 60, CY - 60, fill="#223", width=1)  # Z dijagonala

    # vektor
    cv.create_line(CX, CY, u, v, fill="lime", width=4, capstyle=tk.ROUND)
    cv.create_oval(CX - 4, CY - 4, CX + 4, CY + 4, fill="white", outline="")

    # brzi info
    cv.create_text(8, 8, anchor="nw", fill="#9cf",
                   text=f"ax:{ax:5.1f} ay:{ay:5.1f} az:{az:5.1f}")

    # ~60 FPS (16 ms)
    root.after(16, crtaj)

crtaj()
root.mainloop()
