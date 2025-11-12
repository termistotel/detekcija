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
