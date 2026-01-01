import board
import bmi160 as BMI160


# Postavi I²C bus
i2c = board.I2C()


# Inicijalizacija BMI160 senzora
bmi160 = BMI160.BMI160(i2c, address=0x69)

ax, ay, az = bmi160.acceleration
gx, gy, gz = bmi160.gyro

print(ax, ay, az)
print(gx, gy, gz)


# Definiraj funkciju za citanje
def citaj_podatke():
    ax, ay, az = bmi160.acceleration
    gx, gy, gz = bmi160.gyro
    return ax, ay, az, gx, gy, gz