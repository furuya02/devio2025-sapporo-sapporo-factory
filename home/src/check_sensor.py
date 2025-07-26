from time import sleep
import Jetson.GPIO as GPIO


SENSOR_PIN = 16  # 物理ピン番号16 (GPIO08)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)  # 物理ピン番号で指定
GPIO.setup(SENSOR_PIN, GPIO.IN)


while True:
    sensor = GPIO.input(SENSOR_PIN)
    print("sensor:{}".format(sensor))
    sleep(0.2)  # 1秒待機
