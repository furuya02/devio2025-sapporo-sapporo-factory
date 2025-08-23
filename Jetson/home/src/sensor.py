import Jetson.GPIO as GPIO


class Sensor:
    SENSOR_PIN = 16
    mode = "ready"  # ready | sleep

    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)  # 物理ピン番号で指定
        GPIO.setup(self.SENSOR_PIN, GPIO.IN)

    def reset(self):
        self.mode = "ready"

    def check(self):
        if self.mode == "sleep":
            # センサーがsleep状態のときは、何もしない
            return "sleep"

        sensor = GPIO.input(self.SENSOR_PIN)
        if sensor == 1:
            return "off"
        else:
            self.mode = "sleep"
            return "on"
