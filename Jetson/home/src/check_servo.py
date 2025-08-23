import Jetson.GPIO as GPIO
import time

# 13番では、何故か動作できないため18番を使用する
SERVO_PIN = 18  # 物理ピン番号18


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)  # 物理ピン番号で指定
GPIO.setup(SERVO_PIN, GPIO.OUT, initial=GPIO.HIGH)
p1 = GPIO.PWM(SERVO_PIN, 50)

p1.start(9)

print("start")

while True:
    print("check")

    p1.ChangeDutyCycle(5)
    time.sleep(0.5)
    p1.ChangeDutyCycle(9)

    time.sleep(5)  # 5秒待機
