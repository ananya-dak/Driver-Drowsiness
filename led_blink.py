import time
import RPi.GPIO as GPIO

def bb():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(26, GPIO.OUT)

    GPIO.output(26, True)
    time.sleep(3)

    GPIO.output(26, False)
    time.sleep(3)

    GPIO.cleanup()
