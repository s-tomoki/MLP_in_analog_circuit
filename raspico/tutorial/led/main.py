import time

from machine import Pin

pin = Pin("LED", Pin.OUT)  # Pico基板上のLED(Pico:25, Pico W:"LED"で記載)

print("LED starts flashing...")
while True:
    try:
        pin.high()  # LEDを点灯(3.3Vを出力)
        time.sleep(1.0)
        pin.low()  # LEDを消灯(0Vを出力)
        time.sleep(1.0)
    except KeyboardInterrupt:
        break

pin.off()
print("Finished.")
