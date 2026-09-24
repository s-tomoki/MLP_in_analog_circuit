import time

from machine import UART, Pin

# --- テストベクタ出力ピン ---
# 出力要素 i を OUT_PINS[i] に出力する（primary の MASK で 1 の画素を行優先で並べた順）。
# GP23-25 は Pico W の無線チップ用のため使わない。GP28 は予備（未使用）。
OUT_PINS = list(range(2, 23)) + [26, 27]  # 23 本
N_OUT = len(OUT_PINS)

# --- primary との通信 (UART0: GP0=TX, GP1=RX) ---
UART_TX_PIN = 0
UART_RX_PIN = 1
BAUDRATE = 115200
BLINK_MS = 500  # PING 待ちの間、この周期で基板 LED を反転（1Hz 点滅）
MAX_LINE = 32  # これより長い行は破棄する


# 1. 基板 LED を消灯（消灯 > 1Hz点滅: PING待ち > 点灯: 接続完了）
led = Pin("LED", Pin.OUT)
led.off()

# 2. 出力ピンを初期化（全 0）
pins = [Pin(p, Pin.OUT, value=0) for p in OUT_PINS]

# 3. 待機バッファと UART を初期化（RX は相手が未起動でも High (アイドル) に保つ）
staged = 0
Pin(UART_RX_PIN, Pin.IN, Pin.PULL_UP)
uart = UART(0, baudrate=BAUDRATE, tx=Pin(UART_TX_PIN), rx=Pin(UART_RX_PIN))


def to_hex(bits):
    return "{:06X}".format(bits)


def output(bits):
    """bit i を OUT_PINS[i] に出力する（1 本ずつ書き換える）。"""
    for i, p in enumerate(pins):
        p.value((bits >> i) & 1)


def read_pins():
    """出力ピンの実際の状態を読み、bit i = OUT_PINS[i] の整数にする。"""
    bits = 0
    for i, p in enumerate(pins):
        if p.value():
            bits |= 1 << i
    return bits


# --- コマンド処理 ---
# 1 行 = 1 コマンド（ASCII, 改行終端）。データは 6 桁の 16 進数（bit i = 出力要素 i）。
#   PING       -> PONG
#   INIT       -> OK          待機バッファと出力を 0 にする
#   W hhhhhh   -> OK          待機バッファへ書き込む（出力は変えない）
#   R          -> D hhhhhh    待機バッファを読み返す
#   A          -> OK hhhhhh   待機バッファを出力し、実際のピン状態を返す
def handle(line):
    global staged

    if line == "PING":
        return "PONG"
    if line == "INIT":
        staged = 0
        output(staged)
        return "OK"
    if line.startswith("W "):
        try:
            bits = int(line[2:], 16)
        except ValueError:
            return "ERR bad data"
        if not 0 <= bits < (1 << N_OUT):
            return "ERR out of range"
        staged = bits
        return "OK"
    if line == "R":
        return "D " + to_hex(staged)
    if line == "A":
        output(staged)
        return "OK " + to_hex(read_pins())
    return "ERR unknown command"


def main():
    connected = False
    last_blink = time.ticks_ms()
    line = b""
    print("Waiting for primary...")

    while True:
        # 接続前は 1Hz で点滅
        if not connected and time.ticks_diff(time.ticks_ms(), last_blink) >= BLINK_MS:
            led.toggle()
            last_blink = time.ticks_ms()

        if not uart.any():
            time.sleep_ms(1)
            continue

        c = uart.read(1)
        if c == b"\r":
            continue
        if c != b"\n":
            line += c
            if len(line) > MAX_LINE:
                line = b""  # ノイズなどで行が長すぎる場合は破棄
            continue

        # 1 行受信
        try:
            cmd = line.decode().strip()
        except UnicodeError:
            cmd = None
        line = b""
        if not cmd:
            continue

        resp = handle(cmd)
        uart.write(resp + "\n")
        print("{} -> {}".format(cmd, resp))

        if cmd == "PING" and not connected:
            connected = True
            led.on()
            print("Connected.")


main()
