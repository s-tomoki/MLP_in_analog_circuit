import time

import rp2
from machine import UART, Pin

# --- UART ループバック ---
# 事前に GP16 (TX) と GP17 (RX) をジャンパ線で接続しておく。
# BOOTSEL を押すと TX から "Hello, Pico" を送信し、RX で受信した内容を REPL に表示する。

# --- UART0 の設定 (GP16=TX, GP17=RX) ---
UART_TX_PIN = 16
UART_RX_PIN = 17
BAUDRATE = 115200
TIMEOUT_MS = 500  # 受信待ちのタイムアウト

MESSAGE = "Hello, Pico\n"  # 行末の改行で受信側は 1 行の終わりを判定する


# 1. UART を初期化（RX が未接続でも High (アイドル) に保つ）
Pin(UART_RX_PIN, Pin.IN, Pin.PULL_UP)
uart = UART(0, baudrate=BAUDRATE, tx=Pin(UART_TX_PIN), rx=Pin(UART_RX_PIN))


# --- BOOTSEL ボタン ---
def wait_bootsel_press():
    """BOOTSEL ボタンが押されて離されるまで待つ（チャタリング対策込み）。"""
    while not rp2.bootsel_button():
        time.sleep_ms(20)
    time.sleep_ms(20)
    while rp2.bootsel_button():
        time.sleep_ms(20)
    time.sleep_ms(20)


def read_line(timeout_ms):
    """改行を受信するかタイムアウトするまで 1 バイトずつ読み、受信した bytes を返す。"""
    buf = b""
    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
        if not uart.any():
            continue
        c = uart.read(1)
        buf += c
        if c == b"\n":
            break
    return buf


print("Connect GP16 (TX) and GP17 (RX), then press BOOTSEL.")
try:
    while True:
        # 2. BOOTSEL の押下を待つ
        wait_bootsel_press()

        # 受信バッファに残っているデータを捨てる
        while uart.any():
            uart.read()

        # 3. TX へ送信
        uart.write(MESSAGE)
        print("TX:", MESSAGE.strip())

        # 4. RX で受信して REPL へ出力
        data = read_line(TIMEOUT_MS)
        if data:
            print("RX:", data.decode().strip())
        else:
            print("RX: (no data) - check the wire between GP16 and GP17")
except KeyboardInterrupt:
    pass
finally:
    uart.deinit()
