import time

import rp2
from machine import Pin, Timer

DOT_SIZE = 8  # ドットマトリックスは 8x8

# --- GPIOピンの設定 (tutorial/dotmatrix と同じ) ---

COL_PINS = [4, 1, 6, 0, 11, 7, 13, 8]  # 列(COL)ピン
ROW_PINS = [12, 2, 3, 9, 5, 10, 14, 15]  # 行(ROW)ピン


# ピンオブジェクトの初期化
rows = [Pin(p, Pin.OUT) for p in ROW_PINS]
cols = [Pin(p, Pin.OUT) for p in COL_PINS]


# 初期状態（全消灯）
# ※アノード行/カソード列の場合: ROW=LOW(0), COL=HIGH(1) で消灯
def clear_display():
    for r in rows:
        r.value(0)
    for c in cols:
        c.value(1)


def show(buf):
    """ビットパターンを REPL に表示する（デバッグ用）。"""
    for row_data in buf:
        print(
            "".join("#" if (row_data >> (DOT_SIZE - 1 - c)) & 1 else "." for c in range(DOT_SIZE))
        )


# --- 表示パターン (8x8 ビットパターン, 1: 点灯, 0: 消灯) ---
# 四隅に L 字（腕の長さ 3 ドット）を順に表示し、行・列の接続を確認する
ALL_ON = [0b11111111] * DOT_SIZE
ALL_OFF = [0b00000000] * DOT_SIZE

TOP_LEFT = [
    0b11100000,
    0b10000000,
    0b10000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
]

TOP_RIGHT = [
    0b00000111,
    0b00000001,
    0b00000001,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
]

BOTTOM_RIGHT = [
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000001,
    0b00000001,
    0b00000111,
]

BOTTOM_LEFT = [
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b10000000,
    0b10000000,
    0b11100000,
]

PATTERNS = [
    ("All on", ALL_ON),
    ("Top left", TOP_LEFT),
    ("Top right", TOP_RIGHT),
    ("Bottom right", BOTTOM_RIGHT),
    ("Bottom left", BOTTOM_LEFT),
    ("All off", ALL_OFF),
]

buffer = ALL_ON
current_row = 0


# --- ダイナミック点灯用タイマー割り込みハンドラ ---
def scan_display(timer):
    global current_row

    # 前の行を消灯（ゴースト点灯防止）
    rows[current_row].value(0)

    # 次の行へ更新
    current_row = (current_row + 1) % DOT_SIZE

    # 現在の行のデータを列ピンに出力
    # カソードコモンの場合、bit=1 の時に COL を LOW(0) にして電流を流す
    row_data = buffer[current_row]
    for col_idx in range(DOT_SIZE):
        bit = (row_data >> (DOT_SIZE - 1 - col_idx)) & 1
        cols[col_idx].value(0 if bit else 1)

    # 現在の行を有効化（アノード行を HIGH(1) にする）
    rows[current_row].value(1)


# --- BOOTSEL ボタン ---
def wait_bootsel_press():
    """BOOTSEL ボタンが押されて離されるまで待つ（チャタリング対策込み）。"""
    while not rp2.bootsel_button():
        time.sleep_ms(20)
    time.sleep_ms(20)
    while rp2.bootsel_button():
        time.sleep_ms(20)
    time.sleep_ms(20)


# 初期化して全点灯
clear_display()
timer = Timer()
timer.init(freq=1000, mode=Timer.PERIODIC, callback=scan_display)

try:
    print("Press BOOTSEL to switch the pattern.")
    index = 0
    name, buffer = PATTERNS[index]
    print(name)
    show(buffer)
    while True:
        wait_bootsel_press()
        # 全点灯 > 左上 > 右上 > 右下 > 左下 > 全消灯 > 全点灯 ... と切り替える
        index = (index + 1) % len(PATTERNS)
        name, buffer = PATTERNS[index]
        print(name)
        show(buffer)
except KeyboardInterrupt:
    pass
finally:
    timer.deinit()
    clear_display()
