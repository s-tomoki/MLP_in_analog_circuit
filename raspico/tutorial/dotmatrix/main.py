import time

from machine import Pin, Timer

# --- GPIOピンの設定 ---

COL_PINS = [8, 13, 7, 11, 0, 6, 1, 4]  # 列(COL)ピン
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


clear_display()

# --- 表示データバッファ (8x8 ビットパターン) ---
# 例: 「ハートマーク」のパターン (1: 点灯, 0: 消灯)
buffer = [
    0b00000000,
    0b01100110,
    0b11111111,
    0b11111111,
    0b01111110,
    0b00111100,
    0b00011000,
    0b00000000,
]


current_row = 0


# --- ダイナミック点灯用タイマー割り込みハンドラ ---
def scan_display(timer):
    global current_row

    # 前の行を消灯（ゴースト点灯防止）
    rows[current_row].value(0)

    # 次の行へ更新
    current_row = (current_row + 1) % 8

    # 現在の行のデータを列ピンに出力
    # カソードコモンの場合、bit=1 の時に COL を LOW(0) にして電流を流す
    row_data = buffer[current_row]
    for col_idx in range(8):
        bit = (row_data >> (7 - col_idx)) & 1
        cols[col_idx].value(0 if bit else 1)

    # 現在の行を有効化（アノード行を HIGH(1) にする）
    rows[current_row].value(1)


# --- タイマー起動 (約 1kHz / 各行を約 2ms 周期でスキャン) ---
timer = Timer()
timer.init(freq=1000, mode=Timer.PERIODIC, callback=scan_display)

# メインループ（ダイナミック点灯はタイマーで背景実行されているため自由に使用可能）
try:
    while True:
        # ここで buffer の書き換えを行うことでアニメーションなども可能
        time.sleep(1)
except KeyboardInterrupt:
    timer.deinit()
    clear_display()
