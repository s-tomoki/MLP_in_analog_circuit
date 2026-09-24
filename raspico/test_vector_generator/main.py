import time

import rp2
from machine import Pin, Timer

# --- 表示するテストベクタ ---
# Pico 上のパス（Pico のルート "/" からの相対パス）。
# 事前に REPL や mpremote で CSV_PATH を定義しておけば、そちらを優先する。
#   例: mpremote exec "CSV_PATH='test_vectors/test_1.csv'" run test_vector_generator/main.py
CSV_PATH = "train_0.csv"

IMG_SIZE = 7  # テストベクタは 7x7 = 49 画素
DOT_SIZE = 8  # ドットマトリックスは 8x8

# --- GPIOピンの設定 (tutorial/dotmatrix と同じ) ---

COL_PINS = [4, 1, 6, 0, 11, 7, 13, 8]  # 列(COL)ピン
# COL_PINS = [8, 13, 7, 11, 0, 6, 1, 4]  # 列(COL)ピン
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


# --- テストベクタの読み込み ---
def vector_to_buffer(vec):
    """49 要素の 0/1 リストを 8x8 のビットパターンに変換する。

    k 番目の要素は画素 (row, col) = (k // 7, k % 7)。
    8 列目（最終列）と 8 行目（最終行）はパディングとして常に消灯。
    """
    buf = [0] * DOT_SIZE
    for r in range(IMG_SIZE):
        row_data = 0
        for c in range(IMG_SIZE):
            if vec[r * IMG_SIZE + c]:
                row_data |= 1 << (DOT_SIZE - 1 - c)
        buf[r] = row_data
    return buf


def load_vectors(path):
    """CSV（ヘッダなし, 1 行 49 列の 0/1）を読み込み、8x8 ビットパターンのリストを返す。"""
    frames = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            vec = [int(v) for v in line.split(",")]
            if len(vec) != IMG_SIZE * IMG_SIZE:
                raise ValueError(
                    "{}:{}: expected {} values, got {}".format(
                        path, line_no, IMG_SIZE * IMG_SIZE, len(vec)
                    )
                )
            frames.append(vector_to_buffer(vec))
    return frames


def show(buf):
    """ビットパターンを REPL に表示する（デバッグ用）。"""
    for row_data in buf:
        print(
            "".join("#" if (row_data >> (DOT_SIZE - 1 - c)) & 1 else "." for c in range(DOT_SIZE))
        )


# --- 表示データバッファ (8x8 ビットパターン, 1: 点灯, 0: 消灯) ---
ALL_ON = [0b11111111] * DOT_SIZE

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


# 1. 初期化して全点灯
clear_display()
timer = Timer()
timer.init(freq=1000, mode=Timer.PERIODIC, callback=scan_display)

try:
    # 2. テストベクタを読み込む
    frames = load_vectors(CSV_PATH)
    print("Loaded {} vectors from {}".format(len(frames), CSV_PATH))
    print("Press BOOTSEL to show the next vector.")

    # index = -1 は全点灯状態
    index = -1
    while True:
        wait_bootsel_press()
        index += 1
        if index < len(frames):
            # 3, 4. 次の行のデータを表示
            buffer = frames[index]
            print("[{}/{}]".format(index + 1, len(frames)))
            show(buffer)
        else:
            # 5. 最終行の次は全点灯に戻す
            index = -1
            buffer = ALL_ON
            print("All on")
except KeyboardInterrupt:
    pass
finally:
    timer.deinit()
    clear_display()
