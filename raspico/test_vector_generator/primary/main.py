import time

import rp2
from machine import UART, Pin, Timer

# --- 表示するテストベクタ ---
# Pico 上のパス（Pico のルート "/" からの相対パス）。
CSV_PATH = "train_0.csv"

IMG_SIZE = 7  # テストベクタは 7x7 = 49 画素
DOT_SIZE = 8  # ドットマトリックスは 8x8

# --- テストベクタ出力用のマスク (1: secondary へ出力する画素) ---
# 1 の画素を行優先で並べた順に、出力要素 0, 1, 2, ... となる（secondary の OUT_PINS[i] に対応）。
# ※仮の値。アナログ回路の入力に合わせて書き換えること（1 の数は secondary の出力ピン数 23 以下）。
MASK = [
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
]  # fmt: skip
OUT_INDEX = [k for k in range(IMG_SIZE * IMG_SIZE) if MASK[k]]

# --- GPIOピンの設定 (display/main.py と同じ) ---

COL_PINS = [4, 1, 6, 0, 11, 7, 13, 8]  # 列(COL)ピン
ROW_PINS = [12, 2, 3, 9, 5, 10, 14, 15]  # 行(ROW)ピン

# --- secondary との通信 (UART0: GP16=TX, GP17=RX) ---
UART_TX_PIN = 16
UART_RX_PIN = 17
BAUDRATE = 115200
RESPONSE_TIMEOUT_MS = 200  # 1 回の応答待ち時間
RETRIES = 3  # 応答が無い／期待と異なる場合の再送回数
PING_INTERVAL_MS = 250  # PING 間隔（基板 LED はこの周期で反転 = 2Hz 点滅）
CONNECT_TIMEOUT_MS = 10000  # この時間 PONG が返らなければ終了


# ピンオブジェクトの初期化
rows = [Pin(p, Pin.OUT) for p in ROW_PINS]
cols = [Pin(p, Pin.OUT) for p in COL_PINS]
led = Pin("LED", Pin.OUT)  # 基板上の LED（消灯 > 2Hz点滅: PING中 > 点灯: 接続済み）

# RX は相手が未起動でも High (アイドル) に保つ
Pin(UART_RX_PIN, Pin.IN, Pin.PULL_UP)
uart = UART(0, baudrate=BAUDRATE, tx=Pin(UART_TX_PIN), rx=Pin(UART_RX_PIN))


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
    """CSV（ヘッダなし, 1 行 49 列の 0/1）を読み込む。

    戻り値: (49 要素のリストのリスト, 8x8 ビットパターンのリスト)
    """
    vectors = []
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
            vectors.append(vec)
            frames.append(vector_to_buffer(vec))
    return vectors, frames


def compress(vec):
    """49 要素のベクタをマスクで圧縮し、整数にする（bit i = 出力要素 i）。"""
    bits = 0
    for i, k in enumerate(OUT_INDEX):
        if vec[k]:
            bits |= 1 << i
    return bits


def show(buf):
    """ビットパターンを REPL に表示する（デバッグ用）。"""
    for row_data in buf:
        print(
            "".join("#" if (row_data >> (DOT_SIZE - 1 - c)) & 1 else "." for c in range(DOT_SIZE))
        )


# --- 表示データバッファ (8x8 ビットパターン, 1: 点灯, 0: 消灯) ---
ALL_ON = [0b11111111] * DOT_SIZE
CROSS = [  # 通信エラー表示
    0b10000001,
    0b01000010,
    0b00100100,
    0b00011000,
    0b00011000,
    0b00100100,
    0b01000010,
    0b10000001,
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


# --- secondary との通信 ---
# 1 行 = 1 コマンド（ASCII, 改行終端）。データは 6 桁の 16 進数（bit i = 出力要素 i）。
#   PING       -> PONG
#   INIT       -> OK          待機バッファと出力を 0 にする
#   W hhhhhh   -> OK          待機バッファへ書き込む（出力は変えない）
#   R          -> D hhhhhh    待機バッファを読み返す
#   A          -> OK hhhhhh   待機バッファを出力し、実際のピン状態を返す
def to_hex(bits):
    return "{:06X}".format(bits)


def send_line(line):
    # 以前の応答の残りやノイズを捨ててから送る
    while uart.any():
        uart.read(uart.any())
    uart.write(line + "\n")


def read_line(timeout_ms):
    """1 行受信して文字列で返す。タイムアウト・不正なデータの場合は None。"""
    line = b""
    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
        if not uart.any():
            time.sleep_ms(1)
            continue
        c = uart.read(1)
        if c == b"\n":
            try:
                return line.decode().strip()
            except UnicodeError:
                return None
        line += c
    return None


def transact(cmd, expected):
    """cmd を送り、応答が expected になるまで最大 RETRIES 回試す。

    戻り値: (成功したか, 最後に受信した応答)
    """
    resp = None
    for _ in range(RETRIES):
        send_line(cmd)
        resp = read_line(RESPONSE_TIMEOUT_MS)
        if resp == expected:
            return True, resp
    return False, resp


def wait_connection():
    """PONG が返るまで PING を繰り返す。待っている間は基板 LED を 2Hz で点滅させる。"""
    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < CONNECT_TIMEOUT_MS:
        led.toggle()
        t = time.ticks_ms()
        send_line("PING")
        if read_line(PING_INTERVAL_MS) == "PONG":
            return True
        rest = PING_INTERVAL_MS - time.ticks_diff(time.ticks_ms(), t)
        if rest > 0:
            time.sleep_ms(rest)
    return False


def init_secondary():
    """secondary の待機バッファと出力を 0 にし、読み返して確認する。"""
    for cmd, expected in [("INIT", "OK"), ("R", "D " + to_hex(0))]:
        ok, resp = transact(cmd, expected)
        if not ok:
            print("ERROR: {} -> {} (expected {})".format(cmd, resp, expected))
            return False
    return True


def send_vector(bits):
    """書き込み → 読み返し → 出力切り替え（実際のピン状態を確認）を行う。"""
    h = to_hex(bits)
    for cmd, expected in [("W " + h, "OK"), ("R", "D " + h), ("A", "OK " + h)]:
        ok, resp = transact(cmd, expected)
        if not ok:
            print("ERROR: {} -> {} (expected {})".format(cmd, resp, expected))
            return False
    return True


def main():
    global buffer

    # 1. 初期化して全点灯、基板 LED は消灯
    led.off()
    clear_display()
    timer = Timer()
    timer.init(freq=1000, mode=Timer.PERIODIC, callback=scan_display)

    try:
        # secondary との接続・初期化
        print("Connecting to secondary...")
        if not wait_connection():
            led.off()
            print("ERROR: secondary did not respond within {} ms".format(CONNECT_TIMEOUT_MS))
            return
        if not init_secondary():
            led.off()
            print("ERROR: failed to initialize secondary")
            return
        led.on()
        print("Connected.")

        # 2. テストベクタを読み込む
        vectors, frames = load_vectors(CSV_PATH)
        print("Loaded {} vectors from {}".format(len(vectors), CSV_PATH))
        print("Press BOOTSEL to show the next vector.")

        # index = -1 は全点灯状態（secondary は全 0 を出力）
        index = -1
        failed = False
        while True:
            wait_bootsel_press()
            # 前回失敗していたら同じベクタを再送する
            if not failed:
                index += 1
                if index >= len(vectors):
                    index = -1

            if index >= 0:
                # 3. 表示し、4. マスクで圧縮
                buffer = frames[index]
                bits = compress(vectors[index])
                print(
                    "[{}/{}] output {} ({:023b})".format(
                        index + 1, len(vectors), to_hex(bits), bits
                    )
                )
                show(buffer)
            else:
                # 最終行の次は全点灯、出力は全 0
                buffer = ALL_ON
                bits = 0
                print("All on, output {}".format(to_hex(bits)))

            # 5-8. secondary へ送信・読み返し・切り替え
            failed = not send_vector(bits)
            if failed:
                buffer = CROSS
                print("Send failed. Press BOOTSEL to retry.")
    except KeyboardInterrupt:
        pass
    finally:
        timer.deinit()
        clear_display()


main()
