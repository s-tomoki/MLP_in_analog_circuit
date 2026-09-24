# Raspberry Pi Pico & MicroPython

## 環境構築

### 初回のみ

#### デバイスの接続

初回の設定です。 Raspbery Pi Pico を PC へ接続した後、次の手順で進めます。

1. [usbipd](https://github.com/dorssel/usbipd-winhttps://github.com/dorssel/usbipd-win)
   を Windows へインストール
   1. PowerShell を管理者権限付きで起動
   2. `winget install usbipd` を実行
   3. 次のコマンド列を管理者権限付きの PowerShell で実行:
   ```bash
   usbipd --help # コマンドのインストールの確認
   usbipd list # USB デバイス一覧が表示される。
   ```
   コマンドを実行したとき、このように表示されます。
   ```text
   Connected:
   ```
BUSID VID:PID DEVICE STATE 2-5 2e8a:0003 USB 大容量記憶装置, RP2 Boot Not shared 2-7 27c6:658c
Goodix MOC Fingerprint Not shared 2-9 30c9:005f Integrated Camera, Integrated IR Camera,
Camera DFU Device Not shared 2-10 8087:0033 インテル(R) ワイヤレス Bluetooth(R) Not shared
`4. 次のコマンド列を管理者権限付きPowerShellで実行:` usbipd attach --wsl --busid=<BUSID> usbipd list # ここで
BUSID で指定したデバイスが ‘attached’ になる `これによりWSLから Pico が見えるようになります。 5. 次のコマンド列を WSL2 で実行：`bash
lsusb `ここで次のように Pico が見えていれば成功です。この場合は 2 番目が Pico です。` Bus 001 Device 001: ID 1d6b:0002
Linux Foundation 2.0 root hub Bus 001 Device 002: ID 2e8a:0003 Raspberry Pi RP2 Boot Bus
002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
``` 2. Pico に MicroPython 用ファームウェアを書き込む。 1. [MicroPython 公式ページの Pico W ページ](https://micropython.org/download/RPI_PICO_W/)から、最新の UF2 ファイル（作成時は v1.29.0 (2026-08-24)）をダウンロードする。 2. Pico を PC から切断し、 `BOOTSEL`
プッシュスイッチを押下したまま USB ケーブルを PC へ接続する。 3. エクスプローラが立ち上がる。ここにダウンロードした UF2 ファイルをコピーする。 4. Pico
の接続が自動的に切れる。

その後、[デバイス接続ルーチン]()を実行して Pico が WSL 上で認識されることを確認してください。

#### VSCode のセットアップ

Pico との接続確認を完了し、WSL に接続した VSCode を用意します。

拡張機能から [MicroPico](https://github.com/paulober/MicroPico)をインストールします。 （注：WSL 上であって
Windows 上ではない）

その後、VSCode のウィンドウを全て閉じ、新たに VSCode を立ち上げます。

左下に Pico との接続が表示されますので、 ‘Connected’ になることを確認してください。 ならない場合は ‘Pico Disconnected’ をクリックし、
‘Connect’ を選択してください。

接続すると Pico 上で動作する MicroPython の REPL が立ち上がります。 `print('hello, pico')`
と入力してエンターキーを入力することで、 ‘hello, pico’ と応答があれば環境設定は完了です。

### デバイス接続ルーチン

ここからは毎回の手順です。 Pico を接続する毎に実施してください。

1. 次のコマンド列を管理者権限付きの PowerShell で実行:
   ```bash
   usbipd list # USB デバイス一覧が表示
   usbipd attach --wsl --busid=<BUSID>
   usbipd list # 該当 BUSID デバイスが 'attached' になることを確認
   ```
2. 次のコマンド列を WSL2 で実行：
   ```bash
   lsusb # 'MicroPython Board in FS mode' と認識されていることを確認
   ```

## 使い方

### Tutorial

#### LED 操作

1. Pico と接続済みの VSCode を立ち上げます。
2. `tutorial/led/main.py` に対し、コマンドパレットを開き、 `MicroPico: Uplaod project to Board` を選択します。
3. 書込み完了後、下部ステータスバーの ‘Run’ もしくはコマンドパレットで `MicroPico: Run current file` を選択します。

これで LED が 1 秒ごとに明滅を繰り返すようになれば成功です。

以降のチュートリアルも、対象の `main.py` を開いて同じ手順（Upload → Run）で実行します。

#### ドットマトリックス LED

8x8 ドットマトリックス LED をダイナミック点灯で表示します（`tutorial/dotmatrix/main.py`）。表示面から見た位置と GPIO
の対応は次のとおりです。

| 行 (ROW, アノード) | GPIO | 列 (COL, カソード) | GPIO |
| --- | --- | --- | --- |
| 上から 1 行目 | GP12 | 左から 1 列目 | GP4 |
| 上から 2 行目 | GP2 | 左から 2 列目 | GP1 |
| 上から 3 行目 | GP3 | 左から 3 列目 | GP6 |
| 上から 4 行目 | GP9 | 左から 4 列目 | GP0 |
| 上から 5 行目 | GP5 | 左から 5 列目 | GP11 |
| 上から 6 行目 | GP10 | 左から 6 列目 | GP7 |
| 上から 7 行目 | GP14 | 左から 7 列目 | GP13 |
| 上から 8 行目 | GP15 | 左から 8 列目 | GP8 |

文字「F」が正しい向きで表示されれば成功です。「F」は上下左右非対称なので、左右が反転していれば列 (COL) の、上下が反転していれば行 (ROW)
の接続順が逆になっています。

#### スイッチによる表示切替

基板上の `BOOTSEL`
ボタンを押すたびにドットマトリックスの表示を切り替え、各行・各列の接続を確認します（`tutorial/sw_mat/main.py`）。配線はドットマトリックス LED
と同じです。

`BOOTSEL` を押すごとに次の順で表示が切り替わります。表示中のパターンは REPL にも出力されます。

1. 全点灯
2. 左上の角に L 字（┌）
3. 右上の角に L 字（┐）
4. 右下の角に L 字（┘）
5. 左下の角に L 字（└）
6. 全消灯
7. 1 へ戻る

全点灯で点灯しない LED があればその行・列の配線を、L 字の位置が違えば行・列の接続順を確認してください。

#### UART

UART の送信 (TX) と受信 (RX) をつなぎ、送った文字列を自分で受け取ります（`tutorial/uart/main.py`）。

1. `GP16` (UART0 TX) と `GP17` (UART0 RX) をジャンパ線で接続します。
2. プログラムを実行し、`BOOTSEL` ボタンを押します。
3. TX から `Hello, Pico` が送信され、RX で受信されます。

REPL に次のように表示されれば成功です。

```text
TX: Hello, Pico
RX: Hello, Pico
```

`RX: (no data) - check the wire between GP16 and GP17` と表示された場合は、`GP16` と `GP17`
の配線を確認してください。

### テストベクタ生成器

MNIST を 7x7 に縮小・ 2 値化したテストベクタを、ドットマトリックス LED
に表示し、アナログ回路の入力となるデジタル信号として出力します（`test_vector_generator/`）。

| ディレクトリ | 役割 |
| --- | --- |
| `display` | Pico 1 台で、テストベクタをドットマトリックス LED に表示するだけの確認用 |
| `primary` | テストベクタを表示し、UART で secondary へ送る |
| `secondary` | primary から受け取ったテストベクタを GPIO へ出力する（アナログ回路の入力へ接続） |

#### テストベクタ (CSV)

テストベクタは `model/step_by_step/mnist/export_test_vectors.py` で生成します。`test_vectors/` はその出力先
`model/step_by_step/mnist/test_vectors/` へのシンボリックリンクです。

- ファイル名は `<split>_<class>.csv`（例: `train_0.csv` は学習データのクラス 0）です。
- ヘッダなし、1 行が 1 ベクタで、49 列の 0/1 が並びます。k 列目は画素 (行, 列) = (k // 7, k % 7) です。
- ドットマトリックスには左上 7x7 に表示し、8 行目と 8 列目は常に消灯します。

表示するファイルは、各 `main.py` の `CSV_PATH`（Pico のルート `/` からのパス、既定値は `train_0.csv`）で指定します。CSV
ファイルを Pico へアップロードし、`CSV_PATH` をアップロード先のパスに合わせてください。

#### 表示のみ (display)

Pico 1 台とドットマトリックス LED で、テストベクタの内容を確認します。配線はチュートリアルの「ドットマトリックス LED」と同じです。

1. `test_vector_generator/display/main.py` を実行します。起動直後は全点灯です。
2. `BOOTSEL` ボタンを押すたびに、CSV の次のベクタを表示します。REPL にも `#`/`.` で同じ内容を出力します。
3. 最後のベクタの次は全点灯に戻ります。

#### primary / secondary 構成

Pico 2 台を UART でつなぎ、primary で表示したテストベクタを secondary の GPIO から出力します。

##### 配線

primary のドットマトリックス LED の配線はチュートリアルと同じです（GP0/GP1 をドットマトリックスで使うため、UART は GP16/GP17 を使います）。

| primary | secondary |
| --- | --- |
| GP16 (TX) | GP1 (RX) |
| GP17 (RX) | GP0 (TX) |
| GND | GND |

secondary は、出力要素 i を `OUT_PINS[i]`（GP2〜GP22, GP26, GP27 の 23 本）へ出力します。GP23〜25 は Pico W
の無線チップ用のため使いません。

##### 出力する画素の選択 (MASK)

49 画素のうち secondary へ出力する画素は、`primary/main.py` の `MASK`（7x7, 1: 出力する）で選びます。1
の画素を行優先で並べた順に、出力要素 0, 1, 2, … となります。1 の数は secondary の出力ピン数（23）以下にしてください。現在の `MASK`
は仮の値なので、アナログ回路の入力に合わせて書き換えてください。

##### 実行手順

1. secondary の `test_vector_generator/secondary/main.py` を実行します。基板 LED が 1Hz
   で点滅し、primary からの接続を待ちます。
2. primary の `test_vector_generator/primary/main.py` を実行します。ドットマトリックスが全点灯し、基板 LED が 2Hz
   で点滅しながら secondary へ接続します。
3. 接続すると両方の基板 LED が点灯し、primary の REPL に `Connected.` と表示されます。10 秒以内に接続できなければ primary
   は終了します。
4. primary の `BOOTSEL` ボタンを押すたびに、次のベクタを表示し、secondary の GPIO
   から出力します。最後のベクタの次は全点灯（secondary の出力は全 0）に戻ります。

送信に失敗すると、primary のドットマトリックスに ×（バツ）を表示します。もう一度 `BOOTSEL` を押すと、同じベクタを再送します。

| 基板 LED | primary | secondary |
| --- | --- | --- |
| 消灯 | 起動直後／接続失敗 | 起動直後 |
| 点滅 | 2Hz: 接続中 | 1Hz: 接続待ち |
| 点灯 | 接続済み | 接続済み |

##### 通信プロトコル

UART0、115200 bps で、1 行（ASCII、改行終端）が 1 コマンドです。データは 6 桁の 16 進数で、bit i が出力要素 i に対応します。

| コマンド | 応答 | 動作 |
| --- | --- | --- |
| `PING` | `PONG` | 接続確認 |
| `INIT` | `OK` | 待機バッファと出力を 0 にする |
| `W hhhhhh` | `OK` | 待機バッファへ書き込む（出力は変えない） |
| `R` | `D hhhhhh` | 待機バッファを読み返す |
| `A` | `OK hhhhhh` | 待機バッファを出力し、実際のピン状態を返す |

primary は 1 ベクタごとに `W` → `R` → `A` の順に送り、応答が期待どおりか確認します。応答がない、または期待と異なる場合は、各コマンドを最大 3
回まで再送します。
