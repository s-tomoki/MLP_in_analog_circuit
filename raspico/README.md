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
