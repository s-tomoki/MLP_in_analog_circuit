# Fan-in 加算回路 LTSpice 検証キット (N=2 試運転版)

## 構成

- `gen_fanin_netlist.py` : ネットリスト(.cir)生成スクリプト
- `analyze_results.py` : 単一 run(.log)を解析して理想値との誤差[%FS]を計算
- `out/` : 生成済みネットリスト 4 本(下記コマンド参照)

## 回路トポロジー

N 入力反転加算アンプ。非反転入力は 0V 固定(両電源、真の仮想接地)。 1 回の`.tran`内で以下のパターンを連続して流し、`.meas`で自動的に平均値を抽出します。

```
時間軸: [全Lo] [ワンホットch1] [ワンホットch2] ... [累積k=1] [累積k=2] ... [全Hi]
        1phase  1phase          1phase           1phase      1phase       1phase
```

N=2 の場合、`cum1`はワンホット ch1 と、`cum2`は全 Hi と数値的に一致します(冗長ですが自動抽出の一貫性のため残しています)。

## Step 1: デフォルト(汎用)のまま動作確認

```bash
python3 gen_fanin_netlist.py --n-list 2 --preset generic --mode opamp_err --outdir out
```
→ `out/fanin_N2_generic_opamp_err.cir`

## Step 2: 実 IC 値で確認

```bash
python3 gen_fanin_netlist.py --n-list 2 --preset mcp6232  --mode opamp_err --outdir out
python3 gen_fanin_netlist.py --n-list 2 --preset njm2732d --mode opamp_err --outdir out
```

両 IC とも単電源 1.8〜6.0V 品ですが、仮想接地の分圧回路を作らずに済むよう **両電源±2.5V(合計 5V、定格内)**
を自動設定しています(`--preset`のデフォルト値、 `--vsupply`で上書き可)。

## Step 3: MCP6232 + 抵抗誤差±5% モンテカルロ

```bash
python3 gen_fanin_netlist.py --n-list 2 --preset mcp6232 --mode mc_res --tol 0.05 --mc-runs 100 --outdir out
```
→ `.step param run 1 100 1` が付与され、Rin/Rf に`{mc(Rnom,0.05)}`が適用されます。

## LTSpice での実行(バッチモード)

```bash
# Windows
"C:\Program Files\ADI\LTspice\LTspice.exe" -b out\fanin_N2_generic_opamp_err.cir

# macOS
/Applications/LTspice.app/Contents/MacOS/LTspice -b out/fanin_N2_generic_opamp_err.cir
```
実行後、同名の `.log` ファイルが生成されます(`.meas`結果はこの中)。

## 解析

```bash
python3 analyze_results.py out/fanin_N2_generic_opamp_err.log --n 2 --rin 10e3 --rf 10e3 --vhi 1.0 --vlo -1.0
```

※ `mc_res`モードの`.log`は複数 run 分のステップテーブル形式になるため、上記の単純パーサーでは扱えません。 その場合は
`pip install PyLTSpice` 後、`PyLTSpice.LTSteps` を使ってください。

## トラブルシューティング: “This sub-circuit name is not defined.”

`.cir`テキストネットリストを直接バッチ実行する場合、GUI の回路図経由と違って
`UniversalOpAmp2`のサブサーキットが自動読み込みされない上、`UniversalOpAmp2`という 名前自体はライブラリ内で直接呼び出せる subckt
名ではありませんでした。 実際に動作した構成(実機検証済み)は以下の通りです。

1. `.include UniversalOpAmp2.lib` で読み込む(`.lib xxx.sub`形式ではなく`.include`+`.lib`拡張子)
2. ライブラリ内部の実モデル名(精度レベルにより `level1`/`level2`/... 等)を、
   `params:`キーワード付きで呼び出す独自の`.subckt`(`MyOpAmp`)でラップする
3. ピン順序は **`IN+ IN- VCC VEE OUT`**(5 番目が OUT)

生成スクリプトはこの構成をデフォルトで出力するように修正済みです。 `--opamp-lib`(既定
`UniversalOpAmp2.lib`)と`--opamp-model`(既定 `level2`)で 環境に応じて上書きできます。

## 実行前に確認してほしいこと

1. **精度レベル(`--opamp-model`)**: `level2`は Avol/GBW/Slew/Vos/Ib/Ios/rail/ilimit を
   持つ中間精度レベルです。ノイズ(En/In)まで見たい場合は`level3`系が必要になる可能性があるので、
   その場合はライブラリファイルを開いて対応レベルとパラメータ名を確認してください。
2. **MCP6232 の Ib**: データシートの数値表を確認できず、Ios と同オーダー(1pA)と仮定した推定値です。 Ib
   由来の誤差を厳密に評価したい場合は実際の値を確認の上、`--ib`で上書きしてください。
3. **NJM2732D の Vos と Ilimit**: 情報源によって「1mV(typ)」と「5mV(max)」の記載差がありました。 本スクリプトはワーストケース側の
   5mV を採用しています。また Ilimit はデータシートで 明記箇所を確認できず、25mA の推定値を使っています。

## 既知の設計上の割り切り

- 熱ドリフトや信号源インピーダンスは、SPICE レベルでは基本的に再現されません (電圧源は理想 0Ω、UniversalOpAmp2
  は固定パラメータで自己発熱なし)。これらは実機検証側の 観点として切り分けてください。
- CMRR/PSRR は UniversalOpAmp2 にパラメータが存在しないため、今回の IC 比較では反映されません。
