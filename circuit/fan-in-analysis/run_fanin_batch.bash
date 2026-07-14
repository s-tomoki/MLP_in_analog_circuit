#!/usr/bin/env bash
# run_fanin_batch.sh
#
# WSL上で fan-in 加算回路のネットリスト生成 -> LTspice(Windows側exe)バッチ実行
# を一括で行うスクリプト。
#
# 前提:
#   - gen_fanin_netlist.py / fanin_scaling.py が同じディレクトリにある
#   - LTspiceはWindows側にインストールされている(WSL側からexeパスを直接指定)
#   - wslpath コマンドが使える(WSL標準搭載)
#
# 使い方:
#   1. 下記 CONFIG セクションの LTSPICE_EXE に、Windows側のLTspice実行ファイルの
#      パスをWSL形式(/mnt/c/...)で指定する
#      例: LTSPICE_EXE="/mnt/c/Program Files/ADI/LTspice/LTspice.exe"
#           (旧バージョンなら /mnt/c/Program Files/LTC/LTspiceXVII/XVIIx64.exe など)
#   2. bash run_fanin_batch.sh

set -euo pipefail

# ============================================================
# CONFIG (ここを編集する)
# ============================================================

# LTspice実行ファイルのパス(WSL形式 /mnt/c/... で指定)。
# 未設定のまま実行するとエラーで止まります。
LTSPICE_EXE="/mnt/c/Users/sugiu/AppData/Local/Programs/ADI/LTspice/LTspice.exe"

# ネットリスト生成・解析の対象
N_LIST="4"                 # カンマ区切り。gen_fanin_netlist.py --n-list にそのまま渡す
PRESETS=("mcp6232" "njm2732d")  # 対象IC
SCALE_MODE="rin_scale"          # rin_scale または rf_scale
RF0="10e3"
RIN0="10e3"
TOL="0.05"
MC_RUNS="100"
VHI="1.0"
VLO="-1.0"
FAIL_THRESHOLD_PCT="5.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${SCRIPT_DIR}/out"
SUMMARY_DIR="${SCRIPT_DIR}/summary"
TIMING_LOG="${SCRIPT_DIR}/timing_log.csv"

# ============================================================
# 事前チェック
# ============================================================

if [[ -z "$LTSPICE_EXE" ]]; then
    echo "ERROR: LTSPICE_EXE が未設定です。スクリプト冒頭のCONFIGセクションで" >&2
    echo "       Windows側LTspice実行ファイルのパス(WSL形式 /mnt/c/...)を設定してください。" >&2
    exit 1
fi
if [[ ! -f "$LTSPICE_EXE" ]]; then
    echo "ERROR: LTSPICE_EXE で指定したファイルが見つかりません: $LTSPICE_EXE" >&2
    exit 1
fi
if ! command -v wslpath >/dev/null 2>&1; then
    echo "ERROR: wslpath コマンドが見つかりません。WSL環境で実行してください。" >&2
    exit 1
fi

mkdir -p "$OUTDIR" "$SUMMARY_DIR"

echo "label,n,preset,scale_mode,elapsed_sec" > "$TIMING_LOG"

# ============================================================
# 1. ネットリスト生成
# ============================================================

echo "=== [1/3] ネットリスト生成 (N=${N_LIST}, scale_mode=${SCALE_MODE}) ==="
for preset in "${PRESETS[@]}"; do
    python3 "${SCRIPT_DIR}/gen_fanin_netlist.py" \
        --n-list "$N_LIST" \
        --preset "$preset" \
        --mode mc_res \
        --scale-mode "$SCALE_MODE" \
        --rf0 "$RF0" --rin0 "$RIN0" \
        --tol "$TOL" --mc-runs "$MC_RUNS" \
        --vhi "$VHI" --vlo "$VLO" \
        --outdir "$OUTDIR"
done

# ============================================================
# 2. LTspiceバッチ実行
# ============================================================

echo "=== [2/3] LTspiceバッチ実行 ==="
IFS=',' read -ra N_ARRAY <<< "$N_LIST"

for preset in "${PRESETS[@]}"; do
    for n in "${N_ARRAY[@]}"; do
        base="fanin_N${n}_${preset}_${SCALE_MODE}_mc_res"
        cir_path="${OUTDIR}/${base}.cir"
        log_path="${OUTDIR}/${base}.log"

        if [[ ! -f "$cir_path" ]]; then
            echo "WARNING: ${cir_path} が見つかりません。スキップします。" >&2
            continue
        fi

        # LTspiceに渡す際はWindows形式のパスに変換する
        win_cir_path="$(wslpath -w "$cir_path")"

        echo "--- 実行中: N=${n} preset=${preset} ---"
        start_ts=$(date +%s)

        "$LTSPICE_EXE" -b "$win_cir_path"

        end_ts=$(date +%s)
        elapsed=$((end_ts - start_ts))
        echo "    完了: ${elapsed} 秒"
        echo "${base},${n},${preset},${SCALE_MODE},${elapsed}" >> "$TIMING_LOG"

        if [[ ! -f "$log_path" ]]; then
            echo "WARNING: ${log_path} が生成されませんでした(LTspice側でエラーの可能性)。" >&2
        fi
    done
done

# ============================================================
# 3. 解析 (mc_res_analyze.py を各ログに対して実行)
# ============================================================

echo "=== [3/3] 解析 (mc_res_analyze.py) ==="
for preset in "${PRESETS[@]}"; do
    for n in "${N_ARRAY[@]}"; do
        base="fanin_N${n}_${preset}_${SCALE_MODE}_mc_res"
        log_path="${OUTDIR}/${base}.log"
        csv_path="${SUMMARY_DIR}/${base}_raw.csv"
        summary_path="${SUMMARY_DIR}/${base}_summary.txt"

        if [[ ! -f "$log_path" ]]; then
            continue
        fi

        python3 "${SCRIPT_DIR}/mc_res_analyze.py" "$log_path" \
            --n "$n" \
            --scale-mode "$SCALE_MODE" --rf0 "$RF0" --rin0 "$RIN0" \
            --vhi "$VHI" --vlo "$VLO" \
            --fail-threshold-pct "$FAIL_THRESHOLD_PCT" \
            --csv-out "$csv_path" \
            | tee "$summary_path"
        echo ""
    done
done

echo "=== 完了 ==="
echo "タイミングログ: $TIMING_LOG"
echo "解析結果: $SUMMARY_DIR/"
