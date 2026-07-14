#!/usr/bin/env bash
# run_fanin_batch.sh
#
# WSL上で fan-in 加算回路の「フェーズ単位分割ネットリスト」を生成し、
# xargs -P による並列実行でLTspice(Windows側exe)バッチ処理を行うスクリプト。
#
# 1ジョブ = (1つのN × 1つのIC × 1つのフェーズ: lo/oh<k>/cum<k>/hi) のモンテカルロ100回。
# 出力は <outdir>/<preset>/N<n>/ に整理され、各ジョブ完了後に .raw(波形データ、
# 未使用のため)を削除、N×preset単位で .cir+.log をひとつのzipにまとめてから
# 展開済みファイルは削除する(zipだけが残る)。
#
# 前提:
#   - gen_fanin_netlist.py / fanin_scaling.py / templates/netlist.cir.j2 /
#     mc_res_analyze.py が同じディレクトリにある
#   - jinja2 がインストールされている (pip install jinja2 --break-system-packages)
#   - LTspiceはWindows側にインストールされている(WSL側からexeパスを直接指定)
#   - wslpath / xargs コマンドが使える(WSL標準搭載)
#
# 使い方:
#   1. 下記 CONFIG セクションの LTSPICE_EXE に、Windows側のLTspice実行ファイルの
#      パスをWSL形式(/mnt/c/...)で指定する
#   2. bash run_fanin_batch.sh

set -uo pipefail
# 注: 単一ジョブの失敗で全体を止めたくないので、gen_fanin_netlist.py以降は
#     `set -e` を使わず、個別にリターンコードをチェックする。

# ============================================================
# CONFIG (ここを編集する)
# ============================================================

# LTspice実行ファイルのパス(WSL形式 /mnt/c/... で指定)。
# 未設定のまま実行するとエラーで止まります。
LTSPICE_EXE="/mnt/c/Users/sugiu/AppData/Local/Programs/ADI/LTspice/LTspice.exe"

# 並列実行数。デフォルト8(WSLの `nproc` 値以下を推奨)。
PARALLEL_JOBS=4

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
WORKDIR="${SCRIPT_DIR}/out_phases"        # ジョブ実行中の作業ディレクトリ(最終的に空になる)
ZIPDIR="${SCRIPT_DIR}/zips"               # 最終成果物(zip)の置き場所
SUMMARY_DIR="${SCRIPT_DIR}/summary_phases"
TIMING_LOG="${SCRIPT_DIR}/timing_log_phases.csv"
JOBLIST="${SCRIPT_DIR}/.joblist.txt"

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
for cmd in wslpath xargs python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: ${cmd} コマンドが見つかりません。" >&2
        exit 1
    fi
done
if ! python3 -c "import jinja2" >/dev/null 2>&1; then
    echo "ERROR: jinja2 がインストールされていません。" >&2
    echo "       pip install jinja2 --break-system-packages を実行してください。" >&2
    exit 1
fi

mkdir -p "$WORKDIR" "$ZIPDIR" "$SUMMARY_DIR"
echo "label,elapsed_sec" > "$TIMING_LOG"

# ============================================================
# 1. ネットリスト生成 (フェーズ単位分割, --all-phases)
#    出力先: out_phases/<preset>/N<n>/*.cir
# ============================================================

echo "=== [1/4] ネットリスト生成 (N=${N_LIST}, scale_mode=${SCALE_MODE}, フェーズ分割) ==="
for preset in "${PRESETS[@]}"; do
    python3 "${SCRIPT_DIR}/gen_fanin_netlist.py" \
        --n-list "$N_LIST" \
        --preset "$preset" \
        --mode mc_res \
        --scale-mode "$SCALE_MODE" \
        --rf0 "$RF0" --rin0 "$RIN0" \
        --tol "$TOL" --mc-runs "$MC_RUNS" \
        --vhi "$VHI" --vlo "$VLO" \
        --all-phases \
        --outdir "$WORKDIR"
done

find "$WORKDIR" -name "*.cir" | sort > "$JOBLIST"
n_jobs=$(wc -l < "$JOBLIST")
echo "生成ジョブ数: ${n_jobs}"

# ============================================================
# 2. LTspiceバッチ実行 (xargs -P で並列化)
#    各ジョブ完了後、.raw(波形データ、未使用)は即削除する
# ============================================================

echo "=== [2/4] LTspiceバッチ実行 (並列数=${PARALLEL_JOBS}) ==="

run_one_job() {
    local cir_path="$1"
    local base
    base="$(basename "$cir_path" .cir)"
    local dir
    dir="$(dirname "$cir_path")"
    local log_path="${dir}/${base}.log"
    local raw_path="${dir}/${base}.raw"
    local win_cir_path
    win_cir_path="$(wslpath -w "$cir_path")"

    local start_ts end_ts elapsed
    start_ts=$(date +%s)
    "$LTSPICE_EXE" -b "$win_cir_path" >/dev/null 2>&1
    local rc=$?
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    # .raw(波形データ)は今回使わないので容量削減のため即削除
    rm -f "$raw_path"

    if [[ $rc -ne 0 ]]; then
        echo "FAILED (${elapsed}s): ${base}" >&2
    elif [[ ! -f "$log_path" ]]; then
        echo "NO LOG (${elapsed}s): ${base}" >&2
    else
        echo "ok (${elapsed}s): ${base}"
    fi
    echo "${base},${elapsed}" >> "${TIMING_LOG}"
}
export -f run_one_job
export LTSPICE_EXE TIMING_LOG

start_all=$(date +%s)
cat "$JOBLIST" | xargs -P "$PARALLEL_JOBS" -I{} bash -c 'run_one_job "$@"' _ {}
end_all=$(date +%s)
echo "全ジョブ実行時間(壁時計): $((end_all - start_all)) 秒"

# ============================================================
# 3. preset x N 単位で .cir + .log を1つのzipにまとめ、
#    展開済みファイル(ディレクトリ)は削除する
# ============================================================

echo "=== [3/4] zip化 (preset x N 単位) ==="
IFS=',' read -ra N_ARRAY <<< "$N_LIST"

for preset in "${PRESETS[@]}"; do
    for n in "${N_ARRAY[@]}"; do
        n_dir="${WORKDIR}/${preset}/N${n}"
        [[ -d "$n_dir" ]] || continue
        zip_path="${ZIPDIR}/${preset}_N${n}.zip"

        python3 - "$n_dir" "$zip_path" << 'PYEOF'
import sys, os, glob, zipfile
d, zpath = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(d, "*.cir")) + glob.glob(os.path.join(d, "*.log")))
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, os.path.basename(f))
print(f"zipped {len(files)} files -> {zpath}")
PYEOF

        rm -rf "$n_dir"
    done
done

# out_phases 配下が空になった preset ディレクトリも掃除
find "$WORKDIR" -type d -empty -delete 2>/dev/null || true

# ============================================================
# 4. 解析 (mc_res_analyze.py はzipを直接読める)
# ============================================================

echo "=== [4/4] 解析 (mc_res_analyze.py, zipを直接読み込み) ==="
for preset in "${PRESETS[@]}"; do
    for n in "${N_ARRAY[@]}"; do
        zip_path="${ZIPDIR}/${preset}_N${n}.zip"
        [[ -f "$zip_path" ]] || continue
        summary_path="${SUMMARY_DIR}/${preset}_N${n}_summary.txt"
        csv_path="${SUMMARY_DIR}/${preset}_N${n}_raw.csv"

        python3 "${SCRIPT_DIR}/mc_res_analyze.py" "$zip_path" \
            --n "$n" --scale-mode "$SCALE_MODE" --rf0 "$RF0" --rin0 "$RIN0" \
            --vhi "$VHI" --vlo "$VLO" \
            --fail-threshold-pct "$FAIL_THRESHOLD_PCT" \
            --include-cum \
            --csv-out "$csv_path" \
            | tee "$summary_path"
        echo ""
    done
done

echo "=== 完了 ==="
echo "zip成果物: $ZIPDIR/"
echo "タイミングログ: $TIMING_LOG"
echo "解析結果: $SUMMARY_DIR/"
