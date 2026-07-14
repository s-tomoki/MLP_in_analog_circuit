#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse a MULTI-STEP (mc_res mode, i.e. ".step param run 1 N 1") LTspice .log
file and summarize the Monte Carlo results per .meas label.

This is a SEPARATE script from analyze_results.py because the log format
for multi-step runs is fundamentally different from a single-run log:

    Measurement: vph_lo
      step	AVG(V(out))	FROM	TO
         1	1.86695420742	8.86478898e-05	0.00091135211
         2	1.97342526913	8.86478898e-05	0.00091135211
         ...
    <blank line>
    Measurement: vph_oh1
      step	AVG(V(out))	FROM	TO
         1	...
    ...

(Confirmed against an actual LTspice 24.1.7 log; plain ASCII/CRLF, no
UTF-16 issue in this case.)

By default, "vph_cum*" labels are excluded from the summary (per current
experiment focus: DC-offset behavior of vph_lo/vph_hi and per-channel
resistor-tolerance behavior of vph_oh*). Use --include-cum to include them,
or --labels to pick an explicit subset.

Usage:
    python3 mc_res_analyze.py fanin_N2_mcp6232_rin_scale_mc_res.log \
        --n 2 --scale-mode rin_scale --rf0 10e3 --rin0 10e3 \
        --vhi 1.0 --vlo -1.0 \
        --fail-threshold-pct 5.0 \
        --csv-out mc_raw.csv
"""
import argparse
import csv
import re
import statistics
import sys

import fanin_scaling


def parse_mc_log(path):
    """Returns dict: label -> list of (step:int, value:float)."""
    blocks = {}
    current = None
    header_re = re.compile(r"^\s*Measurement:\s*(\S+)")
    row_re = re.compile(r"^\s*(\d+)\s+([\-0-9.eE+]+)\s+([\-0-9.eE+]+)\s+([\-0-9.eE+]+)\s*$")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            m = header_re.match(line)
            if m:
                current = m.group(1)
                blocks[current] = []
                continue
            if current is None:
                continue
            m2 = row_re.match(line)
            if m2:
                step = int(m2.group(1))
                value = float(m2.group(2))
                blocks[current].append((step, value))
    return blocks


def ideal_for_label(label, n, rin, rf, vhi, vlo):
    """Ideal output voltage for a given phase label, by circuit symmetry."""
    gain = rf / rin
    if label == "vph_lo":
        return -gain * n * vlo
    if label == "vph_hi":
        return -gain * n * vhi
    m = re.match(r"vph_oh(\d+)$", label)
    if m:
        # exactly one channel Hi, the rest Lo -- same ideal for any k by symmetry
        return -gain * (vhi + (n - 1) * vlo)
    m = re.match(r"vph_cum(\d+)$", label)
    if m:
        k = int(m.group(1))
        return -gain * (k * vhi + (n - k) * vlo)
    return None


def select_labels(all_labels, args):
    if args.labels:
        wanted = [s.strip() for s in args.labels.split(",")]
        missing = [w for w in wanted if w not in all_labels]
        if missing:
            print(f"WARNING: requested labels not found in log: {missing}", file=sys.stderr)
        return [w for w in wanted if w in all_labels]
    if args.include_cum:
        return sorted(all_labels)
    return sorted(ls for ls in all_labels if not ls.startswith("vph_cum"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logfile")
    ap.add_argument("--n", type=int, required=True)
    fanin_scaling.add_scaling_args(ap)
    ap.add_argument("--vhi", type=float, default=1.0)
    ap.add_argument("--vlo", type=float, default=-1.0)
    ap.add_argument(
        "--fail-threshold-pct",
        type=float,
        default=5.0,
        help="target spec in %%FS; samples with |err%%FS| beyond " "this are counted as failing",
    )
    ap.add_argument(
        "--labels",
        default=None,
        help="comma-separated explicit label list, e.g. "
        "vph_hi,vph_oh1 (overrides --include-cum)",
    )
    ap.add_argument(
        "--include-cum", action="store_true", help="include vph_cum* labels (excluded by default)"
    )
    ap.add_argument(
        "--csv-out",
        default=None,
        help="write raw per-run data (label,step,value,ideal,err_pct) "
        "to this CSV, for histogram/plotting use",
    )
    args = ap.parse_args()

    blocks = parse_mc_log(args.logfile)
    if not blocks:
        print(
            "No 'Measurement:' blocks found -- is this really a multi-step "
            "(mc_res) log? Single-run logs use analyze_results.py instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    labels = select_labels(set(blocks.keys()), args)
    if not labels:
        print("No labels left to analyze after filtering.", file=sys.stderr)
        sys.exit(1)

    rf, rin = fanin_scaling.compute_rf_rin(args.n, args.scale_mode, args.rf0, args.rin0)
    print(f"# N={args.n} scale_mode={args.scale_mode} -> Rf={rf:.6g} Rin={rin:.6g}")

    full_scale = abs(
        ideal_for_label("vph_hi", args.n, rin, rf, args.vhi, args.vlo)
        - ideal_for_label("vph_lo", args.n, rin, rf, args.vhi, args.vlo)
    )

    csv_rows = []
    fail_col = f"fail%(>+-{args.fail_threshold_pct:.3g}%FS)"
    print(
        f"{'label':<10} {'N':>4} {'mean[V]':>10} {'std[V]':>10} "
        f"{'min[V]':>10} {'max[V]':>10} {'ideal[V]':>10} "
        f"{'mean_err%FS':>12} {fail_col:>20}"
    )

    for label in labels:
        rows = blocks[label]
        if not rows:
            continue
        ideal = ideal_for_label(label, args.n, rin, rf, args.vhi, args.vlo)
        values = [v for _, v in rows]
        mean_v = statistics.fmean(values)
        std_v = statistics.stdev(values) if len(values) > 1 else 0.0
        min_v = min(values)
        max_v = max(values)
        mean_err_pct = (mean_v - ideal) / full_scale * 100 if full_scale else float("nan")

        n_fail = 0
        for step, v in rows:
            err_pct = (v - ideal) / full_scale * 100 if full_scale else float("nan")
            if abs(err_pct) > args.fail_threshold_pct:
                n_fail += 1
            csv_rows.append((label, step, v, ideal, err_pct))
        fail_rate_pct = 100.0 * n_fail / len(rows)

        print(
            f"{label:<10} {len(rows):>4} {mean_v:>10.5g} {std_v:>10.5g} "
            f"{min_v:>10.5g} {max_v:>10.5g} {ideal:>10.5g} "
            f"{mean_err_pct:>12.4f} {fail_rate_pct:>19.2f}%"
        )

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "step", "value_V", "ideal_V", "err_pctFS"])
            w.writerows(csv_rows)
        print(f"\nwrote raw data: {args.csv_out} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
