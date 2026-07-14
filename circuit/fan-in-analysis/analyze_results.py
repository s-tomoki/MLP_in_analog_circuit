#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse a single-run (opamp_err mode) LTSpice .log file for .meas TRAN results
and compare against ideal values.

NOTE: This parser targets the SINGLE-RUN .meas log format:
    vph_lo: AVG(v(out))=-1.2345e-03 FROM ...
For mc_res (Monte Carlo, multi-.step) logs, LTSpice writes a step table
instead of this simple format. For that case, use PyLTSpice's LTSteps
utility (`pip install PyLTSpice`) which is purpose-built for step-table
logs -- a hand-rolled regex here would be fragile across LTSpice versions.

Usage:
    python3 analyze_results.py fanin_N2_mcp6232_rin_scale_opamp_err.log --n 2 \
        --scale-mode rin_scale --rf0 10e3 --rin0 10e3 --vhi 1.0 --vlo -1.0
"""
import argparse
import re
import sys

import fanin_scaling


def parse_log(path):
    """Returns dict: label -> float value, from lines like
    'vph_lo: AVG(v(out))=-1.2345e-03 FROM ...'"""
    pattern = re.compile(r"(?i)^\s*(vph_\w+)\s*:\s*AVG\(v\(out\)\)\s*=\s*([\-0-9.eE+]+)")
    results = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                results[m.group(1).lower()] = float(m.group(2))
    return results


def ideal_values(n, rin, rf, vhi, vlo):
    gain = rf / rin
    ideal = {}
    ideal["vph_lo"] = -gain * n * vlo
    for k in range(1, n + 1):
        ideal[f"vph_oh{k}"] = -gain * (vhi + (n - 1) * vlo)
    for k in range(1, n + 1):
        ideal[f"vph_cum{k}"] = -gain * (k * vhi + (n - k) * vlo)
    ideal["vph_hi"] = -gain * n * vhi
    return ideal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--n", type=int, required=True)
    fanin_scaling.add_scaling_args(ap)
    ap.add_argument("--vhi", type=float, default=1.0)
    ap.add_argument("--vlo", type=float, default=-1.0)
    args = ap.parse_args()

    rf, rin = fanin_scaling.compute_rf_rin(args.n, args.scale_mode, args.rf0, args.rin0)
    print(f"# N={args.n} scale_mode={args.scale_mode} -> Rf={rf:.6g} Rin={rin:.6g}")

    measured = parse_log(args.logfile)
    if not measured:
        print(
            "No '.meas' results found. Is this a single-run opamp_err log? "
            "(mc_res multi-step logs need a different parser -- see PyLTSpice LTSteps.)",
            file=sys.stderr,
        )
        sys.exit(1)

    ideal = ideal_values(args.n, rin, rf, args.vhi, args.vlo)
    full_scale = ideal["vph_hi"] - ideal["vph_lo"]

    print(f"{'label':<10} {'measured[V]':>12} {'ideal[V]':>10} {'err[%FS]':>10}")
    for label in sorted(measured, key=lambda s: (len(s), s)):
        meas_v = measured[label]
        ideal_v = ideal.get(label)
        if ideal_v is None:
            continue
        err_pct = (meas_v - ideal_v) / full_scale * 100 if full_scale != 0 else float("nan")
        print(f"{label:<10} {meas_v:>12.6g} {ideal_v:>10.6g} {err_pct:>9.3f}%")


if __name__ == "__main__":
    main()
