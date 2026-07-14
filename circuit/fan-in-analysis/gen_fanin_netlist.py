#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fan-in summing amplifier netlist generator for LTSpice (UniversalOpAmp2).

Circuit: N-input inverting summing amplifier (weighted-sum / "artificial
neuron" use case -- see fanin_scaling.py for the Rf/Rin(N) scaling rules
that keep the output range constant as N grows).

Generation is SINGLE-PHASE ONLY (the earlier "all 2N+2 phases in one
.tran" mode has been removed): each generated .cir simulates exactly one
test phase (lo, hi, oh<k>, or cum<k>) for one N. This is what allows
N x (2N+2) x preset combinations to be split into many small, independent
jobs and run in parallel (see run_fanin_batch.sh), rather than one big job
per (N, preset) where the slowest single job dominates wall-clock time
regardless of how many CPU cores are available.

Use --phase LABEL for one phase, or --all-phases to generate every phase
(lo, oh1..ohN, cum1..cumN, hi) for each N in --n-list in one call. Output
files are organized as <outdir>/<preset>/N<n>/*.cir.

Rendering is done via a Jinja2 template (templates/netlist.cir.j2); this
script's job is to compute all the PWL/measurement/parameter strings that
go into it. Requires `pip install jinja2` (or `--break-system-packages` if
your pip complains about externally-managed environments).

NOTE / CAVEATS:
  1. Pin order for the opamp instance is "IN+ IN- VCC VEE OUT" -- confirmed
     working (thanks to user testing). This differs from a naive guess of
     "IN+ IN- OUT V+ V-", so if you fork this for a different opamp macro,
     don't assume the pin order without checking.
  2. UniversalOpAmp2 is not directly callable as a subckt name in a bare
     .cir netlist. We wrap the actual internal subckt (default "level2",
     the medium-accuracy tier with Avol/GBW/Slew/Vos/Ib/Ios/rail/ilimit) in
     a local .subckt MyOpAmp so parameters can be passed with "params:".
  3. The library must be loaded with ".include UniversalOpAmp2.lib" --
     ".lib <name>.sub" did not work in the user's LTspice install.
  4. Avol is passed to the model in LINEAR V/V, not dB -- the level2
     model's internal formula (G={Avol/Rout}) uses Avol directly as a
     linear gain. Presets store avol_db (dB, human-familiar) and this
     script converts via 10**(avol_db/20) before emitting Avol=... .
  5. Ib for MCP6232-E/P is an ESTIMATED value (pA-order, same magnitude as
     Ios) -- the exact datasheet table entry could not be confirmed from
     available sources.
  6. In PHASE mode, source waveforms are constant (well, a single ramp to
     the target value over --rise-time, then held) rather than touring
     through all phases -- so per-job simulated duration is much shorter
     than FULL mode. Total CPU-seconds across all phase jobs for a given
     (N, preset) should be about the same as one FULL job for that (N,
     preset); the win is that many small jobs can be spread across CPU
     cores, instead of one big job dominating wall-clock time.
"""

import argparse
import math
import os
import re

import fanin_scaling
import jinja2

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# ----------------------------------------------------------------------
# Device presets. avol is stored in dB (converted to linear V/V at
# render time, see NOTE 4 above).
# Units: vos[V], ib[A], ios[A], avol[dB], gbw[Hz], slew[V/s],
#        rail[V] (dropout from each supply rail), ilimit[A], vsupply[V] (+-)
# ----------------------------------------------------------------------
PRESETS = {
    "generic": dict(
        vos=2e-3,
        ib=80e-9,
        ios=20e-9,
        avol=100,
        gbw=1e6,
        slew=1e6,
        rail=1.0,
        ilimit=25e-3,
        vsupply=15.0,
        note="Generic assumed bipolar op-amp (not a real datasheet).",
    ),
    "mcp6232": dict(
        vos=5e-3,
        ib=1e-12,
        ios=1.0e-12,
        avol=110,
        gbw=300e3,
        slew=0.15e6,
        rail=0.15,
        ilimit=23e-3,
        vsupply=2.5,
        note="Microchip MCP6232-E/P. Ib is an ESTIMATE (pA order); "
        "Rail is an estimate (light-load headroom).",
    ),
    "njm2732d": dict(
        vos=5e-3,
        ib=50e-9,
        ios=5e-9,
        avol=85,
        gbw=1e6,
        slew=0.4e6,
        rail=0.25,
        ilimit=25e-3,
        vsupply=2.5,
        note="Nisshinbo/NJRC NJM2732D. Ilimit not found in available "
        "sources; using generic estimate -- verify if output-current "
        "limiting matters for your test.",
    ),
}

PHASE_RE = re.compile(r"^(lo|hi|oh(\d+)|cum(\d+))$")


def phase_channel_values(label, n, vhi, vlo):
    """Return a list of n voltage values (one per channel) for a given
    phase label, following the same lo/hi/onehot/cumulative convention
    as the FULL-mode timeline."""
    m = PHASE_RE.match(label)
    if not m:
        raise ValueError(
            f"Unrecognized phase label {label!r} " f"(expected lo, hi, oh<k>, or cum<k>)"
        )
    if label == "lo":
        return [vlo] * n
    if label == "hi":
        return [vhi] * n
    if m.group(2):  # oh<k>
        k = int(m.group(2))
        if not (1 <= k <= n):
            raise ValueError(f"oh{k} out of range for N={n}")
        return [vhi if i == k else vlo for i in range(1, n + 1)]
    k = int(m.group(3))  # cum<k>
    if not (1 <= k <= n):
        raise ValueError(f"cum{k} out of range for N={n}")
    return [vhi if i <= k else vlo for i in range(1, n + 1)]


def estimate_settle_time(n, rf, rin, vhi, vlo, slew, gbw, vsupply, rail, n_tau=12, safety=1.5):
    """Estimate worst-case settling time for a step of size (vhi-vlo) through
    this inverting summing amp, given the opamp's Slew[V/s] and GBW[Hz].
    full_swing is capped at the physically achievable output range
    (2*(vsupply-rail)) -- without this cap, the naive noise-gain-scaled
    ideal swing (which grows with N and is NOT what the output can
    actually do once it clips at the rail) blows up settling-time
    estimates for large N, especially for low-slew-rate devices."""
    noise_gain = 1.0 + n * (rf / rin)
    full_swing_ideal = abs(vhi - vlo) * noise_gain
    full_swing_max = 2.0 * max(vsupply - rail, 0.0)
    full_swing = min(full_swing_ideal, full_swing_max) if full_swing_max > 0 else full_swing_ideal
    slew_time = full_swing / slew if slew > 0 else 0.0
    closed_loop_bw = gbw / noise_gain if noise_gain > 0 else gbw
    tau = 1.0 / (2 * math.pi * closed_loop_bw) if closed_loop_bw > 0 else 0.0
    linear_settle = n_tau * tau
    return safety * (slew_time + linear_settle)


def fmt_pwl(pts):
    return "PWL(" + " ".join(f"{t:.9g} {v:.6g}" for t, v in pts) + ")"


def build_phase_pwl(value, rise, t_end):
    """PWL breakpoints for a single constant-value phase (ramp to `value`
    over `rise` seconds, then hold until t_end)."""
    return [(0.0, 0.0), (rise, value), (t_end, value)]


def render_netlist(context):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    template = env.get_template("netlist.cir.j2")
    return template.render(**context)


def build_opamp_context(args, preset, mode):
    vos_expr = "{:.6g}".format(preset["vos"])
    vos_sweep_param_line = None
    if mode == "vos_sweep":
        vos_sweep_param_line = ".param V_offset {:.6g} {:.6g} {:.6g}".format(
            args.vos_start, args.vos_stop, args.vos_step
        )
        vos_expr = "{V_offset}"
    return dict(
        opamp_lib=args.opamp_lib,
        opamp_model=args.opamp_model,
        avol_linear=f"{10 ** (preset['avol'] / 20):.6g}",
        gbw=f"{preset['gbw']:.6g}",
        slew=f"{preset['slew']:.6g}",
        vos_expr=vos_expr,
        vos_sweep_param_line=vos_sweep_param_line,
        ib=f"{preset['ib']:.6g}",
        ios=f"{preset['ios']:.6g}",
        rail=f"{preset['rail']:.6g}",
        ilimit=f"{preset['ilimit']:.6g}",
    )


def gen_phase_netlist(n, mode, args, preset, phase_label):
    rf, rin = fanin_scaling.compute_rf_rin(n, args.scale_mode, args.rf0, args.rin0)
    tol = args.tol
    vsupply = args.vsupply if args.vsupply is not None else preset["vsupply"]
    vhi, vlo, rise = args.vhi, args.vlo, args.rise_time

    values = phase_channel_values(phase_label, n, vhi, vlo)

    settle = estimate_settle_time(
        n, rf, rin, vhi, vlo, preset["slew"], preset["gbw"], vsupply, preset["rail"]
    )
    margin = max(rise * 5, settle)
    T = args.phase_time if args.phase_time is not None else max(1e-3, margin / 0.2)
    margin = min(margin, T * 0.45)

    rf_expr = "{Rf_nom}"
    rin_expr = "{Rin_nom}"
    if mode == "mc_res":
        rf_expr = "{mc(Rf_nom," + f"{tol:.6g}" + ")}"
        rin_expr = "{mc(Rin_nom," + f"{tol:.6g}" + ")}"

    sources = []
    for i, value in enumerate(values, start=1):
        pts = build_phase_pwl(value, rise, T)
        sources.append(dict(idx=i, pwl=fmt_pwl(pts)))

    meas_lines = [f".meas TRAN vph_{phase_label} AVG V(out) FROM={margin:.9g} TO={T - margin:.9g}"]
    step_line = f".step param run 1 {args.mc_runs} 1" if mode == "mc_res" else None

    context = dict(
        n=n,
        mode=mode,
        phase=phase_label,
        scale_mode=args.scale_mode,
        rf0=args.rf0,
        rin0=args.rin0,
        rf=rf,
        rin=rin,
        preset_note=preset["note"],
        vsupply=vsupply,
        rf_expr=rf_expr,
        rin_expr=rin_expr,
        sources=sources,
        tran_line=f".tran 0 {T:.9g} 0 {rise / 5:.3g}",
        step_line=step_line,
        meas_lines=meas_lines,
    )
    context.update(build_opamp_context(args, preset, mode))
    return render_netlist(context)


def all_phase_labels(n):
    labels = ["lo"]
    labels += [f"oh{k}" for k in range(1, n + 1)]
    labels += [f"cum{k}" for k in range(1, n + 1)]
    labels += ["hi"]
    return labels


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-list", default="2", help="comma-separated N values, e.g. 2,4,8,16,32")
    ap.add_argument("--mode", choices=["opamp_err", "mc_res", "vos_sweep"], default="opamp_err")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="generic")
    ap.add_argument("--outdir", default="./ltspice_out")
    ap.add_argument(
        "--opamp-lib", default="UniversalOpAmp2.lib", help="filename passed to .include"
    )
    ap.add_argument(
        "--opamp-model", default="level2", help="internal subckt name for the accuracy level to use"
    )

    fanin_scaling.add_scaling_args(ap)
    ap.add_argument("--vhi", type=float, default=1.0)
    ap.add_argument("--vlo", type=float, default=-1.0)
    ap.add_argument(
        "--vsupply", type=float, default=None, help="override preset supply magnitude (+-V)"
    )

    ap.add_argument(
        "--phase-time",
        type=float,
        default=None,
        help="seconds per test phase; default: auto-sized from "
        "the preset's Slew/GBW so measurement windows are safely settled",
    )
    ap.add_argument("--rise-time", type=float, default=1e-6)

    ap.add_argument(
        "--tol", type=float, default=0.01, help="resistor tolerance fraction for mc_res mode"
    )
    ap.add_argument("--mc-runs", type=int, default=100)

    ap.add_argument("--vos-start", type=float, default=0.0)
    ap.add_argument("--vos-stop", type=float, default=5e-3)
    ap.add_argument("--vos-step", type=float, default=1e-3)

    ap.add_argument(
        "--phase",
        default=None,
        help="generate a SINGLE-PHASE netlist instead of the full "
        "2N+2-phase one (for parallel job-splitting). One of: "
        "lo, hi, oh<k>, cum<k>. Use --all-phases to generate "
        "every phase for the given N in one call.",
    )
    ap.add_argument(
        "--all-phases",
        action="store_true",
        help="generate one single-phase netlist per phase "
        "(lo, oh1..ohN, cum1..cumN, hi) for each N in --n-list",
    )

    for field in ["vos", "ib", "ios", "avol", "gbw", "slew", "rail", "ilimit"]:
        ap.add_argument(f"--{field}", type=float, default=None)

    args = ap.parse_args()

    preset = dict(PRESETS[args.preset])
    for field in ["vos", "ib", "ios", "avol", "gbw", "slew", "rail", "ilimit"]:
        v = getattr(args, field)
        if v is not None:
            preset[field] = v

    if not args.phase and not args.all_phases:
        ap.error(
            "--phase LABEL または --all-phases のどちらかを指定してください "
            "(全フェーズを1本にまとめる旧モードは廃止しました)"
        )

    n_list = [int(x) for x in args.n_list.split(",")]

    for n in n_list:
        n_dir = os.path.join(args.outdir, args.preset, f"N{n}")
        os.makedirs(n_dir, exist_ok=True)

        labels = all_phase_labels(n) if args.all_phases else [args.phase]
        for label in labels:
            text = gen_phase_netlist(n, args.mode, args, preset, label)
            fname = f"fanin_N{n}_{args.preset}_{args.scale_mode}_{args.mode}_phase-{label}.cir"
            path = os.path.join(n_dir, fname)
            with open(path, "w") as f:
                f.write(text)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
