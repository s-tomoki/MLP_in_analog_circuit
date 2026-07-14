#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fan-in summing amplifier netlist generator for LTSpice (UniversalOpAmp2).

Circuit: N-input inverting summing amplifier.
  - Non-inverting input tied to 0V (true virtual ground; dual supply).
  - Each input i: Vin_i -> Rin -> inverting node, feedback Rf.
  - Ideal (Rf=Rin for all i): Vout = -(Rf/Rin) * sum(Vin_i)

A single .tran run sweeps through test patterns back-to-back:
  phase 0            : all inputs = Lo
  phase 1..N          : one-hot (only channel k = Hi)
  phase N+1..2N        : cumulative step-up (channels 1..k = Hi)
  phase 2N+1 (last)     : all inputs = Hi

.meas TRAN statements extract the settled AVG(V(out)) for each phase so the
whole pattern set is obtained from ONE simulation.

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
  4. Ib for MCP6232-E/P is an ESTIMATED value (pA-order, same magnitude as
     Ios) -- the exact datasheet table entry could not be confirmed from
     available sources. Treat step-2/3 MCP6232 results involving Ib as
     approximate until you confirm against the datasheet electrical table.
"""

import argparse
import os

import fanin_scaling

# ----------------------------------------------------------------------
# Device presets (values gathered from datasheets; see chat for sources).
# Units: vos[V], ib[A], ios[A], avol[dB] (converted to linear V/V before
# being passed to the model, since level2's Avol param is linear, not dB),
# gbw[Hz], slew[V/s],
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


def build_pwl(
    channel_idx, n, t_lo, t_oh_start, t_cum_start, t_hi_start, t_end, phase_time, rise, vlo, vhi
):
    """Build PWL breakpoint list (time, value) for input channel `channel_idx`
    (1-indexed) out of `n` channels total, across the 2N+2 phase timeline."""
    i = channel_idx
    pts = [(0.0, vlo)]

    def add_level(t_start, value):
        # ramp to `value` starting at t_start over `rise` seconds
        prev_v = pts[-1][1]
        if abs(prev_v - value) < 1e-15:
            return  # no change, skip redundant point
        pts.append((t_start, prev_v))
        pts.append((t_start + rise, value))

    # phase 0: all-lo (already the initial state)
    add_level(t_lo, vlo)

    # one-hot phases: k = 1..n
    for k in range(1, n + 1):
        t_start = t_oh_start + (k - 1) * phase_time
        add_level(t_start, vhi if k == i else vlo)

    # cumulative phases: k = 1..n (channels 1..k = Hi)
    for k in range(1, n + 1):
        t_start = t_cum_start + (k - 1) * phase_time
        add_level(t_start, vhi if i <= k else vlo)

    # final: all-hi
    add_level(t_hi_start, vhi)

    pts.append((t_end, pts[-1][1]))
    return pts


def fmt_pwl(pts):
    return "PWL(" + " ".join(f"{t:.9g} {v:.6g}" for t, v in pts) + ")"


def estimate_settle_time(n, rf, rin, vhi, vlo, slew, gbw, vsupply, rail, n_tau=12, safety=1.5):
    """Estimate worst-case settling time for a step of size (vhi-vlo) through
    this inverting summing amp, given the opamp's Slew[V/s] and GBW[Hz].
    Returns seconds. Used to size the .meas averaging-window margin so we
    don't measure while the output is still slewing/settling.

    full_swing is capped at the physically achievable output range
    (2*(vsupply-rail)) -- without this cap, the naive noise-gain-scaled
    ideal swing (which grows with N and is NOT what the output can actually
    do once it clips at the rail) blows up settling-time estimates for
    large N, especially for low-slew-rate devices. This was the root cause
    of a large, preset-dependent runtime discrepancy at N=16 (MCP6232 taking
    ~2.6x longer than NJM2732D despite near-identical netlist size)."""
    noise_gain = 1.0 + n * (rf / rin)
    full_swing_ideal = abs(vhi - vlo) * noise_gain
    full_swing_max = 2.0 * max(vsupply - rail, 0.0)
    full_swing = min(full_swing_ideal, full_swing_max) if full_swing_max > 0 else full_swing_ideal
    slew_time = full_swing / slew if slew > 0 else 0.0
    closed_loop_bw = gbw / noise_gain if noise_gain > 0 else gbw
    tau = 1.0 / (2 * 3.141592653589793 * closed_loop_bw) if closed_loop_bw > 0 else 0.0
    linear_settle = n_tau * tau
    return safety * (slew_time + linear_settle)


def gen_netlist(n, mode, args, preset):
    rf, rin = fanin_scaling.compute_rf_rin(n, args.scale_mode, args.rf0, args.rin0)
    tol = args.tol
    vsupply = args.vsupply if args.vsupply is not None else preset["vsupply"]
    vhi = args.vhi
    vlo = args.vlo
    rise = args.rise_time

    # Auto-size phase_time / measurement margin from the opamp's Slew & GBW
    # so we don't average while the output is still slewing/settling
    # (this was the root cause of the ~1-2% errors seen in earlier runs --
    # not a real Vos/Avol DC limitation, just measuring too early).
    settle = estimate_settle_time(
        n, rf, rin, vhi, vlo, preset["slew"], preset["gbw"], vsupply, preset["rail"]
    )
    margin = max(rise * 5, settle)
    if args.phase_time is not None:
        T = args.phase_time
        if margin > T * 0.3:
            print(
                f"WARNING: N={n} preset={args.preset}: estimated settle time "
                f"({settle*1e6:.1f}us) is large relative to --phase-time "
                f"({T*1e6:.1f}us). Measurement window may still be too tight; "
                f"consider --phase-time {max(T, margin/0.2):.2e} or higher."
            )
    else:
        # auto: leave at least 5x settle time as usable averaging window
        T = max(1e-3, margin / 0.2)
    margin = min(margin, T * 0.45)  # never eat the whole window

    # timeline boundaries
    t_lo = 0.0
    t_oh_start = T
    t_cum_start = T + n * T
    t_hi_start = T + 2 * n * T
    t_end = t_hi_start + T

    lines = []
    lines.append(f"* Fan-in summing amplifier, N={n}, mode={mode}")
    lines.append(
        f"* scale_mode={args.scale_mode} Rf0={args.rf0:.6g} Rin0={args.rin0:.6g} "
        f"-> Rf={rf:.6g} Rin={rin:.6g}"
    )
    lines.append(f"* preset note: {preset['note']}")
    # UniversalOpAmp2 must be explicitly loaded when running a bare .cir
    # netlist (schematic capture does this automatically, .cir does not).
    # Confirmed working form: ".include UniversalOpAmp2.lib" (not ".lib ...sub").
    lines.append(f".include {args.opamp_lib}")
    lines.append("")
    # The symbol "UniversalOpAmp2" itself is not a callable subckt name in
    # the library -- the actual subckt inside is named per accuracy level
    # (e.g. "level2"). Wrap it in our own subckt so we can pass named params.
    lines.append(".subckt MyOpAmp 1 2 3 4 5")
    lines.append("* pins: 1=IN+ 2=IN- 3=VCC 4=VEE 5=OUT")
    vos_expr = "{:.6g}".format(preset["vos"])
    if mode == "vos_sweep":
        lines.append(
            ".param V_offset {:.6g} {:.6g} {:.6g}".format(
                args.vos_start, args.vos_stop, args.vos_step
            )
        )
        vos_expr = "{V_offset}"
    lines.append(
        f"X_internal 1 2 3 4 5 {args.opamp_model} params: "
        f"Avol={10**(preset['avol']/20):.6g} GBW={preset['gbw']:.6g} "
        f"Slew={preset['slew']:.6g} Vos={vos_expr} "
        f"Ib={preset['ib']:.6g} Ios={preset['ios']:.6g} "
        f"rail={preset['rail']:.6g} ilimit={preset['ilimit']:.6g}"
    )
    lines.append(".ends MyOpAmp")
    lines.append("")
    lines.append(".param Rin_nom={:.6g} Rf_nom={:.6g}".format(rin, rf))
    lines.append(f"V+ vcc 0 {vsupply:.6g}")
    lines.append(f"V- vee 0 -{vsupply:.6g}")
    lines.append("Vinp inp 0 0")

    # Instance. Pin order: IN+ IN- VCC VEE OUT (confirmed working order).
    lines.append("XU1 inp inn vcc vee out MyOpAmp")

    rf_expr = "{Rf_nom}"
    if mode == "mc_res":
        rf_expr = "{mc(Rf_nom," + f"{tol:.6g}" + ")}"
    lines.append(f"Rf inn out {rf_expr}")

    for i in range(1, n + 1):
        pts = build_pwl(i, n, t_lo, t_oh_start, t_cum_start, t_hi_start, t_end, T, rise, vlo, vhi)
        lines.append(f"Vin{i} n{i} 0 {fmt_pwl(pts)}")
        rin_expr = "{Rin_nom}"
        if mode == "mc_res":
            rin_expr = "{mc(Rin_nom," + f"{tol:.6g}" + ")}"
        lines.append(f"Rin{i} n{i} inn {rin_expr}")

    lines.append(f".tran 0 {t_end:.9g} 0 {rise/5:.3g}")

    if mode == "mc_res":
        lines.append(f".step param run 1 {args.mc_runs} 1")

    # .meas: settled-average window (margin computed above from settle time)

    def meas(label, t0, t1):
        return f".meas TRAN vph_{label} AVG V(out) FROM={t0+margin:.9g} TO={t1-margin:.9g}"

    lines.append(meas("lo", t_lo, t_oh_start))
    for k in range(1, n + 1):
        t0 = t_oh_start + (k - 1) * T
        lines.append(meas(f"oh{k}", t0, t0 + T))
    for k in range(1, n + 1):
        t0 = t_cum_start + (k - 1) * T
        lines.append(meas(f"cum{k}", t0, t0 + T))
    lines.append(meas("hi", t_hi_start, t_end))

    lines.append(".backanno")
    lines.append(".end")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-list", default="2", help="comma-separated N values, e.g. 2,4,8,16,32")
    ap.add_argument("--mode", choices=["opamp_err", "mc_res", "vos_sweep"], default="opamp_err")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="generic")
    ap.add_argument("--outdir", default="./ltspice_out")
    ap.add_argument(
        "--opamp-lib",
        default="UniversalOpAmp2.lib",
        help="filename passed to .include (confirmed working: "
        "UniversalOpAmp2.lib via .include, not .lib+.sub)",
    )
    ap.add_argument(
        "--opamp-model",
        default="level2",
        help="internal subckt name inside the library that "
        "implements the accuracy level to use (level2 "
        "confirmed working: Avol/GBW/Slew/Vos/Ib/Ios/rail/ilimit)",
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
        "the preset's Slew/GBW so measurement windows are "
        "safely settled",
    )
    ap.add_argument("--rise-time", type=float, default=1e-6)

    ap.add_argument(
        "--tol", type=float, default=0.01, help="resistor tolerance fraction for mc_res mode"
    )
    ap.add_argument("--mc-runs", type=int, default=100)

    ap.add_argument("--vos-start", type=float, default=0.0)
    ap.add_argument("--vos-stop", type=float, default=5e-3)
    ap.add_argument("--vos-step", type=float, default=1e-3)

    # allow overriding any preset field directly
    for field in ["vos", "ib", "ios", "avol", "gbw", "slew", "rail", "ilimit"]:
        ap.add_argument(f"--{field}", type=float, default=None)

    args = ap.parse_args()

    preset = dict(PRESETS[args.preset])
    for field in ["vos", "ib", "ios", "avol", "gbw", "slew", "rail", "ilimit"]:
        v = getattr(args, field)
        if v is not None:
            preset[field] = v

    os.makedirs(args.outdir, exist_ok=True)
    n_list = [int(x) for x in args.n_list.split(",")]

    for n in n_list:
        text = gen_netlist(n, args.mode, args, preset)
        fname = f"fanin_N{n}_{args.preset}_{args.scale_mode}_{args.mode}.cir"
        path = os.path.join(args.outdir, fname)
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
