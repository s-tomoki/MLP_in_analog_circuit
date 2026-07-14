#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fanin_scaling.py

Shared logic for how Rf/Rin scale with N in the weighted-sum (artificial
neuron) fan-in circuit. Used by both the netlist generator and the
analysis scripts so the "ideal value" math always matches what was
actually built into the .cir file.

Two scaling modes (no "fixed/none" mode -- removed per design decision):

  rin_scale : Rin(N) = Rin0 * N,   Rf = Rf0           (Rf fixed)
  rf_scale  : Rf(N)  = Rf0 / N,    Rin = Rin0          (Rin fixed)

Both modes give the same per-channel weight w(N) = Rf/Rin = Rf0/(Rin0*N)
when Rf0 == Rin0, i.e. the same noise gain and full-scale output range
regardless of N -- this is what keeps the neuron's output range constant
as fan-in grows. They differ in the *absolute* resistor values used,
which matters for Ib-driven bias-current error and for how resistor
tolerance (Monte Carlo) manifests.
"""

SCALE_MODES = ("rin_scale", "rf_scale")


def compute_rf_rin(n, scale_mode, rf0, rin0):
    """Return (rf, rin) for a given N and scale_mode."""
    if scale_mode == "rin_scale":
        return rf0, rin0 * n
    elif scale_mode == "rf_scale":
        return rf0 / n, rin0
    else:
        raise ValueError(f"Unknown scale_mode {scale_mode!r}; must be one of {SCALE_MODES}")


def add_scaling_args(ap):
    """Attach the shared --scale-mode/--rf0/--rin0 CLI arguments to an
    argparse.ArgumentParser. --scale-mode is required (no default) by
    design decision -- explicit choice avoids accidentally mixing data
    from different scaling regimes."""
    ap.add_argument(
        "--scale-mode",
        choices=list(SCALE_MODES),
        required=True,
        help="how Rf/Rin scale with N (required, no default): "
        "rin_scale = Rf fixed, Rin(N)=Rin0*N; "
        "rf_scale = Rin fixed, Rf(N)=Rf0/N",
    )
    ap.add_argument(
        "--rf0",
        type=float,
        default=10e3,
        help="base Rf [ohm] (see --scale-mode for how it scales with N)",
    )
    ap.add_argument(
        "--rin0",
        type=float,
        default=10e3,
        help="base Rin [ohm] (see --scale-mode for how it scales with N)",
    )
