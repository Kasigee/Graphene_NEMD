#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_quadratic(z_m, v_mps, w=None):
    A = np.vstack([z_m**2, z_m, np.ones_like(z_m)]).T
    if w is None:
        coef, *_ = np.linalg.lstsq(A, v_mps, rcond=None)
        return coef
    W = np.sqrt(w)
    Aw = A * W[:, None]
    yw = v_mps * W
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    return coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nemd_dir", type=Path, help="Path to solvated/nemd_poiseuille_ax... directory")
    ap.add_argument("--z-exclude-A", type=float, default=None, help="Override exclusion thickness (Å)")
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    d = args.nemd_dir
    meta = json.loads((d / "nemd_summary.json").read_text())
    prof = np.genfromtxt(d / "vx_profile_z.csv", delimiter=",", names=True, dtype=None, encoding=None)

    z_A = prof["z_A"].astype(float)
    vx_A_fs = np.array([float(x) if x != "" else np.nan for x in prof["mean_vx_A_per_fs"]], float)
    cnt = prof["count"].astype(float)
    rho_g_cm3 = prof["mass_density_g_cm3"].astype(float)

    z_bot_A = float(meta["graphene_z_bot_A"])
    z_top_A = float(meta["graphene_z_top_A"])
    a_mps2 = float(meta["body_accel_mps2"])
    gap_A = float(meta["gap_A"])

    z_excl = args.z_exclude_A if args.z_exclude_A is not None else float(meta["profile"]["z_exclude_A_recommended"])

    z_m = z_A * 1e-10
    v_mps = vx_A_fs * 1e5
    rho_kg_m3 = rho_g_cm3 * 1000.0

    zmin_fit_A = z_bot_A + z_excl
    zmax_fit_A = z_top_A - z_excl
    fit_mask = (z_A >= zmin_fit_A) & (z_A <= zmax_fit_A) & np.isfinite(v_mps) & (cnt > 10)

    if fit_mask.sum() < 10:
        raise RuntimeError("Too few bins in fit region. Reduce --z-exclude-A or run longer production.")

    rho_fit = float(np.nanmean(rho_kg_m3[fit_mask]))

    coef = fit_quadratic(z_m[fit_mask], v_mps[fit_mask], w=cnt[fit_mask])
    a2, b1, c0 = coef

    curvature = 2.0 * a2
    eta_Pa_s = float(rho_fit * a_mps2 / curvature)

    def vfit(z):
        return a2 * z * z + b1 * z + c0

    def dv_dz(z):
        return 2.0 * a2 * z + b1

    z_bot_m = z_bot_A * 1e-10
    z_top_m = z_top_A * 1e-10

    v_bot = vfit(z_bot_m)
    v_top = vfit(z_top_m)
    g_bot = dv_dz(z_bot_m)
    g_top = dv_dz(z_top_m)

    b_bot = float(v_bot / abs(g_bot)) if abs(g_bot) > 0 else float("nan")
    b_top = float(v_top / abs(g_top)) if abs(g_top) > 0 else float("nan")
    b_m = float(np.nanmean([b_bot, b_top]))
    b_nm = b_m * 1e9

    lam_Pa_s_per_m = float(eta_Pa_s / b_m)

    H_m = (z_top_m - z_bot_m)
    vbar_mps = float(np.trapz(v_mps[np.isfinite(v_mps)], z_m[np.isfinite(v_mps)]) / H_m)
    j_mass = float(rho_fit * vbar_mps)

    out = {
        "tag": meta["tag"],
        "accel_mps2": a_mps2,
        "rho_fit_kg_m3": rho_fit,
        "fit_region_A": [float(zmin_fit_A), float(zmax_fit_A)],
        "gap_A": gap_A,
        "eta_Pa_s": eta_Pa_s,
        "slip_length_nm": b_nm,
        "lambda_Pa_s_per_m": lam_Pa_s_per_m,
        "vbar_mps": vbar_mps,
        "mass_flux_kg_m2_s": j_mass,
        "quadratic_fit": {"a2": float(a2), "b1": float(b1), "c0": float(c0)},
    }

    (d / "transport_fit.json").write_text(json.dumps(out, indent=2))

    txt = []
    txt.append(f"Tag: {out['tag']}")
    txt.append(f"a = {a_mps2:.3e} m/s^2")
    txt.append(f"rho_fit = {rho_fit:.1f} kg/m^3 (avg in fit region)")
    txt.append(f"eta = {eta_Pa_s:.4e} Pa·s")
    txt.append(f"b = {b_nm:.3f} nm (avg of top/bot)")
    txt.append(f"lambda = {lam_Pa_s_per_m:.4e} Pa·s/m")
    txt.append(f"vbar = {vbar_mps:.4e} m/s")
    txt.append(f"mass flux = {j_mass:.4e} kg/(m^2 s)")
    (d / "transport_summary.txt").write_text("\n".join(txt) + "\n")

    print("\n".join(txt))

    if args.make_plots:
        import matplotlib.pyplot as plt

        z_plot = np.linspace(z_m.min(), z_m.max(), 400)
        v_fit = a2 * z_plot * z_plot + b1 * z_plot + c0

        plt.figure()
        plt.plot(z_m * 1e10, v_mps, label="mean v_x(z)")
        plt.plot(z_plot * 1e10, v_fit, label="quadratic fit")
        plt.axvline(zmin_fit_A, linestyle="--")
        plt.axvline(zmax_fit_A, linestyle="--")
        plt.xlabel("z (Å)")
        plt.ylabel("v_x (m/s)")
        plt.legend()
        plt.title(f"{meta['tag']} Poiseuille profile")
        plt.tight_layout()
        plt.savefig(d / "vx_profile_fit.png", dpi=200)


if __name__ == "__main__":
    main()
