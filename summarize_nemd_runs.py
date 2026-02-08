#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_dir(nemd_dir: Path) -> dict:
    summary = {"dir": str(nemd_dir)}
    files = {
        "nemd_summary.json": nemd_dir / "nemd_summary.json",
        "vx_profile_z.csv": nemd_dir / "vx_profile_z.csv",
        "transport_fit.json": nemd_dir / "transport_fit.json",
    }
    summary["files"] = {k: v.is_file() for k, v in files.items()}
    if files["nemd_summary.json"].is_file():
        try:
            meta = json.loads(files["nemd_summary.json"].read_text())
            summary["tag"] = meta.get("tag")
            summary["accel_mps2"] = meta.get("body_accel_mps2")
            summary["prod_ps"] = meta.get("prod_ps")
            summary["freeze_graphene"] = meta.get("freeze_graphene")
        except Exception:
            summary["nemd_summary_error"] = True
    if files["transport_fit.json"].is_file():
        try:
            fit = json.loads(files["transport_fit.json"].read_text())
            summary["eta_Pa_s"] = fit.get("eta_Pa_s")
            summary["slip_length_nm"] = fit.get("slip_length_nm")
            summary["lambda_Pa_s_per_m"] = fit.get("lambda_Pa_s_per_m")
            summary["vbar_mps"] = fit.get("vbar_mps")
        except Exception:
            summary["transport_fit_error"] = True
    summary["status"] = "ready" if all(summary["files"].values()) else "incomplete"
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Summarize NEMD Poiseuille runs and report readiness for analysis."
    )
    ap.add_argument("root", type=Path, help="Root directory to scan (e.g., orbmol/)")
    ap.add_argument("--pattern", default="nemd_poiseuille_ax*_rep*", help="Glob pattern for NEMD dirs")
    ap.add_argument("--write-json", action="store_true", help="Write summary_nemd_runs.json")
    args = ap.parse_args()

    root = args.root
    nemd_dirs = sorted(p for p in root.rglob(args.pattern) if p.is_dir())
    summaries = [summarize_dir(d) for d in nemd_dirs]

    if args.write_json:
        out = root / "summary_nemd_runs.json"
        out.write_text(json.dumps(summaries, indent=2))

    if not summaries:
        print("No NEMD directories found.")
        return

    header = [
        "status",
        "tag",
        "accel_mps2",
        "prod_ps",
        "eta_Pa_s",
        "slip_length_nm",
        "lambda_Pa_s_per_m",
        "vbar_mps",
        "dir",
    ]
    print("\t".join(header))
    for s in summaries:
        row = [str(s.get(k, "")) for k in header]
        print("\t".join(row))


if __name__ == "__main__":
    main()
