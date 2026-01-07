#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#solvate_and_md_orbmol.py
solvate_and_md_onlyGraphene_OrbMol.py

UPDATED for graphene (not graphene oxide) + parameter sweep over:
  - solvent
  - interlayer spacing (Å): e.g., 6, 8, 10, 12, 14

Key additions vs your base script
  1) Supports graphene tags of the form:
        <nx>x<ny>_Gr_<solvent>_gap<gap>A
     e.g. 8x8_Gr_water_gap10A

  2) Optional graphene bilayer builder (pristine C sheets):
        --build-graphene-sweep --nx 8 --ny 8 --gaps 6 8 10 12 14 --solvents water ethanol

     This creates (if missing, or if --rebuild-graphene):
        orbmol/<tag>/final_relaxed.extxyz
        orbmol/<tag>/relax.json (marked converged=True as a “prepared starting structure”)

  3) Optional freezing of graphene atoms during MD (default ON), using per-atom ASE tags:
        graphene atoms are tag=1; solvated atoms default to tag=0
        --no-freeze-graphene to allow sheets to move

Notes
  - Solvation uses the same random rejection packing into the slab between the two graphene planes.
  - For small gaps (e.g., 6 Å), the default sheet padding may be too large for bulky solvents.
    This script auto-reduces pad if it would make the slab thickness non-positive.

Examples
  # Build + solvate + run 50 ps for all solvents and all gaps:
  python solvate_and_md_onlyGraphene_OrbMol.py --orbmol-root orbmol \
      --build-graphene-sweep --nx 8 --ny 8 --gaps 6 8 10 12 14 --solvents water ethanol acetone hexane chlorobenzene \
      --device cuda --time-ps 50

  # Extend all existing runs by another 200 ps (resumes automatically):
  python solvate_and_md_orbmol.py --orbmol-root orbmol --device cuda --time-ps 200

  # Re-solvate only for the 10 Å water system, keep existing graphene:
  python solvate_and_md_orbmol.py --orbmol-root orbmol --tags 8x8_Gr_water_gap10A --resolvate

  # Rebuild graphene structures (overwrites final_relaxed.extxyz) then re-solvate:
  python solvate_and_md_orbmol.py --orbmol-root orbmol --build-graphene-sweep --rebuild-graphene --resolvate
"""

from __future__ import annotations
import argparse, json, math, os, random, re, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("RAPIDS_NO_INITIALIZE", "1")

from ase.io import read, write
from ase import Atoms, units
from ase.geometry import cell_to_cellpar
from ase.data import vdw_radii, atomic_numbers
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixAtoms

import torch
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from orb_models.forcefield import featurization_utilities as feat_util

# sklearn is only needed if ORB_FORCE_CPU_FEAT=1
try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# --------------------------- constants / defaults ---------------------------

SHEET_PADDING_A   = 1.0   # lowered vs base script; still auto-adjusted if needed
DIST_TOL_A        = 2.0
MAX_TRIES_PER_MOL = 4000
MAX_MOLS_CAP      = 1500
RNG_SEED          = 12345

DEFAULT_GAPS_A    = [6, 8, 10, 12, 14]
DEFAULT_SOLVENTS  = ["water", "ethanol", "acetone", "hexane", "chlorobenzene"]

SOLVENT_DENSITIES = {
    "water": 0.997,           # g/cm^3 ~25°C
    "ethanol": 0.789,
    "acetone": 0.790,
    "hexane": 0.655,
    "chlorobenzene": 1.106,
}


# ------------------------------ tag parsing ---------------------------------

def parse_tag(tag: str):
    """
    Supports:
      (A) legacy GO-ish tag:  <nx>x<ny>_<cation>_<solvent>_<ostate>
          e.g. 6x6_Al_hexane_o-
      (B) graphene sweep tag: <nx>x<ny>_Gr_<solvent>_gap<gap>A
          e.g. 8x8_Gr_water_gap10A
    """
    m = re.match(r"^(\d+)x(\d+)_([A-Za-z]+)_([A-Za-z0-9]+)_([a-z+\-]+)$", tag)
    if m:
        nx, ny = int(m.group(1)), int(m.group(2))
        return {"mode": "legacy", "nx": nx, "ny": ny,
                "cation": m.group(3), "solvent": m.group(4).lower(), "ostate": m.group(5),
                "gap_A": None}

    m = re.match(r"^(\d+)x(\d+)_Gr_([A-Za-z0-9]+)_gap(\d+(?:\.\d+)?)A$", tag)
    if m:
        nx, ny = int(m.group(1)), int(m.group(2))
        return {"mode": "graphene", "nx": nx, "ny": ny,
                "cation": None, "solvent": m.group(3).lower(), "ostate": None,
                "gap_A": float(m.group(4))}
    return None


def relax_is_converged(tag_dir: Path) -> bool:
    meta = tag_dir / "relax.json"
    if not meta.is_file():
        return False
    try:
        j = json.loads(meta.read_text())
        return bool(j.get("converged", False))
    except Exception:
        return False


# ------------------------------ geometry utils ------------------------------

def kmeans1d(z: np.ndarray, k: int = 3, max_iter: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = np.quantile(z, np.linspace(0.05, 0.95, k))
    for _ in range(max_iter):
        d = np.abs(z[:, None] - centers[None, :])
        labels = np.argmin(d, axis=1)
        new_centers = np.array([z[labels == i].mean() if np.any(labels == i) else centers[i] for i in range(k)], float)
        if np.allclose(new_centers, centers, atol=1e-8):
            centers = new_centers
            break
        centers = new_centers
    return labels, centers


def sheet_planes_from_carbons(atoms: Atoms, pad_A: float = SHEET_PADDING_A):
    """
    Identify bottom/top graphene planes using carbon z-coordinates (k=3 1D kmeans),
    then return (zlo, zhi, gap_mean) where [zlo, zhi] is the fill region.
    Auto-reduces pad if it would make the fill thickness non-positive.
    """
    sym = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    c_mask = np.array([s == "C" for s in sym], bool)
    zC = pos[c_mask, 2]
    if zC.size < 10:
        raise ValueError("Too few carbon atoms to identify sheets.")

    labels, centers = kmeans1d(zC, k=3)
    counts = np.array([(labels == i).sum() for i in range(3)])
    two = counts.argsort()[-2:]
    two = two[np.argsort(centers[two])]
    bot_id, top_id = int(two[0]), int(two[1])

    z_bot_mean = float(zC[labels == bot_id].mean())
    z_top_mean = float(zC[labels == top_id].mean())
    gap = float(z_top_mean - z_bot_mean)
    if gap <= 0:
        raise ValueError("Could not determine a positive inter-sheet gap from carbon planes.")

    pad_eff = float(pad_A)
    # ensure zhi > zlo
    if (z_top_mean - pad_eff) <= (z_bot_mean + pad_eff):
        pad_eff = max(0.0, 0.49 * gap - 1e-3)
        print(f"[pad] reducing pad to {pad_eff:.3f} Å to keep positive slab thickness (gap={gap:.3f} Å)")

    zlo = z_bot_mean + pad_eff
    zhi = z_top_mean - pad_eff
    if not (zhi > zlo):
        raise ValueError(f"Non-positive fill thickness: {zhi - zlo:.3f} Å (gap={gap:.3f} Å, pad={pad_eff:.3f} Å)")

    return zlo, zhi, gap, pad_eff


def cell_area_and_height(atoms: Atoms):
    a1, a2, a3 = atoms.cell[0], atoms.cell[1], atoms.cell[2]
    return float(np.linalg.norm(np.cross(a1, a2))), float(np.linalg.norm(a3))


def estimate_excluded_volume_A3(atoms: Atoms, exclude_carbon=True, zslab: tuple[float, float] | None = None) -> float:
    sym = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    z = pos[:, 2]
    v = 0.0
    for i, s in enumerate(sym):
        if exclude_carbon and s == "C":
            continue
        if zslab is not None:
            zlo, zhi = zslab
            if not (zlo <= z[i] <= zhi):
                continue
        Z = atomic_numbers.get(s, None)
        if Z is None:
            continue
        r = vdw_radii[Z] if Z < len(vdw_radii) and vdw_radii[Z] > 0 else 1.5
        v += (4.0 / 3.0) * math.pi * (r ** 3)
    return float(v)


def fractional_to_cart(cell: np.ndarray, uvw: np.ndarray) -> np.ndarray:
    return uvw @ cell


def random_rotation_matrix():
    u1, u2, u3 = random.random(), random.random(), random.random()
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    w, x, y, z = q4, q1, q2, q3
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], float)


def min_distance_to_atoms(pt: np.ndarray, other_pos: np.ndarray, cell: np.ndarray) -> float:
    invT = np.linalg.inv(cell.T)
    f_pt = (invT @ pt).reshape(3)
    f_other = (invT @ other_pos.T).T
    df = f_pt - f_other
    df -= np.round(df)
    dcart = (df @ cell)
    d2 = np.sum(dcart * dcart, axis=1)
    return float(np.sqrt(d2.min())) if d2.size else float("inf")


def place_molecule_in_slab(mol: Atoms, cell: np.ndarray, zlo: float, zhi: float,
                           existing_pos: np.ndarray, tol: float, max_tries: int) -> np.ndarray | None:
    R0 = mol.get_positions()
    masses = mol.get_masses()
    com0 = (R0 * masses[:, None]).sum(0) / masses.sum()
    Rrel = R0 - com0

    for _ in range(max_tries):
        U = random_rotation_matrix()
        Rrot = (Rrel @ U.T)

        u, v = random.random(), random.random()
        z = zlo + random.random() * (zhi - zlo)

        com_cart = fractional_to_cart(cell, np.array([u, v, 0.0]))
        com_cart[2] = z

        if min_distance_to_atoms(com_cart, existing_pos, cell) < tol:
            continue

        Rtrial = Rrot + com_cart

        ok = True
        for i in range(Rtrial.shape[0]):
            if min_distance_to_atoms(Rtrial[i], existing_pos, cell) < tol:
                ok = False
                break
        if ok:
            return Rtrial
    return None


# ------------------------------ solvents ------------------------------------

def pick_solvent_file(tag_dir: Path, solvent: str, solvent_lib: Path | None) -> Path | None:
    for where in [tag_dir / "provenance", tag_dir]:
        g = list(where.glob(f"solvent_used_{solvent}.*"))
        if g:
            return g[0]
    if solvent_lib:
        for ext in (".xyz", ".mol", ".sdf", ".pdb", ".mol2"):
            p = solvent_lib / f"{solvent}{ext}"
            if p.is_file():
                return p
    return None


def load_solvent_geom(solvent: str, path: Path | None) -> Atoms:
    if path and path.is_file():
        mol = read(path.as_posix())
        mol.pbc = (False, False, False)
        return mol
    from ase.build import molecule
    aliases = {
        "water": ["water", "H2O"],
        "ethanol": ["ethanol", "C2H5OH", "EtOH", "CH3CH2OH"],
        "acetone": ["acetone", "propanone"],
        "hexane": ["hexane", "n-hexane"],
        "chlorobenzene": ["chlorobenzene", "C6H5Cl"],
    }.get(solvent.lower(), [solvent])
    last = None
    for nm in aliases:
        try:
            return molecule(nm)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not build solvent '{solvent}': {last}")


def load_solvent_template(solvent: str, solvent_source: str | None, solvent_lib: Path | None) -> Atoms:
    path = None
    if solvent_source:
        p = Path(solvent_source)
        if p.is_file():
            path = p
    if path is None and solvent_lib is not None:
        for ext in (".xyz", ".mol", ".sdf", ".pdb", ".mol2"):
            p = solvent_lib / f"{solvent}{ext}"
            if p.is_file():
                path = p
                break
    return load_solvent_geom(solvent, path)


# --------------------------- graphene builder --------------------------------

def build_graphene_bilayer(nx: int, ny: int, gap_A: float,
                           vacuum_A: float = 20.0, cc_A: float = 1.42,
                           stacking: str = "AA") -> Atoms:
    """
    Periodic graphene bilayer in xy (and in z via large vacuum).
    - gap_A is plane-to-plane distance between the two graphene sheets.
    - vacuum_A is empty space below bottom and above top sheet (each side).
    - stacking: "AA" (no xy shift) or "AB" (simple basis-vector shift).
    Graphene atoms are tagged with tag=1.
    """
    a = math.sqrt(3.0) * cc_A  # ~2.46 Å
    Lz = float(gap_A + 2.0 * vacuum_A)

    cell = np.array([
        [a, 0.0, 0.0],
        [0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0],
        [0.0, 0.0, Lz],
    ], float)

    # basis atoms
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5 * a, (math.sqrt(3.0) / 6.0) * a, 0.0]
    ], float)

    mono = Atoms("C2", positions=basis, cell=cell, pbc=(True, True, True)).repeat((nx, ny, 1))
    mono.wrap()
    mono.positions[:, 2] += vacuum_A  # bottom at z=vacuum

    top = mono.copy()
    top.positions[:, 2] += gap_A

    if stacking.upper() == "AB":
        # simple shift by one basis vector projection (not a full graphite model, but a reasonable AB-like offset)
        shift = np.array([0.5 * a, (math.sqrt(3.0) / 6.0) * a, 0.0], float)
        top.positions[:, :2] += shift[:2]
        top.wrap()

    mono.set_tags(np.ones(len(mono), dtype=int))
    top.set_tags(np.ones(len(top), dtype=int))

    bilayer = mono + top
    bilayer.set_cell(mono.cell, scale_atoms=False)
    bilayer.pbc = (True, True, True)
    bilayer.wrap()
    return bilayer


def ensure_graphene_tag_dir(root: Path, tag: str, nx: int, ny: int, gap_A: float,
                            vacuum_A: float, stacking: str, rebuild: bool):
    """
    Create/update orbmol/<tag>/final_relaxed.extxyz for graphene bilayer tags.
    Also writes relax.json with converged=True (this is a prepared starting structure).
    """
    tag_dir = root / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    start = tag_dir / "final_relaxed.extxyz"
    meta = tag_dir / "relax.json"

    if start.is_file() and meta.is_file() and (not rebuild):
        return tag_dir

    atoms = build_graphene_bilayer(nx=nx, ny=ny, gap_A=gap_A, vacuum_A=vacuum_A, stacking=stacking)
    write(start.as_posix(), atoms, format="extxyz")

    relax_meta = {
        "converged": True,
        "note": "Prepared starting structure (graphene bilayer). No geometry relaxation performed here.",
        "tag": tag,
        "nx": nx, "ny": ny, "gap_A": float(gap_A), "vacuum_A": float(vacuum_A),
        "stacking": stacking,
        "cellpar": list(map(float, cell_to_cellpar(atoms.cell))),
        "n_atoms": int(len(atoms)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta.write_text(json.dumps(relax_meta, indent=2))
    print(f"[BUILD] wrote {start} and {meta}")
    return tag_dir


# ------------------------- ORB helpers / calculator -------------------------

class DPWrapper(torch.nn.Module):
    def __init__(self, dp):
        super().__init__()
        self.dp = dp
    def forward(self, *a, **k):
        return self.dp(*a, **k)
    def __getattr__(self, n):
        try:
            return super().__getattr__(n)
        except AttributeError:
            return getattr(self.dp.module, n)

def build_orb_model(model_key: str, device: str, compile_flag: bool):
    fn = getattr(pretrained, model_key, None)
    if fn is None:
        opts = [m for m in dir(pretrained) if m.startswith("orb_v3")]
        raise ValueError(f"Unknown model '{model_key}'. Options: {', '.join(opts)}")
    base = fn(device=device, compile=compile_flag)
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return DPWrapper(torch.nn.DataParallel(base, device_ids=list(range(torch.cuda.device_count()))))
    return base

def get_orbcalc(model_key: str, device: str, compile_flag: bool):
    return ORBCalculator(build_orb_model(model_key, device, compile_flag), device=device)

_cat_charge = {"Na": +1, "Ca": +2, "Al": +3}
_ostate_q   = {"o-": -1, "oh": 0}

def infer_charge_from_tag(tag: str) -> int | None:
    info = parse_tag(tag)
    if info and info.get("mode") == "graphene":
        return 0
    m = re.match(r"^\d+x\d+_([A-Za-z]{1,2})_[^_]+_([a-z\-]+)$", tag)
    if not m:
        return None
    cat, ost = m.groups()
    if cat not in _cat_charge or ost not in _ostate_q:
        return None
    return _cat_charge[cat] + 2 * _ostate_q[ost]

def setup_orb_env_for_atoms(atoms: Atoms, target_radius_default: float = 7.5):
    atoms.wrap()
    Lx, Ly, Lz = atoms.cell.lengths()
    pbc = atoms.get_pbc()
    periodic_lengths = [L for L, p in zip((Lx, Ly, Lz), pbc) if p]
    if not periodic_lengths:
        raise ValueError("No periodic directions set; enable PBC for OrbMol.")
    try:
        target_radius = float(os.getenv("ORB_SEARCH_RADIUS", str(target_radius_default)))
    except Exception:
        target_radius = target_radius_default
    min_periodic = min(periodic_lengths)
    safe_radius = min(target_radius, 0.45 * min_periodic)
    if safe_radius < target_radius - 1e-6:
        print(f"[orb] reducing search radius {target_radius:.2f} → {safe_radius:.2f} Å (2r < min(L))")
    os.environ["ORB_SEARCH_RADIUS"] = f"{safe_radius:.6f}"
    os.environ.setdefault("ORB_MAX_NEIGHBORS", "128")
    return safe_radius

# Optional: CPU neighbour-search patch via env ORB_FORCE_CPU_FEAT=1
def _cpu_compute_supercell_neighbors_flexible(*args, **kwargs):
    if NearestNeighbors is None:
        raise RuntimeError("ORB_FORCE_CPU_FEAT=1 requested but scikit-learn is not available.")
    import numpy as _np, torch as _torch
    device      = kwargs.get("device", "cpu")
    to_torch    = kwargs.get("to_torch", True)
    super_pos   = kwargs.get("supercell_positions", None)
    central_pos = kwargs.get("central_cell_positions", None)
    max_n       = int(os.getenv("ORB_MAX_NEIGHBORS", kwargs.get("max_neighbors", 64)))
    radius      = float(os.getenv("ORB_SEARCH_RADIUS", kwargs.get("search_radius", 6.0)))

    def _to_np(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu().to(_torch.float32).numpy()
        else:
            x = _np.asarray(x, dtype=_np.float32)
        return x

    X_super_orig = _to_np(super_pos)
    X_query_orig = _to_np(central_pos) if central_pos is not None else X_super_orig
    n_query = X_query_orig.shape[0]

    super_mask = _np.all(_np.isfinite(X_super_orig), axis=1)
    query_mask = _np.all(_np.isfinite(X_query_orig), axis=1)
    super_idx_map = _np.nonzero(super_mask)[0]
    query_idx_map = _np.nonzero(query_mask)[0]
    X_super = X_super_orig[super_mask]
    X_query = X_query_orig[query_mask]

    if X_super.shape[0] == 0 or X_query.shape[0] == 0:
        if to_torch:
            zL = _torch.zeros(0, dtype=_torch.long, device=device)
            zV = _torch.zeros((0, 3), dtype=_torch.float32, device=device)
            zC = _torch.zeros(n_query, dtype=_torch.long, device=device)
            return zL, zL.clone(), zV, zC
        else:
            return (_np.zeros(0, _np.int64),) * 2 + (_np.zeros((0, 3), _np.float32), _np.zeros(n_query, _np.int64))

    nn = NearestNeighbors(radius=float(radius), algorithm="kd_tree")
    nn.fit(X_super)
    dists_list, inds_list = nn.radius_neighbors(X_query, radius=float(radius), sort_results=True)

    same_array = (central_pos is None)
    senders, receivers, vec_chunks = [], [], []
    counts = _np.zeros(n_query, dtype=_np.int64)

    for i_f, (ix_f, ds) in enumerate(zip(inds_list, dists_list)):
        i0 = int(query_idx_map[i_f])
        ix0 = super_idx_map[ix_f]
        if same_array and len(ix0) and ix0[0] == i0 and (len(ds) and ds[0] <= 1e-8):
            ix0 = ix0[1:]
            ds  = ds[1:]
        if len(ix0) == 0:
            continue
        if max_n and len(ix0) > max_n:
            ix0 = ix0[:max_n]
        counts[i0] = len(ix0)
        senders.extend([i0] * len(ix0))
        receivers.extend(ix0.tolist())
        vec_chunks.append(X_super_orig[ix0] - X_query_orig[i0])

    vectors = (_np.vstack(vec_chunks).astype(_np.float32) if vec_chunks else _np.zeros((0, 3), _np.float32))
    if to_torch:
        s = _torch.as_tensor(_np.asarray(senders, _np.int64), device=device)
        r = _torch.as_tensor(_np.asarray(receivers, _np.int64), device=device)
        v = _torch.as_tensor(vectors, device=device)
        c = _torch.as_tensor(counts, device=device)
        return s, r, v, c
    else:
        return _np.asarray(senders, _np.int64), _np.asarray(receivers, _np.int64), vectors, counts


if os.getenv("ORB_FORCE_CPU_FEAT", "0") == "1":
    feat_util.compute_supercell_neighbors = _cpu_compute_supercell_neighbors_flexible
    print("[orb] Using CPU neighbour-search patch (ORB_FORCE_CPU_FEAT=1).")
else:
    print("[orb] Using default ORB neighbour search.")


# ------------------------- solvation (skips by default) ---------------------

def solvate_one(tag_dir: Path, solvent_lib: Path, pad_A: float, dist_tol: float,
                tries_per_mol: int, max_mols_cap: int, seed: int, no_progress: bool) -> tuple[Path, Atoms, dict]:
    random.seed(seed)
    np.random.seed(seed)

    start = tag_dir / "final_relaxed.extxyz"
    if not start.is_file():
        raise FileNotFoundError(f"{start} not found")

    atoms = read(start.as_posix())
    atoms.wrap()
    cell = atoms.cell.array.copy()

    zlo, zhi, gap, pad_eff = sheet_planes_from_carbons(atoms, pad_A=pad_A)
    area, _ = cell_area_and_height(atoms)
    slab_thickness = zhi - zlo
    slab_volume = area * slab_thickness

    v_excl_existing = estimate_excluded_volume_A3(atoms, exclude_carbon=True, zslab=(zlo, zhi))

    info = parse_tag(tag_dir.name)
    if not info:
        raise RuntimeError(f"Tag format not recognised: {tag_dir.name}")
    solvent = info["solvent"]

    s_path = pick_solvent_file(tag_dir, solvent, solvent_lib)
    mol = load_solvent_geom(solvent, s_path)

    # Quick feasibility warning for very tight gaps
    mol_z_span = float(mol.get_positions()[:, 2].max() - mol.get_positions()[:, 2].min())
    if slab_thickness < (mol_z_span + 0.5):
        print(f"[warn] slab thickness={slab_thickness:.2f} Å; solvent z-span≈{mol_z_span:.2f} Å. "
              f"Packing may fail for {solvent} at this gap/pad.")

    MW = float(np.sum(mol.get_masses()))  # g/mol
    rho = SOLVENT_DENSITIES.get(solvent)
    if rho is None:
        raise RuntimeError(f"No density for '{solvent}'")

    NA = 6.02214076e23
    n_per_A3 = (rho / MW) * (NA / 1e24)
    V_free = max(0.0, slab_volume - v_excl_existing)
    n_target = int(max(0, math.floor(n_per_A3 * V_free)))
    if n_target > max_mols_cap:
        n_target = max_mols_cap

    pbar = tqdm(total=n_target, desc=f"{tag_dir.name} place", unit="mol") if (tqdm and not no_progress) else None

    pos_existing = atoms.get_positions()
    placed_positions = []
    placed_counts = 0

    for i in range(n_target):
        R = place_molecule_in_slab(
            mol=mol, cell=cell, zlo=zlo, zhi=zhi,
            existing_pos=np.vstack([pos_existing] + placed_positions) if placed_positions else pos_existing,
            tol=dist_tol, max_tries=tries_per_mol
        )
        if R is None:
            print(f"  [place] stop at {i}/{n_target} (max tries reached)")
            break
        placed_positions.append(R)
        placed_counts += 1
        if pbar:
            pbar.update(1)
        if (i + 1) % 50 == 0:
            print(f"  [place] {i+1}/{n_target}")

    if placed_counts == 0:
        raise RuntimeError("No molecules placed (try lowering --dist-tol or increasing --tries-per-mol).")
    if pbar:
        pbar.close()

    solvent_syms = mol.get_chemical_symbols()
    add_syms, add_pos = [], []
    for R in placed_positions:
        add_syms.extend(solvent_syms)
        add_pos.extend(R.tolist())

    add_atoms = Atoms(add_syms, positions=np.array(add_pos))
    add_atoms.set_cell(cell, scale_atoms=False)
    add_atoms.pbc = (True, True, True)
    # Tags for added solvent default to 0; graphene tags (if present) remain 1

    solvated = atoms + add_atoms
    solvated.wrap()

    outdir = tag_dir / "solvated"
    outdir.mkdir(exist_ok=True)
    out_extxyz = outdir / "start_solvated.extxyz"
    write(out_extxyz.as_posix(), solvated, format="extxyz")

    manifest = {
        "tag": tag_dir.name,
        "solvent": solvent,
        "solvent_source": (str(s_path) if s_path else "ASE molecule()"),
        "sheet_padding_A_requested": float(pad_A),
        "sheet_padding_A_used": float(pad_eff),
        "distance_tolerance_A": float(dist_tol),
        "tries_per_mol": int(tries_per_mol),
        "rng_seed": int(seed),
        "cellpar": list(map(float, cell_to_cellpar(solvated.cell))),
        "zlo_A": round(float(zlo), 6), "zhi_A": round(float(zhi), 6), "gap_mean_A": round(float(gap), 6),
        "slab_area_A2": round(float(area), 6),
        "slab_volume_A3": round(float(slab_volume), 6),
        "excluded_volume_existing_A3": round(float(v_excl_existing), 6),
        "density_g_cm3": float(rho),
        "MW_g_mol": round(float(MW), 3),
        "n_per_A3": float(n_per_A3),
        "n_target_initial": int(n_target),
        "n_placed": int(placed_counts),
        "notes": [
            "Placement = random rotation + PBC-aware min-distance rejection.",
            "Excluded volume = vdW spheres for non-C atoms inside slab (overlaps ignored)."
        ],
    }
    with open(outdir / "solvate_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[OK] {tag_dir.name}: wrote {out_extxyz} (placed {placed_counts} molecules)")
    return outdir, solvated, manifest


# ------------------------------ MD / resume ---------------------------------

def _read_last_logged_step(steps_csv: Path) -> int:
    if not steps_csv.is_file():
        return 0
    last = ""
    try:
        with open(steps_csv, "r") as fh:
            for line in fh:
                if line.strip():
                    last = line
    except Exception:
        return 0
    if not last or last.startswith("step"):
        return 0
    try:
        return int(last.split(",")[0].strip())
    except Exception:
        return 0


def _ensure_steps_header(steps_csv: Path):
    if not steps_csv.exists():
        with open(steps_csv, "w") as fh:
            fh.write("step,t_ps,E_eV,T_K,fmax\n")


# ------------------------------ NEMD helpers --------------------------------

AMU_KG = 1.66053906660e-27
ANG_M = 1.0e-10
FS_S = 1.0e-15
EV_J = 1.602176634e-19

# Conversion: (amu*(Å/fs)^2) -> eV
C_Ev_per_amuA2fs2 = (AMU_KG * (ANG_M / FS_S) ** 2) / EV_J  # ~103.6427


def mps2_to_Afs2(a_mps2: float) -> float:
    return float(a_mps2) * 1.0e-20


def _ensure_header(path: Path, header_line: str):
    if not path.exists():
        with open(path, "w") as fh:
            fh.write(header_line.rstrip() + "\n")


def _infer_graphene_mask_from_tags(atoms: Atoms):
    tags = atoms.get_tags()
    if tags is None or len(tags) != len(atoms):
        raise RuntimeError("No per-atom tags found; cannot identify graphene reliably for NEMD.")
    gmask = (tags == 1)
    if not np.any(gmask):
        raise RuntimeError("No tag==1 atoms found; cannot identify graphene reliably for NEMD.")
    return gmask


def _sheet_z_planes_from_mask(atoms: Atoms, graphene_mask: np.ndarray):
    z = atoms.get_positions()[:, 2]
    zG = z[graphene_mask]
    z_mid = float(np.median(zG))
    bot = zG[zG <= z_mid]
    top = zG[zG > z_mid]
    if bot.size < 5 or top.size < 5:
        labels, centers = kmeans1d(zG, k=2)
        z_bot = float(zG[labels == int(np.argmin(centers))].mean())
        z_top = float(zG[labels == int(np.argmax(centers))].mean())
    else:
        z_bot = float(bot.mean())
        z_top = float(top.mean())
    return z_bot, z_top, float(z_top - z_bot), 0.5 * (z_bot + z_top)


def _area_xy_from_cell(cell: np.ndarray) -> float:
    a1 = cell[0]
    a2 = cell[1]
    return float(np.linalg.norm(np.cross(a1, a2)))


def _prepare_solvent_blocks(atoms: Atoms, nat_per_mol: int):
    """
    Assumes solvent atoms are tag!=1 and were appended as contiguous blocks of a single template.
    Returns (solv_start, solv_end_inclusive, n_mols, nat_per_mol, masses_reshaped, weights, msum)
    """
    tags = atoms.get_tags()
    solv_idx = np.where(tags != 1)[0]
    if solv_idx.size == 0:
        raise RuntimeError("No solvent atoms found (tag!=1).")
    sidx = np.sort(solv_idx)

    if not np.all(np.diff(sidx) == 1):
        raise RuntimeError("Solvent indices not contiguous; molecule blocking would need a slower fallback.")
    solv_start = int(sidx[0])
    solv_end = int(sidx[-1])
    n_sol_atoms = solv_end - solv_start + 1
    if n_sol_atoms % nat_per_mol != 0:
        raise RuntimeError(f"Solvent atoms ({n_sol_atoms}) not divisible by nat_per_mol ({nat_per_mol}).")

    n_mols = n_sol_atoms // nat_per_mol
    masses = atoms.get_masses()[solv_start:solv_end + 1].reshape(n_mols, nat_per_mol)
    msum = masses.sum(axis=1)
    weights = masses / msum[:, None]
    return solv_start, solv_end, n_mols, nat_per_mol, masses, weights, msum


def _com_from_block(pos_block: np.ndarray, weights: np.ndarray):
    return (pos_block * weights[:, :, None]).sum(axis=1)


def _compute_T_yz_K(v: np.ndarray, masses_amu: np.ndarray, mobile_mask: np.ndarray) -> float:
    idx = np.where(mobile_mask)[0]
    if idx.size == 0:
        return float("nan")
    m = masses_amu[idx]
    vy = v[idx, 1]
    vz = v[idx, 2]
    KE_eV = 0.5 * np.sum(m * (vy * vy + vz * vz)) * C_Ev_per_amuA2fs2
    Ndof = 2 * idx.size
    return float((2.0 * KE_eV) / (Ndof * units.kB))


def run_nemd_poiseuille(
    out_dir: Path,
    atoms: Atoms,
    tag_name: str,
    solvent_name: str,
    solvent_source: str | None,
    solvent_lib: Path | None,
    device: str,
    model_key: str,
    compile_flag: bool,
    temp_K: float,
    dt_fs: float,
    warmup_ps: float,
    prod_ps: float,
    friction_ps_inv: float,
    body_accel_mps2: float,
    profile_every: int,
    profile_z_bins: int,
    n_profile_blocks: int,
    z_exclude_A: float,
    traj_every: int,
    restart_every: int,
    resumed: bool,
    freeze_graphene: bool,
    seed: int,
):
    out_dir.mkdir(exist_ok=True, parents=True)
    traj_path = out_dir / "traj.extxyz"
    steps_csv = out_dir / "nemd_steps.csv"
    wall_csv = out_dir / "wall_force.csv"
    accum_npz = out_dir / "profile_accum.npz"
    restart_path = out_dir / "restart_last.extxyz"
    summary_js = out_dir / "nemd_summary.json"

    _ensure_header(steps_csv, "step,t_ps,E_pot_eV,T_yz_K,fmax_eV_per_A,mean_vx_A_per_fs")
    _ensure_header(wall_csv, "step,t_ps,Fwall_x_eV_per_A,Fwall_y_eV_per_A")

    rng = np.random.default_rng(int(seed))

    graphene_mask = _infer_graphene_mask_from_tags(atoms)
    mobile_mask = ~graphene_mask if freeze_graphene else np.ones(len(atoms), dtype=bool)

    setup_orb_env_for_atoms(atoms)
    calc = get_orbcalc(model_key, device=device, compile_flag=compile_flag)
    atoms.calc = calc

    v = atoms.get_velocities()
    if v is None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp_K, force_temp=True)
        v = atoms.get_velocities()
    v = np.array(v, float)

    if freeze_graphene:
        v[graphene_mask, :] = 0.0
        atoms.set_velocities(v)

    cell = atoms.cell.array.copy()
    Axy = _area_xy_from_cell(cell)
    z_bot, z_top, gap, z_mid = _sheet_z_planes_from_mask(atoms, graphene_mask)
    if z_top <= z_bot:
        raise RuntimeError("Bad graphene plane ordering for NEMD.")
    z_min = float(z_bot)
    z_max = float(z_top)
    dz = (z_max - z_min) / int(profile_z_bins)
    bin_edges = np.linspace(z_min, z_max, int(profile_z_bins) + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    templ = load_solvent_template(solvent_name, solvent_source, solvent_lib)
    nat_per_mol = int(len(templ))

    solv_start, solv_end, n_mols, nat, masses_mol_atoms, weights, msum = _prepare_solvent_blocks(atoms, nat_per_mol)
    mol_mass_amu = float(np.sum(templ.get_masses()))
    mol_mass_g = mol_mass_amu * 1.66053906660e-24

    a_Afs2 = mps2_to_Afs2(body_accel_mps2)
    gamma_fs = float(friction_ps_inv) / 1000.0
    c = float(np.exp(-gamma_fs * dt_fs))
    masses_amu = atoms.get_masses()
    sigma = np.sqrt((1.0 - c * c) * (units.kB * temp_K) / (masses_amu * C_Ev_per_amuA2fs2))

    if accum_npz.is_file():
        Z = np.load(accum_npz)
        vx_sum_blk = Z["vx_sum_blk"]
        vx_n_blk = Z["vx_n_blk"]
        mol_n_blk = Z["mol_n_blk"]
        mol_m_blk = Z["mol_m_blk"]
        prod_samples_done = int(Z["prod_samples_done"])
        assert vx_sum_blk.shape == (n_profile_blocks, profile_z_bins)
    else:
        vx_sum_blk = np.zeros((n_profile_blocks, profile_z_bins), dtype=float)
        vx_n_blk = np.zeros((n_profile_blocks, profile_z_bins), dtype=float)
        mol_n_blk = np.zeros((n_profile_blocks, profile_z_bins), dtype=float)
        mol_m_blk = np.zeros((n_profile_blocks, profile_z_bins), dtype=float)
        prod_samples_done = 0

    last_step_logged = _read_last_logged_step(steps_csv)
    step0 = int(last_step_logged)

    n_warm_steps = int(round((warmup_ps * 1000.0) / dt_fs))
    n_prod_steps = int(round((prod_ps * 1000.0) / dt_fs))
    if n_prod_steps <= 0:
        raise RuntimeError("prod_ps must be > 0 for NEMD.")
    if profile_every < 1:
        profile_every = 1

    total_prod_samples_target = max(1, n_prod_steps // profile_every)
    samples_per_block = max(1, int(math.floor(total_prod_samples_target / n_profile_blocks)))

    def current_block_index(sample_idx: int) -> int:
        bi = sample_idx // samples_per_block
        return int(min(bi, n_profile_blocks - 1))

    F = atoms.get_forces()

    def log_step(step_abs: int, t_ps: float, v_now: np.ndarray, F_now: np.ndarray):
        E = float(atoms.get_potential_energy())
        fmax = float(np.linalg.norm(F_now, axis=1).max())
        T_yz = _compute_T_yz_K(v_now, masses_amu, mobile_mask & (~graphene_mask))
        mean_vx = float(np.mean(v_now[~graphene_mask, 0]))
        with open(steps_csv, "a") as fh:
            fh.write(f"{step_abs},{t_ps:.6f},{E:.8f},{T_yz:.2f},{fmax:.6f},{mean_vx:.8e}\n")

    def log_wall_force(step_abs: int, t_ps: float, F_now: np.ndarray):
        Fx = float(F_now[graphene_mask, 0].sum())
        Fy = float(F_now[graphene_mask, 1].sum())
        with open(wall_csv, "a") as fh:
            fh.write(f"{step_abs},{t_ps:.6f},{Fx:.8e},{Fy:.8e}\n")

    def maybe_write_traj(step_abs: int):
        if traj_every > 0 and (step_abs % traj_every == 0):
            atoms.set_velocities(v)
            atoms.info["energy"] = float(atoms.get_potential_energy())
            atoms.info["free_energy"] = float(atoms.get_potential_energy())
            write(traj_path.as_posix(), atoms, format="extxyz", append=True)

    print(f"[NEMD] {tag_name}: warmup {warmup_ps} ps, production {prod_ps} ps, a={body_accel_mps2:.3e} m/s^2")
    t0 = time.perf_counter()

    solv_atom_slice = slice(solv_start, solv_end + 1)
    solv_atom_mask = np.zeros(len(atoms), dtype=bool)
    solv_atom_mask[solv_start:solv_end + 1] = True
    log_every = max(1, profile_every)

    step_abs = step0
    for istep in range(n_warm_steps + n_prod_steps):
        acc = F / (masses_amu[:, None] * C_Ev_per_amuA2fs2)
        acc[~graphene_mask, 0] += a_Afs2
        if freeze_graphene:
            acc[graphene_mask, :] = 0.0

        v[mobile_mask, :] += 0.5 * dt_fs * acc[mobile_mask, :]

        pos = atoms.get_positions()
        pos[mobile_mask, :] += 0.5 * dt_fs * v[mobile_mask, :]
        atoms.set_positions(pos)
        atoms.wrap()

        if friction_ps_inv > 0.0:
            noise = rng.normal(size=(len(atoms), 2))
            msk = solv_atom_mask & mobile_mask
            v[msk, 1] = c * v[msk, 1] + sigma[msk] * noise[msk, 0]
            v[msk, 2] = c * v[msk, 2] + sigma[msk] * noise[msk, 1]

            vy_mean = float(np.mean(v[msk, 1]))
            vz_mean = float(np.mean(v[msk, 2]))
            v[msk, 1] -= vy_mean
            v[msk, 2] -= vz_mean

        pos = atoms.get_positions()
        pos[mobile_mask, :] += 0.5 * dt_fs * v[mobile_mask, :]
        atoms.set_positions(pos)
        atoms.wrap()

        atoms.set_velocities(v)
        F = atoms.get_forces()

        acc = F / (masses_amu[:, None] * C_Ev_per_amuA2fs2)
        acc[~graphene_mask, 0] += a_Afs2
        if freeze_graphene:
            acc[graphene_mask, :] = 0.0
        v[mobile_mask, :] += 0.5 * dt_fs * acc[mobile_mask, :]

        if freeze_graphene:
            v[graphene_mask, :] = 0.0

        step_abs += 1
        t_ps = step_abs * dt_fs / 1000.0

        if (step_abs % log_every) == 0:
            log_step(step_abs, t_ps, v, F)
            log_wall_force(step_abs, t_ps, F)

        maybe_write_traj(step_abs)

        if restart_every > 0 and (step_abs % restart_every == 0):
            atoms.set_velocities(v)
            write(restart_path.as_posix(), atoms, format="extxyz")

        if istep >= n_warm_steps:
            prod_step = istep - n_warm_steps
            if (prod_step % profile_every) == 0:
                posS = atoms.get_positions()[solv_atom_slice].reshape(n_mols, nat, 3)
                velS = v[solv_atom_slice].reshape(n_mols, nat, 3)

                com = _com_from_block(posS, weights)
                comv = _com_from_block(velS, weights)

                zc = com[:, 2]
                vx = comv[:, 0]

                bins = np.floor((zc - z_min) / dz).astype(int)
                ok = (bins >= 0) & (bins < profile_z_bins)
                bins = bins[ok]
                vx = vx[ok]

                bi = current_block_index(prod_samples_done)

                np.add.at(vx_sum_blk[bi], bins, vx)
                np.add.at(vx_n_blk[bi], bins, 1.0)
                np.add.at(mol_n_blk[bi], bins, 1.0)
                np.add.at(mol_m_blk[bi], bins, mol_mass_g)

                prod_samples_done += 1

                if (prod_samples_done % 100) == 0:
                    np.savez(
                        accum_npz,
                        vx_sum_blk=vx_sum_blk,
                        vx_n_blk=vx_n_blk,
                        mol_n_blk=mol_n_blk,
                        mol_m_blk=mol_m_blk,
                        prod_samples_done=prod_samples_done,
                    )

    atoms.set_velocities(v)
    write(restart_path.as_posix(), atoms, format="extxyz")

    np.savez(
        accum_npz,
        vx_sum_blk=vx_sum_blk,
        vx_n_blk=vx_n_blk,
        mol_n_blk=mol_n_blk,
        mol_m_blk=mol_m_blk,
        prod_samples_done=prod_samples_done,
    )

    vol_bin_A3 = Axy * dz

    with np.errstate(invalid="ignore", divide="ignore"):
        vx_mean_blk = np.where(vx_n_blk > 0, vx_sum_blk / vx_n_blk, np.nan)

    vx_sum_all = np.nansum(vx_sum_blk, axis=0)
    vx_n_all = np.nansum(vx_n_blk, axis=0)
    vx_mean = np.where(vx_n_all > 0, vx_sum_all / vx_n_all, np.nan)

    vx_sem = np.full(profile_z_bins, np.nan, dtype=float)
    for k in range(profile_z_bins):
        vals = vx_mean_blk[:, k]
        okb = np.isfinite(vals)
        if okb.sum() >= 3:
            vx_sem[k] = float(np.nanstd(vals[okb], ddof=1) / np.sqrt(okb.sum()))

    mol_n_all = np.nansum(mol_n_blk, axis=0)
    mol_m_all = np.nansum(mol_m_blk, axis=0)
    Nsamp = float(max(1, prod_samples_done))
    n_density_per_A3 = mol_n_all / (Nsamp * vol_bin_A3)
    rho_g_cm3 = (mol_m_all / Nsamp) / (vol_bin_A3 * 1e-24)

    prof_csv = out_dir / "vx_profile_z.csv"
    with open(prof_csv, "w") as fh:
        fh.write("z_A,mean_vx_A_per_fs,sem_vx_A_per_fs,number_density_per_A3,mass_density_g_cm3,count\n")
        for zc, vv, se, nd, rg, nn in zip(bin_centers, vx_mean, vx_sem, n_density_per_A3, rho_g_cm3, vx_n_all):
            vv_s = "" if np.isnan(vv) else f"{vv:.8e}"
            se_s = "" if np.isnan(se) else f"{se:.8e}"
            fh.write(f"{zc:.6f},{vv_s},{se_s},{nd:.8e},{rg:.6f},{int(nn)}\n")

    meta = {
        "tag": tag_name,
        "mode": "nemd_poiseuille",
        "solvent": solvent_name,
        "device": device,
        "model": model_key,
        "temp_K": float(temp_K),
        "dt_fs": float(dt_fs),
        "warmup_ps": float(warmup_ps),
        "prod_ps": float(prod_ps),
        "friction_ps_inv": float(friction_ps_inv),
        "body_accel_mps2": float(body_accel_mps2),
        "body_accel_A_fs2": float(a_Afs2),
        "freeze_graphene": bool(freeze_graphene),
        "graphene_z_bot_A": float(z_bot),
        "graphene_z_top_A": float(z_top),
        "gap_A": float(gap),
        "area_A2": float(Axy),
        "profile": {
            "z_bins": int(profile_z_bins),
            "profile_every_steps": int(profile_every),
            "n_profile_blocks": int(n_profile_blocks),
            "z_exclude_A_recommended": float(z_exclude_A),
        },
        "files": {
            "vx_profile_z_csv": str(prof_csv),
            "wall_force_csv": str(wall_csv),
            "steps_csv": str(steps_csv),
            "accum_npz": str(accum_npz),
            "restart": str(restart_path),
            "traj_extxyz": str(traj_path) if traj_path.exists() else None,
        },
        "notes": [
            "Viscosity/slip are extracted by fitting v_x(z) in the continuum-like region (exclude near-wall layers).",
            "Transverse (y,z) Langevin thermostat avoids biasing the streaming direction.",
            "For linear-response defensibility, repeat with 2–3 smaller accelerations and verify eta,b invariance.",
        ],
    }
    summary_js.write_text(json.dumps(meta, indent=2))

    t1 = time.perf_counter()
    print(f"[NEMD DONE] {tag_name}: wrote {out_dir} | wall {t1 - t0:.1f}s | samples={prod_samples_done}")


def run_md(out_dir: Path, atoms: Atoms, device: str, model_key: str, compile_flag: bool,
           temp_K: float, dt_fs: float, add_time_ps: float,
           friction_ps_inv: float, traj_every: int, restart_every: int,
           charge: float | None, spin_mult: float | None, infer_tag: str | None,
           no_progress: bool, progress_every: int,
           resumed: bool, tag_name: str,
           freeze_graphene: bool):

    out_dir.mkdir(exist_ok=True)
    traj_path    = out_dir / "md.traj.extxyz"
    steps_csv    = out_dir / "md_steps.csv"
    restart_path = out_dir / "restart_last.extxyz"
    summary_js   = out_dir / "md_summary.json"

    last_step_logged = _read_last_logged_step(steps_csv)
    _ensure_steps_header(steps_csv)

    q = charge if charge is not None else (infer_charge_from_tag(infer_tag) if infer_tag else 0.0)
    if q is None:
        q = 0.0
    s = spin_mult if spin_mult is not None else (2.0 if (abs(q) % 2 == 1) else 1.0)
    atoms.info["charge"] = float(q)
    atoms.info["spin"]   = float(s)

    atoms.pbc = (True, True, True)
    atoms.wrap()

    # Freeze graphene atoms if requested:
    # We rely on ASE per-atom tags: graphene atoms should be tag==1.
    if freeze_graphene:
        tags = atoms.get_tags()
        if tags is None or len(tags) != len(atoms):
            print("[freeze] no tags found; cannot freeze graphene reliably.")
        else:
            mask = (tags == 1)
            if np.any(mask):
                atoms.set_constraint(FixAtoms(mask=mask))
                print(f"[freeze] fixed {int(mask.sum())} graphene atoms (tag==1)")
            else:
                print("[freeze] no tag==1 atoms found; nothing frozen.")

    setup_orb_env_for_atoms(atoms)
    calc = get_orbcalc(model_key, device=device, compile_flag=compile_flag)
    atoms.calc = calc

    if not resumed:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp_K, force_temp=True)

    dt = dt_fs * units.fs
    dyn = Langevin(atoms, timestep=dt, temperature_K=temp_K, friction=friction_ps_inv)

    nsteps_run = int(round((add_time_ps * 1000.0) / dt_fs))
    if nsteps_run <= 0:
        print(f"[MD SKIP] Requested additional time {add_time_ps} ps → 0 steps at dt={dt_fs} fs.")
        return

    use_pbar = (tqdm is not None) and (not no_progress)
    pbar = tqdm(total=nsteps_run, desc=f"{tag_name} MD", unit="step") if use_pbar else None

    steps_done = 0
    last_write = -1

    def write_frame(step_abs: int):
        v = atoms.get_velocities()
        if v is not None:
            atoms.set_velocities(v)

        E = float(atoms.get_potential_energy())
        N = len(atoms)
        T = float(2.0 * atoms.get_kinetic_energy() / (3.0 * N * units.kB))
        F = atoms.get_forces()
        fmax = float(np.linalg.norm(F, axis=1).max())

        write(traj_path.as_posix(), atoms, format="extxyz", append=True)
        with open(steps_csv, "a") as fh:
            fh.write(f"{step_abs},{step_abs*dt_fs/1000.0:.6f},{E:.8f},{T:.2f},{fmax:.6f}\n")

    if not resumed and last_step_logged == 0:
        write_frame(0)
        last_write = 0

    def progress_hook():
        if pbar:
            pbar.update(progress_every)

    def step_hook():
        nonlocal steps_done, last_write
        steps_done += 1
        step_abs = last_step_logged + steps_done
        if (traj_every <= 1) or (steps_done % traj_every == 0):
            write_frame(step_abs)
            last_write = step_abs
        if (restart_every > 0) and (steps_done % restart_every == 0):
            write(restart_path.as_posix(), atoms, format="extxyz")

    dyn.attach(progress_hook, interval=max(1, progress_every))
    dyn.attach(step_hook,     interval=1)

    t0 = time.perf_counter()
    try:
        dyn.run(nsteps_run)
    finally:
        if pbar:
            pbar.close()
        t1 = time.perf_counter()

    if last_write < (last_step_logged + steps_done):
        write_frame(last_step_logged + steps_done)
        write(restart_path.as_posix(), atoms, format="extxyz")

    Efin = float(atoms.get_potential_energy())
    Ffin = atoms.get_forces()
    fmax_fin = float(np.linalg.norm(Ffin, axis=1).max())

    meta = {
        "added_time_ps": float(add_time_ps),
        "dt_fs": float(dt_fs),
        "temp_K": float(temp_K),
        "friction_ps_inv": float(friction_ps_inv),
        "steps_this_run": int(steps_done),
        "traj_every": int(traj_every),
        "restart_every": int(restart_every),
        "device": device,
        "model": model_key,
        "charge": float(q),
        "spin_multiplicity": float(s),
        "freeze_graphene": bool(freeze_graphene),
        "energy_final_eV": round(Efin, 6),
        "fmax_final_eV_per_A": round(fmax_fin, 6),
        "wall_s": round(t1 - t0, 3),
        "last_step_cumulative": int(last_step_logged + steps_done)
    }
    with open(summary_js, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[MD DONE] {tag_name}: +{steps_done} steps ({add_time_ps} ps) | total steps ≈ {meta['last_step_cumulative']}")


# ------------------------------ tag discovery --------------------------------

def find_converged_tags(orbmol_root: Path, subset: list[str] | None,
                        min_n: int = 1, max_n: int = 999, square_only: bool = False):
    def _ok(info):
        if info is None:
            return False
        nx, ny = info["nx"], info["ny"]
        if not (min_n <= nx <= max_n and min_n <= ny <= max_n):
            return False
        if square_only and nx != ny:
            return False
        return True

    if subset:
        items = []
        for t in subset:
            d = orbmol_root / t
            if (d / "final_relaxed.extxyz").is_file() and relax_is_converged(d):
                info = parse_tag(t)
                if _ok(info):
                    items.append((info["nx"], info["ny"], t))
            else:
                print(f"[SKIP] {t}: missing final_relaxed.extxyz or not converged")
        for _, _, name in sorted(items):
            yield name
        return

    items = []
    for d in (p for p in orbmol_root.iterdir() if p.is_dir()):
        if (d / "final_relaxed.extxyz").is_file() and relax_is_converged(d):
            info = parse_tag(d.name)
            if _ok(info):
                items.append((info["nx"], info["ny"], d.name))
    for _, _, name in sorted(items):
        yield name


# ------------------------------ CLI -----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Solvate OrbMol systems and run resumable NVT MD (append in-place).")

    ap.add_argument("--orbmol-root", default="orbmol",
                    help="Root containing <tag>/ with relax.json + final_relaxed.extxyz")

    # Either provide --tags explicitly OR build a sweep (graphene mode)
    ap.add_argument("--tags", nargs="*", help="Optional subset of tags to run")
    ap.add_argument("--solvent-lib", type=Path, default=Path("/home/kgrego23/solvents"))

    # Graphene sweep builder
    ap.add_argument("--build-graphene-sweep", action="store_true",
                    help="If set, create graphene bilayer starting structures for a solvent×gap sweep.")
    ap.add_argument("--rebuild-graphene", action="store_true",
                    help="If set with --build-graphene-sweep, overwrite existing final_relaxed.extxyz.")
    ap.add_argument("--nx", type=int, default=8, help="Graphene replication in x for sweep tags")
    ap.add_argument("--ny", type=int, default=8, help="Graphene replication in y for sweep tags")
    ap.add_argument("--gaps", type=float, nargs="*", default=DEFAULT_GAPS_A, help="Interlayer gaps (Å) for sweep")
    ap.add_argument("--solvents", nargs="*", default=DEFAULT_SOLVENTS, help="Solvents for sweep tags")
    ap.add_argument("--vacuum-A", type=float, default=20.0, help="Vacuum padding outside sheets (each side) in Å")
    ap.add_argument("--stacking", choices=["AA", "AB"], default="AA", help="Bilayer stacking")

    # Solvation controls
    ap.add_argument("--resolvate", action="store_true",
                    help="Force re-solvation (otherwise skip if solvated/start_solvated.extxyz exists).")
    ap.add_argument("--sheet-pad", type=float, default=SHEET_PADDING_A)
    ap.add_argument("--dist-tol", type=float, default=DIST_TOL_A)
    ap.add_argument("--tries-per-mol", type=int, default=MAX_TRIES_PER_MOL)
    ap.add_argument("--max-mols-cap", type=int, default=MAX_MOLS_CAP)
    ap.add_argument("--seed", type=int, default=RNG_SEED)

    # MD controls
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--model", default="orb_v3_conservative_omol")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--temp-K", type=float, default=300.0)
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--time-ps", type=float, default=100.0)
    ap.add_argument("--friction-ps-inv", type=float, default=0.002)
    ap.add_argument("--traj-interval", type=int, default=100)
    ap.add_argument("--restart-every", type=int, default=5000)
    ap.add_argument("--fresh-start", action="store_true",
                    help="Ignore existing restart_last.extxyz (start from start_solvated.extxyz and reinit velocities).")

    # Freeze graphene (default ON)
    ap.add_argument("--no-freeze-graphene", action="store_true",
                    help="Allow graphene sheets to move during MD (default freezes tag==1 atoms).")

    # NEMD controls
    ap.add_argument("--nemd", action="store_true",
                    help="Run NEMD Poiseuille (body-force) instead of equilibrium Langevin MD.")
    ap.add_argument("--nemd-warmup-ps", type=float, default=20.0,
                    help="Warmup time (ps) before sampling profiles.")
    ap.add_argument("--nemd-prod-ps", type=float, default=50.0,
                    help="Production time (ps) for profile accumulation.")
    ap.add_argument("--body-accel-mps2", type=float, default=1e12,
                    help="Body acceleration (m/s^2) applied to solvent along +x.")
    ap.add_argument("--profile-every", type=int, default=20,
                    help="Accumulate v_x(z) every N steps during production.")
    ap.add_argument("--profile-z-bins", type=int, default=200,
                    help="Number of z bins for v_x(z) and density(z).")
    ap.add_argument("--profile-blocks", type=int, default=10,
                    help="Number of time blocks for SEM via block averaging.")
    ap.add_argument("--z-exclude-A", type=float, default=3.0,
                    help="Recommended near-wall exclusion for continuum fits (Å).")

    # Charge/spin
    ap.add_argument("--charge", type=float, default=None)
    ap.add_argument("--spin",   type=float, default=None)

    # UI / filtering
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    ap.add_argument("--min-n", type=int, default=1, help="Minimum slab replication (nx, ny) to include when scanning")
    ap.add_argument("--max-n", type=int, default=999, help="Maximum slab replication (nx, ny) to include when scanning")
    ap.add_argument("--square-only", action="store_true", help="Only run square systems (nx == ny) when scanning")

    args = ap.parse_args()

    root = Path(args.orbmol_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # If building a graphene sweep, generate tags and ensure starting structures exist.
    if args.build_graphene_sweep:
        tags = []
        for gap in args.gaps:
            for solvent in args.solvents:
                tag = f"{args.nx}x{args.ny}_Gr_{solvent.lower()}_gap{gap:g}A"
                ensure_graphene_tag_dir(
                    root=root, tag=tag,
                    nx=args.nx, ny=args.ny, gap_A=float(gap),
                    vacuum_A=float(args.vacuum_A),
                    stacking=args.stacking,
                    rebuild=args.rebuild_graphene
                )
                tags.append(tag)
    else:
        tags = list(find_converged_tags(root, args.tags, args.min_n, args.max_n, args.square_only))

    if not tags:
        print("[INFO] No converged tags found / built.")
        return

    freeze_graphene = (not args.no_freeze_graphene)

    for tag in tags:
        tag_dir = root / tag
        print(f"\n=== {tag} ===")
        solv_dir = tag_dir / "solvated"
        start_solvated = solv_dir / "start_solvated.extxyz"

        # Solvate only if needed or forced
        if (not start_solvated.is_file()) or args.resolvate:
            try:
                solv_dir, solvated_atoms, _ = solvate_one(
                    tag_dir=tag_dir, solvent_lib=args.solvent_lib,
                    pad_A=args.sheet_pad, dist_tol=args.dist_tol,
                    tries_per_mol=args.tries_per_mol, max_mols_cap=args.max_mols_cap,
                    seed=args.seed, no_progress=args.no_progress
                )
            except Exception as e:
                print(f"[SOLVATE SKIP] {tag}: {e}")
                if not start_solvated.is_file():
                    continue
                solvated_atoms = read(start_solvated.as_posix())
        else:
            solvated_atoms = read(start_solvated.as_posix())

        # Choose start file (default resume)
        restart_path = solv_dir / "restart_last.extxyz"
        if (not args.fresh_start) and restart_path.is_file():
            start_path = restart_path
            resumed = True
        else:
            start_path = start_solvated
            resumed = False

        try:
            atoms0 = read(start_path.as_posix())
        except Exception as e:
            print(f"[MD ERROR] {tag}: failed to read start structure '{start_path}': {e}")
            continue

        atoms0.set_cell(solvated_atoms.cell, scale_atoms=False)
        atoms0.pbc = (True, True, True)

        try:
            manifest_p = solv_dir / "solvate_manifest.json"
            solvent_name = parse_tag(tag)["solvent"]
            solvent_source = None
            if manifest_p.is_file():
                try:
                    man = json.loads(manifest_p.read_text())
                    solvent_name = man.get("solvent", solvent_name)
                    solvent_source = man.get("solvent_source", None)
                except Exception:
                    pass

            if args.nemd:
                ax = args.body_accel_mps2
                nemd_dir = solv_dir / f"nemd_poiseuille_ax{ax:.2e}"
                restart_path = solv_dir / "restart_last.extxyz"
                if (not args.fresh_start) and restart_path.is_file():
                    start_path = restart_path
                    resumed = True
                else:
                    start_path = start_solvated
                    resumed = False

                atoms0 = read(start_path.as_posix())
                atoms0.set_cell(solvated_atoms.cell, scale_atoms=False)
                atoms0.pbc = (True, True, True)

                run_nemd_poiseuille(
                    out_dir=nemd_dir,
                    atoms=atoms0,
                    tag_name=tag,
                    solvent_name=solvent_name,
                    solvent_source=solvent_source,
                    solvent_lib=args.solvent_lib,
                    device=args.device,
                    model_key=args.model,
                    compile_flag=not args.no_compile,
                    temp_K=args.temp_K,
                    dt_fs=args.dt_fs,
                    warmup_ps=args.nemd_warmup_ps,
                    prod_ps=args.nemd_prod_ps,
                    friction_ps_inv=args.friction_ps_inv,
                    body_accel_mps2=args.body_accel_mps2,
                    profile_every=args.profile_every,
                    profile_z_bins=args.profile_z_bins,
                    n_profile_blocks=args.profile_blocks,
                    z_exclude_A=args.z_exclude_A,
                    traj_every=args.traj_interval,
                    restart_every=args.restart_every,
                    resumed=resumed,
                    freeze_graphene=freeze_graphene,
                    seed=args.seed,
                )
            else:
                run_md(
                    out_dir=solv_dir, atoms=atoms0, device=args.device, model_key=args.model,
                    compile_flag=not args.no_compile, temp_K=args.temp_K, dt_fs=args.dt_fs,
                    add_time_ps=args.time_ps, friction_ps_inv=args.friction_ps_inv,
                    traj_every=args.traj_interval, restart_every=args.restart_every,
                    charge=args.charge, spin_mult=args.spin, infer_tag=tag,
                    no_progress=args.no_progress, progress_every=max(1, min(args.traj_interval, 100)),
                    resumed=resumed, tag_name=tag,
                    freeze_graphene=freeze_graphene
                )
        except Exception as e:
            print(f"[MD ERROR] {tag}: {e}")


if __name__ == "__main__":
    main()
