from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import os

import config
from config import Species

# ── Simulation Constants ─────────────────────────────────────────────────────
G = config.G
RHO_F = config.RHO_F
MU = config.MU
N_PARTICLES_TOTAL = config.N_PARTICLES_TOTAL
H_COLUMN = config.H_COLUMN
DT = config.DT
N_CELLS = config.N_CELLS
COLUMN_ASPECT_RATIO = config.COLUMN_ASPECT_RATIO
PHI_BED = config.PHI_BED
PHI_MAX = config.PHI_MAX
ETA_INTRINSIC = config.ETA_INTRINSIC
K_REP = config.K_REP
D_MIN = config.D_MIN
D_MAX = config.D_MAX

# =============================================================================
# Physics Helpers
# =============================================================================

def drag_coefficient(Re: float) -> float:
    """Schiller-Naumann drag coefficient for a sphere."""
    if Re < 1e-9:
        return 0.0
    elif Re < 1000.0:
        return (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)
    else:
        return 0.44

def terminal_velocity(d: float, rho_p: float, rho_f: float = RHO_F, mu: float = MU,
                      tol: float = 1e-10, max_iter: int = 2000) -> tuple[float, float]:
    """Iteratively solve for terminal velocity (vt) and Reynolds number (Re_t)."""
    r  = d / 2.0
    vt = (2.0 * r**2 * (rho_p - rho_f) * G) / (9.0 * mu)
    for _ in range(max_iter):
        Re = rho_f * vt * d / mu
        Cd = drag_coefficient(Re)
        if Cd == 0.0: break
        vt_new = np.sqrt(4.0 / 3.0 * (rho_p - rho_f) / rho_f * G * d / Cd)
        if abs(vt_new - vt) < tol:
            return vt_new, rho_f * vt_new * d / mu
        vt = vt_new
    return vt, rho_f * vt * d / mu

def rz_exponent(Re_t: float) -> float:
    """Richardson-Zaki exponent n (Garside & Al-Dibouni 1977)."""
    if Re_t < 0.2: return 4.65
    elif Re_t < 1.0: return 4.35 + 17.5 * Re_t ** (-0.03) / (1.0 + 0.175 * Re_t ** 0.75)
    elif Re_t < 500.0: return 4.45 * Re_t ** (-0.1)
    else: return 2.39

def krieger_dougherty(phi: np.ndarray, mu_f: float = MU, phi_max: float = PHI_MAX,
                      eta: float = ETA_INTRINSIC) -> np.ndarray:
    """Krieger-Dougherty effective viscosity μ_eff(φ)."""
    phi_safe = np.clip(np.asarray(phi, dtype=float), 0.0, phi_max * 0.9999)
    return mu_f * (1.0 - phi_safe / phi_max) ** (-eta * phi_max)

def apply_collisions(z: np.ndarray, r: np.ndarray, active: np.ndarray,
                     h_bed: float, cell_edges: np.ndarray) -> np.ndarray:
    """Soft-sphere DEM collision correction (vertical only)."""
    dz = np.zeros_like(z)
    act_idx = np.where(active)[0]
    if len(act_idx) < 2: return dz

    n_cells = len(cell_edges) - 1
    ci = np.clip(np.digitize(z[act_idx], cell_edges) - 1, 0, n_cells - 1)
    cell_map: dict[int, list[int]] = {}
    for local, i in enumerate(act_idx):
        cell_map.setdefault(ci[local], []).append(i)

    checked = set()
    for c, members in cell_map.items():
        candidates = members + cell_map.get(c + 1, [])
        for i in members:
            for j in candidates:
                if j <= i or (i, j) in checked: continue
                checked.add((i, j))
                sep, min_d = abs(z[i] - z[j]), r[i] + r[j]
                if sep < min_d:
                    delta = K_REP * (min_d - sep) * 0.5
                    if z[i] > z[j]: dz[i] += delta; dz[j] -= delta
                    else: dz[i] -= delta; dz[j] += delta

    for i in act_idx:
        if z[i] + dz[i] < h_bed: dz[i] = h_bed - z[i]
    return dz

# =============================================================================
# Simulation Engine
# =============================================================================

def run_simulation(species_list: list[Species] | None = None, seed: int = 42,
                   t_max: float | None = None, snap_pcts: list[float] | None = None) -> dict:
    """Main simulation loop for multi-species particle settling."""
    if species_list is None: species_list = config.MIXTURE
    phi_global = sum(sp.phi for sp in species_list)
    if not (0 < phi_global < PHI_BED):
        raise ValueError(f"Total phi = {phi_global:.4f} must be < PHI_BED.")

    n_species, rng = len(species_list), np.random.default_rng(seed)
    raw_counts = np.array([sp.phi / phi_global * N_PARTICLES_TOTAL for sp in species_list])
    counts = np.round(raw_counts).astype(int)
    counts[np.argmax(raw_counts - counts)] += (N_PARTICLES_TOTAL - counts.sum())
    n_total = counts.sum()

    diameters, rho_arr, species_id = np.empty(n_total), np.empty(n_total), np.empty(n_total, dtype=int)
    idx = 0
    for s, (sp, cnt) in enumerate(zip(species_list, counts)):
        mu_ln, sigma_ln = sp.lognormal_params()
        d = np.clip(rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=cnt), D_MIN, D_MAX)
        diameters[idx:idx+cnt], rho_arr[idx:idx+cnt], species_id[idx:idx+cnt] = d, sp.rho_p, s
        idx += cnt

    raw_vols = (np.pi / 6.0) * diameters ** 3
    particle_vols = np.empty(n_total)
    for s, sp in enumerate(species_list):
        mask = species_id == s
        particle_vols[mask] = raw_vols[mask] * (sp.phi * H_COLUMN / raw_vols[mask].sum())

    vt_arr, n_arr = np.empty(n_total), np.empty(n_total)
    for i in range(n_total):
        vt, Re_t = terminal_velocity(diameters[i], rho_arr[i])
        vt_arr[i], n_arr[i] = vt, rz_exponent(Re_t)

    if t_max is None:
        v_hindered = vt_arr * (1.0 - phi_global) ** n_arr
        v_min = max(v_hindered.min(), 1e-9)
        t_max = min(H_COLUMN / v_min * 3.0, 3600.0)

    W_COLUMN = H_COLUMN / COLUMN_ASPECT_RATIO
    x, z, active = rng.uniform(0.0, W_COLUMN, size=n_total), rng.uniform(0.0, H_COLUMN, size=n_total), np.ones(n_total, dtype=bool)
    cell_edges, radii = np.linspace(0.0, H_COLUMN, N_CELLS + 1), diameters / 2.0
    bed_solid_vol, bed_vol_species = 0.0, np.zeros(n_species)

    times = np.arange(0.0, t_max + DT, DT)
    n_steps = len(times)
    settled_frac, settled_frac_by_species = np.zeros(n_steps), np.zeros((n_steps, n_species))
    bed_height, bed_vol_by_species = np.zeros(n_steps), np.zeros((n_steps, n_species))
    phi_field, mu_eff_field = np.zeros((n_steps, N_CELLS)), np.zeros((n_steps, N_CELLS))

    snapshots = []
    _snap_steps = {min(int(round(pct / 100.0 * (n_steps - 1))), n_steps - 1)
                   for pct in (snap_pcts if snap_pcts is not None else range(0, 101, 10))}

    for step in range(n_steps):
        h_bed = bed_solid_vol / PHI_BED
        phi = np.zeros(N_CELLS)
        if active.any():
            ci = np.clip(np.digitize(z[active], cell_edges) - 1, 0, N_CELLS - 1)
            np.add.at(phi, ci, particle_vols[active])
            phi /= H_COLUMN
        phi_field[step] = phi
        mu_eff_cell = krieger_dougherty(phi)
        mu_eff_field[step] = mu_eff_cell

        if active.any():
            act = np.where(active)[0]
            ci_act = np.clip(np.digitize(z[act], cell_edges) - 1, 0, N_CELLS - 1)
            vt_local = np.array([terminal_velocity(diameters[a], rho_arr[a], mu=mu_eff_cell[ci_act[k]])[0]
                                 for k, a in enumerate(act)])
            v_h = vt_local * (1.0 - np.clip(phi[ci_act], 0.0, 0.999)) ** n_arr[act]
            z[act] = np.maximum(z[act] - v_h * DT, h_bed)

            if K_REP > 0.0:
                z += apply_collisions(z, radii, active, h_bed, cell_edges)
                z[act] = np.maximum(z[act], h_bed)

            hit = act[z[act] <= h_bed]
            for i in hit:
                z[i], active[i] = h_bed, False
                bed_solid_vol += particle_vols[i]
                bed_vol_species[species_id[i]] += particle_vols[i]
                h_bed = bed_solid_vol / PHI_BED

        settled_frac[step] = np.sum(~active) / n_total
        for s in range(n_species):
            mask = species_id == s
            settled_frac_by_species[step, s] = np.sum(~active & mask) / mask.sum()
        bed_height[step], bed_vol_by_species[step] = h_bed, bed_vol_species.copy()

        if step in _snap_steps:
            snapshots.append({"step": step, "t": times[step], "pct_time": times[step] / t_max * 100.0,
                              "settled_frac": float(settled_frac[step]), "h_bed": h_bed, "bed_vol_sp": bed_vol_species.copy(),
                              "z": z.copy(), "x": x.copy(), "active": active.copy(), "phi": phi.copy(), "mu_eff": mu_eff_cell.copy()})

    return {"times": times, "settled_frac": settled_frac, "settled_frac_by_species": settled_frac_by_species,
            "bed_height": bed_height, "bed_vol_by_species": bed_vol_by_species, "phi_field": phi_field,
            "mu_eff_field": mu_eff_field, "cell_edges": cell_edges, "diameters": diameters, "radii": radii,
            "vt_arr": vt_arr, "n_arr": n_arr, "species_id": species_id, "species_list": species_list,
            "phi_global": phi_global, "particle_vols": particle_vols, "counts": counts, "snapshots": snapshots,
            "x": x, "W_COLUMN": W_COLUMN}

# =============================================================================
# Plotting & Persistence
# =============================================================================

def plot_results(res: dict) -> None:
    """Generate comprehensive analysis plots."""
    times, sp_list = res["times"] / 60.0, res["species_list"]
    colors, names = [sp.color for sp in sp_list], [sp.name for sp in sp_list]
    cell_centres = 0.5 * (res["cell_edges"][:-1] + res["cell_edges"][1:])

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Multi-Species Particle Settling Analysis", fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.3)

    # 1. Settling Curves
    ax1 = fig.add_subplot(gs[0, 0])
    for s, col in enumerate(colors):
        ax1.plot(times, res["settled_frac_by_species"][:, s] * 100.0, color=col, label=names[s])
    ax1.plot(times, res["settled_frac"] * 100.0, "k--", label="Overall")
    ax1.set_title("Cumulative Settling"); ax1.set_xlabel("Time (min)"); ax1.set_ylabel("Settled (%)"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    # 2. Bed Growth
    ax2 = fig.add_subplot(gs[0, 1]); bottom = np.zeros(len(times))
    h_by_s = res["bed_vol_by_species"] / PHI_BED * 100.0
    for s, col in enumerate(colors):
        ax2.fill_between(times, bottom, bottom + h_by_s[:, s], color=col, alpha=0.6, label=names[s])
        bottom += h_by_s[:, s]
    ax2.set_title("Bed Composition"); ax2.set_xlabel("Time (min)"); ax2.set_ylabel("Height (cm)"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    # 3. Phi Field
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.pcolormesh(times, cell_centres, res["phi_field"].T, cmap="YlOrRd", shading="auto", vmax=PHI_MAX)
    fig.colorbar(im3, ax=ax3, label="phi"); ax3.plot(times, res["bed_height"], "b-", lw=1.5, label="Bed")
    ax3.set_title("Volume Fraction φ(z,t)"); ax3.set_xlabel("Time (min)"); ax3.set_ylabel("Height (m)")

    # 4. Mu Field
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.pcolormesh(times, cell_centres, np.log10(res["mu_eff_field"].T / MU + 1e-6), cmap="plasma")
    fig.colorbar(im4, ax=ax4, label="log10(μ_eff/μ_f)"); ax4.plot(times, res["bed_height"], "c-", lw=1.5, label="Bed")
    ax4.set_title("Effective Viscosity μ_eff(z,t)"); ax4.set_xlabel("Time (min)"); ax4.set_ylabel("Height (m)")

    # 5. PSD
    ax5 = fig.add_subplot(gs[1, 0])
    for s, sp in enumerate(sp_list):
        mask = res["species_id"] == s
        ax5.hist(res["diameters"][mask] * 1e6, bins=25, color=colors[s], alpha=0.6, label=sp.name)
    ax5.set_title("Particle Size Distribution"); ax5.set_xlabel("Diameter (μm)"); ax5.set_ylabel("Count"); ax5.legend(fontsize=6); ax5.grid(True, alpha=0.3)

    # 6. Vt vs D
    ax6 = fig.add_subplot(gs[1, 1])
    for s, sp in enumerate(sp_list):
        mask = res["species_id"] == s
        ax6.scatter(res["diameters"][mask] * 1e6, res["vt_arr"][mask] * 1e3, color=colors[s], alpha=0.5, s=10)
    ax6.set_title("Terminal Velocity vs Size"); ax6.set_xlabel("Diameter (μm)"); ax6.set_ylabel("vt (mm/s)"); ax6.grid(True, alpha=0.3)

    # 7. K-D Curve
    ax7 = fig.add_subplot(gs[1, 2])
    phi_r = np.linspace(0, PHI_MAX * 0.99, 200)
    ax7.semilogy(phi_r, krieger_dougherty(phi_r) / MU, "p-", color="purple", markevery=10)
    ax7.set_title("Krieger-Dougherty Viscosity"); ax7.set_xlabel("φ"); ax7.set_ylabel("μ_eff/μ_f"); ax7.grid(True, alpha=0.3)

    # 8. Bed Pie
    ax8 = fig.add_subplot(gs[1, 3])
    final_vols = res["bed_vol_by_species"][-1]
    if final_vols.sum() > 0:
        ax8.pie(final_vols, labels=[n.split()[0] for n in names], colors=colors, autopct="%1.1f%%", textprops={'fontsize': 8})
    ax8.set_title("Final Bed Composition")

    plt.savefig("settling_results.png", dpi=150, bbox_inches="tight")
    print("Comprehensive plot saved to settling_results.png")

def save_results(res: dict, filepath: str) -> None:
    """Save results to HDF5."""
    with h5py.File(filepath, "w") as f:
        meta = f.create_group("metadata")
        for k in ["H_COLUMN", "N_CELLS", "MU", "RHO_F", "PHI_BED", "PHI_MAX", "ETA_INTRINSIC", "K_REP", "COLUMN_ASPECT_RATIO", "DT"]:
            meta.attrs[k] = getattr(config, k)
        meta.attrs["phi_global"], meta.attrs["W_COLUMN"] = res["phi_global"], res["W_COLUMN"]
        sp_grp = meta.create_group("species")
        for s, sp in enumerate(res["species_list"]):
            sg = sp_grp.create_group(str(s))
            for k in ["name", "rho_p", "phi", "d_mean", "d_sigma", "color"]: sg.attrs[k] = getattr(sp, k)
            for k in ["d10", "d50", "d90"]: sg.attrs[k] = getattr(sp, k) if getattr(sp, k) is not None else float("nan")
        ts, flds, pts, snps = f.create_group("timeseries"), f.create_group("fields"), f.create_group("particles"), f.create_group("snapshots")
        for k in ["times", "settled_frac", "bed_height", "settled_frac_by_species", "bed_vol_by_species"]: ts.create_dataset(k, data=res[k])
        for k in ["phi_field", "mu_eff_field"]: flds.create_dataset(k, data=res[k], compression="gzip")
        flds.create_dataset("cell_edges", data=res["cell_edges"])
        for k in ["diameters", "radii", "vt_arr", "n_arr", "species_id", "particle_vols", "x", "counts"]: pts.create_dataset(k, data=res[k])
        for i, snap in enumerate(res["snapshots"]):
            sg = snps.create_group(f"{i:02d}")
            for k in ["step", "t", "pct_time", "settled_frac", "h_bed"]: sg.attrs[k] = snap[k]
            for k in ["z", "x", "active", "phi", "mu_eff", "bed_vol_sp"]: sg.create_dataset(k, data=snap[k])

def load_results(filepath: str) -> dict:
    """Load results from HDF5."""
    res = {}
    with h5py.File(filepath, "r") as f:
        meta = f["metadata"]
        res["phi_global"], res["W_COLUMN"] = float(meta.attrs["phi_global"]), float(meta.attrs["W_COLUMN"])
        res["species_list"] = [Species(name=str(f[f"metadata/species/{s}"].attrs["name"]), rho_p=float(f[f"metadata/species/{s}"].attrs["rho_p"]),
                                       phi=float(f[f"metadata/species/{s}"].attrs["phi"]), d_mean=float(f[f"metadata/species/{s}"].attrs["d_mean"]),
                                       d_sigma=float(f[f"metadata/species/{s}"].attrs["d_sigma"]), color=str(f[f"metadata/species/{s}"].attrs["color"]),
                                       d10=None if np.isnan(f[f"metadata/species/{s}"].attrs["d10"]) else f[f"metadata/species/{s}"].attrs["d10"],
                                       d50=None if np.isnan(f[f"metadata/species/{s}"].attrs["d50"]) else f[f"metadata/species/{s}"].attrs["d50"],
                                       d90=None if np.isnan(f[f"metadata/species/{s}"].attrs["d90"]) else f[f"metadata/species/{s}"].attrs["d90"])
                               for s in sorted(f["metadata/species"].keys(), key=int)]
        for k in ["times", "settled_frac", "bed_height", "settled_frac_by_species", "bed_vol_by_species"]: res[k] = f["timeseries"][k][:]
        for k in ["phi_field", "mu_eff_field", "cell_edges"]: res[k] = f["fields"][k][:]
        for k in ["diameters", "radii", "vt_arr", "n_arr", "particle_vols", "x", "counts"]: res[k] = f["particles"][k][:]
        res["species_id"] = f["particles/species_id"][:].astype(int)
        res["snapshots"] = [{"step": int(f[f"snapshots/{i}"].attrs["step"]), "t": float(f[f"snapshots/{i}"].attrs["t"]),
                             "pct_time": float(f[f"snapshots/{i}"].attrs["pct_time"]), "settled_frac": float(f[f"snapshots/{i}"].attrs["settled_frac"]),
                             "h_bed": float(f[f"snapshots/{i}"].attrs["h_bed"]), "z": f[f"snapshots/{i}/z"][:], "x": f[f"snapshots/{i}/x"][:],
                             "active": f[f"snapshots/{i}/active"][:].astype(bool), "phi": f[f"snapshots/{i}/phi"][:],
                             "mu_eff": f[f"snapshots/{i}/mu_eff"][:], "bed_vol_sp": f[f"snapshots/{i}/bed_vol_sp"][:]}
                            for i in sorted(f["snapshots"].keys())]
    return res

def print_summary(res: dict) -> None:
    """Print simulation summary."""
    print(f"\n{'MULTI-SPECIES SETTLING SUMMARY':^60}\n" + "="*60)
    print(f"Duration: {res['times'][-1]/60:.1f} min | Total particles: {len(res['diameters'])}")
    print(f"Total phi: {res['phi_global']*100:.2f}% | Final bed: {res['bed_height'][-1]*100:.2f} cm")
    print("-" * 60 + f"\n{'Species':<25} {'rho':>10} {'phi %':>10} {'Settled %':>10}\n" + "-" * 60)
    for s, sp in enumerate(res["species_list"]):
        print(f"{sp.name:<25} {sp.rho_p:>10.0f} {sp.phi*100:>10.1f} {res['settled_frac_by_species'][-1, s]*100:>10.1f}%")
    print("=" * 60)

def plot_2d_snapshots(res: dict, output_dir: str = "outputs") -> None:
    """Generate 2D snapshots with side profiles."""
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    os.makedirs(output_dir, exist_ok=True)
    sp_list, species_id = res["species_list"], res["species_id"]
    particle_rgba = np.array([mcolors.to_rgba(sp.color) for sp in sp_list])[species_id]
    cell_centres = 0.5 * (res["cell_edges"][:-1] + res["cell_edges"][1:])

    for i, snap in enumerate(res["snapshots"]):
        fig = plt.figure(figsize=(10, 8)); gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1], wspace=0.1)
        ax0 = fig.add_subplot(gs[0]) # Column
        ax0.add_patch(mpatches.Rectangle((0, 0), res["W_COLUMN"], snap["h_bed"], color="grey", alpha=0.3))
        ax0.scatter(snap["x"][snap["active"]], snap["z"][snap["active"]], s=2, c=particle_rgba[snap["active"]], alpha=0.7)
        ax0.set_xlim(0, res["W_COLUMN"]); ax0.set_ylim(0, H_COLUMN); ax0.set_title(f"Column (t={snap['t']/60:.1f} min)")

        ax1 = fig.add_subplot(gs[1]) # Phi
        ax1.barh(cell_centres, snap["phi"], height=H_COLUMN/N_CELLS, color="orange", alpha=0.6)
        ax1.set_xlim(0, PHI_MAX); ax1.set_title("φ profile"); ax1.set_yticklabels([])

        ax2 = fig.add_subplot(gs[2]) # Mu
        ax2.barh(cell_centres, np.log10(snap["mu_eff"]/MU + 1e-6), height=H_COLUMN/N_CELLS, color="purple", alpha=0.6)
        ax2.set_title("log10(μ_eff/μ_f)"); ax2.set_yticklabels([])

        plt.savefig(f"{output_dir}/snap_{i:02d}.png"); plt.close(fig)

if __name__ == "__main__":
    if config.MODE == "run":
        from config import N_SNAPS_LOG, SNAP_INTERVAL, LOG_SCALE
        if LOG_SCALE:
            pcts = np.clip(np.logspace(0, np.log10(101), N_SNAPS_LOG) - 1, 0, 100)
        else:
            pcts = np.arange(0, 100 + SNAP_INTERVAL, SNAP_INTERVAL)
        res = run_simulation(snap_pcts=sorted(set(pcts)))
        save_results(res, config.RESULTS_FILE)
    else:
        res = load_results(config.RESULTS_FILE)
    print_summary(res); plot_results(res); plot_2d_snapshots(res)
