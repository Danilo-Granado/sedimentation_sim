"""
Particle Settling Simulation
=============================
Multi-species · Richardson-Zaki hindered settling · Sediment bed tracking
· Concentrated-suspension / paste-transition physics
--------------------------------------------------------------------------
Each solid species is defined by:
  - name          : label used in plots and console output
  - rho_p         : particle density  (kg/m³)
  - phi           : initial bulk volume fraction in the suspension
  - d_mean        : mean diameter of the log-normal size distribution  (m)
  - d_sigma       : log-normal shape parameter  (dimensionless)
  - color         : matplotlib colour string for plots

Physics included
  - Stokes / Schiller-Naumann drag → per-particle terminal velocity
  - Garside & Al-Dibouni (1977) Richardson-Zaki exponent
  - Hindered settling:  v_h = v_t(μ_eff) · (1 − φ)^n
    (φ is the combined volume fraction of ALL species in a cell)
  - Krieger-Dougherty effective viscosity:
      μ_eff(φ) = μ_f · (1 − φ/φ_max)^(−[η]·φ_max)
    Captures lubrication forces and the paste transition as φ → φ_max.
    μ_eff feeds back into both the drag force and the RZ velocity,
    so settling slows sharply once φ exceeds ~0.3.
  - Soft-sphere (DEM) collision repulsion via cell-list neighbour search:
    overlapping particles receive a pairwise Hertzian repulsive velocity
    impulse that prevents unphysical overlap at high concentrations.
  - Sediment bed: particles accumulate at the bottom with packing
    fraction PHI_BED; the rising bed surface is the lower boundary
    for all suspended particles
  - Per-species settled-fraction curves and bed-composition tracking
  - μ_eff(z,t) field stored and plotted alongside φ(z,t)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

# ── Physical constants ────────────────────────────────────────────────────────
G = 9.81        # gravitational acceleration,  m/s²

# ── Fluid properties ─────────────────────────────────────────────────────────
RHO_F = 1100.0  # fluid density,               kg/m³  (water)
MU    = 20e-3  # dynamic viscosity,            Pa·s   (water at ~20 °C)

# ── Simulation settings ───────────────────────────────────────────────────────
N_PARTICLES_TOTAL = 5000   # total particles across all species
H_COLUMN          = 1.0   # column height,            m
DT                = 0.5   # time step,                s
N_CELLS           = 100    # vertical cells for φ field

# ── Column geometry ───────────────────────────────────────────────────────────
# H / W ratio of the physical column.  Controls x-extent in 2D visualisation.
# 8 gives good readability while still reflecting a tall narrow column.
COLUMN_ASPECT_RATIO = 2   # H_COLUMN / W_COLUMN

# ── Bed packing ───────────────────────────────────────────────────────────────
PHI_BED = 0.60   # solid volume fraction inside the settled bed

# ── Concentrated-suspension / paste physics ───────────────────────────────────
# Krieger-Dougherty parameters
PHI_MAX    = 0.64   # random close-packing limit  (φ → φ_max ⟹ μ_eff → ∞)
ETA_INTRINSIC = 2.5 # Einstein intrinsic viscosity  [dimensionless]

# Soft-sphere DEM collision parameters
# Repulsive velocity impulse applied when two particles overlap:
#   Δv = K_REP · overlap / DT  (split equally between the pair)
# K_REP ~ O(1): 1.0 gives one-step correction for a full-diameter overlap.
K_REP = 0.5        # repulsion strength  (dimensionless, 0 = off, 1 = stiff)

# ── Diameter clamps (applied to every species) ───────────────────────────────
D_MIN = 1e-6    # m
D_MAX = 1e-3    # m


# =============================================================================
# Species definition
# =============================================================================

@dataclass
class Species:
    """
    One solid component in the suspension.

    Parameters
    ----------
    name    : display name  (e.g. "ZnO")
    rho_p   : particle density,              kg/m3
    phi     : initial bulk volume fraction   (0 < phi < PHI_BED)
    d_mean  : mean diameter (log-normal μ),  m        (default 100 µm)
              Ignored if d50 is provided.
    d_sigma : log-normal shape parameter               (default 0.4)
              Ignored if d50 and d90 (or d10) are provided.
    d10     : 10th percentile diameter,      m  (optional, for fitting σ)
    d50     : median diameter,               m  (optional, overrides d_mean)
    d90     : 90th percentile diameter,      m  (optional, for fitting σ)
    color   : matplotlib colour for this species

    Size distribution fitting
    -------------------------
    For a log-normal distribution:
        D50 = exp(μ_ln)            →  μ_ln  = ln(D50)
        D90 = exp(μ_ln + 1.282·σ) →  σ     = ln(D90/D50) / 1.282
        D10 = exp(μ_ln − 1.282·σ) →  σ     = ln(D50/D10) / 1.282

    If both d10 and d90 are given, σ is averaged from both estimates.
    If only d90 is given, σ is derived from D90/D50.
    If only d10 is given, σ is derived from D50/D10.
    """
    name   : str
    rho_p  : float
    phi    : float
    d_mean : float = 100e-6
    d_sigma: float = 0.4
    d10    : float | None = None
    d50    : float | None = None
    d90    : float | None = None
    color  : str   = "steelblue"

    def lognormal_params(self) -> tuple[float, float]:
        """
        Return (mu_ln, sigma_ln) for np.random.lognormal().

        Priority:
          1. d50 / d90 / d10  (fit from percentile data)
          2. d_mean / d_sigma (direct parameters)
        """
        if self.d50 is not None:
            mu_ln = np.log(self.d50)
            sigmas = []
            if self.d90 is not None:
                sigmas.append(np.log(self.d90 / self.d50) / 1.2816)
            if self.d10 is not None:
                sigmas.append(np.log(self.d50 / self.d10) / 1.2816)
            sigma_ln = float(np.mean(sigmas)) if sigmas else self.d_sigma
            return mu_ln, sigma_ln
        return np.log(self.d_mean), self.d_sigma


# ── Default mixture: three metal oxides ──────────────────────────────────────
DEFAULT_SPECIES = [
    Species("Fe2O3 (Hematite)",  rho_p=5260, phi=0.04, d_mean=80e-6,  color="firebrick"),
    Species("Al2O3 (Alumina)",   rho_p=3950, phi=0.03, d_mean=120e-6, color="steelblue"),
    Species("SiO2 (Silica)",     rho_p=2650, phi=0.03, d_mean=150e-6, color="goldenrod"),
]


# =============================================================================
# Physics helpers
# =============================================================================

def drag_coefficient(Re: float) -> float:
    """Schiller-Naumann drag coefficient for a sphere."""
    if Re < 1e-9:
        return 0.0
    elif Re < 1000.0:
        return (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)
    else:
        return 0.44


def terminal_velocity(d: float,
                      rho_p: float,
                      rho_f: float = RHO_F,
                      mu: float    = MU,
                      tol: float   = 1e-10,
                      max_iter: int = 2000) -> tuple[float, float]:
    """
    Iteratively solve for single-particle terminal velocity and Re_t.

    Returns
    -------
    vt  : terminal velocity,  m/s
    Re  : particle Reynolds number at terminal velocity
    """
    r  = d / 2.0
    vt = (2.0 * r**2 * (rho_p - rho_f) * G) / (9.0 * mu)  # Stokes guess
    for _ in range(max_iter):
        Re    = rho_f * vt * d / mu
        Cd    = drag_coefficient(Re)
        if Cd == 0.0:
            break
        vt_new = np.sqrt(4.0 / 3.0 * (rho_p - rho_f) / rho_f * G * d / Cd)
        if abs(vt_new - vt) < tol:
            return vt_new, rho_f * vt_new * d / mu
        vt = vt_new
    return vt, rho_f * vt * d / mu


def rz_exponent(Re_t: float) -> float:
    """
    Richardson-Zaki exponent n  (Garside & Al-Dibouni 1977).
    n -> 4.65 (Stokes),  n -> 2.39 (Newton).
    """
    if Re_t < 0.2:
        return 4.65
    elif Re_t < 1.0:
        return 4.35 + 17.5 * Re_t ** (-0.03) / (1.0 + 0.175 * Re_t ** 0.75)
    elif Re_t < 500.0:
        return 4.45 * Re_t ** (-0.1)
    else:
        return 2.39


def krieger_dougherty(phi: np.ndarray,
                      mu_f: float = MU,
                      phi_max: float = PHI_MAX,
                      eta: float = ETA_INTRINSIC) -> np.ndarray:
    """
    Krieger-Dougherty effective viscosity of a concentrated suspension.

        μ_eff(φ) = μ_f · (1 − φ / φ_max)^(−η · φ_max)

    Physical meaning
    ----------------
    At low φ this recovers Einstein's linear result (μ_eff ≈ μ_f·(1+2.5φ)).
    As φ → φ_max (random close packing ≈ 0.64) the viscosity diverges,
    representing the transition from a suspension to a paste/gel where
    lubrication films between particles dominate and bulk flow ceases.

    Parameters
    ----------
    phi     : local volume fraction, scalar or array  (clipped to [0, φ_max)
    mu_f    : pure-fluid dynamic viscosity,  Pa·s
    phi_max : maximum packing fraction (divergence point)
    eta     : intrinsic viscosity  (2.5 for hard spheres)

    Returns
    -------
    mu_eff : effective suspension viscosity, same shape as phi,  Pa·s
    """
    phi_safe = np.clip(np.asarray(phi, dtype=float), 0.0, phi_max * 0.9999)
    return mu_f * (1.0 - phi_safe / phi_max) ** (-eta * phi_max)


def apply_collisions(z: np.ndarray,
                     r: np.ndarray,
                     active: np.ndarray,
                     h_bed: float,
                     cell_edges: np.ndarray) -> np.ndarray:
    """
    Soft-sphere DEM collision correction (1-D, vertical only).

    For each pair of active particles that overlap (|z_i - z_j| < r_i + r_j),
    compute a repulsive displacement proportional to the overlap depth
    (Hertz-like: F ∝ δ, split equally) and add it to a correction array.

    Uses a cell-list for O(N) neighbour search: only particles in the same
    cell or adjacent cells are tested as candidates.

    Parameters
    ----------
    z          : particle positions (m), all particles
    r          : particle radii     (m), all particles
    active     : boolean mask of unsettled particles
    h_bed      : current bed height (m) — settled particles are excluded
    cell_edges : grid cell boundaries for the cell-list

    Returns
    -------
    dz : position correction array (m), same length as z.
         Add this to z after the call.
    """
    dz        = np.zeros_like(z)
    act_idx   = np.where(active)[0]
    if len(act_idx) < 2:
        return dz

    n_cells   = len(cell_edges) - 1
    cell_h    = cell_edges[1] - cell_edges[0]

    # Build cell list: map each active particle to its cell
    ci = np.clip(np.digitize(z[act_idx], cell_edges) - 1, 0, n_cells - 1)

    # Group particle indices by cell
    cell_map: dict[int, list[int]] = {}
    for local, i in enumerate(act_idx):
        c = ci[local]
        cell_map.setdefault(c, []).append(i)

    # Check pairs within same cell and adjacent cells only
    checked = set()
    for c, members in cell_map.items():
        candidates = members[:]
        if c + 1 in cell_map:
            candidates += cell_map[c + 1]
        for a in range(len(members)):
            i = members[a]
            for j in candidates:
                if j <= i:
                    continue
                pair = (i, j)
                if pair in checked:
                    continue
                checked.add(pair)

                sep    = abs(z[i] - z[j])
                min_d  = r[i] + r[j]         # sum of radii = contact distance
                if sep < min_d:
                    overlap = min_d - sep
                    # Repulsive displacement: push apart along z
                    # Upper particle (larger z) moves up, lower moves down
                    delta = K_REP * overlap * 0.5
                    if z[i] > z[j]:
                        dz[i] += delta
                        dz[j] -= delta
                    else:
                        dz[i] -= delta
                        dz[j] += delta

    # Prevent pushing particles below bed
    for i in act_idx:
        if z[i] + dz[i] < h_bed:
            dz[i] = h_bed - z[i]

    return dz




def run_simulation(species_list: list[Species] | None = None,
                   seed: int = 42,
                   t_max: float | None = None,
                   snap_pcts: list[int] | None = None) -> dict:
    """
    Run the multi-species settling simulation.

    Parameters
    ----------
    species_list : list of Species
        Each entry defines one solid component.  Their phi values set the
        per-species volume fractions; the total must be < PHI_BED.
    seed : int
        RNG seed for reproducibility.
    t_max : float or None
        Simulation duration in seconds.  None -> adaptive (capped at 1 h).

    Returns
    -------
    dict with keys:
        times, settled_frac, settled_frac_by_species,
        bed_height, bed_vol_by_species,
        phi_field, cell_edges,
        diameters, vt_arr, n_arr, species_id,
        species_list, phi_global, counts
    """
    if species_list is None:
        species_list = DEFAULT_SPECIES

    phi_global = sum(sp.phi for sp in species_list)
    if not (0 < phi_global < PHI_BED):
        raise ValueError(
            f"Total phi = {phi_global:.4f} must be in (0, PHI_BED={PHI_BED}). "
            "Reduce the per-species phi values.")

    n_species = len(species_list)
    rng       = np.random.default_rng(seed)

    # ── Allocate particles proportional to each species' phi ─────────────
    raw_counts = np.array([sp.phi / phi_global * N_PARTICLES_TOTAL
                           for sp in species_list])
    counts     = np.round(raw_counts).astype(int)
    diff       = N_PARTICLES_TOTAL - counts.sum()
    counts[np.argmax(raw_counts - counts)] += diff
    n_total = counts.sum()

    # ── Generate particle arrays ──────────────────────────────────────────
    diameters  = np.empty(n_total)
    rho_arr    = np.empty(n_total)
    species_id = np.empty(n_total, dtype=int)

    idx = 0
    for s, (sp, cnt) in enumerate(zip(species_list, counts)):
        mu_ln, sigma_ln = sp.lognormal_params()   # honours d50/d90/d10 if set
        d = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=cnt)
        d = np.clip(d, D_MIN, D_MAX)
        diameters [idx:idx+cnt] = d
        rho_arr   [idx:idx+cnt] = sp.rho_p
        species_id[idx:idx+cnt] = s
        idx += cnt

    # ── Particle volumes, scaled per species to match their phi ───────────
    raw_vols      = (np.pi / 6.0) * diameters ** 3
    particle_vols = np.empty(n_total)
    for s, sp in enumerate(species_list):
        mask      = species_id == s
        scale     = sp.phi * H_COLUMN / raw_vols[mask].sum()
        particle_vols[mask] = raw_vols[mask] * scale

    # ── Terminal velocity and RZ exponent per particle ────────────────────
    # FIX (bug 4): pass rho_f and mu explicitly so that any runtime change
    # to the module-level RHO_F / MU constants is honoured correctly.
    vt_arr = np.empty(n_total)
    n_arr  = np.empty(n_total)
    for i in range(n_total):
        vt, Re_t  = terminal_velocity(diameters[i], rho_arr[i],
                                      rho_f=RHO_F, mu=MU)
        vt_arr[i] = vt
        n_arr [i] = rz_exponent(Re_t)

    # ── Adaptive t_max ────────────────────────────────────────────────────
    # FIX (bug 2): pair vt and n per-particle before taking the minimum.
    # Using vt.min() with n.max() (unpaired) can wildly overestimate the
    # worst-case settling time when the slowest and most-hindered particles
    # are not the same particle.
    if t_max is None:
        v_hindered = vt_arr * (1.0 - phi_global) ** n_arr   # per-particle
        v_min      = max(v_hindered.min(), 1e-9)
        t_max      = min(H_COLUMN / v_min * 3.0, 3600.0)

    # ── Initial positions ─────────────────────────────────────────────────
    # x is assigned once and never updated (pseudo-2D visualisation layer).
    # W_COLUMN controls the visual width; see COLUMN_ASPECT_RATIO constant.
    W_COLUMN = H_COLUMN / COLUMN_ASPECT_RATIO
    x      = rng.uniform(0.0, W_COLUMN, size=n_total)
    z      = rng.uniform(0.0, H_COLUMN, size=n_total)
    active = np.ones(n_total, dtype=bool)

    # ── Grid ─────────────────────────────────────────────────────────────
    cell_edges  = np.linspace(0.0, H_COLUMN, N_CELLS + 1)
    cell_height = H_COLUMN / N_CELLS

    # ── Particle radii (true physical, for collision detection) ───────────
    radii = diameters / 2.0

    # ── Bed state ─────────────────────────────────────────────────────────
    bed_solid_vol   = 0.0
    bed_vol_species = np.zeros(n_species)

    # ── Snapshot bookkeeping ──────────────────────────────────────────────
    # Snapshots are collected at the step indices closest to the requested
    # percentage marks.  Stored compactly: only z, active mask, h_bed, phi.
    snapshots: list[dict] = []
    _snap_steps: set[int] = set()   # populated below after times is built

    # ── Time loop ─────────────────────────────────────────────────────────
    times                   = np.arange(0.0, t_max + DT, DT)
    n_steps                 = len(times)
    settled_frac            = np.zeros(n_steps)
    settled_frac_by_species = np.zeros((n_steps, n_species))
    bed_height              = np.zeros(n_steps)
    bed_vol_by_species      = np.zeros((n_steps, n_species))
    phi_field               = np.zeros((n_steps, N_CELLS))
    mu_eff_field            = np.zeros((n_steps, N_CELLS))   # ← new

    # Resolve snapshot step indices now that times array is known.
    # snap_pcts is passed in via the closure; default to every 10%.
    _pcts = list(snap_pcts) if snap_pcts is not None else list(range(0, 101, 10))
    for pct in _pcts:
        idx = int(round(pct / 100.0 * (n_steps - 1)))
        _snap_steps.add(min(idx, n_steps - 1))

    for step in range(n_steps):
        h_bed = bed_solid_vol / PHI_BED

        # φ field — all species combined (RZ uses total local concentration)
        # FIX (bug 1): normalise by H_COLUMN, not cell_height.
        #
        # particle_vols[i] was scaled so that sum(particle_vols) = phi * H_COLUMN,
        # i.e. each particle_vols[i] carries units of [m³ per m² cross-section = m].
        # The true local volume fraction in a cell is:
        #
        #   phi_cell = (solid volume in cell [m]) / H_COLUMN
        #            = sum(particle_vols[i] in cell) / H_COLUMN
        #
        # Dividing by cell_height instead multiplied phi by N_CELLS,
        # causing massive over-estimation of hindrance at high N_CELLS.
        phi = np.zeros(N_CELLS)
        if active.any():
            az = z[active]
            av = particle_vols[active]
            ci = np.clip(np.digitize(az, cell_edges) - 1, 0, N_CELLS - 1)
            np.add.at(phi, ci, av)
            phi /= H_COLUMN          # ← was: phi /= cell_height
        phi_field[step] = phi

        # ── Krieger-Dougherty effective viscosity field ────────────────────
        # μ_eff(z) rises steeply as φ → φ_max, capturing lubrication forces
        # and the paste transition.  This feeds into the hindered velocity
        # via terminal_velocity(mu=μ_eff), so settling slows dramatically
        # in dense regions without requiring explicit pairwise fluid solving.
        mu_eff_cell          = krieger_dougherty(phi, mu_f=MU)
        mu_eff_field[step]   = mu_eff_cell

        # Move all active particles (vectorised)
        if active.any():
            act     = np.where(active)[0]
            ci_act  = np.clip(np.digitize(z[act], cell_edges) - 1, 0, N_CELLS - 1)
            phi_loc    = phi[ci_act]
            mu_eff_loc = mu_eff_cell[ci_act]   # local effective viscosity

            # Re-compute terminal velocity at local μ_eff for each particle.
            # This couples the paste viscosity directly into the drag force:
            # as μ_eff → ∞, vt → 0 and particles freeze in place.
            vt_local = np.array([
                terminal_velocity(diameters[act[k]], rho_arr[act[k]],
                                  rho_f=RHO_F, mu=mu_eff_loc[k])[0]
                for k in range(len(act))
            ])

            # Richardson-Zaki collective hindrance on top of μ_eff correction
            v_h = vt_local * (1.0 - np.clip(phi_loc, 0.0, 0.999)) ** n_arr[act]
            z[act] -= v_h * DT

            # FIX (bug 3): clamp positions to h_bed before checking for
            # settling. Fast particles can overshoot the bed by several cell
            # widths in one timestep; without clamping they end up below the
            # bed surface and are never detected by the z <= h_bed test,
            # causing them to "skip through" the bed and remain unsettled.
            z[act] = np.maximum(z[act], h_bed)

            # ── Soft-sphere DEM collision correction ──────────────────────
            # Pairwise repulsion pushes overlapping particles apart.
            # Cell-list search keeps this O(N) rather than O(N²).
            # This prevents unphysical stacking that would otherwise occur
            # in dense regions where multiple particles occupy the same cell.
            if K_REP > 0.0:
                dz = apply_collisions(z, radii, active, h_bed, cell_edges)
                z += dz
                # Re-clamp after collision correction
                z[act] = np.maximum(z[act], h_bed)

            # Settle particles that reached the bed surface
            hit = act[z[act] <= h_bed]
            for i in hit:
                z[i]                    = h_bed
                active[i]               = False
                s                       = species_id[i]
                bed_solid_vol          += particle_vols[i]
                bed_vol_species[s]     += particle_vols[i]
                h_bed                   = bed_solid_vol / PHI_BED

        settled_frac[step] = np.sum(~active) / n_total
        for s in range(n_species):
            mask = species_id == s
            settled_frac_by_species[step, s] = np.sum(~active & mask) / mask.sum()
        bed_height[step]         = h_bed
        bed_vol_by_species[step] = bed_vol_species.copy()

        # Capture snapshot if this step was requested
        if step in _snap_steps:
            snapshots.append({
                "step":          step,
                "t":             times[step],
                "pct_time":      times[step] / t_max * 100.0,
                "settled_frac":  float(settled_frac[step]),
                "h_bed":         h_bed,
                "bed_vol_sp":    bed_vol_species.copy(),
                "z":             z.copy(),
                "x":             x.copy(),
                "active":        active.copy(),
                "phi":           phi_field[step].copy(),
                "mu_eff":        mu_eff_field[step].copy(),
            })

    return {
        "times":                   times,
        "settled_frac":            settled_frac,
        "settled_frac_by_species": settled_frac_by_species,
        "bed_height":              bed_height,
        "bed_vol_by_species":      bed_vol_by_species,
        "phi_field":               phi_field,
        "mu_eff_field":            mu_eff_field,
        "cell_edges":              cell_edges,
        "diameters":               diameters,
        "radii":                   radii,
        "vt_arr":                  vt_arr,
        "n_arr":                   n_arr,
        "species_id":              species_id,
        "species_list":            species_list,
        "phi_global":              phi_global,
        "particle_vols":           particle_vols,
        "counts":                  counts,
        "snapshots":               snapshots,          # ← new
        "x":                       x,                  # ← new
        "W_COLUMN":                W_COLUMN,           # ← new
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_results(res: dict) -> None:
    times      = res["times"] / 60.0
    sp_list    = res["species_list"]
    colors     = [sp.color for sp in sp_list]
    names      = [sp.name  for sp in sp_list]

    cell_edges   = res["cell_edges"]
    cell_centres = 0.5 * (cell_edges[:-1] + cell_edges[1:])

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "Multi-Species Particle Settling  —  RZ + Krieger-Dougherty + DEM Collisions",
        fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.44, wspace=0.38)

    # ── 1. Per-species cumulative settling curves ─────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sfbs = res["settled_frac_by_species"]
    for s, (name, col) in enumerate(zip(names, colors)):
        ax1.plot(times, sfbs[:, s] * 100.0, color=col, linewidth=2, label=name)
    ax1.plot(times, res["settled_frac"] * 100.0,
             color="black", linewidth=1.5, linestyle="--", label="Overall")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Settled (%)")
    ax1.set_title("Cumulative Settling — by Species")
    ax1.set_xlim(left=0)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.4)

    # ── 2. Bed height stacked by species ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bvbs   = res["bed_vol_by_species"]
    h_by_s = bvbs / PHI_BED * 100.0          # cm per species

    bottom = np.zeros(len(times))
    for s, (name, col) in enumerate(zip(names, colors)):
        ax2.fill_between(times, bottom, bottom + h_by_s[:, s],
                         color=col, alpha=0.65, label=name)
        bottom += h_by_s[:, s]

    h_max_cm = res["phi_global"] * H_COLUMN / PHI_BED * 100.0
    ax2.axhline(h_max_cm, color="black", linestyle="--", linewidth=1.0,
                label=f"Max ({h_max_cm:.1f} cm)")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Bed height (cm)")
    ax2.set_title("Bed Growth — Species Composition")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.4)

    # ── 3. φ(z,t) heatmap with bed surface ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    n_snap  = min(200, res["phi_field"].shape[0])
    step_s  = max(1, res["phi_field"].shape[0] // n_snap)
    phi_sub = res["phi_field"][::step_s].T
    t_sub   = times[::step_s]
    bed_sub = res["bed_height"][::step_s]

    im3 = ax3.pcolormesh(t_sub, cell_centres, phi_sub,
                         cmap="YlOrRd", shading="auto", vmin=0, vmax=PHI_MAX)
    fig.colorbar(im3, ax=ax3, label="Volume fraction φ")
    ax3.plot(t_sub, bed_sub, color="royalblue", linewidth=1.5, label="Bed surface")
    ax3.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax3.set_xlabel("Time (min)")
    ax3.set_ylabel("Height (m)")
    ax3.set_title("φ(z,t) Field + Bed Surface")
    ax3.legend(fontsize=7, loc="upper right")

    # ── 4. μ_eff(z,t) heatmap — paste transition visible ─────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    mu_sub = res["mu_eff_field"][::step_s].T / MU   # normalised: μ_eff / μ_f
    im4 = ax4.pcolormesh(t_sub, cell_centres, np.log10(mu_sub + 1e-6),
                         cmap="plasma", shading="auto")
    cbar4 = fig.colorbar(im4, ax=ax4, label="log₁₀(μ_eff / μ_f)")
    ax4.plot(t_sub, bed_sub, color="cyan", linewidth=1.5, label="Bed surface")
    ax4.set_xlabel("Time (min)")
    ax4.set_ylabel("Height (m)")
    ax4.set_title("Effective Viscosity μ_eff(z,t)\n(Krieger-Dougherty)")
    ax4.legend(fontsize=7, loc="upper right")

    # ── 5. Particle size distributions ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 0])
    for s, (sp, col) in enumerate(zip(sp_list, colors)):
        mask = res["species_id"] == s
        ax5.hist(res["diameters"][mask] * 1e6, bins=25,
                 color=col, alpha=0.65, edgecolor="white",
                 linewidth=0.4, label=sp.name)
        # Overlay D50 and D90 lines using the effective distribution params
        mu_ln, sigma_ln = sp.lognormal_params()
        d50_um = np.exp(mu_ln) * 1e6
        d90_um = np.exp(mu_ln + 1.2816 * sigma_ln) * 1e6
        ax5.axvline(d50_um, color=col, linewidth=1.4,
                    linestyle="--", alpha=0.9,
                    label=f"{sp.name.split()[0]} D50={d50_um:.1f}µm")
        ax5.axvline(d90_um, color=col, linewidth=1.0,
                    linestyle=":", alpha=0.8,
                    label=f"{sp.name.split()[0]} D90={d90_um:.1f}µm")
    ax5.set_xlabel("Diameter (µm)")
    ax5.set_ylabel("Count")
    ax5.set_title("Particle Size Distributions\n(-- D50, ··· D90)")
    ax5.legend(fontsize=6)
    ax5.grid(True, alpha=0.4)

    # ── 6. Terminal velocity vs diameter (free-settling) ─────────────────
    ax6 = fig.add_subplot(gs[1, 1])
    for s, (sp, col) in enumerate(zip(sp_list, colors)):
        mask = res["species_id"] == s
        ax6.scatter(res["diameters"][mask] * 1e6,
                    res["vt_arr"][mask] * 1e3,
                    color=col, alpha=0.5, s=12, label=sp.name)
    ax6.set_xlabel("Diameter (µm)")
    ax6.set_ylabel("Free-settling vt (mm/s)")
    ax6.set_title("Size-Velocity by Species")
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.4)

    # ── 7. Krieger-Dougherty curve (analytical) ───────────────────────────
    ax7 = fig.add_subplot(gs[1, 2])
    phi_range = np.linspace(0, PHI_MAX * 0.99, 400)
    mu_kd     = krieger_dougherty(phi_range, mu_f=MU)
    ax7.semilogy(phi_range, mu_kd / MU, color="purple", linewidth=2)
    ax7.axvline(0.30, color="orange", linestyle="--", linewidth=1,
                label="φ = 0.30 (onset)")
    ax7.axvline(PHI_MAX, color="firebrick", linestyle=":", linewidth=1,
                label=f"φ_max = {PHI_MAX} (RCP)")
    ax7.set_xlabel("Volume fraction φ")
    ax7.set_ylabel("μ_eff / μ_f  (log scale)")
    ax7.set_title("Krieger-Dougherty\nViscosity Divergence")
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.4, which="both")

    # ── 8. Final bed composition pie ──────────────────────────────────────
    ax8 = fig.add_subplot(gs[1, 3])
    final_vols = res["bed_vol_by_species"][-1]
    total_vol  = final_vols.sum()

    if total_vol > 0:
        fracs = final_vols / total_vol
        wedges, texts, autotexts = ax8.pie(
            fracs, labels=names, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 7})
        for at in autotexts:
            at.set_fontsize(7)

    ax8_in = ax8.inset_axes([0.68, 0.0, 0.32, 0.45])
    phis = [sp.phi * 100.0 for sp in sp_list]
    ax8_in.barh(range(len(sp_list)), phis, color=colors, alpha=0.8, height=0.6)
    ax8_in.set_yticks(range(len(sp_list)))
    ax8_in.set_yticklabels([sp.name.split()[0] for sp in sp_list], fontsize=6)
    ax8_in.set_xlabel("φ (%)", fontsize=6)
    ax8_in.tick_params(labelsize=6)
    ax8_in.set_title("Init. φ", fontsize=6)
    ax8.set_title("Final Bed Composition")

    plt.savefig("/outputs/settling_results.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to settling_results.png")


# =============================================================================
# Summary
# =============================================================================

def print_summary(res: dict) -> None:
    sp_list    = res["species_list"]
    times      = res["times"]
    t_min      = times / 60.0
    sf         = res["settled_frac"]
    sfbs       = res["settled_frac_by_species"]
    bed_height = res["bed_height"]
    phi_global = res["phi_global"]
    counts     = res["counts"]

    h_bed_max = phi_global * H_COLUMN / PHI_BED

    # Peak effective viscosity reached anywhere in the column
    mu_eff_peak = res["mu_eff_field"].max()
    phi_peak    = res["phi_field"].max()

    def time_to(arr, pct):
        idx = np.searchsorted(arr, pct / 100.0)
        return f"{t_min[idx]:.1f}" if idx < len(times) else ">sim"

    W = 72
    print("\n" + "=" * W)
    print("  MULTI-SPECIES SETTLING SIMULATION SUMMARY")
    print("=" * W)
    print(f"  Total particles          : {counts.sum()}")
    print(f"  Column height            : {H_COLUMN:.2f} m")
    print(f"  Simulation duration      : {times[-1]/60:.1f} min")
    print(f"  Time step                : {DT:.1f} s")
    print(f"  Total phi (global)       : {phi_global*100:.2f} %")
    print(f"  Bed packing fraction     : {PHI_BED:.2f}")
    print(f"  Theoretical max bed ht.  : {h_bed_max*100:.2f} cm")
    print(f"  Final bed height         : {bed_height[-1]*100:.2f} cm")
    print(f"  Peak φ reached           : {phi_peak:.4f}  "
          f"({'paste regime' if phi_peak > 0.30 else 'dilute/hindered'})")
    print(f"  Peak μ_eff reached       : {mu_eff_peak*1000:.3f} mPa·s  "
          f"({mu_eff_peak/MU:.1f}× μ_fluid)")
    print("-" * W)
    header = (f"  {'Species':<22} {'rho':>9} {'phi%':>5} "
              f"{'N':>5} {'t50':>8} {'t90':>8} {'Final%':>7}")
    print(header)
    print(f"  {'':22} {'(kg/m3)':>9} {'':5} {'':5} "
          f"{'(min)':>8} {'(min)':>8} {'settled':>7}")
    print("-" * W)
    for s, sp in enumerate(sp_list):
        print(f"  {sp.name:<22} {sp.rho_p:>9.0f} {sp.phi*100:>5.1f} "
              f"{counts[s]:>5} "
              f"{time_to(sfbs[:, s], 50):>8} "
              f"{time_to(sfbs[:, s], 90):>8} "
              f"{sfbs[-1, s]*100:>6.1f}%")
    print("-" * W)
    print(f"  {'Overall':<22} {'—':>9} {phi_global*100:>5.1f} "
          f"{counts.sum():>5} "
          f"{time_to(sf, 50):>8} "
          f"{time_to(sf, 90):>8} "
          f"{sf[-1]*100:>6.1f}%")
    print("=" * W)

    print("\n  Final bed composition (by volume):")
    final_vols = res["bed_vol_by_species"][-1]
    total_vol  = final_vols.sum()
    for s, sp in enumerate(sp_list):
        pct = final_vols[s] / total_vol * 100.0 if total_vol > 0 else 0.0
        bar = "=" * int(pct / 2)
        print(f"    {sp.name:<22}  {pct:5.1f}%  {bar}")
    print()




# =============================================================================
# HDF5 persistence — save / load results
# =============================================================================

# HDF5 file layout
# ─────────────────
# /metadata/
#     attrs: sim_constants  (H_COLUMN, N_CELLS, MU, RHO_F, PHI_BED, PHI_MAX,
#                            ETA_INTRINSIC, K_REP, COLUMN_ASPECT_RATIO, DT)
#     /species/
#         /<s>/  (one group per species, named "0", "1", …)
#             attrs: name, rho_p, phi, d_mean, d_sigma, color
#             attrs: d10, d50, d90  (stored as NaN when not set)
# /timeseries/
#     times                    (n_steps,)
#     settled_frac             (n_steps,)
#     bed_height               (n_steps,)
#     settled_frac_by_species  (n_steps, n_species)
#     bed_vol_by_species       (n_steps, n_species)
# /fields/
#     phi_field                (n_steps, N_CELLS)
#     mu_eff_field             (n_steps, N_CELLS)
#     cell_edges               (N_CELLS + 1,)
# /particles/
#     diameters                (n_total,)
#     radii                    (n_total,)
#     vt_arr                   (n_total,)
#     n_arr                    (n_total,)
#     species_id               (n_total,)
#     particle_vols            (n_total,)
#     x                        (n_total,)
#     counts                   (n_species,)
# /snapshots/
#     /<i>/  (one group per snapshot, zero-padded: "00", "01", …)
#         attrs: step, t, pct_time, settled_frac, h_bed
#         z, x, active, phi, mu_eff
#         bed_vol_sp            (n_species,)


def save_results(res: dict, filepath: str) -> None:
    """
    Persist the full results dict produced by run_simulation() to an HDF5 file.

    The saved file is self-contained: it includes all simulation constants and
    species definitions needed to reconstruct plots without re-running the
    simulation.  load_results() returns a dict that is a drop-in replacement
    for the live run_simulation() output.

    Parameters
    ----------
    res      : dict returned by run_simulation()
    filepath : path to write, e.g. "settling_results.h5"
               Existing file is overwritten.
    """
    sp_list   = res["species_list"]
    snapshots = res["snapshots"]

    with h5py.File(filepath, "w") as f:

        # ── /metadata ─────────────────────────────────────────────────────
        meta = f.create_group("metadata")
        meta.attrs["H_COLUMN"]           = H_COLUMN
        meta.attrs["N_CELLS"]            = N_CELLS
        meta.attrs["MU"]                 = MU
        meta.attrs["RHO_F"]              = RHO_F
        meta.attrs["PHI_BED"]            = PHI_BED
        meta.attrs["PHI_MAX"]            = PHI_MAX
        meta.attrs["ETA_INTRINSIC"]      = ETA_INTRINSIC
        meta.attrs["K_REP"]              = K_REP
        meta.attrs["COLUMN_ASPECT_RATIO"]= COLUMN_ASPECT_RATIO
        meta.attrs["DT"]                 = DT
        meta.attrs["phi_global"]         = res["phi_global"]
        meta.attrs["W_COLUMN"]           = res["W_COLUMN"]

        sp_grp = meta.create_group("species")
        for s, sp in enumerate(sp_list):
            sg = sp_grp.create_group(str(s))
            sg.attrs["name"]    = sp.name
            sg.attrs["rho_p"]   = sp.rho_p
            sg.attrs["phi"]     = sp.phi
            sg.attrs["d_mean"]  = sp.d_mean
            sg.attrs["d_sigma"] = sp.d_sigma
            sg.attrs["color"]   = sp.color
            # Store optional percentile fields as NaN when absent so the
            # dataset schema stays consistent regardless of which fields
            # the user originally provided.
            sg.attrs["d10"] = sp.d10 if sp.d10 is not None else float("nan")
            sg.attrs["d50"] = sp.d50 if sp.d50 is not None else float("nan")
            sg.attrs["d90"] = sp.d90 if sp.d90 is not None else float("nan")

        # ── /timeseries ────────────────────────────────────────────────────
        ts = f.create_group("timeseries")
        ts.create_dataset("times",                   data=res["times"])
        ts.create_dataset("settled_frac",            data=res["settled_frac"])
        ts.create_dataset("bed_height",              data=res["bed_height"])
        ts.create_dataset("settled_frac_by_species", data=res["settled_frac_by_species"])
        ts.create_dataset("bed_vol_by_species",      data=res["bed_vol_by_species"])

        # ── /fields ────────────────────────────────────────────────────────
        flds = f.create_group("fields")
        flds.create_dataset("phi_field",    data=res["phi_field"],
                            compression="gzip", compression_opts=4)
        flds.create_dataset("mu_eff_field", data=res["mu_eff_field"],
                            compression="gzip", compression_opts=4)
        flds.create_dataset("cell_edges",   data=res["cell_edges"])

        # ── /particles ─────────────────────────────────────────────────────
        pts = f.create_group("particles")
        for key in ("diameters", "radii", "vt_arr", "n_arr",
                    "species_id", "particle_vols", "x", "counts"):
            pts.create_dataset(key, data=res[key])

        # ── /snapshots ─────────────────────────────────────────────────────
        snp_grp = f.create_group("snapshots")
        for i, snap in enumerate(snapshots):
            sg = snp_grp.create_group(f"{i:02d}")
            sg.attrs["step"]         = snap["step"]
            sg.attrs["t"]            = snap["t"]
            sg.attrs["pct_time"]     = snap["pct_time"]
            sg.attrs["settled_frac"] = snap["settled_frac"]
            sg.attrs["h_bed"]        = snap["h_bed"]
            sg.create_dataset("z",          data=snap["z"])
            sg.create_dataset("x",          data=snap["x"])
            sg.create_dataset("active",     data=snap["active"])
            sg.create_dataset("phi",        data=snap["phi"])
            sg.create_dataset("mu_eff",     data=snap["mu_eff"])
            sg.create_dataset("bed_vol_sp", data=snap["bed_vol_sp"])

    import os
    size_mb = os.path.getsize(filepath) / 1024 / 1024 if os.path.exists(filepath) else 0.0
    print(f"Results saved → {filepath}  ({size_mb:.2f} MB)")


def load_results(filepath: str) -> dict:
    """
    Load results from an HDF5 file written by save_results().

    Returns a dict that is structurally identical to the output of
    run_simulation(), so all plotting and summary functions work unchanged.

    Parameters
    ----------
    filepath : path to an HDF5 file written by save_results()

    Returns
    -------
    dict — drop-in replacement for live run_simulation() output.
    """
    res = {}

    with h5py.File(filepath, "r") as f:

        # ── /metadata ─────────────────────────────────────────────────────
        meta = f["metadata"]
        res["phi_global"] = float(meta.attrs["phi_global"])
        res["W_COLUMN"]   = float(meta.attrs["W_COLUMN"])

        sp_grp    = meta["species"]
        sp_list   = []
        n_species = len(sp_grp)
        for s in range(n_species):
            sg  = sp_grp[str(s)]
            d10 = float(sg.attrs["d10"])
            d50 = float(sg.attrs["d50"])
            d90 = float(sg.attrs["d90"])
            sp  = Species(
                name    = str(sg.attrs["name"]),
                rho_p   = float(sg.attrs["rho_p"]),
                phi     = float(sg.attrs["phi"]),
                d_mean  = float(sg.attrs["d_mean"]),
                d_sigma = float(sg.attrs["d_sigma"]),
                color   = str(sg.attrs["color"]),
                d10     = None if np.isnan(d10) else d10,
                d50     = None if np.isnan(d50) else d50,
                d90     = None if np.isnan(d90) else d90,
            )
            sp_list.append(sp)
        res["species_list"] = sp_list

        # ── /timeseries ────────────────────────────────────────────────────
        ts = f["timeseries"]
        res["times"]                   = ts["times"][:]
        res["settled_frac"]            = ts["settled_frac"][:]
        res["bed_height"]              = ts["bed_height"][:]
        res["settled_frac_by_species"] = ts["settled_frac_by_species"][:]
        res["bed_vol_by_species"]      = ts["bed_vol_by_species"][:]

        # ── /fields ────────────────────────────────────────────────────────
        flds = f["fields"]
        res["phi_field"]    = flds["phi_field"][:]
        res["mu_eff_field"] = flds["mu_eff_field"][:]
        res["cell_edges"]   = flds["cell_edges"][:]

        # ── /particles ─────────────────────────────────────────────────────
        pts = f["particles"]
        for key in ("diameters", "radii", "vt_arr", "n_arr",
                    "particle_vols", "x", "counts"):
            res[key] = pts[key][:]
        res["species_id"] = pts["species_id"][:].astype(int)

        # ── /snapshots ─────────────────────────────────────────────────────
        snp_grp   = f["snapshots"]
        snapshots = []
        for i in range(len(snp_grp)):
            sg   = snp_grp[f"{i:02d}"]
            snap = {
                "step":         int(sg.attrs["step"]),
                "t":            float(sg.attrs["t"]),
                "pct_time":     float(sg.attrs["pct_time"]),
                "settled_frac": float(sg.attrs["settled_frac"]),
                "h_bed":        float(sg.attrs["h_bed"]),
                "z":            sg["z"][:],
                "x":            sg["x"][:],
                "active":       sg["active"][:].astype(bool),
                "phi":          sg["phi"][:],
                "mu_eff":       sg["mu_eff"][:],
                "bed_vol_sp":   sg["bed_vol_sp"][:],
            }
            snapshots.append(snap)
        res["snapshots"] = snapshots

    print(f"Results loaded ← {filepath}  "
          f"({len(res['times'])} steps, "
          f"{len(res['species_list'])} species, "
          f"{len(snapshots)} snapshots)")
    return res


# =============================================================================
# 2-D column visualisation
# =============================================================================

def _snap_percentages(n_snaps: int,
                      interval: float,
                      log_scale: bool) -> list[float]:
    """
    Generate the list of time-percentage marks at which to capture snapshots.

    Parameters
    ----------
    n_snaps   : total number of snapshots (ignored when interval is given
                explicitly as a list — used only for log mode)
    interval  : spacing between snapshots in % of total time  (linear mode)
    log_scale : if True, space snapshots logarithmically so early settling
                (fast) is captured more densely than late settling (slow)

    Returns
    -------
    Sorted list of percentages in [0, 100].
    """
    if log_scale:
        # Log spacing: dense near t=0, coarse near t=t_max.
        # np.logspace(log10(1), log10(101), n_snaps) maps to ~[0, 100]%.
        raw = np.logspace(np.log10(1), np.log10(101), n_snaps) - 1
        pcts = np.clip(raw / raw[-1] * 100.0, 0, 100)
    else:
        pcts = np.arange(0, 100 + interval, interval)
        pcts = np.clip(pcts, 0, 100)
    # Always include 0 % and 100 %
    pcts = sorted(set([0.0, 100.0] + list(pcts)))
    return pcts


def _bed_colour(bed_vol_sp: np.ndarray,
                species_list: list[Species]) -> np.ndarray:
    """
    Compute the average RGB colour of the settled bed at one timestep
    by blending species colours weighted by their volume fraction in the bed.

    Returns an (3,) RGB array in [0, 1].
    """
    import matplotlib.colors as mcolors
    total = bed_vol_sp.sum()
    if total == 0:
        return np.array([0.75, 0.65, 0.45])   # neutral sandy colour
    rgb = np.zeros(3)
    for s, sp in enumerate(species_list):
        frac = bed_vol_sp[s] / total
        rgb += frac * np.array(mcolors.to_rgb(sp.color))
    return np.clip(rgb, 0, 1)


def _marker_size_pts(radii: np.ndarray,
                     ax,
                     W_COLUMN: float,
                     H_COLUMN: float,
                     base_scale: float = 0.6) -> np.ndarray:
    """
    Convert physical radii (m) to matplotlib scatter marker sizes (points²).

    Strategy: compute how many display points correspond to one metre in data
    coordinates, then scale radii accordingly.  This preserves relative size
    differences between particles while making them visible.

    base_scale : fraction of the "natural" point size to use.  <1 prevents
                 very large particles from completely occluding neighbours.
    """
    fig = ax.get_figure()
    bbox   = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    ax_w_pts = bbox.width    # axis width in display points
    ax_h_pts = bbox.height

    # Points per metre in x and y
    pts_per_m_x = ax_w_pts  / W_COLUMN
    pts_per_m_y = ax_h_pts  / H_COLUMN

    pts_per_m = min(pts_per_m_x, pts_per_m_y)   # use smaller to stay within axes

    # Marker size in scatter is the area in points² of the marker circle.
    # diameter in points = 2 * radius * pts_per_m * base_scale
    # area = π * (diameter/2)² — but matplotlib's s= uses full area in pts²
    diam_pts  = 2.0 * radii * pts_per_m * base_scale
    s_vals    = np.pi * (diam_pts / 2.0) ** 2
    return np.clip(s_vals, 0.5, 5000.0)   # clamp to visible range


def plot_2d_snapshots(res: dict,
                      interval: float = 10.0,
                      log_scale: bool = False,
                      n_snaps_log: int = 11,
                      output_dir: str = "outputs",
                      filename_prefix: str = "settling_2d") -> list[str]:
    """
    Produce a series of 2-D column visualisation PNGs, one per snapshot.

    Each PNG shows the column at one instant with:
      - Suspended particles as coloured circles (size ∝ physical radius)
      - Settled bed as a filled rectangle coloured by species blend
      - φ(z) and μ_eff(z) profiles as side panels
      - Annotated axes: time, % settled, bed height

    Parameters
    ----------
    res            : dict returned by run_simulation()
    interval       : snapshot spacing in % of t_max  (linear mode only)
    log_scale      : if True, use logarithmic time spacing
    n_snaps_log    : number of snapshots in log mode
    output_dir     : directory to write PNG files
    filename_prefix: prefix for output filenames

    Returns
    -------
    List of file paths written.
    """
    import os
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker

    sp_list    = res["species_list"]
    species_id = res["species_id"]
    diameters  = res["diameters"]
    radii      = res["radii"]
    snapshots  = res["snapshots"]
    cell_edges = res["cell_edges"]
    W_COLUMN   = res["W_COLUMN"]
    t_max      = res["times"][-1]

    colors     = [sp.color for sp in sp_list]
    names      = [sp.name  for sp in sp_list]
    n_species  = len(sp_list)

    cell_centres = 0.5 * (cell_edges[:-1] + cell_edges[1:])
    cell_height  = cell_edges[1] - cell_edges[0]

    # ── Determine which snapshots to render ───────────────────────────────
    pct_targets = _snap_percentages(n_snaps_log, interval, log_scale)

    # Match each target % to the closest available snapshot
    snap_times_pct = np.array([s["pct_time"] for s in snapshots])
    render_snaps   = []
    seen_idx       = set()
    for pt in pct_targets:
        idx = int(np.argmin(np.abs(snap_times_pct - pt)))
        if idx not in seen_idx:
            render_snaps.append(snapshots[idx])
            seen_idx.add(idx)
    render_snaps.sort(key=lambda s: s["t"])

    # ── Per-particle colour array (constant across snapshots) ─────────────
    particle_rgba = np.array([mcolors.to_rgba(sp.color)
                               for sp in sp_list])[species_id]

    os.makedirs(output_dir, exist_ok=True)
    written = []

    for snap_num, snap in enumerate(render_snaps):
        t_min    = snap["t"] / 60.0
        pct_t    = snap["pct_time"]
        pct_s    = snap["settled_frac"] * 100.0
        h_bed    = snap["h_bed"]
        phi      = snap["phi"]
        mu_eff   = snap["mu_eff"]
        active   = snap["active"]
        z_snap   = snap["z"]
        x_snap   = snap["x"]
        bvsp     = snap["bed_vol_sp"]

        # ── Figure layout: column (wide) + phi panel + mu_eff panel ──────
        # Aspect ratio: column panel height = H_COLUMN display units,
        # width = W_COLUMN display units — then phi/mu panels are narrow strips.
        col_width  = 3.0          # inches for the column panel
        side_width = 1.1          # inches for each side profile panel
        fig_w      = col_width + 2 * side_width + 0.8   # total + margins
        fig_h      = col_width * COLUMN_ASPECT_RATIO + 1.2   # tall

        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.suptitle(
            f"Settling Column — t = {t_min:.1f} min  "
            f"({pct_t:.0f}% of sim time)  |  "
            f"{pct_s:.1f}% settled",
            fontsize=9, fontweight="bold", y=0.995)

        # GridSpec: [column | phi | mu_eff]
        gs = gridspec.GridSpec(
            1, 3,
            figure=fig,
            width_ratios=[col_width, side_width, side_width],
            wspace=0.05,
            left=0.08, right=0.97, top=0.97, bottom=0.04)

        ax_col  = fig.add_subplot(gs[0, 0])   # main column
        ax_phi  = fig.add_subplot(gs[0, 1])   # φ(z) profile
        ax_mu   = fig.add_subplot(gs[0, 2])   # μ_eff(z) profile

        # ── Main column panel ─────────────────────────────────────────────

        # Settled bed: filled rectangle, colour = species-weighted blend
        bed_color = _bed_colour(bvsp, sp_list)
        if h_bed > 0:
            bed_rect = mpatches.Rectangle(
                (0, 0), W_COLUMN, h_bed,
                facecolor=bed_color, edgecolor="none", zorder=1,
                label="Settled bed")
            ax_col.add_patch(bed_rect)
            # Hatching overlay to make it clearly "solid"
            hatch_rect = mpatches.Rectangle(
                (0, 0), W_COLUMN, h_bed,
                facecolor="none", edgecolor="white",
                hatch="////", linewidth=0.3, zorder=2, alpha=0.25)
            ax_col.add_patch(hatch_rect)
            # Bed surface line
            ax_col.axhline(h_bed, color="white", linewidth=1.2,
                           linestyle="--", zorder=3, alpha=0.9)

        # Suspended particles
        susp_mask = active
        if susp_mask.any():
            # Compute marker sizes after figure is drawn (needs renderer)
            # Use a fixed physical scale: 1 m = col_width / W_COLUMN inches
            # → 1 m = col_width / W_COLUMN * 72 pts
            pts_per_m  = col_width / W_COLUMN * 72.0 * 0.5   # 0.5 = visual scale
            diam_pts   = 2.0 * radii[susp_mask] * pts_per_m
            s_vals     = np.pi * (diam_pts / 2.0) ** 2
            s_vals     = np.clip(s_vals, 0.8, 4000.0)

            ax_col.scatter(
                x_snap[susp_mask],
                z_snap[susp_mask],
                s=s_vals,
                c=particle_rgba[susp_mask],
                linewidths=0.0,
                zorder=4,
                alpha=0.75)

        # Column walls
        for xv in [0, W_COLUMN]:
            ax_col.axvline(xv, color="black", linewidth=1.5, zorder=5)

        # Annotations
        ax_col.set_xlim(-0.02 * W_COLUMN, 1.02 * W_COLUMN)
        ax_col.set_ylim(-0.01 * H_COLUMN, 1.01 * H_COLUMN)
        ax_col.set_ylabel("Height (m)", fontsize=7)
        ax_col.set_xlabel("Width (m)", fontsize=7)
        ax_col.tick_params(labelsize=6)
        ax_col.xaxis.set_major_locator(mticker.MaxNLocator(3))

        # Bed height annotation
        if h_bed > 0.01 * H_COLUMN:
            ax_col.text(W_COLUMN * 0.98, h_bed + 0.008 * H_COLUMN,
                        f"bed = {h_bed*100:.1f} cm",
                        ha="right", va="bottom", fontsize=6,
                        color="white", fontweight="bold", zorder=6)

        # Species legend
        legend_handles = [
            mpatches.Patch(facecolor=col, label=name, alpha=0.85)
            for col, name in zip(colors, names)
        ]
        legend_handles.append(
            mpatches.Patch(facecolor=bed_color, edgecolor="grey",
                           hatch="////", label="Settled bed", alpha=0.7))
        ax_col.legend(handles=legend_handles, fontsize=5.5,
                      loc="upper right", framealpha=0.75)

        # ── φ(z) side profile ─────────────────────────────────────────────
        phi_plot = np.clip(phi, 0, None)
        phi_max_display = max(phi_plot.max() * 1.25, res["phi_global"] * 1.5, 0.01)

        ax_phi.barh(cell_centres, phi_plot,
                    height=cell_height * 0.92,
                    color="steelblue", alpha=0.7, linewidth=0)
        ax_phi.axhline(h_bed, color="sienna", linewidth=1.0,
                       linestyle="--", alpha=0.8)
        ax_phi.set_xlim(0, phi_max_display)
        ax_phi.set_ylim(-0.01 * H_COLUMN, 1.01 * H_COLUMN)
        ax_phi.set_xlabel("φ", fontsize=7)
        ax_phi.set_title("Vol. frac.", fontsize=6, pad=2)
        ax_phi.tick_params(labelsize=5)
        ax_phi.yaxis.set_ticklabels([])
        ax_phi.xaxis.set_major_locator(mticker.MaxNLocator(3))
        # φ = 0.30 onset line
        if phi_max_display > 0.30:
            ax_phi.axvline(0.30, color="orange", linewidth=0.8,
                           linestyle=":", alpha=0.7)

        # ── μ_eff(z) side profile ─────────────────────────────────────────
        mu_ratio = mu_eff / MU
        # Use log scale if range spans more than one decade
        mu_max = mu_ratio.max()
        use_log = mu_max > 10.0

        if use_log:
            ax_mu.barh(cell_centres, np.log10(np.clip(mu_ratio, 1, None)),
                       height=cell_height * 0.92,
                       color="purple", alpha=0.6, linewidth=0)
            ax_mu.set_xlabel("log₁₀(μ/μ₀)", fontsize=7)
        else:
            ax_mu.barh(cell_centres, mu_ratio,
                       height=cell_height * 0.92,
                       color="purple", alpha=0.6, linewidth=0)
            ax_mu.set_xlabel("μ_eff/μ₀", fontsize=7)

        ax_mu.axhline(h_bed, color="sienna", linewidth=1.0,
                      linestyle="--", alpha=0.8)
        ax_mu.set_ylim(-0.01 * H_COLUMN, 1.01 * H_COLUMN)
        ax_mu.set_title("Eff. visc.", fontsize=6, pad=2)
        ax_mu.tick_params(labelsize=5)
        ax_mu.yaxis.set_ticklabels([])
        ax_mu.xaxis.set_major_locator(mticker.MaxNLocator(3))

        # ── Save ──────────────────────────────────────────────────────────
        fname  = f"{filename_prefix}_{snap_num:02d}_t{pct_t:05.1f}pct.png"
        fpath  = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(fpath)
        print(f"  Saved: {fname}  "
              f"(t={t_min:.1f} min, {pct_s:.1f}% settled, bed={h_bed*100:.1f} cm)")

    print(f"\n2D snapshots written: {len(written)} files → {output_dir}/")
    return written


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    # =========================================================================
    # USER SETTINGS
    # =========================================================================

    # ── Mode ─────────────────────────────────────────────────────────────────
    # "run"  : run the simulation, save results to RESULTS_FILE, then plot
    # "plot" : load results from RESULTS_FILE and plot only (no simulation)
    MODE         = "run"
    RESULTS_FILE = "settling_results.h5"

    # ── Mixture (only used in "run" mode) ─────────────────────────────────────
    # Only constraint: sum of all phi values must be < PHI_BED (0.60).
    mixture = [
        Species("ZnO (Zinc Oxide)",  rho_p=5610, phi=0.0124,
                d_mean=50e-6,  d_sigma=0.4, color="firebrick"),
        Species("MnO (Manganese Oxide)",   rho_p=5030, phi=0.0323,
                d_mean=100e-6, d_sigma=0.4, color="steelblue"),
        Species("Zinc Borate",     rho_p=2670, phi=0.03,
                d50=7e-6, d90=20e-6, color="goldenrod"),
    ]

    # ── 2D snapshot settings ──────────────────────────────────────────────
    SNAP_INTERVAL = 2.0    # % of simulation time between snapshots (linear)
    LOG_SCALE     = True   # True = log-spaced snapshots (dense early)
    N_SNAPS_LOG   = 51      # number of snapshots when LOG_SCALE = True

    # =========================================================================
    # MODE DISPATCH
    # =========================================================================

    if MODE not in ("run", "plot"):
        raise ValueError(f"MODE must be 'run' or 'plot', got '{MODE}'")

    W = 72
    snap_pcts_list = _snap_percentages(N_SNAPS_LOG, SNAP_INTERVAL, LOG_SCALE)

    if MODE == "run":
        # ── Simulate ──────────────────────────────────────────────────────────
        print("=" * W)
        print("  MODE: run  —  simulating, saving, then plotting")
        print("=" * W)
        print("  Mixture definition")
        print("-" * W)
        for sp in mixture:
            mu_ln, sigma_ln = sp.lognormal_params()
            d50_eff = np.exp(mu_ln) * 1e6
            d90_eff = np.exp(mu_ln + 1.2816 * sigma_ln) * 1e6
            src = "fitted" if sp.d50 is not None else "d_mean"
            print(f"  {sp.name:<24}  rho={sp.rho_p} kg/m3  "
                  f"phi={sp.phi*100:.1f}%  "
                  f"D50={d50_eff:.1f}µm  D90={d90_eff:.1f}µm  ({src})")
        total_phi = sum(sp.phi for sp in mixture)
        print(f"  {'Total phi':<24}  {total_phi*100:.1f}%")
        print(f"  Snapshot mode: {'log-spaced' if LOG_SCALE else 'linear'}, "
              f"every {SNAP_INTERVAL:.0f}% → {len(snap_pcts_list)} snapshots")
        print("=" * W)

        results = run_simulation(species_list=mixture, seed=42,
                                 snap_pcts=snap_pcts_list)
        save_results(results, RESULTS_FILE)

    else:
        # ── Load only — simulation is skipped entirely ─────────────────────
        print("=" * W)
        print("  MODE: plot  —  loading results, skipping simulation")
        print("=" * W)
        results = load_results(RESULTS_FILE)

    # ── Plot (runs in both modes) ──────────────────────────────────────────
    print_summary(results)
    plot_results(results)

    print("\nGenerating 2D column snapshots …")
    plot_2d_snapshots(
        results,
        interval=SNAP_INTERVAL,
        log_scale=LOG_SCALE,
        n_snaps_log=N_SNAPS_LOG,
    )
