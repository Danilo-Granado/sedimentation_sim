from __future__ import annotations
from dataclasses import dataclass

# =============================================================================
# USER SETTINGS & CONFIGURATION
# =============================================================================

# ── Mode ─────────────────────────────────────────────────────────────────
# "run"  : run the simulation, save results to RESULTS_FILE, then plot
# "plot" : load results from RESULTS_FILE and plot only (no simulation)
MODE = "run"
RESULTS_FILE = "outputs/settling_results.h5"

# ── Physical constants ────────────────────────────────────────────────────────
G = 9.81        # gravitational acceleration,  m/s²

# ── Fluid properties ─────────────────────────────────────────────────────────
RHO_F = 1100.0  # fluid density,               kg/m³
MU    = 20e-3  # dynamic viscosity,            Pa·s

# ── Simulation settings ───────────────────────────────────────────────────────
N_PARTICLES_TOTAL = 10000   # total particles across all species
H_COLUMN          = 0.3   # column height,            m
DT                = 1.0   # time step,                s
N_CELLS           = 500    # vertical cells for φ field

# ── Column geometry ───────────────────────────────────────────────────────────
COLUMN_ASPECT_RATIO = 2   # H_COLUMN / W_COLUMN

# ── Bed packing ───────────────────────────────────────────────────────────────
PHI_BED = 0.60   # solid volume fraction inside the settled bed

# ── Concentrated-suspension / paste physics ───────────────────────────────────
PHI_MAX    = 0.64   # random close-packing limit
ETA_INTRINSIC = 2.5 # Einstein intrinsic viscosity

# Soft-sphere DEM collision parameters
K_REP = 0.5        # repulsion strength (0 = off, 1 = stiff)

# ── Diameter clamps ──────────────────────────────────────────────────────────
D_MIN = 1e-6    # m
D_MAX = 1e-3    # m

# ── 2D snapshot settings ──────────────────────────────────────────────────────
SNAP_INTERVAL = 10.0    # % of simulation time between snapshots (linear)
LOG_SCALE     = True    # True = log-spaced snapshots (dense early)
N_SNAPS_LOG   = 5       # number of snapshots when LOG_SCALE = True


@dataclass
class Species:
    """
    One solid component in the suspension.
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
        import numpy as np
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


# ── Mixture definition ────────────────────────────────────────────────────────
# Only constraint: sum of all phi values must be < PHI_BED.
MIXTURE = [
    Species("ZnO (Zinc Oxide)",  rho_p=5610, phi=0.0124,
            d_mean=50e-6,  d_sigma=0.4, color="firebrick"),
    Species("MnO (Manganese Oxide)",   rho_p=5030, phi=0.0323,
            d_mean=100e-6, d_sigma=0.4, color="steelblue"),
    Species("Zinc Borate",     rho_p=2670, phi=0.03,
            d50=7e-6, d90=20e-6, color="goldenrod"),
]
