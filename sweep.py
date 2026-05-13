"""
sweep.py — Sequential parameter sweep runner
See README.md for a detailed tutorial and output descriptions.
"""

from __future__ import annotations

import copy
import csv
import os
import sys
import traceback
from dataclasses import dataclass, field

import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap

# ── Locate config and simulation relative to this script ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import simulation as sim
from config import Species


# =============================================================================
# USER SETTINGS
# =============================================================================

# Output directory for all sweep files
SWEEP_DIR = "outputs/sweep"

# Total number of runs
N_RUNS = 5

# Snapshot percentages captured per run (keeps HDF5 files compact)
SNAP_PCTS = [0, 50, 100]

@dataclass
class SweepParam:
    """
    Defines a parameter to vary during the sweep.
    target : "fluid", "sim", or "species"
    attr   : attribute name to change
    species: name of the species (only for target="species")
    """
    target : str
    attr   : str
    species: str | None
    start  : float
    end    : float
    scale  : str = "linear"   # "linear" or "log"

    def values(self, n: int) -> np.ndarray:
        if n == 1:
            return np.array([self.start])
        if self.scale == "log":
            return np.logspace(np.log10(self.start), np.log10(self.end), n)
        return np.linspace(self.start, self.end, n)

    @property
    def label(self) -> str:
        if self.target == "species":
            sp_short = self.species.split()[0] if self.species else "?"
            return f"{sp_short}.{self.attr}"
        return self.attr


# ── Define your sweeps here ───────────────────────────────────────────────────
# All entries in this list co-vary together across N_RUNS.
SWEEPS: list[SweepParam] = [
    SweepParam(
        target  = "fluid",
        attr    = "MU",
        species = None,
        start   = 20e-3,
        end     = 100e-3,
        scale   = "linear",
    ),
]


# =============================================================================
# Internal helpers
# =============================================================================

def _generate_run_configs(sweeps: list[SweepParam],
                          n_runs: int) -> list[list[tuple[SweepParam, float]]]:
    value_arrays = [sw.values(n_runs) for sw in sweeps]
    configs = []
    for i in range(n_runs):
        configs.append([(sw, float(value_arrays[s][i]))
                        for s, sw in enumerate(sweeps)])
    return configs


def _run_label(run_idx: int,
               run_cfg: list[tuple[SweepParam, float]]) -> str:
    parts = [f"{sw.label}_{val:.4g}" for sw, val in run_cfg]
    return f"{run_idx:02d}_" + "_".join(parts)


def _apply_overrides(run_cfg: list[tuple[SweepParam, float]],
                     base_mixture: list[Species]) -> tuple[list[Species], dict]:
    patched_mixture = copy.deepcopy(base_mixture)
    module_overrides = {}

    for sw, val in run_cfg:
        if sw.target in ("fluid", "sim"):
            module_overrides[sw.attr] = val
        elif sw.target == "species":
            matched = [sp for sp in patched_mixture if sp.name == sw.species]
            if not matched:
                raise ValueError(
                    f"Species '{sw.species}' not found in mixture. "
                    f"Available: {[sp.name for sp in patched_mixture]}")
            setattr(matched[0], sw.attr, val)

    return patched_mixture, module_overrides


def _patch_module(overrides: dict) -> dict:
    original = {}
    for attr, val in overrides.items():
        original[attr] = getattr(sim, attr)
        setattr(sim, attr, val)
    return original


def _restore_module(original: dict) -> None:
    for attr, val in original.items():
        setattr(sim, attr, val)


def _time_to_pct(times: np.ndarray,
                 settled_frac: np.ndarray,
                 pct: float) -> float:
    idx = np.searchsorted(settled_frac, pct / 100.0)
    if idx >= len(times):
        return float("nan")
    return float(times[idx]) / 60.0


# =============================================================================
# Log helpers
# =============================================================================

def _init_log(log_path: str, sweeps: list[SweepParam], n_runs: int) -> None:
    with open(log_path, "w") as f:
        f.write("Settling Simulation — Sweep Run Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"N_RUNS : {n_runs}\n")
        for sw in sweeps:
            f.write(f"Sweep  : {sw.label}  {sw.start:.4g} → {sw.end:.4g}"
                    f"  ({sw.scale})\n")
        f.write("=" * 60 + "\n\n")


def _log_run(log_path: str, run_idx: int, label: str,
             run_cfg: list[tuple[SweepParam, float]], status: str,
             res: dict | None = None,
             error: str | None = None) -> None:
    with open(log_path, "a") as f:
        f.write(f"Run {run_idx:02d} | {label}\n")
        for sw, val in run_cfg:
            f.write(f"  {sw.label} = {val:.6g}\n")
        f.write(f"  Status : {status}\n")
        if res is not None:
            t50 = _time_to_pct(res["times"], res["settled_frac"], 50)
            t90 = _time_to_pct(res["times"], res["settled_frac"], 90)
            f.write(f"  Settled: {res['settled_frac'][-1]*100:.1f}%\n")
            f.write(f"  t50    : {t50:.2f} min\n")
            f.write(f"  t90    : {t90:.2f} min\n")
            f.write(f"  Bed ht : {res['bed_height'][-1]*100:.2f} cm\n")
        if error is not None:
            f.write(f"  Error  : {error}\n")
        f.write("\n")


# =============================================================================
# CSV summary
# =============================================================================

def _write_csv(csv_path: str,
               run_labels: list[str],
               run_cfgs: list[list[tuple[SweepParam, float]]],
               results_list: list[dict | None],
               sweeps: list[SweepParam]) -> None:
    sweep_headers = [sw.label for sw in sweeps]
    sp_names = []
    for res in results_list:
        if res is not None:
            sp_names = [sp.name for sp in res["species_list"]]
            break

    header = (["run", "label"]
              + sweep_headers
              + ["t50_min", "t90_min", "final_bed_cm",
                 "final_settled_pct", "duration_min"]
              + [f"settled_{sp.split()[0]}_pct" for sp in sp_names])

    def cfg_val(run_cfg, sw):
        for s, v in run_cfg:
            if s is sw:
                return v
        return float("nan")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (label, cfg, res) in enumerate(
                zip(run_labels, run_cfgs, results_list)):
            if res is None:
                row = [i, label] + [cfg_val(cfg, sw) for sw in sweeps]
                row += ["ERROR"] * (len(header) - len(row))
                writer.writerow(row)
                continue

            t50 = _time_to_pct(res["times"], res["settled_frac"], 50)
            t90 = _time_to_pct(res["times"], res["settled_frac"], 90)
            sp_settled = [
                f"{res['settled_frac_by_species'][-1, s]*100:.1f}"
                for s in range(len(sp_names))
            ]
            row = (
                [i, label]
                + [f"{cfg_val(cfg, sw):.6g}" for sw in sweeps]
                + [f"{t50:.2f}", f"{t90:.2f}",
                   f"{res['bed_height'][-1]*100:.2f}",
                   f"{res['settled_frac'][-1]*100:.1f}",
                   f"{res['times'][-1]/60:.1f}"]
                + sp_settled
            )
            writer.writerow(row)

    print(f"  CSV summary → {csv_path}")


# =============================================================================
# Comparison plot
# =============================================================================

def _plot_comparison(results_list: list[dict | None],
                     run_cfgs: list[list[tuple[SweepParam, float]]],
                     run_labels: list[str],
                     sweeps: list[SweepParam],
                     output_path: str) -> None:
    # Filter to successful runs only
    valid = [(i, res, run_cfgs[i], run_labels[i])
             for i, res in enumerate(results_list) if res is not None]
    if not valid:
        print("  No successful runs — skipping comparison plot.")
        return

    def _val(cfg, sw):
        for s, v in cfg:
            if s is sw:
                return v
        return float("nan")

    def _leg(cfg):
        return ", ".join(f"{sw.label}={_val(cfg, sw):.3g}" for sw in sweeps)

    n_valid  = len(valid)
    cmap     = get_cmap("viridis")
    colors   = [cmap(i / max(n_valid - 1, 1)) for i in range(n_valid)]

    sweep_axis_label = " / ".join(sw.label for sw in sweeps)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Parameter Sweep — Settling Comparison",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1: cumulative settling curves ───────────────────────────────
    for color, (i, res, cfg, label) in zip(colors, valid):
        t_min = res["times"] / 60.0
        ax1.plot(t_min, res["settled_frac"] * 100.0,
                 color=color, linewidth=1.8, label=_leg(cfg))

    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Overall settled (%)")
    ax1.set_title("Cumulative Settling")
    ax1.set_xlim(left=0)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=6, loc="lower right")
    ax1.grid(True, alpha=0.35)

    # ── Panel 2: bed height growth curves ─────────────────────────────────
    for color, (i, res, cfg, label) in zip(colors, valid):
        t_min = res["times"] / 60.0
        ax2.plot(t_min, res["bed_height"] * 100.0,
                 color=color, linewidth=1.8, label=_leg(cfg))

    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Bed height (cm)")
    ax2.set_title("Sediment Bed Growth")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=6, loc="upper left")
    ax2.grid(True, alpha=0.35)

    # ── Panel 3: final bed composition (stacked horizontal bars) ──────────
    sp_names  = [sp.name  for sp in valid[0][1]["species_list"]]
    sp_colors = [sp.color for sp in valid[0][1]["species_list"]]
    n_species = len(sp_names)

    y_pos        = np.arange(n_valid)
    y_tick_labels = [_leg(cfg) for _, _, cfg, _ in valid]

    lefts = np.zeros(n_valid)
    for s in range(n_species):
        widths = np.array([
            res["bed_vol_by_species"][-1, s] /
            max(res["bed_vol_by_species"][-1].sum(), 1e-12) * 100.0
            for _, res, _, _ in valid
        ])
        ax3.barh(y_pos, widths, left=lefts,
                 color=sp_colors[s], alpha=0.85,
                 label=sp_names[s].split()[0], height=0.6)
        lefts += widths

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(y_tick_labels, fontsize=6)
    ax3.set_xlabel("Bed volume fraction (%)")
    ax3.set_title("Final Bed Composition by Species")
    ax3.set_xlim(0, 100)
    ax3.legend(fontsize=7, loc="lower right")
    ax3.grid(True, alpha=0.3, axis="x")

    # ── Panel 4: key metrics vs swept parameter value ─────────────────────
    primary_sw = sweeps[0]
    x_vals = np.array([_val(cfg, primary_sw) for _, _, cfg, _ in valid])

    t50_vals = [_time_to_pct(res["times"], res["settled_frac"], 50)
                for _, res, _, _ in valid]
    t90_vals = [_time_to_pct(res["times"], res["settled_frac"], 90)
                for _, res, _, _ in valid]
    bed_vals = [res["bed_height"][-1] * 100.0 for _, res, _, _ in valid]

    ax4b = ax4.twinx()

    ax4.plot(x_vals, t50_vals, "o--", color="steelblue",
             linewidth=1.5, markersize=5, label="t₅₀ (min)")
    ax4.plot(x_vals, t90_vals, "s--", color="firebrick",
             linewidth=1.5, markersize=5, label="t₉₀ (min)")
    ax4.set_xlabel(sweep_axis_label)
    ax4.set_ylabel("Settling time (min)", color="black")
    ax4.tick_params(axis="y")
    ax4.grid(True, alpha=0.3)

    ax4b.plot(x_vals, bed_vals, "^-", color="sienna",
              linewidth=1.8, markersize=6, label="Bed ht (cm)")
    ax4b.set_ylabel("Final bed height (cm)", color="sienna")
    ax4b.tick_params(axis="y", labelcolor="sienna")

    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4b.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
    ax4.set_title(f"Key Metrics vs {sweep_axis_label}")

    # ── Colour bar to show run position ───────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(0, n_valid - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation="vertical",
                        fraction=0.015, pad=0.02)
    cbar.set_label("Run index (0 = first)", fontsize=7)
    cbar.set_ticks(range(n_valid))
    cbar.set_ticklabels([str(i) for i, *_ in valid], fontsize=6)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot → {output_path}")


# =============================================================================
# Main sweep runner
# =============================================================================

def run_sweep(sweeps: list[SweepParam] = SWEEPS,
              n_runs: int = N_RUNS,
              snap_pcts: list[float] = SNAP_PCTS,
              sweep_dir: str = SWEEP_DIR) -> list[dict | None]:
    """Execute the full parameter sweep."""
    os.makedirs(sweep_dir, exist_ok=True)
    log_path  = os.path.join(sweep_dir, "run_log.txt")
    csv_path  = os.path.join(sweep_dir, "sweep_summary.csv")
    plot_path = os.path.join(sweep_dir, "sweep_comparison.png")

    run_cfgs   = _generate_run_configs(sweeps, n_runs)
    run_labels = [_run_label(i, cfg) for i, cfg in enumerate(run_cfgs)]

    _init_log(log_path, sweeps, n_runs)

    W = 70
    print("=" * W)
    print("  PARAMETER SWEEP")
    print("=" * W)
    for sw in sweeps:
        vals = sw.values(n_runs)
        print(f"  {sw.label:<20}  {sw.start:.4g} → {sw.end:.4g}"
              f"  ({sw.scale})  values: "
              + ", ".join(f"{v:.4g}" for v in vals))
    print(f"  N_RUNS = {n_runs}  |  output → {sweep_dir}/")
    print("=" * W + "\n")

    results_list = []

    for i, (cfg, label) in enumerate(zip(run_cfgs, run_labels)):
        print(f"Run {i+1}/{n_runs}  [{label}]")

        try:
            patched_mixture, module_overrides = _apply_overrides(
                cfg, config.MIXTURE)
        except ValueError as e:
            msg = str(e)
            print(f"  SKIP — {msg}")
            _log_run(log_path, i, label, cfg, "SKIPPED", error=msg)
            results_list.append(None)
            continue

        total_phi = sum(sp.phi for sp in patched_mixture)
        if total_phi >= config.PHI_BED:
            msg = (f"Total phi={total_phi:.4f} >= PHI_BED={config.PHI_BED} "
                   f"after sweep override — skipping run.")
            print(f"  SKIP — {msg}")
            _log_run(log_path, i, label, cfg, "SKIPPED", error=msg)
            results_list.append(None)
            continue

        original = _patch_module(module_overrides)

        try:
            res = sim.run_simulation(
                species_list=patched_mixture,
                snap_pcts=snap_pcts,
            )

            h5_path = os.path.join(sweep_dir, f"sweep_{label}.h5")
            sim.save_results(res, h5_path)
            print(f"  OK — settled={res['settled_frac'][-1]*100:.1f}%  "
                  f"bed={res['bed_height'][-1]*100:.2f}cm  "
                  f"→ {os.path.basename(h5_path)}")
            _log_run(log_path, i, label, cfg, "OK", res=res)
            results_list.append(res)

        except Exception:
            msg = traceback.format_exc().strip().splitlines()[-1]
            full = traceback.format_exc()
            print(f"  ERROR — {msg}")
            _log_run(log_path, i, label, cfg, "ERROR", error=full)
            results_list.append(None)

        finally:
            _restore_module(original)

    # ── Post-sweep outputs ─────────────────────────────────────────────────
    n_ok   = sum(1 for r in results_list if r is not None)
    n_fail = n_runs - n_ok
    print(f"\nSweep complete — {n_ok}/{n_runs} runs succeeded"
          + (f", {n_fail} skipped/errored (see run_log.txt)" if n_fail else ""))

    _write_csv(csv_path, run_labels, run_cfgs, results_list, sweeps)
    _plot_comparison(results_list, run_cfgs, run_labels, sweeps, plot_path)

    print(f"\nAll outputs written to: {sweep_dir}/")
    return results_list


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_sweep(
        sweeps    = SWEEPS,
        n_runs    = N_RUNS,
        snap_pcts = SNAP_PCTS,
        sweep_dir = SWEEP_DIR,
    )
