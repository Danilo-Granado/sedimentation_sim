# Particle Settling Simulation

This repository contains a multi-species sedimentation simulation for solid particles in a viscous medium. It accounts for hindered settling, concentrated-suspension physics, and sediment bed formation.

## Features

- **Multi-species Support**: Define multiple solid species with unique densities, volume fractions, and size distributions.
- **Size Distributions**: Log-normal particle size distributions can be directly specified or fitted from $D_{10}$, $D_{50}$, and $D_{90}$ percentiles.
- **Physics-based Drag**: Implementation of Stokes and Schiller-Naumann drag models to compute terminal velocities.
- **Hindered Settling**: Uses the [Richardson-Zaki (1954)](#) hindered settling model.
- **Viscosity Divergence**: Employs the [Krieger-Dougherty](#) effective viscosity model to capture the transition from a suspension to a paste as volume fraction approaches maximum packing.
- **Collision Detection**: Soft-sphere Discrete Element Method (DEM) repulsive impulses prevent unphysical particle overlap in dense regions.
- **Bed Tracking**: Real-time tracking of the sediment bed height and its species composition.
- **Visualisation**: Comprehensive analysis plots including:
    - Cumulative settling curves by species.
    - Bed growth and composition over time.
    - Volume fraction $\phi(z,t)$ and effective viscosity $\mu_{eff}(z,t)$ heatmaps.
    - Particle size distributions (PSD).
    - Terminal velocity vs. diameter scatter plots.
    - Krieger-Dougherty viscosity divergence curve.
    - Final bed composition pie chart.
- **2D snapshots**: Visual representations of the settling column with side profiles for $\phi$ and $\mu_{eff}$.

## Physics and Equations

The simulation incorporates several key physical models:

1. **Terminal Velocity ($v_t$)**: Solved iteratively using the Schiller-Naumann drag coefficient.
2. **Richardson-Zaki Hindrance**: $v_h = v_t (1 - \phi)^n$, where $n$ is the [Garside & Al-Dibouni (1977)](#) exponent.
3. **Krieger-Dougherty Viscosity**: $\mu_{eff} = \mu_f (1 - \frac{\phi}{\phi_{max}})^{-[\eta]\phi_{max}}$, representing the [viscosity divergence](#) at high concentrations.
4. **DEM Collisions**: Pairwise Hertzian-like repulsive impulses applied during particle overlaps.

## Configuration

All simulation parameters are centralized in `config.py`. You can adjust:

- **Fluid Properties**: Density ($\rho_f$) and dynamic viscosity ($\mu$).
- **Simulation Settings**: Number of particles, column height, time step ($dt$), and grid resolution.
- **Mixture Definition**: Define species using the `Species` dataclass.
- **Visualisation Settings**: Snapshot intervals and scaling.

```python
# Example Species definition in config.py
Species("ZnO (Zinc Oxide)", rho_p=5610, phi=0.0124, d_mean=50e-6, d_sigma=0.4, color="firebrick")
```

## Running the Simulation

1.  Configure your parameters in `config.py`.
2.  Set `MODE = "run"` in `config.py`.
3.  Run the main script:
    ```bash
    python simulation.py
    ```

## Output Files

- `settling_results.h5`: A self-contained HDF5 file containing all simulation data, metadata, and snapshots.
- `settling_results.png`: A comprehensive plot with 8 subplots analyzing the simulation results.
- `outputs/`: A directory containing 2D snapshots of the settling column.

### HDF5 Structure Reference

The `settling_results.h5` file is organized as follows:
- `/metadata/`: Contains simulation constants and species definitions.
- `/timeseries/`: Global metrics over time (settled fraction, bed height, etc.).
- `/fields/`: Spatial fields like volume fraction ($\phi$) and effective viscosity ($\mu_{eff}$).
- `/particles/`: Initial properties of every generated particle.
- `/snapshots/`: Detailed state (positions, active masks, fields) at specific time intervals for 2D visualisation.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- h5py
