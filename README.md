

# OPF-Riehen-Data: Distribution Network Optimization

This repository contains an optimization framework for analyzing the feasible region of flexible power flows (FFOR) in a distribution network. It uses Gurobi to solve optimal power flow (OPF) problems with renewable energy integration (PV), heat pumps (HP), battery energy storage systems (BESS), and electric vehicles (EV).

## Core Modules

### `main.py`
Entry point for the optimization analysis. Orchestrates the workflow by:
- Defining a list of scenarios via `Config` objects (year, month, day, start hour, time steps, PV weather type, (no_bess), (no_ev))
- Creating output folders for each scenario
- Executing the optimization analysis (`_run_analysis`) for each configuration

Each scenario runs a detailed OPF problem that computes the feasible region of flexible active (P) and reactive (Q) power injections at each node.

### `config.py`
Central configuration module containing the `Config` class. Manages:
- **Input file structure and paths**: Specifies locations of geospatial data (network edges/nodes), load profiles, PV generation profiles (sunny/cloudy/average), EV charging constraints, BESS parameters and HP parameters
- **Technology parameters**: PV reactive power ratio, heat pump operating temperatures and COP, BESS state-of-charge bounds, EV charging limits
- **Optimization settings**: Convergence tolerance for polygon approximation of FFOR, initial optimization directions
- **Per-unit (pu) system bases**: Voltage (0.4 kV), apparent power (0.63 MVA), impedance, and current bases
- **Derived parameters**: Preprocesses time indices, loads all data from files, calculates heat pump temperature bounds based on outdoor conditions

### `functions.py`
Low-level optimization building blocks. Provides functions to define:
- **Variable definitions and constraints** for each technology:
  - `define_pv_vars_and_bcs()`: PV active/reactive power with upper bounds from generation profiles
  - `define_hp_vars_and_bcs()`: Heat pump power draw with thermal dynamics and temperature constraints
  - `define_bess_vars_and_bcs()`: Battery energy storage with power/energy limits and octagon approximation
  - `define_ev_vars_and_bcs()`: EV charging with lower/upper bound profiles
- **Network constraints**: Kirchhoff's current law (KCL) for active/reactive power, voltage drop equations, line thermal limits (convex hull approximation)

These functions are called sequentially in the main optimization workflow to build the complete Gurobi MIP model.

### `utils.py`
Utility functions for post-processing and data extraction:
- Parsing result folder names to extract scenario parameters
- Extracting minimum/maximum flexible power values from result files
- Helper functions for loading and transforming results data

### `plots.py`
Visualization functions for optimization results:
- `_plot_FFOR()`: Plots the feasible region (convex hull) of flexible power injections using scipy.spatial.ConvexHull
- `plot_seasonal_p_flex()`: Line plots showing seasonal variations of minimum/maximum active power flexibility

## Post-Processing & Analysis

### `explorations.py`
Exploratory analysis scripts for investigating results (not very clean):
- `exploration_seasonality()`: Extracts and visualizes how flexible power ranges vary seasonally
- `exploration_pv_weather_seasonality()`: Compares FFOR size across different PV weather scenarios (sunny, cloudy, average)

These are self-contained functions useful for understanding flexibility properties across different time periods and weather conditions.

### `dual_and_plots.py`
Advanced post-processing for detailed constraint analysis:
- **Dual variable extraction**: Fixes optimal solutions from OPF and re-solves the LP dual to identify which constraints are binding
- **Time-horizon comparison**: Overlays FFORs computed for different time horizons (0.5h to 8h) to show how flexibility changes with planning window
- **EV hull analysis**: Visualizes FFOR boundaries colored by their binding constraints, with optional EV charging envelope comparison

Can be run as a standalone script to regenerate plots from cached CSV data without requiring Gurobi.

## Data Preparation

### `misc/` Folder
Contains helper scripts for preparing input data profiles (not all are actively used):
- `pv_profil.py` / `pv_profile_erklaerung.txt`: PV generation profile processing
- `bess_profil.py` / `bess_alternativ_erklaerung.txt`: BESS parameter definitions and allocation (NOT ACTUALLY USED!)
- `ev_profile.py`: EV charging profile generation
- `create_baseprofil_hp.py`: Heat pump base load profile creation (NOT ACTUALLY USED!)
- `dummy_functions.py`: Legacy or temporary helper functions

## Input Data Structure

Input data is organized as:
```
base_folder / scenario_year / [input files]
```
Where `base_folder` defaults to `2703_23_homogen` (a specific grid district). Input files include:
- Network topology (edges.geojson, nodes.geojson, edges_metadata.csv)
- Component allocations (bess_allocation.csv, pv_p_installed.csv, hp_allocation.csv, ev_allocation.csv)
- Time-series profiles (load_profiles.csv, pv_generation.csv, ev_charging.csv, temperature_profiles.csv)

## Sign Convention

**Positive power flows into the grid; negative power flows out of the grid.**

This convention applies to all load, generation, and storage power variables throughout the optimization.

## Known Issues

There are no results due to an error when the FFOR is two-dimensional, because the current iterative algorithm expects a two-dimensional area for the convex hull of the FFOR.