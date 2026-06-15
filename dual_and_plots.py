"""Dual analysis, FFOR time-horizon comparison, and EV hull comparison plots.

Three self-contained capabilities (each callable independently):

  1. analyse_from_results   — fix-and-re-solve LP dual extraction from MIP results
  2. FFOR time-horizon plot — overlay FFORs for T = 0.5 h … 8 h (summer / winter)
  3. EV hull comparison     — FFOR hull coloured by binding constraint, with/without EV halo

Run as a script to regenerate all plots from cached CSV data (no Gurobi required
if dual CSVs already exist in the duals/ directory).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.spatial import ConvexHull

import config
import functions as funcs
import utils


# Constraint classification

_CONSTR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^V_root"),                  "V_root"),
    (re.compile(r"^volt_drop"),               "voltage_drop"),
    (re.compile(r"^volt_lb"),                 "volt_lb"),
    (re.compile(r"^volt_ub"),                 "volt_ub"),
    (re.compile(r"^kclP"),                    "kcl_P"),
    (re.compile(r"^kclQ"),                    "kcl_Q"),
    (re.compile(r"^S_limit"),                 "line_S_limit"),
    (re.compile(r"^pv_balance"),              "pv_balance"),
    (re.compile(r"^pv_ub"),                   "pv_ub"),
    (re.compile(r"^pv_lb"),                   "pv_lb"),
    (re.compile(r"^q_pv_ub"),                 "q_pv_ub"),
    (re.compile(r"^q_pv_lb"),                 "q_pv_lb"),
    (re.compile(r"^hp_temp_balance"),         "hp_temp_balance"),
    (re.compile(r"^hp_balance"),              "hp_balance"),
    (re.compile(r"^hp_min"),                  "hp_power_lb"),
    (re.compile(r"^hp_max"),                  "hp_power_ub"),
    (re.compile(r"^q_hp_ratio"),              "q_hp_ratio"),
    (re.compile(r"^q_hp_base"),               "q_hp_base"),
    (re.compile(r"^t_hp_ub"),                 "t_hp_ub"),
    (re.compile(r"^t_hp_lb"),                 "t_hp_lb"),
    (re.compile(r"^soc_bess_balance"),        "soc_bess_balance"),
    (re.compile(r"^soc_bess_ub"),             "soc_bess_ub"),
    (re.compile(r"^soc_bess_lb"),             "soc_bess_lb"),
    (re.compile(r"^p_bess_balance"),          "bess_balance"),
    (re.compile(r"^p_bess_pos_charge"),       "bess_no_simult_charge"),
    (re.compile(r"^p_bess_neg_discharge"),    "bess_no_simult_discharge"),
    (re.compile(r"^p_bess_pos_lb"),           "bess_pos_lb"),
    (re.compile(r"^p_bess_neg_ub"),           "bess_neg_ub"),
    (re.compile(r"^bess_power_octagon"),      "bess_apparent_power"),
    (re.compile(r"^ev_balance"),              "ev_balance"),
    (re.compile(r"^ev_lb"),                   "ev_lb"),
    (re.compile(r"^ev_ub"),                   "ev_ub"),
    (re.compile(r"^p_flex_total"),            "p_flex_total"),
    (re.compile(r"^q_flex_total"),            "q_flex_total"),
]

# Scaling: raw_dual × scale → kVA FFOR per [unit]
# Derivation (obj = -(a·P_flex + b·Q_flex)/S_base, S_base = 630 kVA):
#   kW constraint:   Δ(a·P+b·Q) = |Pi| × S_base      → scale = 630
#   p.u. constraint: Δ(a·P+b·Q) = |Pi| × 1            → scale = 1
#   SOC (0–1):       Δ(a·P+b·Q) = |Pi| × S_base/100   → scale = 6.3
_CONSTR_SCALE: dict[str, tuple[float, str]] = {
    "soc_bess_lb":              (6.3,  "% SOC"),
    "soc_bess_ub":              (6.3,  "% SOC"),
    "line_S_limit":             (1.0,  "kVA"),
    "pv_ub":                    (630,  "kW"),
    "pv_lb":                    (630,  "kW"),
    "q_pv_ub":                  (630,  "kVAr"),
    "q_pv_lb":                  (630,  "kVAr"),
    "bess_apparent_power":      (630,  "kVA"),
    "ev_lb":                    (630,  "kW"),
    "ev_ub":                    (630,  "kW"),
    "hp_power_lb":              (630,  "kW"),
    "hp_power_ub":              (630,  "kW"),
    "t_hp_lb":                  (630,  "°C"),
    "t_hp_ub":                  (630,  "°C"),
    "bess_no_simult_charge":    (630,  "kW"),
    "bess_no_simult_discharge": (630,  "kW"),
    "bess_pos_lb":              (630,  "kW"),
    "bess_neg_ub":              (630,  "kW"),
}

# Physically not relaxable (weather-determined)
_WEATHER_TYPES: set[str] = {"pv_lb", "q_pv_lb"}

# Binding → hardware upgrade needed
_INVESTMENT_TYPES: set[str] = {
    "bess_apparent_power", "line_S_limit",
    "hp_power_lb", "hp_power_ub",
    "volt_lb", "volt_ub",
    "pv_ub", "q_pv_ub",
}

# Equality constraints — dual ≠ 0 here is a nodal price, not a limit
_EQUALITY_TYPES: set[str] = {
    "V_root", "voltage_drop",
    "kcl_P", "kcl_Q",
    "pv_balance",
    "hp_balance", "hp_temp_balance", "q_hp_ratio", "q_hp_base",
    "soc_bess_balance", "bess_balance",
    "ev_balance",
    "p_flex_total", "q_flex_total",
}

# Human-readable LaTeX labels for constraint types
_CONSTR_LABEL: dict[str, str] = {
    "line_S_limit":             r"$S_\mathrm{line}$",
    "bess_apparent_power":      r"$S_\mathrm{BESS}$",
    "hp_power_lb":              r"$P_\mathrm{HP}^{\min}$",
    "hp_power_ub":              r"$\bar{P}_\mathrm{HP}$",
    "volt_lb":                  r"$V^{\min}$",
    "volt_ub":                  r"$\bar{V}$",
    "pv_ub":                    r"$\bar{P}_\mathrm{PV}$",
    "q_pv_ub":                  r"$\bar{Q}_\mathrm{PV}$",
    "t_hp_ub":                  r"$\bar{T}_\mathrm{HP}$",
    "t_hp_lb":                  r"$T_\mathrm{HP}^{\min}$",
    "soc_bess_lb":              r"$\mathrm{SOC}^{\min}$",
    "soc_bess_ub":              r"$\bar{\mathrm{SOC}}$",
    "ev_lb":                    r"$P_\mathrm{EV}^{\min}$",
    "ev_ub":                    r"$\bar{P}_\mathrm{EV}$",
    "bess_no_simult_charge":    r"$\delta_\mathrm{BESS}^{\mathrm{ch}}$",
    "bess_no_simult_discharge": r"$\delta_\mathrm{BESS}^{\mathrm{dis}}$",
    "bess_pos_lb":              r"$P_\mathrm{BESS}^{+,\min}$",
    "bess_neg_ub":              r"$\bar{P}_\mathrm{BESS}^{-}$",
}


def _classify(name: str) -> str:
    for pat, label in _CONSTR_PATTERNS:
        if pat.match(name):
            return label
    return "other"


# 1. Dual analysis: fix-and-re-solve LP from saved MIP results

# Parses "results_node_t_a-0.726_b-0.687.csv" → (-0.726, -0.687)
_DIR_FILE_RE = re.compile(
    r"results_node_t_a([+-]?[\d.]+(?:e[+-]?\d+)?)_b([+-]?[\d.]+(?:e[+-]?\d+)?)\.csv$"
)


def _build_model(conf: config.Config, a: float, b: float):
    """Build the full OPF LP/MIP model for direction (a, b) — unsolved."""
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("OPF_dual")
    model.Params.MIPGap = 0.01
    model.Params.NoRelHeurWork = 5
    model.Params.OutputFlag = 0

    p_pv, p_pv_flex, q_pv, q_pv_flex         = funcs.define_pv_vars_and_bcs(model, conf)
    p_hp, p_hp_flex, q_hp, q_hp_flex, t_hp   = funcs.define_hp_vars_and_bcs(model, conf)
    p_bess_pos, p_bess_neg, p_bess_flex, q_bess, q_bess_flex, soc_bess, b_bess_charge = \
        funcs.define_bess_vars_and_bcs(model, conf)
    p_ev, p_ev_flex                           = funcs.define_ev_vars_and_bcs(model, conf)

    edges_df_pu = funcs.per_unit_edges(
        conf.edges_metadata_df.copy(), V_base_kV=conf.V_base, S_base_MVA=conf.S_base
    )
    root_osmid = 97
    root_idx = conf.node_metadata_df.index[
        conf.node_metadata_df["LV_osmid"] == root_osmid
    ].tolist()[0]

    N = conf.node_metadata_df.shape[0]
    T = len(conf.time_index_list)
    tree     = funcs.build_radial_tree_from_edges(n_nodes=N, edges_df_pu=edges_df_pu, root_idx=root_idx)
    ldf_data = funcs.assemble_lindistflow_data(tree, T=T, V_min=0.95, V_max=1.05)

    Sbase_kW = conf.S_base * 1000.0
    P_inj_expr, Q_inj_expr = {}, {}
    for i in conf.node_metadata_df.index:
        for tcol in conf.time_index_list:
            P_inj_expr[(i, tcol)] = (
                conf.p_load[i, tcol]
                + p_pv.get((i, tcol), 0)
                + p_hp.get((i, tcol), 0)
                + p_bess_pos.get((i, tcol), 0)
                + p_bess_neg.get((i, tcol), 0)
                + p_ev.get((i, tcol), 0)
            ) / Sbase_kW
            Q_inj_expr[(i, tcol)] = (
                q_pv.get((i, tcol), 0)
                + q_hp.get((i, tcol), 0)
                + q_bess.get((i, tcol), 0)
            ) / Sbase_kW

    V, Pf, Qf = funcs.add_lindistflow_to_model(
        model, conf.time_index_list, ldf_data, P_inj_expr, Q_inj_expr,
        fix_root_voltage=1.0, use_soc_lines=True,
    )

    # Replace variable voltage bounds with named constraints so duals are accessible
    for (j, t), v in V.items():
        if j == root_idx:
            continue
        v.LB = -GRB.INFINITY
        v.UB = GRB.INFINITY
        model.addConstr(v >= 0.95, name=f"volt_lb[{j},{t}]")
        model.addConstr(v <= 1.05, name=f"volt_ub[{j},{t}]")

    p_flex_total = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p_flex_total")
    q_flex_total = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="q_flex_total")

    for t in conf.time_index_list:
        model.addConstr(
            p_flex_total == gp.quicksum(
                p_pv_flex.get((n, t), 0) + p_hp_flex.get((n, t), 0)
                + p_bess_flex.get((n, t), 0) + p_ev_flex.get((n, t), 0)
                for n in conf.node_group_dict["ALL NODES"]
            ),
            name=f"p_flex_total[{t}]",
        )
        model.addConstr(
            q_flex_total == gp.quicksum(
                q_pv_flex.get((n, t), 0) + q_hp_flex.get((n, t), 0) + q_bess_flex.get((n, t), 0)
                for n in conf.node_group_dict["ALL NODES"]
            ),
            name=f"q_flex_total[{t}]",
        )

    model.setObjective(
        -a * p_flex_total / Sbase_kW - b * q_flex_total / Sbase_kW,
        GRB.MINIMIZE,
    )
    return model


def _load_b_bess_charge(node_csv: Path, time_cols: list[str]) -> dict[tuple[int, int], int]:
    """Read b_bess_charge per (node, timestep) from a saved direction result CSV."""
    df  = pd.read_csv(node_csv)
    bdf = df[df["Variable"] == "b_bess_charge"].copy()
    result = {}
    for _, row in bdf.iterrows():
        node = int(row["Node"])
        for t_col in time_cols:
            if t_col in row and pd.notna(row[t_col]):
                hour = int(t_col[11:13])
                result[(node, hour)] = int(round(float(row[t_col])))
    return result


def _compute_scaled_pct(df: pd.DataFrame) -> pd.Series:
    """kVA FFOR gain per 1% relaxation of each constraint's own bound.

    For constraints with RHS ≠ 0:  |Pi| × 0.01 × |RHS| × 630
    For RHS = 0 (e.g. ratio constraints as f(x) ≤ 0):
        |Pi| × type_scale × 0.01  (1% of one natural unit)
    """
    rhs_abs  = df["rhs"].abs()
    has_rhs  = rhs_abs > 1e-6
    fallback = df["constr_type"].map(
        lambda c: _CONSTR_SCALE.get(c, (630, ""))[0]
    ) * 0.01
    return np.where(
        has_rhs,
        df["dual"].abs() * 0.01 * rhs_abs * 630,
        df["dual"].abs() * fallback,
    )


def analyse_from_results(
    results_folder: str, conf: config.Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract LP shadow prices from saved MIP direction results.

    Strategy — fix and re-solve:
      1. Build the full OPF model once (direction a=1, b=0 as placeholder).
      2. Relax all binary/integer variables to continuous.
      3. For each saved direction CSV: fix b_bess_charge from MIP solution,
         update objective (a, b), solve LP → read c.Pi shadow prices.

    Returns
    -------
    full_df  : one row per (constraint, direction)
    summary  : aggregated statistics per constraint type, sorted by sum_scaled_pct
    """
    import gurobipy as gp
    from gurobipy import GRB

    folder = Path(results_folder)
    direction_files = sorted(folder.glob("results_node_t_a*_b*.csv"))
    if not direction_files:
        raise FileNotFoundError(f"No direction result files found in {folder}")

    Sbase_kW = conf.S_base * 1000.0

    model = _build_model(conf, 1.0, 0.0)
    model.update()
    model.Params.DualReductions = 0

    # Relax all binary/integer variables; track BESS charge indicators by key
    bess_vars: dict[tuple[int, int], gp.Var] = {}
    for v in model.getVars():
        if v.VType in (GRB.BINARY, GRB.INTEGER):
            mm = re.match(r"b_bess_charge\[(\d+),(\d+)\]", v.VarName)
            if mm:
                bess_vars[(int(mm.group(1)), int(mm.group(2)))] = v
            v.LB = round(v.LB)
            v.UB = round(v.UB)
            v.VType = GRB.CONTINUOUS
    model.update()

    p_flex_var = model.getVarByName("p_flex_total")
    q_flex_var = model.getVarByName("q_flex_total")

    frames = []
    for csv_path in direction_files:
        m = _DIR_FILE_RE.search(csv_path.name)
        if not m:
            continue
        a, b = float(m.group(1)), float(m.group(2))
        print(f"  direction a={a:+.4f}, b={b:+.4f} ...", end=" ", flush=True)

        # Fix BESS binaries from this direction's saved MIP values
        b_charge_vals = _load_b_bess_charge(csv_path, conf.time_col_list)
        for (node, hour), v in bess_vars.items():
            val = b_charge_vals.get((node, hour), 0)
            v.LB = val
            v.UB = val

        model.setObjective(
            -a * p_flex_var / Sbase_kW - b * q_flex_var / Sbase_kW,
            GRB.MINIMIZE,
        )
        model.update()
        model.optimize()
        if model.status != GRB.OPTIMAL:
            print(f"failed (status={model.status})")
            continue

        df = pd.DataFrame([
            {
                "constr_name": c.ConstrName,
                "constr_type": _classify(c.ConstrName),
                "dual":  c.Pi,
                "slack": c.Slack,
                "rhs":   c.RHS,
            }
            for c in model.getConstrs()
        ])
        df["a"] = a
        df["b"] = b
        frames.append(df)
        print("done")

    full_df = pd.concat(frames, ignore_index=True)
    full_df["scaled_pct"] = _compute_scaled_pct(full_df)

    summary = (
        full_df.groupby("constr_type")
        .agg(
            mean_abs_dual   = ("dual", lambda x: x.abs().mean()),
            max_abs_dual    = ("dual", lambda x: x.abs().max()),
            mean_scaled_pct = ("scaled_pct", "mean"),
            max_scaled_pct  = ("scaled_pct", "max"),
            sum_scaled_pct  = ("scaled_pct", "sum"),
            n_binding       = ("dual", lambda x: (x.abs() > 1e-6).sum()),
            n_total         = ("dual", "count"),
        )
        .assign(pct_binding=lambda d: 100 * d["n_binding"] / d["n_total"])
        .sort_values("sum_scaled_pct", ascending=False)
        .reset_index()
    )
    return full_df, summary


# 2. FFOR time-horizon plot

def plot_ffor_time_horizons(save_dir: str = "2703_23_homogen/explorations/time_horizon"):
    """Overlay FFOR hulls for T = 0.5 h … 8 h for summer and winter.

    One figure per (year, no_ev) combination, two subplots (summer / winter).
    Reads results_pq_flex_points.csv from each matching results folder.
    """
    results = utils.extract_results_parameters_from_scenario(r"2703_23_homogen/results")

    DURATIONS   = [0.5, 1.0, 2.0, 4.0, 8.0]   # 12 h excluded
    DAYS        = [(7, 26), (2, 5)]     # (month, day): summer, winter
    START_HOURS = [12]
    YEARS       = [2030, 2050]
    NO_EV_LIST  = [False, True]

    color_map = {
        0.5: "#1f77b4", 1.0: "#ff7f0e", 2.0: "#2ca02c",
        4.0: "#d62728", 8.0: "#9467bd",
    }
    label_map    = {0.5: "0.5 h", 1.0: "1 h", 2.0: "2 h", 4.0: "4 h", 8.0: "8 h"}
    season_label = {7: "Summer", 2: "Winter"}

    lookup = {}
    for r in results:
        if r["pv_weather"] != "pvavg":
            continue
        key = (
            r["analysis_year"], r["analysis_month"], r["analysis_day"],
            r["analysis_start_hour"], r["analysis_duration_hours"], r["no_ev"],
        )
        lookup[key] = r

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        for no_ev in NO_EV_LIST:
            # Shared axis limits across all subplots
            all_x, all_y = [], []
            for month, day in DAYS:
                for start_hour in START_HOURS:
                    for duration in DURATIONS:
                        key = (year, month, day, start_hour, duration, no_ev)
                        if key not in lookup:
                            continue
                        pts_file = Path(lookup[key]["output_folder"]) / "results_pq_flex_points.csv"
                        if not pts_file.is_file():
                            continue
                        pts = np.loadtxt(pts_file, delimiter=",", skiprows=1)
                        if len(pts) < 3:
                            continue
                        all_x.extend(pts[:, 0])
                        all_y.extend(pts[:, 1])

            if all_x:
                margin = 0.05
                xc   = (min(all_x) + max(all_x)) / 2
                yc   = (min(all_y) + max(all_y)) / 2
                half = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2 * (1 + margin)
                shared_xlim = (xc - half, xc + half)
                shared_ylim = (yc - half, yc + half)
            else:
                shared_xlim = shared_ylim = None

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))

            for ax_idx, (month, day) in enumerate(DAYS):
                ax = axes[ax_idx]
                for start_hour in START_HOURS:
                    ax.set_title(
                        f"{season_label[month]} ({month:02d}/{day:02d}), start = {start_hour:02d}:00"
                    )
                    ax.set_xlabel(r"$P_\mathrm{flex}$ [kW]")
                    ax.set_ylabel(r"$Q_\mathrm{flex}$ [kVAr]")
                    ax.grid(True, alpha=0.3)
                    ax.axhline(0, color="black", linewidth=0.5)
                    ax.axvline(0, color="black", linewidth=0.5)

                    for duration in DURATIONS:
                        key = (year, month, day, start_hour, duration, no_ev)
                        if key not in lookup:
                            continue
                        pts_file = Path(lookup[key]["output_folder"]) / "results_pq_flex_points.csv"
                        if not pts_file.is_file():
                            continue
                        pts = np.loadtxt(pts_file, delimiter=",", skiprows=1)
                        if len(pts) < 3:
                            continue
                        hull  = ConvexHull(pts)
                        verts = np.append(hull.vertices, hull.vertices[0])
                        color = color_map[duration]
                        ax.plot(pts[verts, 0], pts[verts, 1],
                                color=color, linewidth=2, label=label_map[duration])
                        ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                                color=color, alpha=0.08)

                    if shared_xlim is not None:
                        ax.set_xlim(shared_xlim)
                        ax.set_ylim(shared_ylim)
                    ax.legend(title="Duration", fontsize=9)
                    ax.set_aspect("equal")

            suffix = "_noev" if no_ev else ""
            fig.suptitle(
                f"FFOR Time-Horizon Comparison — {year}{suffix.replace('_', ' ')}",
                fontsize=13,
            )
            fig.tight_layout()
            out_path = out_dir / f"ffor_time_horizons_{year}{suffix}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")


# Helper: hull geometry

def _hull_ordered_verts(hull_pts: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Return hull vertex indices sorted counterclockwise by angle from centroid."""
    cx, cy = hull_pts[vertices].mean(axis=0)
    angles = np.arctan2(hull_pts[vertices, 1] - cy, hull_pts[vertices, 0] - cx)
    return vertices[np.argsort(angles)]


def _dominant_per_direction(
    full_df: pd.DataFrame,
    results_folder: str,
    conf: config.Config,
    agg: str = "sum",
) -> pd.DataFrame:
    """Dominant binding constraint per FFOR boundary direction.

    Returns DataFrame with columns: a, b, P_flex, Q_flex, dominant.
    Co-dominance: types reaching ≥ 70% of the top are joined as 'type_a + type_b'.

    agg options: 'max', 'sum', 'mean_binding'
    """
    folder   = Path(results_folder)
    year_str = str(conf.analysis_year)
    records  = []

    for csv_path in sorted(folder.glob("results_node_t_a*_b*.csv")):
        m = _DIR_FILE_RE.search(csv_path.name)
        if not m:
            continue
        a, b = float(m.group(1)), float(m.group(2))

        df     = pd.read_csv(csv_path)
        t_col  = next((c for c in df.columns if year_str in c), None)
        if t_col is None:
            continue

        def flex_sum(variables):
            mask = df["Variable"].isin(variables)
            return pd.to_numeric(df.loc[mask, t_col], errors="coerce").sum(skipna=True)

        p_flex = flex_sum(["p_pv_flex", "p_hp_flex", "p_bess_flex", "p_ev_flex"])
        q_flex = flex_sum(["q_pv_flex", "q_hp_flex", "q_bess_flex"])

        dir_mask = np.isclose(full_df["a"], a) & np.isclose(full_df["b"], b)
        dir_ineq = full_df[
            dir_mask
            & ~full_df["constr_type"].isin(_EQUALITY_TYPES)
            & ~full_df["constr_type"].isin(_WEATHER_TYPES)
        ].copy()

        if agg == "max":
            by_type = dir_ineq.groupby("constr_type")["scaled_pct"].max()
        elif agg == "sum":
            by_type = dir_ineq.groupby("constr_type")["scaled_pct"].sum()
        elif agg == "mean_binding":
            binding = dir_ineq[dir_ineq["dual"].abs() > 1e-6]
            by_type = binding.groupby("constr_type")["scaled_pct"].mean()
        else:
            raise ValueError(f"Unknown agg: {agg!r}")

        if by_type.empty or by_type.max() <= 1e-8:
            dominant = "unconstrained"
        else:
            threshold = 0.7 * by_type.max()
            co        = sorted(by_type[by_type >= threshold].index.tolist())
            dominant  = " + ".join(co)

        records.append({"a": a, "b": b, "P_flex": p_flex, "Q_flex": q_flex, "dominant": dominant})

    return pd.DataFrame(records)


# 3a. Binding constraint summary bar chart

def plot_binding_summary(summary: pd.DataFrame, conf: config.Config, save_dir: str | None = None):
    """Two-panel bar chart: investment signals vs. operational signals."""
    actionable = summary[
        ~summary["constr_type"].isin(_EQUALITY_TYPES)
        & ~summary["constr_type"].isin(_WEATHER_TYPES)
        & (summary["n_binding"] > 0)
    ].copy()

    if actionable.empty:
        print("No binding inequality constraints found.")
        return

    invest = actionable[actionable["constr_type"].isin(_INVESTMENT_TYPES)].head(6)
    operat = actionable[~actionable["constr_type"].isin(_INVESTMENT_TYPES)].head(6)

    year  = conf.analysis_year
    month = conf.analysis_month
    day   = conf.analysis_day
    sh    = conf.analysis_start_hour
    dur   = conf.analysis_n_timesteps * conf.delta_t

    n_rows = max(len(invest), len(operat), 2)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n_rows * 0.9 + 1.5)))

    panels = [
        (axes[0], invest, "#E53935", "Investment Signal\n(hardware upgrade required)"),
        (axes[1], operat, "#1E88E5", "Operational Signal\n(relax parameter today)"),
    ]
    for ax, df, color, title in panels:
        ax.set_title(title, fontsize=11)
        if df.empty:
            ax.text(0.5, 0.5, "no binding\nconstraints", ha="center",
                    va="center", transform=ax.transAxes, color="gray")
            ax.set_xlabel(r"$\Sigma$ kVA FFOR gain per 1% bound relaxation")
            ax.grid(axis="x", alpha=0.3)
            continue
        raw_labels = df["constr_type"].values[::-1]
        labels = [_CONSTR_LABEL.get(l, l) for l in raw_labels]
        vals   = df["sum_scaled_pct"].values[::-1]
        ax.barh(labels, vals, color=color)
        ax.set_xlabel(r"$\Sigma$ kVA FFOR gain per 1% bound relaxation")
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ev_suffix = "_noev" if getattr(conf, "no_ev", False) else ""
        fname = f"{save_dir}/dual_binding_{year}{month:02d}{day:02d}_{sh:02d}_{dur}{ev_suffix}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"Saved {fname}")
    plt.close(fig)


# 3b. EV comparison: binding constraint bar chart + clean hull overlay

def plot_ev_comparison(scenario_results: list[dict], save_dir: str):
    """Grouped bar chart comparing binding constraints with vs. without EV.

    scenario_results: list of two dicts, each with keys:
        conf, full_df, summary, results_folder, ev_label ('withev' / 'noev')
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    conf0   = scenario_results[0]["conf"]
    dur     = conf0.analysis_n_timesteps * conf0.delta_t
    date_str = f"{conf0.analysis_year}-{conf0.analysis_month:02d}-{conf0.analysis_day:02d}"
    tag      = f"{conf0.analysis_year}{conf0.analysis_month:02d}{conf0.analysis_day:02d}_12_{dur}h"

    summaries = []
    for sr in scenario_results:
        s = sr["summary"][
            ~sr["summary"]["constr_type"].isin(_EQUALITY_TYPES)
            & ~sr["summary"]["constr_type"].isin(_WEATHER_TYPES)
            & (sr["summary"]["n_binding"] > 0)
        ][["constr_type", "sum_scaled_pct"]].copy()
        s["label"] = sr["ev_label"]
        summaries.append(s)
    df_cmp = pd.concat(summaries, ignore_index=True)

    totals = df_cmp.groupby("constr_type")["sum_scaled_pct"].sum().sort_values(ascending=False)
    nosimt_total   = totals.filter(like="bess_no_simult").max()
    always_include = {"ev_lb"}
    totals = totals[(totals > nosimt_total) | totals.index.isin(always_include)]
    totals = totals[totals > 0]
    order  = totals.sort_values(ascending=True).index.tolist()

    n     = len(order)
    y     = np.arange(n)
    bar_h = 0.35
    readable_order = [_CONSTR_LABEL.get(t, t) for t in order]
    colors = {"withev": "#2196F3", "noev": "#FF5722"}

    fig, ax = plt.subplots(figsize=(11, max(4, n * 1.0 + 1.5)))
    for i, sr in enumerate(scenario_results):
        lbl  = sr["ev_label"]
        vals = [
            df_cmp.loc[(df_cmp["constr_type"] == t) & (df_cmp["label"] == lbl), "sum_scaled_pct"].sum()
            for t in order
        ]
        offset = (i - 0.5) * bar_h
        ax.barh(y + offset, vals, height=bar_h, color=colors[lbl],
                label="with EV" if lbl == "withev" else "without EV")

    ax.set_yticks(y)
    ax.set_yticklabels(readable_order, fontsize=10)
    ax.set_xlabel(
        r"$\Sigma$ kVA FFOR gain per 1% bound relaxation (summed over all directions & instances)"
    )
    ax.set_title(f"Binding Constraint Comparison with / without EV — {date_str}, T={dur}h")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    fname = f"{save_dir}/constraints_ev_compare_{tag}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_ev_hull_clean(
    conf: config.Config,
    results_folder_ev: str,
    results_folder_noev: str,
    full_df_ev: pd.DataFrame,
    save_dir: str | None = None,
):
    """FFOR hull overlay: with-EV boundary coloured by dominant binding constraint.

    The without-EV hull is shown as a light orange fill with a dashed outline.
    The with-EV hull is filled white, creating a visible orange 'halo' where the
    EV obligations reduce the FFOR.  The with-EV boundary is segmented by colour
    according to the dominant constraint in each direction.
    """
    folder_ev   = Path(results_folder_ev)
    folder_noev = Path(results_folder_noev)

    hull_pts_ev   = np.loadtxt(folder_ev   / "results_pq_flex_points.csv", delimiter=",", skiprows=1)
    hull_pts_noev = np.loadtxt(folder_noev / "results_pq_flex_points.csv", delimiter=",", skiprows=1)

    hull_ev   = ConvexHull(hull_pts_ev)
    hull_noev = ConvexHull(hull_pts_noev)
    ordered_ev   = _hull_ordered_verts(hull_pts_ev,   hull_ev.vertices)
    ordered_noev = _hull_ordered_verts(hull_pts_noev, hull_noev.vertices)

    pts_df = _dominant_per_direction(full_df_ev, results_folder_ev, conf, "sum")

    # Per-constraint-family colors (semantic, independent of tab10 ordering)
    def _family(part: str) -> str:
        p = part.lower()
        if "line_s"       in p: return "line"
        if "t_hp"         in p: return "hp_temp"
        if "hp_power"     in p: return "hp_pwr"
        if "ev"           in p: return "ev"
        if "bess_apparent"in p: return "bess_s"
        if "soc"          in p: return "soc"
        if "pv"           in p: return "pv"
        if "bess"         in p: return "bess_op"
        return "other"

    _FAM_COLORS: dict[str, np.ndarray] = {
        "line":    np.array([0.78, 0.10, 0.10]),
        "hp_temp": np.array([0.90, 0.42, 0.00]),
        "hp_pwr":  np.array([0.95, 0.60, 0.10]),
        "ev":      np.array([0.00, 0.47, 0.45]),
        "bess_s":  np.array([0.45, 0.10, 0.60]),
        "soc":     np.array([0.08, 0.38, 0.75]),
        "pv":      np.array([0.30, 0.55, 0.15]),
        "bess_op": np.array([0.70, 0.45, 0.72]),
        "other":   np.array([0.60, 0.60, 0.60]),
    }

    def _label_color(lbl: str) -> tuple:
        if lbl == "unconstrained":
            return (0.7, 0.7, 0.7)
        parts = lbl.split(" + ")
        colors = [_FAM_COLORS.get(_family(p), _FAM_COLORS["other"]) for p in parts]
        return tuple(np.mean(colors, axis=0))

    def _clean_label(raw: str) -> str:
        return " + ".join(_CONSTR_LABEL.get(p, p) for p in raw.split(" + "))

    all_labels = sorted(pts_df["dominant"].unique())
    color_map  = {lbl: _label_color(lbl) for lbl in all_labels}

    # Assign dominant constraint to each hull edge via angle-matching from centroid
    cx, cy = hull_pts_ev[ordered_ev].mean(axis=0)
    n      = len(ordered_ev)

    edge_angles = np.array([
        np.arctan2(
            (hull_pts_ev[ordered_ev[i], 1] + hull_pts_ev[ordered_ev[(i + 1) % n], 1]) / 2 - cy,
            (hull_pts_ev[ordered_ev[i], 0] + hull_pts_ev[ordered_ev[(i + 1) % n], 0]) / 2 - cx,
        )
        for i in range(n)
    ])
    dir_angles = np.arctan2(pts_df["Q_flex"].values - cy, pts_df["P_flex"].values - cx)

    edge_dominants: list[str] = []
    for ea in edge_angles:
        diffs = np.abs(dir_angles - ea)
        diffs = np.minimum(diffs, 2 * np.pi - diffs)   # wraparound
        edge_dominants.append(pts_df.iloc[int(diffs.argmin())]["dominant"])

    # Draw
    fig, ax = plt.subplots(figsize=(8, 8))

    # Without EV: orange fill + dashed outline
    ax.fill(hull_pts_noev[ordered_noev, 0], hull_pts_noev[ordered_noev, 1],
            color="#FF5722", alpha=0.22, zorder=1)
    noev_verts = np.append(ordered_noev, ordered_noev[0])
    ax.plot(hull_pts_noev[noev_verts, 0], hull_pts_noev[noev_verts, 1],
            color="#FF5722", linewidth=1.8, linestyle="--", zorder=4)

    # With EV: white fill to create orange halo in the difference zone
    ax.fill(hull_pts_ev[ordered_ev, 0], hull_pts_ev[ordered_ev, 1],
            color="white", alpha=0.88, zorder=2)

    # With EV boundary: segment-by-segment coloured by dominant constraint
    seen_labels: set[str] = set()
    for i, dominant in enumerate(edge_dominants):
        v1 = hull_pts_ev[ordered_ev[i]]
        v2 = hull_pts_ev[ordered_ev[(i + 1) % n]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]],
                color=color_map[dominant], linewidth=3.5,
                solid_capstyle="round", zorder=5)
        seen_labels.add(dominant)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$P_\mathrm{flex}$ [kW]")
    ax.set_ylabel(r"$Q_\mathrm{flex}$ [kVAr]")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    legend_handles: list = [
        mpatches.Patch(facecolor="#FF5722", alpha=0.55, edgecolor="#FF5722",
                       linewidth=1.5, label="without EV"),
        mpatches.Patch(facecolor="white", edgecolor="gray",
                       linewidth=1, label="with EV"),
        mpatches.Patch(color="none", label=""),
        mpatches.Patch(color="none", label="Binding Constraint (with EV):"),
    ]
    for lbl in sorted(seen_labels):
        legend_handles.append(
            mlines.Line2D([], [], color=color_map[lbl], linewidth=4,
                          label=_clean_label(lbl))
        )
    ax.legend(handles=legend_handles, fontsize=9, loc="center", framealpha=0.90)

    fig.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        dur  = conf.analysis_n_timesteps * conf.delta_t
        tag  = (f"{conf.analysis_year}{conf.analysis_month:02d}{conf.analysis_day:02d}"
                f"_{conf.analysis_start_hour:02d}_{dur}h")
        fname = f"{save_dir}/ffor_ev_hull_clean_{tag}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
    plt.close(fig)


# Entry point

if __name__ == "__main__":
    DUALS_DIR   = "2703_23_homogen/explorations/duals"
    COMPARE_DIR = "2703_23_homogen/explorations/EV_compare"

    # Scenarios for dual analysis.  Add no_ev=True variants to enable EV comparison.
    SCENARIOS = [
        {
            "results_folder": "2703_23_homogen/results/results_20500205_12_8.0_pvavg_20260609_00_08_39",
            "no_ev": False,
            "year": 2050, "month": 2, "day": 5, "start_hour": 12,
            "n_timesteps": 8, "delta_t": 1.0,
        },
        {
            "results_folder": "2703_23_homogen/results/results_20500205_12_8.0_pvavg_noev_20260609_00_08_39",
            "no_ev": True,
            "year": 2050, "month": 2, "day": 5, "start_hour": 12,
            "n_timesteps": 8, "delta_t": 1.0,
        },
    ]

    # Dual analysis — load from cache or run
    collected = []
    for sc in SCENARIOS:
        conf = config.Config(
            year=sc["year"], month=sc["month"], day=sc["day"],
            start_hour=sc["start_hour"], n_timesteps=sc["n_timesteps"],
            delta_t=sc["delta_t"], pv_weather="pvavg", no_ev=sc["no_ev"],
        )
        ev_label = "noev" if sc["no_ev"] else "withev"
        dur_h    = conf.analysis_n_timesteps * conf.delta_t
        tag      = (f"{conf.analysis_year}{conf.analysis_month:02d}{conf.analysis_day:02d}"
                    f"_{conf.analysis_start_hour:02d}_{dur_h}h_{ev_label}")
        full_csv = Path(f"{DUALS_DIR}/dual_full_{tag}.csv")
        summ_csv = Path(f"{DUALS_DIR}/dual_summary_{tag}.csv")

        if full_csv.exists() and summ_csv.exists():
            print(f"Loading cached: {tag}")
            full_df = pd.read_csv(full_csv)
            summary = pd.read_csv(summ_csv)
        else:
            print(f"\nRunning dual analysis: {tag}")
            full_df, summary = analyse_from_results(sc["results_folder"], conf)
            Path(DUALS_DIR).mkdir(parents=True, exist_ok=True)
            full_df.to_csv(full_csv, index=False)
            summary.to_csv(summ_csv, index=False)
            print(f"Cached to {full_csv}")

        plot_binding_summary(summary, conf, save_dir=DUALS_DIR)

        collected.append({
            "conf":           conf,
            "full_df":        full_df,
            "summary":        summary,
            "results_folder": sc["results_folder"],
            "ev_label":       ev_label,
        })

    # EV comparison: bar chart (with/without EV) + clean hull overlay
    from collections import defaultdict
    date_groups: dict = defaultdict(list)
    for item in collected:
        c     = item["conf"]
        dur_h = c.analysis_n_timesteps * c.delta_t
        date_groups[(c.analysis_year, c.analysis_month, c.analysis_day, dur_h)].append(item)

    for _, group in date_groups.items():
        ev_labels = [g["ev_label"] for g in group]
        if len(group) == 2 and len(set(ev_labels)) == 2:
            plot_ev_comparison(group, save_dir=COMPARE_DIR)

    # Clean hull overlay — winter T=8h
    _NOEV_FOLDERS = {
        (2050, 2, 5, 12, 8.0): "2703_23_homogen/results/results_20500205_12_8.0_pvavg_noev_20260609_00_08_39",
    }
    for item in collected:
        c = item["conf"]
        if item["ev_label"] != "withev":
            continue
        dur_h    = c.analysis_n_timesteps * c.delta_t
        noev_key = (c.analysis_year, c.analysis_month, c.analysis_day, c.analysis_start_hour, dur_h)
        noev_folder = _NOEV_FOLDERS.get(noev_key)
        if noev_folder:
            plot_ev_hull_clean(
                conf=c,
                results_folder_ev=item["results_folder"],
                results_folder_noev=noev_folder,
                full_df_ev=item["full_df"],
                save_dir=COMPARE_DIR,
            )

    # FFOR time-horizon plot
    plot_ffor_time_horizons()
