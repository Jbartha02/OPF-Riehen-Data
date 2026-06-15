import re
from pathlib import Path

import datetime as dt
import numpy as np
import pandas as pd

import plots


# Regular expression to parse results folder names
_RESULTS_FOLDER_RE = re.compile(
    r"^results_"
    r"(?P<analysis_date>\d{8})_"
    r"(?P<analysis_start_hour>\d{2})_"
    r"(?P<analysis_duration_hours>\d+(?:\.\d+)?)_"
    r"(?P<pv_weather>pvavg|pvcld|pvsun)"
    r"(?P<no_ev>_noev)?"
    r"(?P<no_bess>_nobess)?"
    r"(?P<run_simple_ffor>_simpleffor)?_"
    r"(?P<created_at>\d{8}_\d{2}_\d{2}_\d{2})$"
)

def _parse_results_output_folder_name(output_folder_name: str,) -> dict[str, object] | None:
    """Parse a single results folder name into the parameters encoded in it."""
    match = _RESULTS_FOLDER_RE.match(output_folder_name)
    if match is None:
        print(f"Warning: '{output_folder_name}' does not match the expected results folder naming convention, skipping.")
        return None

    analysis_date = match.group("analysis_date")
    analysis_duration_hours = float(match.group("analysis_duration_hours"))
    rounded_duration_hours = round(analysis_duration_hours)

    if (
        np.isclose(analysis_duration_hours, rounded_duration_hours)
        and analysis_duration_hours >= 1
    ):
        analysis_n_timesteps = int(rounded_duration_hours)
        delta_t = 1.0
    else:
        analysis_n_timesteps = 1
        delta_t = analysis_duration_hours

    return {
        "analysis_year": int(analysis_date[:4]),
        "analysis_month": int(analysis_date[4:6]),
        "analysis_day": int(analysis_date[6:8]),
        "analysis_start_hour": int(match.group("analysis_start_hour")),
        "analysis_n_timesteps": analysis_n_timesteps,
        "delta_t": delta_t,
        "analysis_duration_hours": analysis_duration_hours,
        "pv_weather": match.group("pv_weather"),
        "no_ev": match.group("no_ev") is not None,
        "no_bess": match.group("no_bess") is not None,
        "run_simple_ffor": match.group("run_simple_ffor") is not None,
        "created_at": match.group("created_at"),
    }


def extract_results_parameters_from_scenario(scenario_folder: str) -> list[dict[str, object]]:
    """Return the encoded parameters for every results folder found under a scenario folder.

    The function walks the scenario folder recursively, finds directories whose
    names follow the results naming convention, and returns the decoded
    parameters together with the output folder path.
    """
    scenario_path = Path(scenario_folder)
    results: list[dict[str, object]] = []

    for output_folder in sorted(
        path for path in scenario_path.rglob("results_*") if path.is_dir()
    ):
        parsed = _parse_results_output_folder_name(output_folder.name)
        if parsed is None:
            print(f"Warning: Skipping '{output_folder}' because its name could not be parsed.")
            continue

        parsed["output_folder"] = str(output_folder)
        results.append(parsed)
    
    # check for duplicate results
    start_index = len(scenario_folder) + 1  # f"{remove scenario_folder}\" prefix of output_folder
    end_index = -13  # ignore the created_at suffix (except for year)
    for result in results:
        _find_duplicate_results(scenario_folder, result["output_folder"][start_index:end_index])  # check for duplicates with same prefix (without created_at)

    return results

def _find_duplicate_results(scenario_results_folder: str, result_prefix: str):
    """Check if there are multiple files with the same prefix in the results folder."""
    results_path = Path(scenario_results_folder)
    matching_dirs = [dir for dir in results_path.glob(f"{result_prefix}*") if dir.is_dir()]
    if len(matching_dirs) > 1:
        print(f"Warning: Found multiple files with prefix '{result_prefix}' in '{scenario_results_folder}':")
        for file in matching_dirs:
            print(f" - {file}")
    elif len(matching_dirs) == 0:
        print(f"Warning: No files found with prefix '{result_prefix}' in '{scenario_results_folder}'.")

def _extract_min_max_p_flex_from_points_file(points_file: str) -> tuple[float, float] | None:
    """Extract the minimum and maximum P_flex values from a results_pq_flex_points.csv file."""
    try:
        df = pd.read_csv(points_file)
        if df.empty:
            print(f"Warning: {points_file} is empty.")
            return None
        return (df["P_flex"].min(), df["P_flex"].max())
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(f"Error reading {points_file}")
        return None

def _extract_nodal_results_to_df(conf, var, varname) -> pd.DataFrame:
    """Create a node-wise result table from a 2D Gurobi variable indexed by (node, timestep).

    The returned DataFrame has one row per node (same order/index as node_metadata_df),
    includes LV_grid and LV_osmid, as well as Variable and Node columns, and one column per timestep.
    Missing (node, timestep) entries are kept as null values.
    """

    # Start from node metadata so every node is present exactly once.
    df = conf.node_metadata_df[["LV_grid", "LV_osmid"]].copy()
    
    # Add columns 'Variable' and 'Node' for identification of values
    df = df.reset_index().rename(columns={"index": "Node"})
    df["Variable"] = varname

    # Add one column per timestep and initialize as null.
    for t in conf.time_index_list:
        df[conf.time_col_list[t]] = pd.NA

    # Fill values for (node, timestep).
    if var is not None:
        for node in df["Node"]:
            for t in conf.time_index_list:
                value = var.get((node, t)) if hasattr(var, "get") else var[node, t]
                df.at[node, conf.time_col_list[t]] = value.X if hasattr(value, "X") else value

    return df

def _extract_edge_results_to_df(conf, var, varname) -> pd.DataFrame:
    """Create an edge-wise result table from a 3D Gurobi variable indexed by (u, v, timestep).

    The returned DataFrame has one row per edge (same order/index as edges_df),
    includes u_osmid and v_osmid, as well as Variable, u_idx, and v_idx columns, and one column per timestep.
    Missing (u, v, timestep) entries are kept as null values.
    """

    # Start from edges_df so every edge is present exactly once.
    df = conf.edges_metadata_df[["u_osmid", "v_osmid", "u_idx", "v_idx", "s_nom"]].copy()
    
    # Add column 'Variable' for identification of values
    df["Variable"] = varname

    # Add one column per timestep and initialize as null.
    for t in conf.time_index_list:
        df[conf.time_col_list[t]] = pd.NA

    # Fill values for (u, v, timestep).
    if var is not None:
        for idx, row in df.iterrows():
            u, v = int(row["u_idx"]), int(row["v_idx"])
            for t in conf.time_index_list:
                value = var.get((u, v, t)) if hasattr(var, "get") else var[u, v, t]
                df.at[idx, conf.time_col_list[t]] = value.X if hasattr(value, "X") else value

    return df



if __name__ == "__main__":
    # USAGE EXAMPLE to extract all result folders and filter them by some parameters:
    results = extract_results_parameters_from_scenario(r"2703_23_homogen/results")
    filtered_results = [
        result 
        for result in results 
        if result["analysis_year"] == 2050 and result["analysis_start_hour"] == 12 and result["pv_weather"] == "pvsun" and result["analysis_n_timesteps"] == 1 and result["delta_t"] == 0.5
    ]

    for result in filtered_results:
        print(f"Processing {result['output_folder']}...")
        print(f"PQ_result_file: {Path(result['output_folder']) / 'results_pq_flex_points.csv'}")
        
        # do something (e.g. read min P_FLEX and max P_FLEX from results_pq_flex_points.csv and use them for a plot)
