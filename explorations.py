import datetime as dt
import pandas as pd
from pathlib import Path

import utils
import plots


def exploration_seasonality():
    """Ugly function to explore seasonality for different parameters."""
    # get result dictionaries
    results = utils.extract_results_parameters_from_scenario(r"2703_23_homogen/results")
    filtered_results = [
        result 
        for result in results 
        if result["analysis_year"] == 2050 and result["analysis_start_hour"] == 12 and result["pv_weather"] == "pvcld" and result["analysis_n_timesteps"] == 2
    ]
    print(filtered_results[0])
    
    # get min and max P_flex for every result
    min_max_p_flex = {}
    for result in filtered_results:
        print(f"Processing {result['output_folder']}...")
        print(f"PQ_result_file: {Path(result['output_folder']) / 'results_pq_flex_points.csv'}")

        date = dt.date(result["analysis_year"], result["analysis_month"], result["analysis_day"])
        output_file = Path(result["output_folder"]) / "results_pq_flex_points.csv"
        if not output_file.is_file():
            print(f"Warning: {output_file} does not exist, skipping.")
            continue
        else:
            min_max_p_flex[date] = utils._extract_min_max_p_flex_from_points_file(output_file)
    
    # convert to df and plot seasonality
    df = pd.DataFrame.from_dict(min_max_p_flex, orient="index", columns=["min_p_flex", "max_p_flex"])
    plots.plot_seasonal_p_flex(df)


if __name__ == "__main__":
    exploration_seasonality()