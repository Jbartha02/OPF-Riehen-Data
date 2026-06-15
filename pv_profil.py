"""
pv_profil.py
create sunny and cloudy pv generation profiles from mean ± k*stddev.

  sunny:   pv_sun    = min(μ + K_SUN  * σ,  P_installed)
  cloudy:   pv_clouds = max(μ - K_RAIN * σ,  0)

output (same format as pv_generation.csv):
  <Jahresordner>/pv_sun.csv    — sunny
  <Jahresordner>/pv_clouds.csv — cloudy

usage in config.py:
  "PV_ub":   "pv_sun.csv"    (instead of "pv_generation.csv")
  "PV_base": "pv_clouds.csv" (instead of "pv_generation.csv")
"""

import os
import numpy as np
import pandas as pd

# Parameter 

BASE_FOLDERS = [
    "2703_23_homogen",
]

YEARS = [2030, 2040, 2050]

# Multiplikator für die Standardabweichung
#
#  k        sunny                           cloudy
#  -----    ----------------------------    ----------------------------
#  k = 0.5  leicht sonnig  (Top 31 %)      leicht trüb    (Bot. 31 %)
#  k = 1.0  sonnig         (Top 16 %)      bewölkt        (Bot. 16 %)
#  k = 1.5  gut sonnig     (Top  7 %)      trüb           (Bot.  7 %)
#  k = 2.0  sehr sonnig    (Top  2 %)      Schlechtwetter (Bot.  2 %)
#
K_SUN   = 1.0   # sunny:   μ + K_SUN  * σ
K_RAIN  = 1.0   # cloudy:   μ - K_RAIN * σ  (floor: 0)

# ─────────────────────────────────────────────────────────────────────────────


def create_pv_profiles(
    base_folders: list[str],
    years: list[int],
    k_sun: float,
    k_rain: float,
) -> None:
    print(f"Sonnenprofil:  μ + {k_sun} · σ")
    print(f"Wolkenprofil:  max(μ - {k_rain} · σ, 0)\n")

    for base in base_folders:
        for year in years:
            folder = os.path.join(base, str(year))

            path_gen = os.path.join(folder, "pv_generation.csv")
            path_std = os.path.join(folder, "pv_std.csv")
            path_ins = os.path.join(folder, "pv_p_installed.csv")

            for p in [path_gen, path_std, path_ins]:
                if not os.path.exists(p):
                    print(f"  Übersprungen — Datei fehlt: {p}")
                    continue

            df_gen = pd.read_csv(path_gen)
            df_std = pd.read_csv(path_std)
            df_ins = pd.read_csv(path_ins)

            id_cols   = ["LV_grid", "LV_osmid"]
            time_cols = [c for c in df_gen.columns if c not in id_cols]

            mu    = df_gen[time_cols].to_numpy()
            sigma = df_std[time_cols].to_numpy()

            # P_installed as upper bound: shape (n_nodes, 1) for broadcast
            p_max = (
                df_gen[id_cols]
                .merge(df_ins, on=id_cols, how="left")["P_installed_kW"]
                .to_numpy()[:, np.newaxis]
            )
            p_max = np.where(np.isnan(p_max), np.inf, p_max)

            sun    = np.minimum(mu + k_sun * sigma, p_max)
            clouds = np.maximum(mu - k_rain * sigma, 0)

            def _save(array: np.ndarray, filename: str) -> str:
                df_out = pd.concat(
                    [df_gen[id_cols].reset_index(drop=True),
                     pd.DataFrame(array, columns=time_cols)],
                    axis=1,
                )
                out_path = os.path.join(folder, filename)
                df_out.to_csv(out_path, index=False)
                return out_path

            _save(sun,    "pv_sun.csv")
            _save(clouds, "pv_clouds.csv")

            noon_idx   = [i for i, c in enumerate(time_cols) if "12:00" in c]
            mean_sun   = sun[:,    noon_idx].mean()
            mean_base  = mu[:,     noon_idx].mean()
            mean_cloud = clouds[:, noon_idx].mean()

            print(
                f"  {folder}/\n"
                f"    pv_sun.csv    (Mittag ø {mean_sun:.3f} kW  |  "
                f"Base {mean_base:.3f} kW  |  +{mean_sun - mean_base:.3f} kW)\n"
                f"    pv_clouds.csv (Mittag ø {mean_cloud:.3f} kW  |  "
                f"Base {mean_base:.3f} kW  |  {mean_cloud - mean_base:.3f} kW)"
            )


if __name__ == "__main__":
    create_pv_profiles(BASE_FOLDERS, YEARS, K_SUN, K_RAIN)
