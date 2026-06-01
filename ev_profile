"""
EV Profil Erstellung — Session-Scheduling
==========================================

Erzeugt knotenspezifische EV-Baseload-Profile (ev_baseload.csv)
für OPF-Simulationen des NS-Netzes Riehen (2703-23_0_4).

Methode:
  - Jeder aktive EV-Knoten lädt mit maximal P_MAX kW.
  - Tagesenergie am PoCC (Base × sum_shares) bestimmt, wie viele Knoten
    heute laden: n_aktiv = round(e_day / (P_MAX × TARGET_SESSION_H))
  - Knotenauswahl via Bresenham-Akkumulator: proportionale Verteilung
    über alle Tage ohne Zufallselement.
  - Jeder ausgewählte Knoten lädt exakt TARGET_SESSION_H Stunden mit P_MAX.

Benötigt: pandas, numpy
"""

import os
import csv
import numpy as np
import pandas as pd

# =============================================================================
# KONFIGURATION
# =============================================================================

BASE_DIR     = r"C:\Users\josua\OPF-Riehen-Data\2703_23_homogen"
ALT_SUBDIR   = "ev_alternativ"
OUTPUT_SUBDIR = "ev_7kw"

YEARS = [2030, 2040, 2050]

P_MAX = {
    2030: 7.4,
    2040: 7.4,
    2050: 7.4,
}

DECIMAL_PLACES   = 6
MIN_BUDGET_KWH   = 0.5
MAX_SESSION_H    = 9
TARGET_SESSION_H = 2

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def read_ev_allocation(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["LV_osmid"] = df["LV_osmid"].astype(int)
    return df


def read_ev_power_profiles(path: str) -> tuple[dict[str, np.ndarray], list[str]]:
    df = pd.read_csv(path).set_index("Profile_type")
    ts_cols = [c for c in df.columns if c != "BFS_municipality_code"]
    profiles = {}
    for label in ("Lower", "Base", "Upper"):
        if label not in df.index:
            raise ValueError(f"Profil-Typ '{label}' nicht in {path} gefunden.")
        profiles[label] = df.loc[label, ts_cols].values.astype(float)
    return profiles, ts_cols


def _get_day_slices(ts_headers: list[str]) -> list[tuple[int, int]]:
    slices, current_day, start = [], None, 0
    for i, header in enumerate(ts_headers):
        date_part = header.split(" ")[0]
        if date_part != current_day:
            if current_day is not None:
                slices.append((start, i))
            current_day, start = date_part, i
    if current_day is not None:
        slices.append((start, len(ts_headers)))
    return slices


def session_schedule(
    profiles: dict[str, np.ndarray],
    active_nodes: list[dict],
    sum_shares: float,
    p_max: float,
    ts_headers: list[str],
    min_budget_kwh: float,
    max_session_h: int,
    target_session_h: int,
) -> dict:
    n_t = len(profiles["Base"])
    day_slices = _get_day_slices(ts_headers)
    node_base   = {n["lv_osmid"]: np.zeros(n_t) for n in active_nodes}
    accumulator = {n["lv_osmid"]: 0.0           for n in active_nodes}

    for ds, de in day_slices:
        p_cap     = profiles["Base"][ds:de] * sum_shares
        n_per_day = de - ds
        e_day     = float(p_cap.sum())

        if e_day < min_budget_kwh:
            continue

        n_charging = min(max(1, int(np.round(e_day / (p_max * target_session_h)))),
                         len(active_nodes))

        for n in active_nodes:
            accumulator[n["lv_osmid"]] += n_charging * (n["ev_share"] / sum_shares)

        charging_today = sorted(active_nodes,
                                key=lambda n: accumulator[n["lv_osmid"]],
                                reverse=True)[:n_charging]

        for node in charging_today:
            nid = node["lv_osmid"]
            n_h = min(target_session_h, max_session_h, n_per_day)

            best_t, best_cap = 0, -1.0
            for t_start in range(n_per_day - n_h + 1):
                cap = float(p_cap[t_start:t_start + n_h].sum())
                if cap > best_cap:
                    best_cap, best_t = cap, t_start

            for i in range(n_h):
                t = best_t + i
                node_base[nid][ds + t]  = p_max
                p_cap[t] = max(0.0, p_cap[t] - p_max)

            accumulator[nid] -= 1.0

    return node_base


def compute_lb_profile(
    profiles: dict[str, np.ndarray],
    active_nodes: list[dict],
    node_base: dict,
    sum_shares: float,
) -> dict:
    n_t    = len(profiles["Base"])
    node_lb = {n["lv_osmid"]: np.zeros(n_t) for n in active_nodes}

    for t in range(n_t):
        lb_pocc  = profiles["Lower"][t] * sum_shares
        charging = [n["lv_osmid"] for n in active_nodes if node_base[n["lv_osmid"]][t] > 0]
        if not charging:
            continue
        lb_per_node = lb_pocc / len(charging)
        for nid in charging:
            node_lb[nid][t] = min(lb_per_node, node_base[nid][t])

    return node_lb


def _write_node_csv(
    output_dir: str,
    filename: str,
    all_nodes: pd.DataFrame,
    lv_grid: str,
    node_profiles: dict,
    decimal_places: int,
    ts_headers: list[str],
):
    os.makedirs(output_dir, exist_ok=True)
    fmt  = f".{decimal_places}f"
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["LV_grid", "LV_osmid"] + ts_headers)
        for _, row in all_nodes.sort_values("LV_osmid").iterrows():
            osmid = int(row["LV_osmid"])
            if osmid in node_profiles:
                vals = [format(v, fmt) for v in node_profiles[osmid]]
            else:
                vals = ["0" + "." + "0" * decimal_places] * len(ts_headers)
            writer.writerow([lv_grid, osmid] + vals)
    print(f"  Geschrieben: {path}")


def write_profiles(output_dir, all_nodes, lv_grid, node_base, decimal_places, ts_headers):
    _write_node_csv(output_dir, "ev_baseload.csv",   all_nodes, lv_grid, node_base, decimal_places, ts_headers)


def write_lb_profiles(output_dir, all_nodes, lv_grid, node_lb, decimal_places, ts_headers):
    _write_node_csv(output_dir, "ev_lowerbound.csv", all_nodes, lv_grid, node_lb,   decimal_places, ts_headers)


def print_summary(year, active_nodes, node_base, p_max, ts_headers):
    total_energy = sum(node_base[n["lv_osmid"]].sum() for n in active_nodes)
    max_load     = max(node_base[n["lv_osmid"]].max() for n in active_nodes)
    day_slices   = _get_day_slices(ts_headers)
    avg_max_daily = np.mean([
        max(np.count_nonzero(node_base[n["lv_osmid"]][ds:de]) for ds, de in day_slices)
        for n in active_nodes
    ])
    print(f"\n  Zusammenfassung {year}:")
    print(f"    Aktive Knoten:                   {len(active_nodes)}")
    print(f"    Max. Knotenleistung:             {max_load:.2f} kW  (Limit: {p_max} kW)")
    print(f"    Ø max. Ladestunden/Tag/Knoten:   {avg_max_daily:.1f} h")
    print(f"    Gesamtenergie (Netz, alle Tage): {total_energy:.1f} kWh")


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def main():
    for year in YEARS:
        print(f"\n{'='*60}")
        print(f"  Jahr {year}")
        print(f"{'='*60}")

        year_dir   = os.path.join(BASE_DIR, str(year), ALT_SUBDIR)
        alloc_path = os.path.join(year_dir, "ev_allocation.csv")
        prof_path  = os.path.join(year_dir, "ev_power_profiles.csv")
        out_dir    = os.path.join(BASE_DIR, str(year), OUTPUT_SUBDIR)

        alloc_df = read_ev_allocation(alloc_path)
        lv_grid  = alloc_df["LV_grid"].iloc[0]
        profiles, ts_headers = read_ev_power_profiles(prof_path)

        active_df    = alloc_df[alloc_df["EV_share"] > 0]
        active_nodes = [
            {"lv_osmid": int(r["LV_osmid"]), "ev_share": float(r["EV_share"])}
            for _, r in active_df.iterrows()
        ]
        sum_shares = active_df["EV_share"].sum()
        p_max      = P_MAX[year]

        print(f"  Aktive Knoten:   {len(active_nodes)}")
        print(f"  sum(EV_shares):  {sum_shares:.6f}")
        print(f"  P_MAX:           {p_max} kW")

        node_base = session_schedule(
            profiles, active_nodes, sum_shares, p_max, ts_headers,
            min_budget_kwh=MIN_BUDGET_KWH,
            max_session_h=MAX_SESSION_H,
            target_session_h=TARGET_SESSION_H,
        )
        node_lb = compute_lb_profile(profiles, active_nodes, node_base, sum_shares)

        print_summary(year, active_nodes, node_base, p_max, ts_headers)

        print(f"\n  Schreibe Ausgabe nach: {out_dir}")
        write_profiles(out_dir, alloc_df, lv_grid, node_base, DECIMAL_PLACES, ts_headers)
        write_lb_profiles(out_dir, alloc_df, lv_grid, node_lb, DECIMAL_PLACES, ts_headers)

    print(f"\n{'='*60}")
    print("  Alle Jahre abgeschlossen.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
