import os
import pandas as pd

# configurable parameters

BASE_FOLDERS = [
    "2703_23_homogen",
]

YEARS = [2030, 2040, 2050]

# storage capacity [h]: Battery_capacity_kWh = Nominal_power_kW * DURATION_H[year]
# (Paper): 2.5 h for all years
DURATION_H = {
    2030: 1.5,   # conservative, actual market reality
    2040: 2,   # moderate
    2050: 2.5,   # ambitious but possible
}


def create_bess_alternativ(base_folders: list[str], years: list[int], duration_h: dict[int, float]) -> None:
    print("Originalmultiplikator (Paper): 2.5 h für alle Jahre")
    print(f"Neuer Multiplikator:           { {y: duration_h[y] for y in years} }\n")

    for base in base_folders:
        for year in years:
            src = os.path.join(base, str(year), "bess_allocation.csv")
            if not os.path.exists(src):
                print(f"  Übersprungen (nicht gefunden): {src}")
                continue

            df = pd.read_csv(src)
            multiplier = duration_h[year]

            df_alt = df.copy()
            df_alt["Battery_capacity_kWh"] = df["Nominal_power_kW"] * multiplier

            out_dir = os.path.join(base, str(year), "bess_alternativ")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "bess_allocation.csv")
            df_alt.to_csv(out_path, index=False)

            orig_cap = df["Battery_capacity_kWh"].sum()
            new_cap  = df_alt["Battery_capacity_kWh"].sum()
            print(
                f"  {out_path}\n"
                f"    Speicherdauer: {multiplier} h  |  "
                f"Kapazität: {orig_cap:.1f} → {new_cap:.1f} kWh  ({new_cap/orig_cap*100:.0f} %)\n"
                f"    Leistung: unverändert ({df['Nominal_power_kW'].sum():.1f} kW)"
            )


if __name__ == "__main__":
    create_bess_alternativ(BASE_FOLDERS, YEARS, DURATION_H)
