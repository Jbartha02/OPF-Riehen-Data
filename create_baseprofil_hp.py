import pandas as pd
import numpy as np
import os
import argparse

def calculate_hp_profiles(scenario_name_0, scenario_name_1, year, t_room_set=20.0):
    base_path = os.path.join(str(scenario_name_0), str(scenario_name_1), str(year))
    hp_file = os.path.join(base_path, 'hp_allocation.csv')
    temp_file = os.path.join(base_path, 'temperature_profiles.csv')
    
    print(f"Lade Daten aus: {base_path}...")

    if not os.path.exists(hp_file) or not os.path.exists(temp_file):
        print(f"Fehler: Dateien in {base_path} nicht gefunden!")
        return

    hp_df = pd.read_csv(hp_file)
    temp_df = pd.read_csv(temp_file)

    hp_df.columns = hp_df.columns.str.strip()
    temp_df.columns = temp_df.columns.str.strip()
    
    time_columns = temp_df.columns[1:]
    results = []

    for _, hp in hp_df.iterrows():

        profile_id = hp['Temperature_profile_name']
        temp_row = temp_df[temp_df['Temperature_profile_name'] == profile_id]
        
        if temp_row.empty:
            print(f"Überspringe: Profil '{profile_id}' nicht in Temp-Datei gefunden.")
            continue
            
        p_nom = hp['Nominal_power_kW']
        h_i = hp['Thermal_conductivity_kW/K']
        c0, c1, c2 = hp['COP_0'], hp['COP_1'], hp['COP_2']
        
        hp_series = {'LV_osmid': hp['LV_osmid']}
        
        for t_col in time_columns:
            t_ambient = temp_row[t_col].values[0]
            delta_t = t_room_set - t_ambient
            
            if delta_t <= 0:
                p_elec_final = 0
            else:
                q_therm_req = h_i * delta_t
                
                cop_t = c0 + c1 * delta_t + c2 * (delta_t**2)
                cop_t = max(cop_t, 1.0)
                
                p_elec_req = q_therm_req / cop_t
                p_elec_final = min(p_elec_req, p_nom)
            
            hp_series[t_col] = p_elec_final
            
        results.append(hp_series)

    # Output im gleichen Szenario-Ordner speichern
    output_df = pd.DataFrame(results)
    output_name = f"hp_power_profiles_{scenario_name_0}_{scenario_name_1}_{year}.csv"
    output_path = os.path.join(base_path, output_name)
    
    output_df.to_csv(output_path, index=False)
    print(f"Erfolg! Gespeichert unter: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BESS Heatpump Profile Generator')
    parser.add_argument('--scenario_0', type=str, default='branch_riehen', help='Name des Szenario-Ordners')
    parser.add_argument('--scenario_1', default='2703_23_homogen')
    parser.add_argument('--year', default='2050')
    
    args = parser.parse_args()

    print(f"DEBUG: Starte mit Scenario={args.scenario_0}, Year={args.year}")
    
    calculate_hp_profiles(args.scenario_0, args.scenario_1, args.year)