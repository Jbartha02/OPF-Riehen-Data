import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt

class Config:
    
    # directories
    base_folder: str = "2703_23_homogen" 

    # contains the filenames of the input files for each type of data
    filename_dict: dict[str, dict[str, str]] = {
        "edges": {
            "2703-23_0_4": "edges.geojson"
        },
        "edges_metadata": {
            "2703-23_0_4": "edges_metadata.csv"
        },
        "nodes": {
            "2703-23_0_4": "nodes.geojson"
        },
        "node_metadata": {
            "BESS": "bess_allocation.csv",
            "PV": "pv_p_installed.csv",
            "HP": "hp_allocation.csv",
            "EV": "ev_alternativ/ev_allocation.csv"
        },
        "loadprofiles": {
            "load": "load_profiles.csv",
            "PV_ub": {
                "pvavg": "pv_generation.csv",
                "pvcld": "pv_clouds.csv",
                "pvsun": "pv_sun.csv"
            },
            "PV_base": {
                "pvavg": "pv_generation.csv",
                "pvcld": "pv_clouds.csv",
                "pvsun": "pv_sun.csv"
            },
            "Ev_base": "ev_7kw/ev_baseload.csv",
            "Ev_lb": "ev_7kw/ev_lowerbound.csv",
            "Ev_ub": "ev_7kw/ev_baseload.csv",
            "t_outdoor": "temperature_profiles.csv"
        }
    } 

    # optimization parameters
    eta_polygon_area: float = 0.01  # convergence parameter for the polygon approximation of the FFOR
    optimization_dirs_init: list[tuple[int, int]] = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # initial optimization directions for the FFOR algorithm, define coefficients (a,b) of minimization objective a*P + b*Q

    # technology parameters
    pv_max_q_p_ratio: float = 0.3
    no_hp_month_list: list[int] = [5, 6, 7, 8, 9]  # summer months in which the HP is deactivated
    hp_lb_temp: int = 20  #°C, minimum temperature inside the houses
    hp_base_temp_dict: dict[int, int] = {
        1: 21,
        2: 21,
        3: 21,
        4: 21.5,
        5: 21.8,  # deactivated in no_hp_month_list
        6: 21.8,  # deactivated in no_hp_month_list
        7: 21.8,  # deactivated in no_hp_month_list
        8: 21.8,  # deactivated in no_hp_month_list
        9: 21.8,  # deactivated in no_hp_month_list
        10: 21.5,
        11: 21,
        12: 21,
    }  #°C per month of the year, base temperature inside the houses
    hp_ub_temp: int = 22  #°C, maximum temperature inside the houses
    hp_output_temp: int = 30  #°C the temperature to which the HP heats the water for the heating system, used for cop calculation (assumed to be constant over the year)
    hp_q_p_ratio: float = 0.3  
    bess_soc_lb: float = 0.3  # min state of charge of bess
    bess_soc_base: float = 0.5  # base state of charge of bess
    bess_soc_ub: float = 0.7  # max state of charge of bess
    bess_power_octagon_approximation: list[tuple[float, float]] = [
        (1, np.sqrt(2)-1), (1, -(np.sqrt(2)-1)), (-1, np.sqrt(2)-1), (-1, -(np.sqrt(2)-1)),
        (np.sqrt(2)-1, 1), (-(np.sqrt(2)-1), 1), (np.sqrt(2)-1, -1), (-(np.sqrt(2)-1), -1)
    ]  # list of tuples (a,b) with the coefficients of the linear constraints a*P + b*Q <= S_bess that approximate the circle P^2 + Q^2 <= S_bess^2 with an octagon
    
    # bases for pu calculation
    V_base: int = 0.4 # kV
    S_base: float = 0.63 # MVA
    Z_base_ohm: float = (V_base**2 / S_base) * 1000
    I_base_A: float = (S_base / (np.sqrt(3)*V_base)) * 1000
    

    # --------- Derived parameters and data structures (calculated in __init__) --------- #
    run_simple_ffor: bool  # runs only four initial optimization directions if True, otherwise determines detailed FFOR
    
    # directories
    analysis_folder: str  # folder with the input data
    output_folder: str  # folder where outputs are stored
    
    analysis_year: int  # year of the scenario and subfolder of base_folder

    # time parameters
    analysis_month: int  # month, e.g. 8 for August
    analysis_day: int  # day of the month, e.g. 9
    analysis_start_hour: int
    analysis_n_timesteps: int  # hours, only if 1 timestep it is delta_t hours

    delta_t: float  # hours, ATTENTION: only used if analysis_n_timesteps == 1, otherwise overwritten by '1 hour'
    
    analysis_date_mm_dd: str
    time_index_list: list[int] # lists the time indexes according to start_hour and n_quarterhours
    time_col_list: list[str] # lists the column names of the time columns in the output files (i.e. timestamps)

    # grid information
    node_group_dict: dict[str, list] # e.g., node_group_dict["PV"] is a list with the indexes of the nodes that have PV

    node_metadata_df: pd.DataFrame  # this df defines the indexes of the nodes

    edges_metadata_df: pd.DataFrame # this df defines the indexes of the edges, and contains the parameters of the edges 
    
    # Uncontrollable Load
    p_load: np.ndarray
    
    # PV
    p_pv_ub: np.ndarray
    p_pv_base: np.ndarray
    p_pv_lb: np.ndarray
    q_pv_base: np.ndarray # assumed to be zero
    
    # HP
    t_hp_base: np.ndarray
    t_hp_ub: np.ndarray
    t_hp_lb: np.ndarray
    t_outdoor: np.ndarray
    p_hp_base: np.ndarray
    q_hp_base: np.ndarray
    cop_hp: np.ndarray
    
    # BESS
    soc_bess_lb: np.ndarray
    soc_bess_base: np.ndarray
    soc_bess_ub: np.ndarray
    p_bess_base_neg: np.ndarray
    p_bess_base_pos: np.ndarray
    q_bess_base: np.ndarray # assumed to be zero
    bess_power_constraints: pd.DataFrame # df with coefficients a,b,c to create approximation of P^2 + Q^2 <= S^2 with an octagon of 8 linear constraints a*P + b*Q <= c  
    
    # EV
    p_ev_lb: np.ndarray
    p_ev_base: np.ndarray
    p_ev_ub: np.ndarray
    

    def __init__(self, year: int, month: int, day: int, start_hour: int, n_timesteps: int, delta_t: float = 1.0, pv_weather: str = "pvavg", no_ev: bool = False, no_bess: bool = False, run_simple_ffor: bool = False):
        """Initialize configuration for an analysis run.

        Parameters
        - year (int): Analysis year (e.g., 2030, 2040 or 2050); must correspond to a subfolder name with the input data.
        - month (int): Analysis month (1-12).
        - day (int): Analysis day of month (1-31).
        - start_hour (int): Starting hour of the analysis (0-23).
        - n_timesteps (int): Number of timesteps to simulate.
        - delta_t (float): Timestep length in hours; if n_timesteps != 1, it is overridden to 1.0.
        - pv_weather (str): PV weather profile to use; one of 'pvavg', 'pvcld', 'pvsun'.
        - no_ev (bool): If True, exclude EVs from the simulation (default False).
        - no_bess (bool): If True, exclude BESS from the simulation (default False).
        - run_simple_ffor (bool): If True, run only the first four optimization directions (+/- P-flex, +/- Q_flex) (default False).
        """
        assert pv_weather in self.filename_dict["loadprofiles"]["PV_ub"].keys(), "Check that pv is one of the options: 'pvavg', 'pvcld', 'pvsun'"
        assert pv_weather in self.filename_dict["loadprofiles"]["PV_base"].keys(), "Check that pv is one of the options: 'pvavg', 'pvcld', 'pvsun'"
        
        # Initialize init parameters
        self.analysis_year = year
        self.analysis_month = month
        self.analysis_day = day
        self.analysis_start_hour = start_hour
        self.analysis_n_timesteps = n_timesteps
        self.delta_t = float(delta_t)
        self.run_simple_ffor = run_simple_ffor

        # overwrite delta_t if n_timesteps is not 1
        if self.analysis_n_timesteps != 1 and self.delta_t != 1.0:
            print("WARNING: Overwriting delta_t to 1 hour.")
            self.delta_t = 1.0  # hours

        # directories
        self.analysis_folder = f"{self.base_folder}/{self.analysis_year}"
        suffix_no_ev = "_noev" if no_ev else ""
        suffix_no_bess = "_nobess" if no_bess else ""
        suffix_simple_ffor = "_simpleffor" if run_simple_ffor else ""
        self.output_folder = f"{self.base_folder}/results/results_{self.analysis_year}{self.analysis_month:02d}{self.analysis_day:02d}_{self.analysis_start_hour:02d}_{self.analysis_n_timesteps*self.delta_t}_{pv_weather}{suffix_no_ev}{suffix_no_bess}{suffix_simple_ffor}_{dt.datetime.now().strftime('%Y%m%d_%H_%M_%S')}"

        # time parameters
        self.analysis_date_mm_dd = f"{self.analysis_month:02d}-{self.analysis_day:02d}"
        self.time_index_list = list(self.analysis_start_hour + np.arange(self.analysis_n_timesteps))
        self.time_col_list = [f"{(dt.datetime(self.analysis_year, self.analysis_month, self.analysis_day) + dt.timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S')}" for hour in range(48)]
        
        # Nodes and node groups
        self.node_metadata_df = self._ingest_node_metadata(analysis_folder=self.analysis_folder, fn_node_metadata=self.filename_dict["node_metadata"], fn_nodes=self.filename_dict["nodes"])
        self.node_group_dict = self._create_node_groups(node_metadata_df=self.node_metadata_df, fn_node_metadata=self.filename_dict["node_metadata"])
        
        # Edges metadata
        self.edges_metadata_df = self._ingest_network_edges(analysis_folder=self.analysis_folder, fn_edges_metadata=self.filename_dict["edges_metadata"], fn_edges=self.filename_dict["edges"], ordered_node_metadata=self.node_metadata_df)


        # --- Profiles ---
        # Uncontrollable Load
        self.p_load = -1 * self._ingest_load_profile(analysis_folder=self.analysis_folder, filename=self.filename_dict["loadprofiles"]["load"], analysis_day=self.analysis_date_mm_dd, node_metadata=self.node_metadata_df)
        
        # PV
        self.p_pv_ub = self._ingest_load_profile(analysis_folder=self.analysis_folder, filename=self.filename_dict["loadprofiles"]["PV_ub"][pv_weather], analysis_day=self.analysis_date_mm_dd, node_metadata=self.node_metadata_df)
        self.p_pv_base = self._ingest_load_profile(analysis_folder=self.analysis_folder, filename=self.filename_dict["loadprofiles"]["PV_base"][pv_weather], analysis_day=self.analysis_date_mm_dd, node_metadata=self.node_metadata_df)
        self.p_pv_lb = np.zeros_like(self.p_load)
        self.q_pv_base = np.zeros_like(self.p_load) # assumed to be zero
        
        # HP
        self.t_hp_ub = self.hp_ub_temp * np.ones_like(self.p_load)
        self.t_hp_base = self.hp_base_temp_dict[self.analysis_month] * np.ones_like(self.p_load)
        self.t_hp_lb = self.hp_lb_temp * np.ones_like(self.p_load)
        t_outdoor_raw = self._loadprofile_df_filter_convert_to_np(pd.read_csv(f"{self.analysis_folder}/{self.filename_dict['loadprofiles']['t_outdoor']}"), analysis_day=self.analysis_date_mm_dd).squeeze() #TODO
        self.t_outdoor = np.minimum(t_outdoor_raw, self.hp_lb_temp) # make sure that t_outdoor is always smaller than hp_lb_temp to avoid negative delta_t and thus negative p_hp_base
        self.cop_hp, self.p_hp_base = self._calculate_hp_cop_and_p(node_metadata_df=self.node_metadata_df, hp_output_temp=self.hp_output_temp, t_outdoor=self.t_outdoor, t_hp_base=self.t_hp_base)
        self.q_hp_base = self.p_hp_base * self.hp_q_p_ratio
        no_hp: bool = month in self.no_hp_month_list # deactivate HP in summer months, e.g. due to low heating demand and thus low economic viability of the HP
        if no_hp:
            # overwrite the HP profiles with zeros to deactivate the HP
            self.t_hp_ub = np.zeros_like(self.t_hp_ub)
            self.t_hp_base = np.zeros_like(self.t_hp_base)
            self.t_hp_lb = np.zeros_like(self.t_hp_lb)
            self.cop_hp = np.zeros_like(self.cop_hp)
            self.p_hp_base = np.zeros_like(self.p_hp_base)
            self.q_hp_base = np.zeros_like(self.q_hp_base)
            # set the HP metadata to zero
            self.node_metadata_df["HP_Nominal_power_kW"] = 0 * self.node_metadata_df["HP_Nominal_power_kW"]
            self.node_metadata_df["HP_Thermal_capacitance_KWh/K"] = 0 * self.node_metadata_df["HP_Thermal_capacitance_KWh/K"]
            self.node_metadata_df["HP_Thermal_conductivity_kW/K"] = 0 * self.node_metadata_df["HP_Thermal_conductivity_kW/K"]
        
        # BESS
        self.soc_bess_ub = self.bess_soc_ub * np.ones_like(self.p_load)
        self.soc_bess_base = self.bess_soc_base * np.ones_like(self.p_load)
        self.soc_bess_lb = self.bess_soc_lb * np.ones_like(self.p_load)
        self.p_bess_base_neg, self.p_bess_base_pos = self._calculate_bess_p(node_metadata_df=self.node_metadata_df, soc_bess_base=self.soc_bess_base)
        self.q_bess_base = np.zeros_like(self.p_bess_base_neg) # assumed to be zero
        if no_bess: # deactivate BESS
            # overwrite the BESS profiles with zeros
            self.soc_bess_ub = np.zeros_like(self.soc_bess_ub)
            self.soc_bess_base = np.zeros_like(self.soc_bess_base)
            self.soc_bess_lb = np.zeros_like(self.soc_bess_lb)
            self.p_bess_base_neg = np.zeros_like(self.p_bess_base_neg)
            self.p_bess_base_pos = np.zeros_like(self.p_bess_base_pos)
            # set the BESS powers to zero
            self.node_metadata_df["BESS_Nominal_power_kW"] = 0 * self.node_metadata_df["BESS_Nominal_power_kW"] 
        
        # EV
        self.p_ev_lb = (-1) * self._ingest_load_profile(analysis_folder=self.analysis_folder, filename=self.filename_dict["loadprofiles"]["Ev_lb"], analysis_day=self.analysis_date_mm_dd, node_metadata=self.node_metadata_df)
        self.p_ev_base = (-1) * self._ingest_load_profile(analysis_folder=self.analysis_folder, filename=self.filename_dict["loadprofiles"]["Ev_base"], analysis_day=self.analysis_date_mm_dd, node_metadata=self.node_metadata_df)
        self.p_ev_ub = (-1) * self._ingest_load_profile(analysis_folder=self.analysis_folder, filename=self.filename_dict["loadprofiles"]["Ev_ub"], analysis_day=self.analysis_date_mm_dd, node_metadata=self.node_metadata_df)
        if no_ev: # deactivate EV
            # overwrite the EV profiles with zeros
            self.p_ev_lb = np.zeros_like(self.p_ev_lb)
            self.p_ev_base = np.zeros_like(self.p_ev_base)
            self.p_ev_ub = np.zeros_like(self.p_ev_ub)
        
        # Some data checks
        self._post_init_checks(no_hp=no_hp)


    def _calculate_bess_p(self, node_metadata_df: pd.DataFrame, soc_bess_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns p_bess_base_neg and p_bess_base_pos for every node for every timestep."""
        # prepare params
        soc_bess_base_prev = np.roll(soc_bess_base, shift=1, axis=1)
        capacity_kWh = node_metadata_df["BESS_Battery_capacity_kWh"].to_numpy()[:, np.newaxis]
        eta_ch = node_metadata_df["BESS_Charging_efficiency"].to_numpy()[:, np.newaxis]
        eta_disch = node_metadata_df["BESS_Discharging_efficiency"].to_numpy()[:, np.newaxis]
        
        # calculate p_bess_base_neg and p_bess_base_pos
        p_bess_base_neg = np.minimum(0, (soc_bess_base_prev - soc_bess_base) * capacity_kWh / self.delta_t / eta_ch) # charging of battery, only negative values
        p_bess_base_pos = np.maximum(0, (soc_bess_base_prev - soc_bess_base) * capacity_kWh / self.delta_t * eta_disch) # discharging of battery, only positive values

        return p_bess_base_neg, p_bess_base_pos


    def _calculate_hp_cop_and_p(self, node_metadata_df: pd.DataFrame, hp_output_temp: int, t_outdoor: np.ndarray, t_hp_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the cop_hp and p_hp_base for every node for every timestep."""
        # calculate cop_hp
        cop_0 = node_metadata_df["HP_COP_0"].to_numpy()[:, np.newaxis]
        cop_1 = node_metadata_df["HP_COP_1"].to_numpy()
        cop_2 = node_metadata_df["HP_COP_2"].to_numpy()
        delta_temp = hp_output_temp - t_outdoor
        
        cop_hp = cop_0 + np.outer(cop_1, delta_temp) + np.outer(cop_2, delta_temp**2) # COP = COP_0 + COP_1*(T_output - T_outdoor) + COP_2*(T_output - T_outdoor)^2
        
        # calculate p_hp_base
        capacity_kwh_K = node_metadata_df["HP_Thermal_capacitance_KWh/K"].to_numpy()[:, np.newaxis]
        conductivity_kw_K = node_metadata_df["HP_Thermal_conductivity_kW/K"].to_numpy()[:, np.newaxis]
        t_hp_base_prev = np.roll(t_hp_base, shift=1, axis=1)
        
        p_hp_base = np.divide(-(capacity_kwh_K / self.delta_t * (t_hp_base - t_hp_base_prev) + conductivity_kw_K * (t_hp_base - t_outdoor)), cop_hp) # cop_hp * p_hp_base = -capacity_kwh_K / delta_t * (t_hp_base - t_hp_base_prev) - conductivity_kw_K * (t_hp_base - t_outdoor)
        
        return cop_hp, p_hp_base
        

    def _create_node_groups(self, node_metadata_df: pd.DataFrame, fn_node_metadata: dict[str, str]) -> dict[str, list[int]]:
        """Creates a dictionary with lists of node indexes of a node group (e.g. PV, BESS, EV, etc) with the same keys as self.filenames_node_metadata, where each list contains all nodes having that technology."""
        node_group_dict = {}
        # add groups of technology
        for tech in fn_node_metadata.keys():
            node_group_dict[tech] = node_metadata_df.index[node_metadata_df[tech] == True].tolist()
        # add group of all nodes
        node_group_dict["ALL NODES"] = node_metadata_df.index.tolist()
        return node_group_dict


    def _ingest_load_profile(self, analysis_folder: str, filename: str, analysis_day: str, node_metadata: pd.DataFrame) -> np.ndarray:
        """Reads a csv file with load profiles for every node, and returns a filtered and prepared np.ndarray with the load profiles for the analysis day for every node."""
        # df with all nodes (also nodes without technology) for correct indexing
        all_nodes_ordered_df = node_metadata.loc[:, ["LV_grid", "LV_osmid"]]
        
        # merge load_profiles onto nodes df
        profiles = pd.read_csv(f"{analysis_folder}/{filename}")
        df = all_nodes_ordered_df.merge(profiles, on=["LV_grid", "LV_osmid"], how="left")
        
        return self._loadprofile_df_filter_convert_to_np(loadprofile_df=df, analysis_day=analysis_day)
    
    
    def _loadprofile_df_filter_convert_to_np(self, loadprofile_df: pd.DataFrame, analysis_day: str) -> np.ndarray:
        """Filters the loadprofile_df to the columns of the analysis_day, converts to np.nd_array, and extends the data two days (copy cols)."""
        # filter columns of the analysis day
        time_column_list = [col for col in loadprofile_df.columns if col.startswith(analysis_day)]
        print("WARNING: No time columns found for the analysis_date_mm_dd. Check input!") if len(time_column_list) == 0 else None
        assert len(time_column_list) == 24, f"Expected 24 time columns for the analysis day {analysis_day}, but found {len(time_column_list)}. Check input and column filtering logic."
        np_array = loadprofile_df.loc[:, time_column_list].to_numpy()

        return np.hstack([np_array, np_array]) # copy the profiles to the next day, to enable periods that cross midnight
        

    def _ingest_node_metadata(self, analysis_folder: str, fn_node_metadata: dict[str, str], fn_nodes: dict[str, str]) -> pd.DataFrame:
        """Ingests all non-timedependent node metadata (e.g., installed capacities) from the csv files in self.filenames_node_metadata, and merges them with the OSM IDs of the nodes in self.filenames_nodes."""
        # create pd.DataFrame of all nodes
        all_nodes_df = self._ingest_all_node_osmids(analysis_folder=analysis_folder, fn_nodes=fn_nodes)
        
        # merge the node metadata from each csv file 
        for tech, filename in fn_node_metadata.items():
            node_metadata_df = pd.read_csv(f"{analysis_folder}/{filename}")
            node_metadata_df = node_metadata_df.rename(
                columns={
                    column_name: f"{tech}_{column_name}"
                    for column_name in node_metadata_df.columns
                    if column_name not in ["LV_grid", "LV_osmid"]
                }
            )
            share_col = next((c for c in node_metadata_df.columns if c.startswith(f"{tech}_") and c.endswith("_share")), None)
            node_metadata_df[tech] = (node_metadata_df[share_col] > 0) if share_col else True
            all_nodes_df = all_nodes_df.merge(node_metadata_df, on=["LV_grid", "LV_osmid"], how="left")
        
        return all_nodes_df.sort_values(["LV_grid", "LV_osmid"]).reset_index(drop=True) # order the nodes by LV grid and OSM ID
    
    
    def _ingest_all_node_osmids(self, analysis_folder: str, fn_nodes: dict[str, str]) -> pd.DataFrame:
        """
        Extracts all OSM IDs of the nodes in the LV grid(s) of the geojson file(s) in self.filenames_nodes.
        Returns a pd.DataFrame with the columns "LV_grid" and "LV_osmid".
        """
        records: list[dict[str, object]] = []

        # iterate over each geojson file (one for each LV grid)
        for lv_grid, filename in fn_nodes.items():
            # read the geojson file
            geojson_path = f"{analysis_folder}/{filename}"
            with open(geojson_path, "r", encoding="utf-8") as geojson_file:
                geojson_data = json.load(geojson_file)

            # extract the OSM IDs and append to records
            for feature in geojson_data.get("features", []):
                properties = feature.get("properties", {})
                osmid = properties.get("osmid")
                if osmid is None or osmid == "":
                    print(f"WARNING: Missing OSM ID for a node in {filename} with features {feature} and properties {properties}.")
                    continue

                records.append(
                    {
                        "LV_grid": lv_grid,
                        "LV_osmid": int(osmid),
                    }
                )

        return pd.DataFrame(records, columns=["LV_grid", "LV_osmid"])
    


    def _ingest_network_edges(self, analysis_folder: str, fn_edges_metadata: dict[str, str], fn_edges: dict[str, str], ordered_node_metadata: pd.DataFrame) -> pd.DataFrame:
        """ 
        extracts all edges from the geojson, merges them with the csv metadata of the edges, and maps u and v to matrix indices 
        based on the existing node_metadata_df. Returns a pd.DataFrame with all edge parameters and the mapped u and v indices.
        """
        records: list[dict[str, object]] = []

        # iterate over each geojson file (one for each LV grid)
        for lv_grid, filename in fn_edges.items():
            geojson_path = f"{analysis_folder}/{filename}"
            with open(geojson_path, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)

            # read .geojson files + properties
            for feature in geojson_data.get("features", []):
                props = feature.get("properties", {})
                if "u" not in props or "v" not in props:
                    continue

                records.append({
                    "LV_grid": lv_grid,
                    "u_osmid": int(props["u"]),
                    "v_osmid": int(props["v"]),
                    "r": float(props["r"]),
                    "x": float(props["x"]),
                    "b": float(props["b"]),
                    "s_nom": float(props["s_nom"]),
                    "length": float(props.get("length", 0))
                })

        all_edges_df = pd.DataFrame(records)

        # 2. csv-metadata of the edges for later indexing
        for lv_grid, filename in fn_edges_metadata.items():
            try:
                metadata_path = f"{analysis_folder}/{filename}"
                edges_meta_df = pd.read_csv(metadata_path)
                
                edges_meta_df = edges_meta_df.rename(
                    columns={col: f"meta_{col}" for col in edges_meta_df.columns if col not in ["u", "v", "key"]}
                )
                all_edges_df = all_edges_df.merge(edges_meta_df, left_on=["u_osmid", "v_osmid"], right_on=["u", "v"], how="left")
            except FileNotFoundError:
                pass

        # 3. mapping via node_metadata_df Index
        osmid_to_matrix_idx = {int(row["LV_osmid"]): idx for idx, row in ordered_node_metadata.iterrows()}

        all_edges_df["u_idx"] = all_edges_df["u_osmid"].map(osmid_to_matrix_idx)
        all_edges_df["v_idx"] = all_edges_df["v_osmid"].map(osmid_to_matrix_idx)

        all_edges_df["u_idx"] = all_edges_df["u_idx"].astype(int)
        all_edges_df["v_idx"] = all_edges_df["v_idx"].astype(int)

        return all_edges_df.reset_index(drop=True)

    def _plot_input_profiles(self):
        """Plots p_load, p_pv_base, p_bess_base, p_hp_base, p_ev_base and their total sum."""
        x_values = [
            dt.datetime(self.analysis_year, self.analysis_month, self.analysis_day)
            + dt.timedelta(hours=int(hour))
            for hour in range(24 * 2)  # 48 hours to cover the copied profiles for the next day
        ]

        p_hp_base = np.nansum(self.p_hp_base, axis=0)
        p_ev_base = np.nansum(self.p_ev_base, axis=0)
        p_load = np.nansum(self.p_load, axis=0)
        p_pv_base = np.nansum(self.p_pv_base, axis=0)
        p_bess_base_neg = np.nansum(self.p_bess_base_neg, axis=0)
        p_bess_base_pos = np.nansum(self.p_bess_base_pos, axis=0)

        stacked_profiles_neg = [p_hp_base, p_ev_base, p_load, p_bess_base_neg]
        labels_neg = ["p_hp_base", "p_ev_base", "p_load", "p_bess_base"]
        stacked_profiles_pos = [p_pv_base, p_bess_base_pos]
        labels_pos = ["p_pv_base", "p_bess_base"]
        net_sum = np.sum(stacked_profiles_neg + stacked_profiles_pos, axis=0)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.stackplot(x_values, stacked_profiles_neg, labels=labels_neg, alpha=0.85)
        ax.stackplot(x_values, stacked_profiles_pos, labels=labels_pos, alpha=0.85)
        ax.plot(x_values, net_sum, color="black", linewidth=2.5, label="net sum")
        ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
        ax.axvline(x_values[self.analysis_start_hour], color="gray", linewidth=0.5, linestyle="--", alpha=0.6, label="analysis start")
        ax.axvline(x_values[self.analysis_start_hour + self.analysis_n_timesteps], color="gray", linewidth=0.5, linestyle="--", alpha=0.6, label="analysis end") if self.delta_t == 1.0 else None
        ax.set_title("Input profiles")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power [kW]")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(f"{self.output_folder}/input_profiles.png", bbox_inches="tight", dpi=300)
        plt.close(fig)


    def _post_init_checks(self, no_hp: bool):
        # q_base
        assert (self.q_pv_base == np.zeros_like(self.p_load)).all(), "q_pv_base is assumed to be zero, check input"
        assert (self.q_bess_base == np.zeros_like(self.p_load)).all(), "q_bess_base is assumed to be zero, check input"
        
        # p_bess_base pos and neg
        assert np.logical_or(self.p_bess_base_neg <= 0, np.isnan(self.p_bess_base_neg)).all(), "p_bess_base_neg is assumed to be negative (charging), check input"
        assert np.logical_or(self.p_bess_base_pos >= 0, np.isnan(self.p_bess_base_pos)).all(), "p_bess_base_pos is assumed to be positive (discharging), check input"
        
        # soc and hp: lb, base, ub
        assert 0 <= self.bess_soc_lb <= self.bess_soc_base <= self.bess_soc_ub <= 1, "Check that bess_soc_lb < bess_soc_base < bess_soc_ub and that they are between 0 and 1"
        assert no_hp or (17 <= self.hp_lb_temp <= min(self.hp_base_temp_dict.values()) <= max(self.hp_base_temp_dict.values()) <= self.hp_ub_temp <= 26), "Check that hp_base_temp <= self.hp_base_temp <= self.hp_ub_temp and that they are reasonable values for temperatures in °C (17-26°C)"
        
        assert np.logical_and(self.p_ev_ub <= self.p_ev_base, self.p_ev_base <= self.p_ev_lb, self.p_ev_lb <= 0).all(), "p_ev must be negative"
        
        # t_outdoor
        assert no_hp or (self.t_outdoor.max() <= self.hp_lb_temp), "Check that the outdoor temperature profile is always smaller than the HP lower bound temperature to avoid negative delta_t and thus negative p_hp_base"
        assert np.logical_or(self.p_hp_base <= 0, np.isnan(self.p_hp_base)).all(), "Check that the HP power profile is always negative (consumption) to avoid issues with the optimization and interpretation of results"
        assert self.hp_output_temp >= self.hp_ub_temp, "hp_output_temp should be greater than or equal to hp_ub_temp to avoid negative delta_t and thus negative p_hp_base"
        
        # time parameters
        assert 1 <= self.analysis_n_timesteps <= 24, "Check that analysis_n_timesteps is between 1 and 24, otherwise the time indexing and profiles need to be adapted"
        assert isinstance(self.analysis_n_timesteps, int), "Check that analysis_n_timesteps is an integer number of hours, otherwise the time indexing and profiles need to be adapted"
        assert 0 < self.delta_t <= 1, "Check that delta_t is positive and less than or equal to 1, otherwise the time indexing and profiles need to be adapted"
        assert (self.analysis_n_timesteps == 1) or (self.delta_t == 1.0), "Check that if analysis_n_timesteps is not 1, then delta_t is 1, otherwise the time indexing and profiles need to be adapted"
        assert 0 <= self.analysis_start_hour <= 23, "Check that analysis_start_hour is between 0 and 23, otherwise the time indexing and profiles need to be adapted"
        assert isinstance(self.analysis_start_hour, int), "Check that analysis_start_hour is an integer number of hours, otherwise the time indexing and profiles need to be adapted"
        assert 1 <= self.analysis_day <= 31, "Check that analysis_day is between 1 and 31, otherwise the time indexing and profiles need to be adapted"
        assert 1 <= self.analysis_month <= 12, "Check that analysis_month is between 1 and 12, otherwise the time indexing and profiles need to be adapted"
        assert self.analysis_year in [2030, 2040, 2050], "Check that analysis_year is one of the years for which we have data (2030, 2040, 2050), otherwise the time indexing and profiles need to be adapted"

    def create_output_folder(self):
        """Create the output folder and store a snapshot of the inputs for this scenario."""

        os.makedirs(self.output_folder, exist_ok=True)
        shutil.copy2(os.path.abspath(__file__), f"{self.output_folder}/config.py")
        self.node_metadata_df.to_csv(f"{self.output_folder}/node_metadata_df.csv", index=False)
        self.edges_metadata_df.to_csv(f"{self.output_folder}/edges_metadata_df.csv", index=False)
        self._plot_input_profiles()
    
