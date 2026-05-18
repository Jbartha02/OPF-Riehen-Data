import json

import numpy as np
import pandas as pd

# TODO: profiles to p.u.?
# TODO: check sign convention of power (positive = injection into grid, negative = withdrawal from grid)
# TODO: correct timestep handling (currently mix of hours and quarterhours, and not implemented correctly yet)


class Config:
    
    analysis_folder: str = r"2703_23_homogen/2050"
    analysis_day: str = "01-08" # "dd-mm"
    analysis_start_hour: int = 9
    analysis_n_quarterhours: int = 4 * 4 # 6 hours 
    
    delta_t = 0.25 # hours; TODO: check or update or implement correctly
    
    eta_polygon_area: float = 0.01  # convergence parameter for the polygon approximation of the FFOR #TODO: define/update this value
    optimization_dirs_init: list[tuple[int, int]] = [(1, 0), (0, 1), (-1, 0), (0, -1)] # initial optimization directions for the FFOR algorithm, define coefficients (a,b) of minimization objective a*P + b*Q #TODO: define/update this value
    
    filename_dict: dict[str, dict[str, str]] = {
        "nodes": {
            "2703-23_0_4": "nodes.geojson"
        },
        "node_metadata": {
            "BESS": "bess_allocation.csv",
            "PV": "pv_p_installed.csv",
            "HP": "hp_allocation.csv",
        },
        "loadprofiles": {
            "load": "load_profiles.csv",
            "PV_ub": "pv_generation.csv",
            "PV_base": "pv_generation.csv",
            "t_outdoor": "temperature_profiles.csv"
        } # TODO: preprocess these files; finish implementation
    } # contains the filenames of the input files for each type of data
    
    pv_max_q_p_ratio: float = 0.3 #TODO: define/update this value
    hp_lb_temp: int = 19 #°C
    hp_base_temp: int = 21 #°C
    hp_ub_temp: int = 23 #°C
    hp_output_temp: int = 30 #°C the temperature to which the HP heats the water for the heating system, used for cop calculation (assumed to be constant over the year)
    hp_q_p_ratio: float = 0.3 # TODO: define/update value (maybe also as cos_phi or similar)
    bess_soc_lb: float = 0.3
    bess_soc_base: float = 0.5
    bess_soc_ub: float = 0.7
    bess_power_octagon_approximation: list[tuple[float, float]] = [
        (1, np.sqrt(2)-1), (1, -(np.sqrt(2)-1)), (-1, np.sqrt(2)-1), (-1, -(np.sqrt(2)-1)),
        (np.sqrt(2)-1, 1), (-(np.sqrt(2)-1), 1), (np.sqrt(2)-1, -1), (-(np.sqrt(2)-1), -1)
    ] # list of tuples (a,b) with the coefficients of the linear constraints a*P + b*Q <= S_bess that approximate the circle P^2 + Q^2 <= S_bess^2 with an octagon
    
    node_metadata_df: pd.DataFrame  # this df defines the indexes of the nodes
    
    # Uncontrollable Load
    p_load: np.ndarray
    
    # PV
    p_pv_ub: np.ndarray
    p_pv_base: np.ndarray
    p_pv_lb: np.ndarray
    q_pv_base: np.ndarray # TODO: assumed to be zero -> not necessary
    
    # HP
    t_hp_base: np.ndarray
    t_hp_ub: np.ndarray
    t_hp_lb: np.ndarray
    t_outdoor: np.ndarray # TODO: make a profile that is always smaller than t_hp to avoid negative delta_t that would result in negative p_hp
    p_hp_base: np.ndarray
    q_hp_base: np.ndarray
    cop_hp: np.ndarray
    
    # BESS
    soc_bess_lb: np.ndarray
    soc_bess_base: np.ndarray
    soc_bess_ub: np.ndarray
    p_bess_base_neg: np.ndarray
    p_bess_base_pos: np.ndarray
    q_bess_base: np.ndarray # TODO: assumed to be zero -> not necessary
    bess_power_constraints: pd.DataFrame # df with coefficients a,b,c to create approximation of P^2 + Q^2 <= S^2 with an octagon of 8 linear constraints a*P + b*Q <= c  
    
    # EV
    p_ev_lb: np.ndarray
    p_ev_base: np.ndarray
    p_ev_ub: np.ndarray
    
    
    time_index_list: list[int] # lists the time indexes according to start_hour and n_quarterhours
    node_group_dict: dict[str, list] # e.g., node_group_dict["PV"] is a list with the indexes of the nodes that have PV
    
    def __init__(self):
        
        # Nodes and node groups
        self.node_metadata_df = self._ingest_node_metadata(analysis_folder=self.analysis_folder, fn_node_metadata=self.filename_dict["node_metadata"], fn_nodes=self.filename_dict["nodes"])
        self.node_group_dict = self._create_node_groups(node_metadata_df=self.node_metadata_df, fn_node_metadata=self.filename_dict["node_metadata"])
        
        # Uncontrollable Load
        self.p_load = -1 * self._ingest_load_profile(analysis_folder=self.analysis_folder, filename=self.filename_dict["loadprofiles"]["load"], analysis_day=self.analysis_day, node_metadata=self.node_metadata_df)
        
        # PV
        self.p_pv_ub = self._ingest_load_profile(analysis_folder=self.analysis_folder, filename=self.filename_dict["loadprofiles"]["PV_ub"], analysis_day=self.analysis_day, node_metadata=self.node_metadata_df)
        self.p_pv_base = self._ingest_load_profile(analysis_folder=self.analysis_folder, filename=self.filename_dict["loadprofiles"]["PV_base"], analysis_day=self.analysis_day, node_metadata=self.node_metadata_df)
        self.p_pv_lb = np.zeros_like(self.p_pv_ub)
        self.q_pv_base = np.zeros_like(self.p_pv_base)
        
        # HP
        self.t_outdoor = self._loadprofile_df_filter_convert_to_np(pd.read_csv(f"{self.analysis_folder}/{self.filename_dict['loadprofiles']['t_outdoor']}"), analysis_day=self.analysis_day).squeeze() #TODO
        self.t_hp_ub = self.hp_ub_temp * np.ones_like(self.p_load)
        self.t_hp_base = self.hp_base_temp * np.ones_like(self.p_load)
        self.t_hp_lb = self.hp_lb_temp * np.ones_like(self.p_load)
        self.cop_hp, self.p_hp_base = self._calculate_hp_cop_and_p(node_metadata_df=self.node_metadata_df, hp_output_temp=self.hp_output_temp, t_outdoor=self.t_outdoor, t_hp_base=self.t_hp_base)
        self.q_hp_base = self.p_hp_base * self.hp_q_p_ratio
        
        # BESS
        self.soc_bess_ub = self.bess_soc_ub * np.ones_like(self.p_load)
        self.soc_bess_base = self.bess_soc_base * np.ones_like(self.p_load)
        self.soc_bess_lb = self.bess_soc_lb * np.ones_like(self.p_load)
        self.p_bess_base_neg, self.p_bess_base_pos = self._calculate_bess_p(node_metadata_df=self.node_metadata_df, soc_bess_base=self.soc_bess_base)
        self.q_bess_base = np.zeros_like(self.p_bess_base_neg) # TODO: assumed to be zero -> not necessary
        
        # TODO: add other profiles
        
        self.time_index_list = list(self.analysis_start_hour + np.arange(int(self.analysis_n_quarterhours/4))) # TODO: implement correctly
        
        self._post_init_checks()


    def _calculate_bess_p(self, node_metadata_df: pd.DataFrame, soc_bess_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns p_bess_base_neg and p_bess_base_pos for every node for every timestep."""
        # prepare params
        soc_bess_base_prev = np.roll(soc_bess_base, shift=1, axis=1)
        bess_capacity = node_metadata_df["BESS_Battery_capacity_kWh"].to_numpy()[:, np.newaxis]
        bess_charging_efficiency = node_metadata_df["BESS_Charging_efficiency"].to_numpy()[:, np.newaxis]
        bess_discharging_efficiency = node_metadata_df["BESS_Discharging_efficiency"].to_numpy()[:, np.newaxis]
        
        # calculate p_bess_base_neg and p_bess_base_pos
        p_bess_base_neg = np.minimum(0, (soc_bess_base_prev - soc_bess_base) * bess_capacity / bess_charging_efficiency) # charging of battery, only negative values # TODO: implement delta_t?
        p_bess_base_pos = np.maximum(0, (soc_bess_base_prev - soc_bess_base) * bess_capacity * bess_discharging_efficiency) # discharging of battery, only positive values # TODO: implement delta_t?

        return p_bess_base_neg, p_bess_base_pos


    def _calculate_hp_cop_and_p(self, node_metadata_df: pd.DataFrame, hp_output_temp: int, t_outdoor: np.ndarray, t_hp_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the cop_hp and p_hp_base for every node for every timestep."""
        # calculate cop_hp
        cop_0 = node_metadata_df["HP_COP_0"].to_numpy()[:, np.newaxis]
        cop_1 = node_metadata_df["HP_COP_1"].to_numpy()
        cop_2 = node_metadata_df["HP_COP_2"].to_numpy()
        delta_t = hp_output_temp - t_outdoor
        
        cop_hp = cop_0 + np.outer(cop_1, delta_t) + np.outer(cop_2, delta_t**2) # COP = COP_0 + COP_1*(T_output - T_outdoor) + COP_2*(T_output - T_outdoor)^2
        
        # calculate p_hp_base
        capacity_kwh_K = node_metadata_df["HP_Thermal_capacitance_KWh/K"].to_numpy()[:, np.newaxis]
        conductivity_kw_K = node_metadata_df["HP_Thermal_conductivity_kW/K"].to_numpy()[:, np.newaxis]
        t_hp_base_prev = np.roll(t_hp_base, shift=1, axis=1)
        
        p_hp_base = np.divide(-(capacity_kwh_K * (t_hp_base - t_hp_base_prev) + conductivity_kw_K * (t_hp_base - t_outdoor)), cop_hp) # cop_hp * p_hp_base = -capacity_kwh_K * (t_hp_base - t_hp_base_prev) - conductivity_kw_K * (t_hp_base - t_outdoor)
        
        return cop_hp, p_hp_base
        

    def _create_node_groups(self, node_metadata_df: pd.DataFrame, fn_node_metadata: dict[str, str]) -> dict[str, list[int]]:
        """Creates a dictionary with lists of node indexes of a node group (e.g. PV, BESS, EV, etc) with the same keys as self.filenames_node_metadata, where each list contains all nodes having that technology."""
        node_group_dict = {}
        for tech in fn_node_metadata.keys():
            node_group_dict[tech] = node_metadata_df.index[node_metadata_df[tech] == True].tolist()
        #TODO: add group 'ALL NODES'
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
            node_metadata_df[tech] = True # add a column to indicate that these nodes have this technology
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
        
    def _post_init_checks(self):
        
        # TODO: implement data inputchecks
        assert self.hp_output_temp >= self.hp_ub_temp, "hp_output_temp should be greater than or equal to hp_ub_temp to avoid negative delta_t and thus negative p_hp_base"
        
    
    
