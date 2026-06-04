import numpy as np
import pandas as pd

# Import physical layer and topology functions from the src directory
from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.system_model import distance_3D, path_loss, free_space_path_loss
import src.constants as const

from src.risk_profiles import generate_bs_failure_scenario
from mc_scheduler import allocate_backup_paths

def run_isolated_bs_failure_simulation(total_backup_rbs=10):
    # --- Step 1: Initialize Network Geometry & Topology ---
    bs_coords = get_hexagonal_bs(radius=2000)
    hap_coord, leo_coord = get_ntn_nodes()
    
    np.random.seed(42) 
    ue_coords = get_random_users(n=100)

    # --- Step 2: Inject the Isolated BS Failure Profile ---
    affected_users_profile = generate_bs_failure_scenario(total_users=100, total_gbs=7)
    affected_ue_ids = [user["ue_id"] for user in affected_users_profile]
    user_to_failed_bs_map = {user["ue_id"]: user["primary_gbs"] for user in affected_users_profile}

    simulation_results = []

    # --- Step 3: Downlink Path Association Audit ---
    for ue_id, ue_pos in enumerate(ue_coords):
        best_node_type, best_node_idx = None, None
        max_rx_power_dbm = -np.inf
        best_distance, best_path_loss = 0, 0

        if ue_id in affected_ue_ids:
            primary_bs_risk = user_to_failed_bs_map[ue_id]
            best_node_type = None 
        else:
            primary_bs_risk = "None"
            for bs_id, bs_pos in enumerate(bs_coords):
                if f"GBS_{bs_id}" in ["GBS_0", "GBS_1", "GBS_2", "GBS_3"]:
                    continue
                    
                dist = distance_3D(bs_pos, ue_pos)
                pl_db = path_loss(dist, const.CARRIER_FREQ_GHZ)
                rx_power_dbm = const.TX_POWER_GBS_HAP - pl_db
                
                if rx_power_dbm > max_rx_power_dbm:
                    max_rx_power_dbm = rx_power_dbm
                    best_node_type = "BS"
                    best_node_idx = bs_id
                    best_distance = dist
                    best_path_loss = pl_db

        # --- Step 4: Route Stranded Users to NTN Signal Scan ---
        if best_node_type is None:
            dist_hap = distance_3D(hap_coord, ue_pos)
            pl_hap_db = free_space_path_loss(dist_hap, const.CARRIER_FREQ_GHZ)
            rx_power_hap_dbm = const.TX_POWER_GBS_HAP - pl_hap_db
            
            if rx_power_hap_dbm > max_rx_power_dbm:
                max_rx_power_dbm, best_node_type, best_node_idx = rx_power_hap_dbm, "HAP", 0
                best_distance, best_path_loss = dist_hap, pl_hap_db

            dist_leo = distance_3D(leo_coord, ue_pos)
            pl_leo_db = free_space_path_loss(dist_leo, const.CARRIER_FREQ_GHZ)
            rx_power_leo_dbm = const.TX_POWER_LEO - pl_leo_db
            
            if rx_power_leo_dbm > max_rx_power_dbm:
                max_rx_power_dbm, best_node_type, best_node_idx = rx_power_leo_dbm, "LEO", 0
                best_distance, best_path_loss = dist_leo, pl_leo_db

        simulation_results.append({
            "UE_ID": ue_id,
            "Primary_BS_Risk": primary_bs_risk,
            "Connected_To": f"{best_node_type}_{best_node_idx}" if best_node_type else "None",
            "Distance (m)": round(best_distance, 2),
            "Rx Power (dBm)": round(max_rx_power_dbm, 2),
            "Status": "Healthy" if primary_bs_risk == "None" else "Stranded"
        })

    df = pd.DataFrame(simulation_results)

    # --- Step 5: Execute Algorithm 1 (Multi-Connectivity Resource Packing) ---
    resilience_pct, scheduled_rbs = allocate_backup_paths(affected_users_profile, total_backup_rbs)
    
    recovered_ue_ids = []
    for rb in scheduled_rbs:
        recovered_ue_ids.extend(rb["users"])

    df.loc[df['UE_ID'].isin(recovered_ue_ids), 'Status'] = 'Rescued (Shared NTN RB)'
    df.loc[(df['Status'] == 'Stranded') & (~df['UE_ID'].isin(recovered_ue_ids)), 'Connected_To'] = 'None'
    df.loc[(df['Status'] == 'Stranded') & (~df['UE_ID'].isin(recovered_ue_ids)), 'Status'] = 'Dropped (NTN Capacity Exceeded)'

    # --- Step 6: Print Consolidated Output Summary ---
    print(f"\n========= SIMULATION RESULTS: {total_backup_rbs} BACKUP RBs =========")
    print("--- Disaster Profile ---")
    print(f"Total Network Users      : {len(df)}")
    print(f"Terrestrial Safe Users   : {len(df[df['Status'] == 'Healthy'])}")
    print(f"Disaster Victims         : {len(affected_ue_ids)}")
    print("\n--- Recovery Telemetry ---")
    print(df['Connected_To'].apply(lambda x: x.split('_')[0]).value_counts().to_string())
    print(f"Successfully Recovered   : {len(recovered_ue_ids)}")
    print(f"Dropped (No Capacity)    : {len(df[df['Status'].str.contains('Dropped')])}")
    print(f"Network Resilience       : {resilience_pct:.1f}%")

    return df

if __name__ == "__main__":
    run_isolated_bs_failure_simulation(total_backup_rbs=10)
