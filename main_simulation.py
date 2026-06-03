# import numpy as np
# import pandas as pd
# import src.constants as const
# from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
# from src.system_model import (
#     distance_3D, path_loss, free_space_path_loss,
#     channel_coefficient, sinr, rate
# )
# # Import your modular risk profiles
# from src.risk_profiles import (
#     apply_hardware_and_weather_risks, 
#     apply_capacity_congestion,
#     SINR_OUTAGE_THRESHOLD_DB
# )

# def run_downlink_simulation(active_risk="None"):
    
#     bs_coords = get_hexagonal_bs(radius=2000)
#     hap_coord, leo_coord = get_ntn_nodes()
    
#     # Fixed seed so the 100 users are in the exact same spot for all 5 tests
#     np.random.seed(42) 
#     ue_coords = get_random_users(n=100)

#     simulation_results = []

#     for ue_id, ue_pos in enumerate(ue_coords):
#         best_node_type, best_node_idx = None, None
#         max_rx_power_dbm = -np.inf
#         best_distance, best_path_loss = 0, 0
#         bs_rx_powers_watts = []

#         # --- A. Evaluate Terrestrial Network (Primary Path) ---
#         for bs_id, bs_pos in enumerate(bs_coords):
#             dist = distance_3D(bs_pos, ue_pos)
#             pl_db = path_loss(dist, const.CARRIER_FREQ_GHZ)
            
#             rx_power_dbm = const.TX_POWER_GBS_HAP - pl_db
#             rx_power_watts = 10 ** ((rx_power_dbm - 30) / 10)
#             bs_rx_powers_watts.append(rx_power_watts)
            
#             # Lock in the healthy baseline first
#             if rx_power_dbm > max_rx_power_dbm:
#                 max_rx_power_dbm = rx_power_dbm
#                 best_node_type = "BS"
#                 best_node_idx = bs_id
#                 best_distance = dist
#                 best_path_loss = pl_db

#         # --- B. Apply Risk Overrides to Primary Path ---
#         if best_node_type == "BS":
#             modified_pl_db = apply_hardware_and_weather_risks(active_risk, best_node_idx, ue_pos, best_path_loss)
            
#             if modified_pl_db != best_path_loss:
#                 best_path_loss = modified_pl_db
#                 max_rx_power_dbm = const.TX_POWER_GBS_HAP - modified_pl_db
#                 if max_rx_power_dbm < -100 or modified_pl_db == np.inf:
#                     best_node_type = None  # Connection completely severed

#         # --- C. Calculate Interim SINR & Apply Cell-Edge Risk ---
#         if best_node_type == "BS":
#             p_watts = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)
#             interference_watts = sum(bs_rx_powers_watts) - bs_rx_powers_watts[best_node_idx]
#             noise_watts = 10 ** ((const.NOISE_SPECTRAL_DENSITY_DBM - 30) / 10)
#             h_coef = channel_coefficient(1.0, best_path_loss, 10)
            
#             sinr_linear = sinr(p_watts, np.abs(h_coef)**2, interference_watts, noise_watts, const.BANDWIDTH_HZ)
#             sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else 0
            
#             if active_risk == "Low_SINR" and sinr_db < SINR_OUTAGE_THRESHOLD_DB:
#                 max_rx_power_dbm = -np.inf
#                 best_node_type = None 

#         # --- D. Evaluate NTN Backup Options (If Ground Failed) ---
#         if best_node_type is None:
#             dist_hap = distance_3D(hap_coord, ue_pos)
#             pl_hap_db = free_space_path_loss(dist_hap, const.CARRIER_FREQ_GHZ)
#             rx_power_hap_dbm = const.TX_POWER_GBS_HAP - pl_hap_db
            
#             if rx_power_hap_dbm > max_rx_power_dbm:
#                 max_rx_power_dbm, best_node_type, best_node_idx = rx_power_hap_dbm, "HAP", 0
#                 best_distance, best_path_loss = dist_hap, pl_hap_db

#             dist_leo = distance_3D(leo_coord, ue_pos)
#             pl_leo_db = free_space_path_loss(dist_leo, const.CARRIER_FREQ_GHZ)
#             rx_power_leo_dbm = const.TX_POWER_LEO - pl_leo_db
            
#             if rx_power_leo_dbm > max_rx_power_dbm:
#                 max_rx_power_dbm, best_node_type, best_node_idx = rx_power_leo_dbm, "LEO", 0
#                 best_distance, best_path_loss = dist_leo, pl_leo_db

#         # --- E. FINAL TELEMETRY CALCULATIONS ---
#         final_interference = 0.0
#         final_p_watts = 0.0
        
#         if best_node_type == "BS":
#             final_p_watts = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)
#             final_interference = sum(bs_rx_powers_watts) - bs_rx_powers_watts[best_node_idx]
#         elif best_node_type == "HAP":
#             final_p_watts = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)
#         elif best_node_type == "LEO":
#             final_p_watts = 10 ** ((const.TX_POWER_LEO - 30) / 10)
            
#         noise_watts = 10 ** ((const.NOISE_SPECTRAL_DENSITY_DBM - 30) / 10)
#         final_h_coef = channel_coefficient(1.0, best_path_loss, 10)
        
#         if best_node_type is not None:
#             final_sinr_linear = sinr(final_p_watts, np.abs(final_h_coef)**2, final_interference, noise_watts, const.BANDWIDTH_HZ)
#             final_sinr_db = 10 * np.log10(final_sinr_linear) if final_sinr_linear > 0 else 0
#             final_rate_mbps = rate(const.BANDWIDTH_HZ, final_sinr_linear) / 1e6
#         else:
#             final_sinr_db, final_rate_mbps = 0, 0

#         # Append the rich data payload
#         simulation_results.append({
#             "UE_ID": ue_id,
#             "Connected_To": f"{best_node_type}_{best_node_idx}",
#             "Distance (m)": round(best_distance, 2),
#             "Path Loss (dB)": round(best_path_loss, 2),
#             "Rx Power (dBm)": round(max_rx_power_dbm, 2),
#             "SINR (dB)": round(final_sinr_db, 2),
#             "Rate (Mbps)": round(final_rate_mbps, 2),
#             "Status": "Healthy"
#         })

#     # --- F. Post-Processing & Capacity Limits ---
#     df = pd.DataFrame(simulation_results)
#     df = apply_capacity_congestion(active_risk, df)
    
#     # Update status for users who were routed to the sky
#     df.loc[(df['Connected_To'].str.contains('HAP|LEO')) & (df['Status'] == 'Healthy'), 'Status'] = f'Rescued ({active_risk})'

#     # --- Generate Terminal Output ---
#     print(f"\n[SCENARIO: {active_risk.upper()}]")
#     print(df['Connected_To'].apply(lambda x: x.split('_')[0]).value_counts().to_string())
    
#     rescued = df[df['Status'].str.contains('Rescued')]
#     if not rescued.empty:
#         print(f"\n -> {len(rescued)} users safely offloaded to NTN. Here is a sample of their telemetry:")
#         print(rescued[['UE_ID', 'Connected_To', 'Distance (m)', 'SINR (dB)', 'Rate (Mbps)', 'Status']].head().to_string(index=False))
#     print("-" * 60)

#     return df

# if __name__ == "__main__":
#     print("=" * 60)
#     print("   O-RAN NTN RESILIENCE SIMULATION SUITE")
#     print("=" * 60)
    
#     scenarios = [
#         "None",           
#         "GBS_Failure",    
#         "Weather_Hazard", 
#         "Low_SINR",       
#         "Limited_RBs"     
#     ]
    
#     for scenario in scenarios:
#         run_downlink_simulation(active_risk=scenario)


import random
import numpy as np
import pandas as pd
import src.constants as const
from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.system_model import (
    distance_3D, path_loss, free_space_path_loss,
    channel_coefficient, sinr, rate
)

from src.risk_profiles import (
    generate_failed_towers, 
    apply_bs_failure_risk,
    apply_risk_disjoint_grouping
)

def run_downlink_simulation(active_risk="None"):
    
    bs_coords = get_hexagonal_bs(radius=2000)
    total_bs_count = len(bs_coords)
    hap_coord, leo_coord = get_ntn_nodes()
    
    np.random.seed(42) 
    ue_coords = get_random_users(n=100)

    # ---------------------------------------------------------
    # 🚨 PRE-COMPUTE THE SYSTEMIC HARDWARE FAILURE
    # ---------------------------------------------------------
    failed_bs_ids = []
    if active_risk == "GBS_Failure":
        random.seed(42) 
        failed_bs_ids = generate_failed_towers(total_bs_count)
        print(f"\n[NETWORK ALERT] 50% Systemic Failure Initiated.")
        print(f"Offline Towers: {failed_bs_ids}")
    # ---------------------------------------------------------

    simulation_results = []

    for ue_id, ue_pos in enumerate(ue_coords):
        best_node_type, best_node_idx = None, None
        max_rx_power_dbm = -np.inf
        best_distance, best_path_loss = 0, 0
        bs_rx_powers_watts = []

        # --- A. Evaluate Terrestrial Network (Primary Path) ---
        for bs_id, bs_pos in enumerate(bs_coords):
            dist = distance_3D(bs_pos, ue_pos)
            pl_db = path_loss(dist, const.CARRIER_FREQ_GHZ)
            
            rx_power_dbm = const.TX_POWER_GBS_HAP - pl_db
            rx_power_watts = 10 ** ((rx_power_dbm - 30) / 10)
            bs_rx_powers_watts.append(rx_power_watts)
            
            if rx_power_dbm > max_rx_power_dbm:
                max_rx_power_dbm = rx_power_dbm
                best_node_type = "BS"
                best_node_idx = bs_id
                best_distance = dist
                best_path_loss = pl_db

        primary_bs_risk = f"BS_{best_node_idx}"

        # --- B. Apply 50% BS Failure Risk Overrides ---
        if best_node_type == "BS" and active_risk == "GBS_Failure":
            modified_pl_db = apply_bs_failure_risk(best_node_idx, failed_bs_ids, best_path_loss)
            
            if modified_pl_db != best_path_loss:
                best_path_loss = modified_pl_db
                max_rx_power_dbm = const.TX_POWER_GBS_HAP - modified_pl_db
                
                if max_rx_power_dbm < -100 or modified_pl_db == np.inf:
                    best_node_type = None  

        # --- C. Evaluate NTN Backup Options (If Ground Failed) ---
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

        # --- D. FINAL TELEMETRY CALCULATIONS ---
        final_interference = 0.0
        final_p_watts = 0.0
        
        if best_node_type == "BS":
            final_p_watts = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)
            final_interference = sum(bs_rx_powers_watts) - bs_rx_powers_watts[best_node_idx]
        elif best_node_type == "HAP":
            final_p_watts = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)
        elif best_node_type == "LEO":
            final_p_watts = 10 ** ((const.TX_POWER_LEO - 30) / 10)
            
        noise_watts = 10 ** ((const.NOISE_SPECTRAL_DENSITY_DBM - 30) / 10)
        final_h_coef = channel_coefficient(1.0, best_path_loss, 10)
        
        if best_node_type is not None:
            final_sinr_linear = sinr(final_p_watts, np.abs(final_h_coef)**2, final_interference, noise_watts, const.BANDWIDTH_HZ)
            final_sinr_db = 10 * np.log10(final_sinr_linear) if final_sinr_linear > 0 else 0
            final_rate_mbps = rate(const.BANDWIDTH_HZ, final_sinr_linear) / 1e6
        else:
            final_sinr_db, final_rate_mbps = 0, 0

        # Append the rich data payload
        simulation_results.append({
            "UE_ID": ue_id,
            "Primary_BS_Risk": primary_bs_risk,
            "Connected_To": f"{best_node_type}_{best_node_idx}",
            "Distance (m)": round(best_distance, 2),
            "Rx Power (dBm)": round(max_rx_power_dbm, 2),
            "SINR (dB)": round(final_sinr_db, 2),
            "Rate (Mbps)": round(final_rate_mbps, 2),
            "Status": "Healthy"
        })

    # --- E. Post-Processing & NTN Capacity Scheduling ---
    df = pd.DataFrame(simulation_results)
    
    if active_risk == "GBS_Failure":
        # 1. Run Algorithm 1
        successful_rescues, dropped_ues, _ = apply_risk_disjoint_grouping(df, total_hap_rbs=10)
        
        # 2. Update the DataFrame Statuses
        df.loc[df['UE_ID'].isin(successful_rescues), 'Status'] = 'Rescued (Shared NTN RB)'
        df.loc[df['UE_ID'].isin(dropped_ues), 'Connected_To'] = 'None'
        df.loc[df['UE_ID'].isin(dropped_ues), 'Status'] = 'Dropped (NTN Capacity Exceeded)'

    # --- Generate Terminal Output ---
    print("\n      FINAL NETWORK ASSOCIATION SUMMARY")
    print("=" * 45)
    print(df['Connected_To'].apply(lambda x: x.split('_')[0]).value_counts().to_string())
    print("-" * 45)
    
    rescued = df[df['Status'].str.contains('Rescued')]
    dropped = df[df['Status'].str.contains('Dropped')]
    
    print(f"Total Rescued by HAP: {len(rescued)}")
    print(f"Total Dropped (No RBs): {len(dropped)}")
    
    if not dropped.empty:
        print("\n[Sample of Dropped Users due to Capacity Constraints]")
        print(dropped[['UE_ID', 'Primary_BS_Risk', 'Status']].head().to_string(index=False))
    print("=" * 45 + "\n")

    return df

if __name__ == "__main__":
    print("=" * 60)
    print("   O-RAN NTN RESILIENCE SIMULATION SUITE")
    print("=" * 60)
    
    scenarios = [
        "None",           
        "GBS_Failure"    
    ]
    
    for scenario in scenarios:
        run_downlink_simulation(active_risk=scenario)