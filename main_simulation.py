import numpy as np
import pandas as pd

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.system_model import (
    distance_3D, 
    path_loss, 
    free_space_path_loss, 
    channel_coefficient, 
    sinr, 
    rate, 
    error_probability
)
import src.constants as const

from src.risk_profiles import generate_bs_failure_scenario
from mc_scheduler import allocate_backup_paths

def run_integrated_simulation(total_backup_rbs_per_node = 20):
    # --- Step 1: Initialize Network Geometry & Topology ---
    bs_coords = get_hexagonal_bs(radius=2000)
    hap_coord, leo_coord = get_ntn_nodes()
    
    np.random.seed(42) 
    ue_coords = get_random_users(n=100)

    # --- Step 2: Inject the Isolated BS Failure Profile ---
    affected_users_profile = generate_bs_failure_scenario(total_users=100)
    affected_ue_ids = [user["ue_id"] for user in affected_users_profile]
    user_to_failed_bs_map = {user["ue_id"]: user["primary_gbs"] for user in affected_users_profile}

    simulation_results = []

    # Map the active infrastructure according to risk_profiles.py (4 active, 3 failed)
    active_gbs_names = ["GBS_3", "GBS_4", "GBS_5", "GBS_6"] 
    active_nodes = ["HAP", "LEO"] + active_gbs_names

    # Convert constants from dBm to linear Watts for accurate physics calculations
    noise_density_watts = 10 ** ((const.NOISE_SPECTRAL_DENSITY_DBM - 30) / 10)
    p_hap_watts = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)
    p_leo_watts = 10 ** ((const.TX_POWER_LEO - 30) / 10)
    p_gbs_watts = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)
    
    M = const.MODULATION_M # 16-QAM

    # --- Step 3: Compute Physical Link Reliability & Capacity for ALL users ---
    for ue_id, ue_pos in enumerate(ue_coords):
        is_stranded = ue_id in affected_ue_ids
        primary_bs_risk = user_to_failed_bs_map.get(ue_id, "None")
        
        # 1. NTN Calculations 
        
        # HAP
        dist_hap = distance_3D(hap_coord, ue_pos)
        pl_hap_db = free_space_path_loss(dist_hap, const.CARRIER_FREQ_GHZ)
        h_hap = channel_coefficient(antenna_gain_db=32.0, path_loss_db=pl_hap_db, k_factor_db=15.0)
        
        snr_hap_linear = sinr(p_hap_watts, np.abs(h_hap)**2, interference_power=0, 
                              noise_density=noise_density_watts, bandwidth=const.BANDWIDTH_HZ)
        eps_hap = error_probability(snr_hap_linear, M)
        
        if is_stranded and (1 - eps_hap) >= const.RELIABILITY_THRESHOLD:
            for u in affected_users_profile:
                if u["ue_id"] == ue_id:
                    u["a_in_HAP"] = (1 - eps_hap) * (1 - const.BACKHAUL_ERROR_PROB)
        
        # LEO 
        dist_leo = distance_3D(leo_coord, ue_pos)
        pl_leo_db = free_space_path_loss(dist_leo, const.CARRIER_FREQ_GHZ)
        h_leo = channel_coefficient(antenna_gain_db=38.0, path_loss_db=pl_leo_db, k_factor_db=15.0)
        
        snr_leo_linear = sinr(p_leo_watts, np.abs(h_leo)**2, interference_power=0, 
                              noise_density=noise_density_watts, bandwidth=const.BANDWIDTH_HZ)
        eps_leo = error_probability(snr_leo_linear, M)
        
        if is_stranded and (1 - eps_leo) >= const.RELIABILITY_THRESHOLD:
            for u in affected_users_profile:
                if u["ue_id"] == ue_id:
                    u["a_in_LEO"] = (1 - eps_leo) * (1 - const.BACKHAUL_ERROR_PROB)

        # Track max received power in dBm for analytics
        max_rx_power_dbm = max(
            10 * np.log10(p_hap_watts * np.abs(h_hap)**2) + 30,
            10 * np.log10(p_leo_watts * np.abs(h_leo)**2) + 30
        )

        # 2. Terrestrial Calculations (3GPP TR 38.901 - Dynamic K)
        active_rx_powers_watts = {}
        active_h_sq = {}
        
        for bs_id, bs_pos in enumerate(bs_coords):
            gbs_name = f"GBS_{bs_id}"
            if gbs_name in active_gbs_names:
                dist_gbs = distance_3D(bs_pos, ue_pos)
                pl_gbs_db = path_loss(dist_gbs, const.CARRIER_FREQ_GHZ)
                
                # 3GPP UMa LOS Distribution: Mean=9 dB, Std=3.5 dB
                k_db = np.random.normal(9.0, 3.5)
                
                # Evaluate terrestrial link (GBS antenna gain = 16.84 dBi)
                h_gbs_mag = channel_coefficient(antenna_gain_db=16.84, path_loss_db=pl_gbs_db, 
                                                k_factor_db=k_db)
                
                rx_power_w = p_gbs_watts * (h_gbs_mag**2)
                active_rx_powers_watts[gbs_name] = rx_power_w
                active_h_sq[gbs_name] = (h_gbs_mag**2)

        best_sinr_linear = 0

        for target_gbs, rx_power_w in active_rx_powers_watts.items():
            # Interference is the sum of received power from all other active terrestrial towers
            interference_w = sum(p for name, p in active_rx_powers_watts.items() if name != target_gbs)
            
            sinr_lin = sinr(p_gbs_watts, active_h_sq[target_gbs], interference_w, 
                            noise_density_watts, const.BANDWIDTH_HZ)
            
            eps_gbs = error_probability(sinr_lin, M)
            
            if is_stranded and (1 - eps_gbs) >= const.RELIABILITY_THRESHOLD:
                for u in affected_users_profile:
                    if u["ue_id"] == ue_id:
                        u[f"a_in_{target_gbs}"] = (1 - eps_gbs) * (1 - const.BACKHAUL_ERROR_PROB)
                    
            if sinr_lin > best_sinr_linear:
                best_sinr_linear = sinr_lin
                
            rx_dbm = 10 * np.log10(rx_power_w) + 30
            if rx_dbm > max_rx_power_dbm:
                max_rx_power_dbm = rx_dbm

        # 3. Capacity Assignment 
        
        ue_capacity_mbps = 0.0
        
        if is_stranded:
            status = "Stranded"
            final_conn = "None"
        else:
            status = "Healthy"
            final_conn = "Terrestrial_BS_Primary"
            if best_sinr_linear > 0:
                # Calculate Shannon Capacity in Mbps
                ue_capacity_bps = rate(const.BANDWIDTH_HZ, best_sinr_linear)
                ue_capacity_mbps = ue_capacity_bps / 1e6 

        simulation_results.append({
            "UE_ID": ue_id,
            "Primary_BS_Risk": primary_bs_risk,
            "Max_Rx_Power_dBm": round(max_rx_power_dbm, 2),
            "Capacity_Mbps": round(ue_capacity_mbps, 2),
            "Status": status,
            "Final_Connection": final_conn
        })

    df = pd.DataFrame(simulation_results)

    # --- Step 4: Execute Algorithm 1 (Capacity Allocator) ---
    resilience_pct, scheduled_paths, recovered_users_profile = allocate_backup_paths(
        affected_users_profile, 
        active_nodes,
        total_backup_rbs_per_node
    )
    
    for user in recovered_users_profile:
        idx = df.index[df['UE_ID'] == user['ue_id']]
        if user.get("final_connection") == "Dropped":
            df.loc[idx, 'Status'] = 'Dropped (Threshold Failed / Capacity Exceeded)'
            df.loc[idx, 'Final_Connection'] = 'None'
        else:
            df.loc[idx, 'Status'] = 'Rescued'
            df.loc[idx, 'Final_Connection'] = user.get("final_connection")

    # --- Step 5: Print Consolidated Output Summary ---
    system_sum_capacity = df['Capacity_Mbps'].sum()

    print(f"\nSIMULATION RESULTS: {total_backup_rbs_per_node} RBs PER NODE")
    print("Terrestrial : UMa Stochastic Fading (μ=9dB, σ=3.5dB)")
    
    print("\nCapacity & Telemetry ")
    print(f"Total Network Users      : {len(df)}")
    print(f"Terrestrial Safe Users   : {len(df[df['Status'] == 'Healthy'])}")
    print(f"Affected Users           : {len(affected_ue_ids)}")
    print(f"System Sum Capacity      : {system_sum_capacity:.2f} Mbps")
    
    print("\nRecovery Breakdown ")
    recovered_df = df[df['Status'] == 'Rescued']
    if not recovered_df.empty:
        print(recovered_df['Final_Connection'].apply(lambda x: str(x).split('_Shared_Pool')[0]).value_counts().to_string())
    else:
        print("No users recovered.")
        
    print(f"\nNetwork Resilience       : {resilience_pct:.1f}%")
    return df

if __name__ == "__main__":
    run_integrated_simulation(total_backup_rbs_per_node = 10)