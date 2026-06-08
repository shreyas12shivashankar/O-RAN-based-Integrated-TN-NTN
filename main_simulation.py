import numpy as np
import pandas as pd
import scipy.special as sp

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.system_model import distance_3D
import src.constants as const

from src.risk_profiles import generate_bs_failure_scenario
from mc_scheduler import allocate_backup_paths

def q_function(x):
    # Standard Q-function implementation via complementary error function
    return 0.5 * sp.erfc(x / np.sqrt(2))

def get_rician_fading_db(K=0.1):
    # Equation 4: w_jn represents small-scale Rician fading
    # Generates a random power gain penalty based on the Rician K-factor
    mu = np.sqrt(K / (K + 1))
    sigma = np.sqrt(1 / (2 * (K + 1)))
    x = np.random.normal(mu, sigma)
    y = np.random.normal(0, sigma)
    power_gain = x**2 + y**2
    return 10 * np.log10(power_gain)

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

    # Define the active network infrastructure (1 Surviving GBS + NTN to match 60/70% scenario)
    active_gbs_names = ["GBS_4"] 
    active_nodes = ["HAP", "LEO"] + active_gbs_names

    # --- Step 3: Compute Physical Link Reliability (Equations 2, 3, 4, 5, 7) ---
    noise_power_dbm = const.NOISE_SPECTRAL_DENSITY_DBM + 10 * np.log10(const.BANDWIDTH_HZ)
    M = const.MODULATION_M # 16-QAM

    for ue_id, ue_pos in enumerate(ue_coords):
        if ue_id in affected_ue_ids:
            primary_bs_risk = user_to_failed_bs_map[ue_id]
            max_rx_power_dbm = -np.inf
            
            # --- NTN Calculations: HAP (Eq 3 & Eq 4 Fading) ---
            dist_hap = distance_3D(hap_coord, ue_pos)
            pl_hap_db = 32.45 + 20 * np.log10(dist_hap) + 20 * np.log10(const.CARRIER_FREQ_GHZ)
            
            # Add dynamic Rician fading (w_jn) to simulate physical signal degradation
            rx_power_hap_dbm = const.TX_POWER_GBS_HAP - pl_hap_db + get_rician_fading_db(K=0.1)
            snr_hap_linear = 10 ** ((rx_power_hap_dbm - noise_power_dbm) / 10)
            
            val_hap = np.sqrt((3 * snr_hap_linear * np.log2(M)) / (M - 1))
            eps_hap = (4 / np.log2(M)) * q_function(val_hap)
            
            # PHYSICAL LINK FILTER: Ensure reliability meets Table I threshold
            if (1 - eps_hap) >= const.RELIABILITY_THRESHOLD:
                a_in_hap = (1 - eps_hap) * (1 - const.BACKHAUL_ERROR_PROB)
                for u in affected_users_profile:
                    if u["ue_id"] == ue_id:
                        u["a_in_HAP"] = a_in_hap
            
            # --- NTN Calculations: LEO (Eq 3 & Eq 4 Fading) ---
            dist_leo = distance_3D(leo_coord, ue_pos)
            pl_leo_db = 32.45 + 20 * np.log10(dist_leo) + 20 * np.log10(const.CARRIER_FREQ_GHZ)
            
            rx_power_leo_dbm = const.TX_POWER_LEO - pl_leo_db + get_rician_fading_db(K=0.1)
            snr_leo_linear = 10 ** ((rx_power_leo_dbm - noise_power_dbm) / 10)
            
            val_leo = np.sqrt((3 * snr_leo_linear * np.log2(M)) / (M - 1))
            eps_leo = (4 / np.log2(M)) * q_function(val_leo)
            
            # PHYSICAL LINK FILTER
            if (1 - eps_leo) >= const.RELIABILITY_THRESHOLD:
                a_in_leo = (1 - eps_leo) * (1 - const.BACKHAUL_ERROR_PROB)
                for u in affected_users_profile:
                    if u["ue_id"] == ue_id:
                        u["a_in_LEO"] = a_in_leo
                        
            max_rx_power_dbm = max(rx_power_hap_dbm, rx_power_leo_dbm)

            # --- Surviving Terrestrial Calculations (Eq 2, 4, and 5) ---
            active_rx_powers_linear = {}
            for bs_id, bs_pos in enumerate(bs_coords):
                gbs_name = f"GBS_{bs_id}"
                if gbs_name in active_gbs_names:
                    dist_gbs = distance_3D(bs_pos, ue_pos)
                    pl_gbs_db = 28 + 22 * np.log10(dist_gbs) + 20 * np.log10(const.CARRIER_FREQ_GHZ)
                    
                    rx_power_dbm = const.TX_POWER_GBS_HAP - pl_gbs_db + get_rician_fading_db(K=0.1)
                    active_rx_powers_linear[gbs_name] = 10 ** (rx_power_dbm / 10)

            noise_linear = 10 ** (noise_power_dbm / 10)
            
            for target_gbs, signal_power in active_rx_powers_linear.items():
                # Eq 5 Denominator: Interference + Noise
                interference_linear = sum(
                    power for name, power in active_rx_powers_linear.items() if name != target_gbs
                )
                
                sinr_linear = signal_power / (interference_linear + noise_linear)
                
                val_gbs = np.sqrt((3 * sinr_linear * np.log2(M)) / (M - 1))
                eps_gbs = (4 / np.log2(M)) * q_function(val_gbs)
                
                # PHYSICAL LINK FILTER
                if (1 - eps_gbs) >= const.RELIABILITY_THRESHOLD:
                    a_in_gbs = (1 - eps_gbs) * (1 - const.BACKHAUL_ERROR_PROB)
                    for u in affected_users_profile:
                        if u["ue_id"] == ue_id:
                            u[f"a_in_{target_gbs}"] = a_in_gbs
                        
                rx_dbm = 10 * np.log10(signal_power)
                if rx_dbm > max_rx_power_dbm:
                    max_rx_power_dbm = rx_dbm

            simulation_results.append({
                "UE_ID": ue_id,
                "Primary_BS_Risk": primary_bs_risk,
                "Max_Rx_Power (dBm)": round(max_rx_power_dbm, 2),
                "Status": "Stranded"
            })
            
        else:
            simulation_results.append({
                "UE_ID": ue_id,
                "Primary_BS_Risk": "None",
                "Max_Rx_Power (dBm)": -1, 
                "Status": "Healthy",
                "Final_Connection": "Terrestrial_BS_Primary"
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
    print(f"\n========= SIMULATION RESULTS: {total_backup_rbs_per_node} RBs PER NODE =========")
    print("--- Disaster Profile ---")
    print(f"Total Network Users      : {len(df)}")
    print(f"Terrestrial Safe Users   : {len(df[df['Status'] == 'Healthy'])}")
    print(f"Disaster Victims         : {len(affected_ue_ids)}")
    print("\n--- Recovery Telemetry ---")
    
    recovered_df = df[df['Status'] == 'Rescued']
    if not recovered_df.empty:
        print(recovered_df['Final_Connection'].apply(lambda x: str(x).split('_Shared_Pool')[0]).value_counts().to_string())
    else:
        print("No users recovered.")
        
    print(f"\nNetwork Resilience       : {resilience_pct:.1f}%")

    return df

if __name__ == "__main__":
    # Test with 20 RBs to see the Rician Physics Bottleneck (caps around 70%)
    run_integrated_simulation(total_backup_rbs_per_node = 20)