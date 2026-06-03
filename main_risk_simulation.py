import numpy as np
import pandas as pd
import src.constants as const
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

def run_downlink_simulation():
    # 1. Initialize Topology
    bs_coords = get_hexagonal_bs(radius=2000)
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=100)

    # Prepare a list to hold the simulation data
    simulation_results = []

    # 2. Iterate through each UE to find its strongest signal
    for ue_id, ue_pos in enumerate(ue_coords):
        
        # Keep track of the best connection for this UE
        best_node_type = None
        best_node_idx = None
        max_rx_power_dbm = -np.inf
        best_distance = 0
        best_path_loss = 0
        
        # Store all Rx powers to calculate interference later (if connected to BS)
        bs_rx_powers_watts = []

        # --- A. Check all Terrestrial Base Stations (BS) ---
        for bs_id, bs_pos in enumerate(bs_coords):
            dist = distance_3D(bs_pos, ue_pos)
            pl_db = path_loss(dist, const.CARRIER_FREQ_GHZ)
            rx_power_dbm = const.TX_POWER_GBS_HAP - pl_db
            
            # Convert Rx power to linear Watts for potential interference math later
            rx_power_watts = 10 ** ((rx_power_dbm - 30) / 10)
            bs_rx_powers_watts.append(rx_power_watts)
            
            if rx_power_dbm > max_rx_power_dbm:
                max_rx_power_dbm = rx_power_dbm
                best_node_type = "BS"
                best_node_idx = bs_id
                best_distance = dist
                best_path_loss = pl_db

        # --- B. Check HAP ---
        dist_hap = distance_3D(hap_coord, ue_pos)
        pl_hap_db = free_space_path_loss(dist_hap, const.CARRIER_FREQ_GHZ)
        rx_power_hap_dbm = const.TX_POWER_GBS_HAP - pl_hap_db
        
        if rx_power_hap_dbm > max_rx_power_dbm:
            max_rx_power_dbm = rx_power_hap_dbm
            best_node_type = "HAP"
            best_node_idx = 0
            best_distance = dist_hap
            best_path_loss = pl_hap_db

        # --- C. Check LEO ---
        dist_leo = distance_3D(leo_coord, ue_pos)
        pl_leo_db = free_space_path_loss(dist_leo, const.CARRIER_FREQ_GHZ)
        rx_power_leo_dbm = const.TX_POWER_LEO - pl_leo_db
        
        if rx_power_leo_dbm > max_rx_power_dbm:
            max_rx_power_dbm = rx_power_leo_dbm
            best_node_type = "LEO"
            best_node_idx = 0
            best_distance = dist_leo
            best_path_loss = pl_leo_db

        # --- 3. Calculate SINR and Rate for the Best Connection ---
        # The paper assumes g_jn (antenna gain) is 1 for simplicity in basic setup, 
        # and we need the Rician factor K. Let's assume K=10 for line of sight.
        h_coef = channel_coefficient(antenna_gain = 1.0, path_loss_db = best_path_loss, rician_factor_k = 10)
        h_sq = np.abs(h_coef)**2
        
        # Convert Tx power of the best node to Watts
        if best_node_type == "BS" or best_node_type == "HAP":
            p_watts = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)
        else:
            p_watts = 10 ** ((const.TX_POWER_LEO - 30) / 10)

        # Calculate Interference
        if best_node_type == "BS":
            # Interference is sum of all OTHER BS signals
            interference_watts = sum(bs_rx_powers_watts) - bs_rx_powers_watts[best_node_idx]
        else:
            # Paper states HAP and satellite links are interference-free
            interference_watts = 0.0 

        # Calculate Noise density in Watts/Hz
        noise_density_watts = 10 ** ((const.NOISE_SPECTRAL_DENSITY_DBM - 30) / 10)
        
        # Final SINR and Rate
        sinr_linear = sinr(p_watts, h_sq, interference_watts, noise_density_watts, const.BANDWIDTH_HZ)
        sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else 0
        rate_bps = rate(const.BANDWIDTH_HZ, sinr_linear)

        # 4. Save results for this UE
        simulation_results.append({
            "UE_ID": ue_id,
            "Connected_To": f"{best_node_type}_{best_node_idx}",
            "Distance (m)": round(best_distance, 2),
            "Path Loss (dB)": round(best_path_loss, 2),
            "Rx Power (dBm)": round(max_rx_power_dbm, 2),
            "SINR (dB)": round(sinr_db, 2),
            "Rate (Mbps)": round(rate_bps / 1e6, 2)
        })

    # Convert to Pandas DataFrame for a beautiful printed table
    df_results = pd.DataFrame(simulation_results)
    
    print("\n--- Downlink Simulation Results ---")
    print(df_results.to_string(index=False))
    
    # Print a quick summary of how many UEs connected to each node type
    print("\n--- Connection Summary ---")
    print(df_results['Connected_To'].apply(lambda x: x.split('_')[0]).value_counts())

if __name__ == "__main__":
    run_downlink_simulation()