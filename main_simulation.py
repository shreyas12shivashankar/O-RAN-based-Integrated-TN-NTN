import numpy as np
import pandas as pd

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.system_model import (
    distance_3D, path_loss, free_space_path_loss, channel_coefficient, 
    sinr, rate, error_probability, check_transmission_success
)
import src.constants as const
from src.risk_profiles import generate_bs_failure_scenario
from mc_scheduler import allocate_backup_paths

def run_simulation(num_users=const.NUM_UE, num_gbs=const.NUM_GBS, fixed_rb_value=10, seed_val=None, verbose=True):
    # Added verbose check to prevent terminal flooding
    if verbose:
        print(f"RUNNING SIMULATION WITH {num_gbs} GBS | FIXED RBs = {fixed_rb_value}")
    
    # 1. Initialize Parametric Network Geometry
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=num_gbs)
    hap_coord, leo_coord = get_ntn_nodes()
    
    # Apply the dynamic seed for Monte Carlo averaging
    if seed_val is not None:
        np.random.seed(seed_val)
    
    # Generate the dynamic number of users
    ue_coords = get_random_users(n=num_users)

    # Inject Failure: Drop GBS_0
    failed_bs = ["GBS_6"]
    
    # Recalculate affected pool dynamically based on new GBS limits
    affected_users_profile = []
    for ue_id, ue_pos in enumerate(ue_coords):
        distances = [np.linalg.norm(ue_pos - bs_pos) for bs_pos in bs_coords]
        closest_bs_idx = np.argmin(distances)
        primary_gbs = f"GBS_{closest_bs_idx}"
        if primary_gbs in failed_bs:
            affected_users_profile.append({"ue_id": ue_id, "primary_gbs": primary_gbs})
            
    affected_ids = [u["ue_id"] for u in affected_users_profile]

    # Map healthy nodes
    active_bs_names = [f"GBS_{i}" for i in range(num_gbs) if f"GBS_{i}" not in failed_bs]
    active_nodes = active_bs_names + ["HAP", "LEO"]
    
    # Enforce tight, uniform fixed resource constraints 
    rb_limits = {node: fixed_rb_value for node in active_nodes}

    simulation_results = []

    # 2. Physical Layer & Quality of Service Evaluation
    for ue_id, ue_pos in enumerate(ue_coords):
        is_affected = ue_id in affected_ids
        user_dict = next((u for u in affected_users_profile if u["ue_id"] == ue_id), None)
        
        max_rx_power_w = 0.0
        best_healthy_sinr = 0.0
        
        # Evaluate NTN Links
        ntn_configs = [("HAP", hap_coord, 32.0, const.TX_POWER_HAP_W), ("LEO", leo_coord, 38.0, const.TX_POWER_LEO_W)]
        for name, coord, gain, p_tx in ntn_configs:
            dist = distance_3D(coord, ue_pos)
            pl = free_space_path_loss(dist, const.CARRIER_FREQ_GHZ)
            h_mag = channel_coefficient(gain, pl, 15.0)
            
            if is_affected:
                snr_lin = sinr(p_tx, h_mag**2, 0, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_HZ)
                cap_mbps = rate(const.BANDWIDTH_HZ, snr_lin) / 1e6
                latency_success, _ = check_transmission_success(
                    cap_mbps, dist, packet_size_bytes=64, max_latency_ms=const.LATENCY_THRESHOLD * 1000
                )
                eps = error_probability(snr_lin, const.MODULATION_M)
                if (1 - eps) >= const.RELIABILITY_THRESHOLD and latency_success: 
                    user_dict[f"a_in_{name}"] = (1 - eps) * (1 - const.BACKHAUL_ERROR_PROB)

        # Evaluate Terrestrial Links
        gbs_powers = {}
        h_sq_dict = {}
        dists_dict = {}
        
        for bs_id, bs_pos in enumerate(bs_coords):
            gbs_name = f"GBS_{bs_id}"
            dist = distance_3D(bs_pos, ue_pos)
            pl = path_loss(dist, const.CARRIER_FREQ_GHZ)
            k_db = np.random.normal(9.0, 3.5)
            h_mag = channel_coefficient(8.0, pl, k_db)
            
            gbs_powers[gbs_name] = const.TX_POWER_GBS_W * (h_mag**2)
            h_sq_dict[gbs_name] = (h_mag**2)
            dists_dict[gbs_name] = dist
            
            if gbs_name in active_bs_names:
                max_rx_power_w = max(max_rx_power_w, gbs_powers[gbs_name])

        for target_gbs, rx_w in gbs_powers.items():
            if target_gbs not in active_bs_names:
                continue 
            interference = sum(p for name, p in gbs_powers.items() if name != target_gbs)
            snr_lin = sinr(const.TX_POWER_GBS_W, h_sq_dict[target_gbs], interference, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_HZ)
            cap_mbps = rate(const.BANDWIDTH_HZ, snr_lin) / 1e6
            
            if is_affected:
                eps = error_probability(snr_lin, const.MODULATION_M)
                latency_success, _ = check_transmission_success(
                    cap_mbps, dists_dict[target_gbs], packet_size_bytes=64, max_latency_ms=const.LATENCY_THRESHOLD * 1000
                )
                if (1 - eps) >= const.RELIABILITY_THRESHOLD and latency_success:
                    user_dict[f"a_in_{target_gbs}"] = (1 - eps) * (1 - const.BACKHAUL_ERROR_PROB)
            
            best_healthy_sinr = max(best_healthy_sinr, snr_lin)

        ue_cap_mbps = (rate(const.BANDWIDTH_HZ, best_healthy_sinr) / 1e6) if best_healthy_sinr > 0 else 0.0

        simulation_results.append({
            "UE_ID": ue_id,
            "Capacity_Mbps": round(ue_cap_mbps, 2) if not is_affected else 0.0,
            "Status": "Affected" if is_affected else "Healthy",
            "Final_Connection": "None" if is_affected else "Terrestrial_BS_Primary"
        })

    df = pd.DataFrame(simulation_results)

    # 3. Process Matrix Allocation under tight constraints
    resilience_perct, backup_paths, recovered_users = allocate_backup_paths(
        affected_users_profile, active_nodes, rb_distribution=rb_limits
    )
    
    for user in recovered_users:
        idx = df.index[df['UE_ID'] == user['ue_id']]
        if user.get("final_connection") == "Dropped":
            df.loc[idx, 'Status'] = 'Dropped'
        else:
            df.loc[idx, 'Status'] = 'Rescued'
            df.loc[idx, 'Final_Connection'] = user.get("final_connection")

    # 4. Display Core Metrics (Only if verbose is True)
    if verbose:
        print(f"Failed Base Stations : {', '.join(failed_bs) if failed_bs else 'None'}")
        print(f"Active Backup Nodes  : {', '.join(active_nodes)}")
        print(f"Total Affected Users : {len(affected_ids)}")
        print(f"Successfully Rescued : {len(df[df['Status'] == 'Rescued'])}")
        print(f"Dropped Users        : {len(df[df['Status'] == 'Dropped'])}")
        
        rec_df = df[df['Status'] == 'Rescued']
        if not rec_df.empty:
            print("\n--- RECOVERY BREAKDOWN ---")
            print(rec_df['Final_Connection'].apply(lambda x: str(x).split('_Shared_Pool')[0]).value_counts().to_string())
            
        print(f"\n[CRITICAL_METRIC] Network Resilience : {resilience_perct:.1f}%")
    
    return len(affected_ids), resilience_perct # Check point

def run_monte_carlo_averaging(num_gbs=7, fixed_rbs=10, runs_per_scenario=50):
    user_counts = [100, 250, 500, 750, 1000]
    
    print(f"\n--- MONTE CARLO SIMULATION ({runs_per_scenario} RUNS PER DATA POINT) ---")
    print(f"Topology: {num_gbs} GBS | Backup RBs: {fixed_rbs} | Failed Risk: GBS_0")
    print(f"{'Total UEs':<12} | {'Avg Affected UEs':<18} | {'Avg Network Resilience':<22}")
    
    total_affected = 0
    total_recovered = 0
    resilience_results = []
    
    for total_users in user_counts:
        cumulative_affected = 0
        cumulative_resilience = 0.0
        
        for iteration in range(runs_per_scenario):
            # Pass verbose=False to silence the individual run prints
            affected_count, resilience_perct = run_simulation(
                num_users=total_users, 
                num_gbs=num_gbs, 
                fixed_rb_value=fixed_rbs, 
                seed_val=iteration,
                verbose=False
            )
            
            cumulative_affected += affected_count
            cumulative_resilience += resilience_perct
            
            recovered_count = affected_count * (resilience_perct / 100)
            
            total_affected += affected_count
            total_recovered += recovered_count
            
        # Calculate the averages
        avg_affected = round(cumulative_affected / runs_per_scenario)
        avg_resilience = cumulative_resilience / runs_per_scenario
        resilience_results.append(avg_resilience)
        
        print(f"{total_users:<12} | {avg_affected:<18.0f} | {avg_resilience:<20.2f}%")

    if total_affected > 0:
        true_weighted_avg = (total_recovered / total_affected) * 100
    else:
        true_weighted_avg = 100.0
        
    print(f"TRUE WEIGHTED AVERAGE RESILIENCE : {true_weighted_avg:.2f}%")
    
    return resilience_results
        
if __name__ == "__main__":
    # Run the simulation exactly as the paper dictates
    res_10 = run_monte_carlo_averaging(num_gbs=7, fixed_rbs=10, runs_per_scenario=50)
    
    res_20 = run_monte_carlo_averaging(num_gbs=7, fixed_rbs=20, runs_per_scenario=50)
    
     # Save the output for plotting
    df = pd.DataFrame({
        "Total_UEs": [100, 250, 500, 750, 1000],
        "Resilience_10_RB": res_10,
        "Resilience_20_RB": res_20
    })
    df.to_csv("simulation_results.csv", index=False)
    print("Data successfully saved to simulation_results.csv")