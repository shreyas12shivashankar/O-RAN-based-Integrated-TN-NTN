import numpy as np
import matplotlib.pyplot as plt

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.system_model import (
    distance_3D, path_loss, free_space_path_loss, channel_coefficient, 
    sinr, rate, error_probability, check_transmission_success
)
import src.constants as const
from mc_scheduler import allocate_backup_paths


def evaluate_link(p_tx, h_sq, interference, dist):
    
    """Evaluates physical URLLC constraints and returns success status and E2E physical availability (a_in)."""
    
    snr_lin = sinr(p_tx, h_sq, interference, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_HZ)
    cap_mbps = rate(const.BANDWIDTH_HZ, snr_lin) / 1e6
    
    eps = error_probability(snr_lin, const.MODULATION_M)
    a_in = (1 - eps) * (1 - const.BACKHAUL_ERROR_PROB)
    
    lat_success, _ = check_transmission_success(cap_mbps, dist, 64 , const.LATENCY_THRESHOLD * 1000)
    is_successful = (a_in >= const.RELIABILITY_THRESHOLD) and lat_success
    
    return is_successful, a_in


def run_simulation(num_users=const.NUM_UE, num_gbs=const.NUM_GBS, fixed_rb_value=10, seed_val=None, verbose=True):
    
    if seed_val is not None:
        np.random.seed(seed_val)
    
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=num_gbs)
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=num_users)
    
    failed_bs_indices = [6] 
    
    # 1. Map link availabilities(a_in) for affected users
    affected_users = []
    
    for ue_id, ue_pos in enumerate(ue_coords):
        dists_gbs = [distance_3D(bs_pos, ue_pos) for bs_pos in bs_coords]
        primary_idx = np.argmin(dists_gbs)
        
        if primary_idx not in failed_bs_indices:
            continue
            
        user_links = {"ue_id": ue_id, "primary_gbs": primary_idx}
        
        # Check NTN Links
        ntn_configs = [('HAP', hap_coord, 32.0, const.TX_POWER_HAP_W), 
                       ('LEO', leo_coord, 38.0, const.TX_POWER_LEO_W)]
        for name, coord, gain, p_tx in ntn_configs:
            d_ntn = distance_3D(coord, ue_pos)
            h_sq = channel_coefficient(gain, free_space_path_loss(d_ntn, const.CARRIER_FREQ_GHZ), 15.0)**2
            success, a_in = evaluate_link(p_tx, h_sq, 0.0, d_ntn)
            if success:
                user_links[name] = a_in
                
        # Check Neighboring GBS Links
        rx_powers = []
        for d_gbs in dists_gbs:
            h_sq = channel_coefficient(8.0, path_loss(d_gbs, const.CARRIER_FREQ_GHZ), np.random.normal(9.0, 3.5))**2
            rx_powers.append(const.TX_POWER_GBS_W * h_sq)
            
        for i in range(num_gbs):
            if i in failed_bs_indices: 
                continue 
            interference = sum(rx_powers) - rx_powers[i]
            success, a_in = evaluate_link(const.TX_POWER_GBS_W, rx_powers[i] / const.TX_POWER_GBS_W, interference, dists_gbs[i])
            if success:
                user_links[f'GBS_{i}'] = a_in
                
        affected_users.append(user_links)

    # 2. Call the Scheduler
    active_nodes = ['HAP', 'LEO'] + [f'GBS_{i}' for i in range(num_gbs) if i not in failed_bs_indices]
    
    recovered_count, allocated_loads = allocate_backup_paths(
        affected_users=affected_users, 
        active_nodes=active_nodes, 
        failed_bs_indices=failed_bs_indices,
        fixed_rb_value=fixed_rb_value,
        verbose=verbose
    )

    affected_count = len(affected_users)
    resilience = (recovered_count / affected_count * 100) if affected_count > 0 else 100.0
    
    if verbose:
        print("\n Recovery Summary ")
        print(f"Total Affected: {affected_count} | Total Rescued: {recovered_count}")
        for node, load in allocated_loads.items():
            if load > 0:
                print(f"  - {node:<6} rescued {load} UEs")
        print(f"Final Resilience: {resilience:.2f}%\n")
        
    return affected_count, resilience


def run_monte_carlo_averaging(num_gbs=7, fixed_rbs=10, runs_per_scenario=50):
    user_counts = [20, 50, 100, 250, 500, 750, 1000]
    resilience_results = []
    
    global_affected = 0
    global_recovered = 0
    
    print(f"\n Monte Carlo Simulation ({runs_per_scenario} Runs/Point) | {num_gbs} GBS | {fixed_rbs} RBs")
    print(f"{'Total UEs':<10} | {'Avg Affected UEs':<10} | {'Average Network Resilience':<15}")
    
    for total_users in user_counts:
        runs = [run_simulation(num_users=total_users, num_gbs=num_gbs, fixed_rb_value=fixed_rbs, seed_val=i, verbose=False)
                for i in range(runs_per_scenario)]
            
        avg_affected = np.mean([r[0] for r in runs])
        avg_resilience = np.mean([r[1] for r in runs])
        resilience_results.append(avg_resilience)
        
        global_affected += sum(r[0] for r in runs)
        global_recovered += sum(r[0] * (r[1] / 100) for r in runs)
        
        print(f"{total_users:<10} | {round(avg_affected):<10} | {avg_resilience:<15.2f}%")

    weighted_avg = (global_recovered / global_affected * 100) if global_affected > 0 else 100.0
    print(f"TRUE WEIGHTED AVERAGE RESILIENCE : {weighted_avg:.2f}%\n")
    
    return resilience_results


def generate_report(total_users, res_10_rb, res_20_rb):
    plt.figure(figsize=(10, 6))
    plt.plot(total_users, res_10_rb, marker='o', linestyle='-', color='#1f77b4', label='10 RBs per Backup Node')
    plt.plot(total_users, res_20_rb, marker='s', linestyle='-', color='#ff7f0e', label='20 RBs per Backup Node')
    plt.title('Average Network Resilience vs. Total Users (7 GBS Topology)', fontsize=14, pad=15)
    plt.xlabel('Total Users in Network', fontsize=12)
    plt.ylabel('Network Resilience (%)', fontsize=12)
    plt.xlim(100, 1000)
    plt.ylim(0, 105) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(" Running Single Verbose Debug Scenario ")
    run_simulation(num_users=250, num_gbs=7, fixed_rb_value=10, seed_val=42, verbose=True)
    
    print("\n Running Full Monte Carlo Batch ")
    res_10 = run_monte_carlo_averaging(num_gbs=7, fixed_rbs=10, runs_per_scenario=50)
    res_20 = run_monte_carlo_averaging(num_gbs=7, fixed_rbs=20, runs_per_scenario=50)
    
    total_users = [20, 50, 100, 250, 500, 750, 1000]
    generate_report(total_users, res_10, res_20)