import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.system_model import (
    distance_3D, path_loss, free_space_path_loss, channel_coefficient, 
    sinr, rate, error_probability, check_transmission_success,
        GAIN_HAP_DBI, GAIN_LEO_DBI,
        K_HAP_STATIC, K_LEO_STATIC
)
import src.constants as const
from src.risk_profiles import inject_bs_failure
from src.scheduler_final import allocate_backup_paths

from src.link_evaluator import get_all_link_budgets
from src.primary_path import evaluate_primary_connection
from src.mc_scheme import apply_mc_scheme
from src.visualization import plot_topology, plot_deterministic_sinr_curve
from src.parameter_sweep import execute_design_contour_sweep, run_mc_vs_bc_sweep, plot_mc_vs_bc_comparison

NUM_DROPS = 50 
all_drops_data = []


def evaluate_link(p_tx, h_sq, interference, dist, rho_wireless=1.0, rho_backhaul=1.0):
   
    # Calculate Capacity
    sinr_lin = sinr(p_tx, h_sq, interference, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_RB)
    
    safe_sinr_lin = max(sinr_lin, 1e-12)
    sinr_db = 10 * math.log10(safe_sinr_lin)
    
    if sinr_db < -5.0:
        return -1.0, 0.0, dist, sinr_db, 1.0
    
    MAX_SE = 8.0 # Maximum Spectral Efficiency of 8 bps/Hz with 256-QAM modulation
    capped_se  = min(MAX_SE, math.log2(1 + sinr_lin))
    
    # Calculate Reliability (psi_s) for each segment
    ep_wireless = error_probability(sinr_lin, const.MODULATION_M)
    psi_wireless = 1 - ep_wireless
    psi_backhaul = 1 - const.BACKHAUL_ERROR_PROB
     
    # Calculate E2E Availability 
    """For simplicty physical availablity (rho) is chosen to be unity (that means, 100% up-time) and 
    overall a_jn depends on reliability of segment (psi_s)"""
    a_jn = (psi_wireless * rho_wireless) * (psi_backhaul * rho_backhaul)
    
    sinr_db = 10 * math.log10(sinr_lin) 
            
    return a_jn, capped_se, dist, sinr_db, ep_wireless


def run_simulation(num_users=const.NUM_UE, num_gbs=const.NUM_GBS, num_failed_gbs=3, fixed_backup_rbs=10, seed_val=None, verbose=True):
    if seed_val is not None:
        np.random.seed(seed_val)
    
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=num_gbs)
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=num_users)
    
    failed_bs_indices = list(range(num_gbs - num_failed_gbs, num_gbs)) # Dynamic failure of ground base stations
    active_nodes = ['HAP', 'LEO'] + [f'GBS_{i}' for i in range(num_gbs) if i not in failed_bs_indices]
    
    # Evaluate the primary connection phase first to establish the baseline network state
    primary_df = evaluate_primary_connection(
        ue_coords, bs_coords, hap_coord, leo_coord, get_all_link_budgets
    )

    affected_users = inject_bs_failure(
        primary_df=primary_df,
        failed_bs_indices=failed_bs_indices,
        evaluate_link_func=evaluate_link
    )
    
    # Output from scheduler
    recovered_count, allocated_loads, node_composition, assigned_scores, recovery_details = allocate_backup_paths(
        affected_users=affected_users, 
        active_nodes=active_nodes, 
        failed_bs_indices=failed_bs_indices,
        fixed_backup_rbs=fixed_backup_rbs,
        verbose=verbose
    )

    affected_count = len(affected_users)
    resilience = ((recovered_count / affected_count) * 100) if affected_count > 0 else 100.0
    
    return affected_count, resilience, allocated_loads, node_composition, assigned_scores, recovery_details


def plot_risk_disjoint_proof(node_composition, num_users):
    """A function to show the distrubution of recovered users among the backup nodes"""
    backup_nodes = ['HAP', 'LEO', 'GBS_0', 'GBS_1', 'GBS_2', 'GBS_3']
    
    all_primaries = set()
    for node in backup_nodes:
        if node in node_composition:
            all_primaries.update(node_composition[node].keys())
    
    all_primaries = sorted(list(all_primaries))
    
    node_names = backup_nodes
    bottoms = np.zeros(len(node_names))
    
    plt.figure(figsize=(8, 6))
    
    for primary in all_primaries:
        values = [node_composition.get(node, {}).get(primary, 0) for node in node_names]
        plt.bar(node_names, values, bottom=bottoms, label=f'Failed {primary}')
        bottoms += values

    plt.title(f'Risk-Disjoint Backup Allocation ({num_users} UEs in the network)', fontsize=14, pad=15)
    plt.ylabel('Number of Allocated Users', fontsize=12)
    plt.yticks(range(0, int(max(bottoms)) + 5, 2))
    plt.legend(title="Original Primary Node")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def run_monte_carlo_averaging(user_counts, num_gbs=7, fixed_backup_rbs=10, runs_per_scenario=50):

    resilience_results = []
    
    global_affected = 0
    global_recovered = 0
    
    print(f"\nMonte Carlo Simulation ({runs_per_scenario} Runs/Point) | {num_gbs} GBS | {fixed_backup_rbs} RBs")
    print(f"{'Total UEs':<10} | {'Affected':<10} | {'Recovered':<10} | {'HAP':<6} | {'LEO':<6} | {'GBS':<6} | {'Network Resilience'}")
    
    for total_users in user_counts:
        runs = [run_simulation(num_users=total_users, num_gbs=num_gbs, fixed_backup_rbs=fixed_backup_rbs, seed_val=i, verbose=False)
                for i in range(runs_per_scenario)]
        
        if total_users == 100:
            first_run_details = runs[0][5] # Index 5 is recovery_details
            
            print("\n" + "-"*60)
            print(" NETWORK CONNECTION SUMMARY (BACKUP SCHEME with 100 UEs) ")
            print("-" * 60)
            print(f"{'UE_ID':<9} {'Failed_Pri_RU':<15} {'Backup_RU':<11} {'SINR(dB)':<10} {'a_in':<9} {'phi_i':<9} {'Final_b_in':<10}")

            sorted_details = sorted(first_run_details, key=lambda x: int(str(x["ue_id"]).split("_")[1]) if "_" in str(x["ue_id"]) else int(x["ue_id"]))
            
            for detail in sorted_details:
                display_ue = detail['ue_id'] if isinstance(detail['ue_id'], str) else f"UE_{detail['ue_id']:02d}"
                print(f"{display_ue:<9} {detail['primary_node']:<15} {detail['backup_node']:<11} "
                      f"{detail['sinr']:<10.2f} {detail['a_in']:<9.5f} {detail['phi_i']:<9.5f} {detail['b_in']:<9.5f}")
            print("-" * 60 + "\n")
            
        avg_affected = round(np.mean([r[0] for r in runs]))
        avg_resilience = np.mean([r[1] for r in runs])
        avg_recovered = round(np.mean([r[0] * (r[1] / 100) for r in runs]))
        
        avg_hap = round(np.mean([r[2].get('HAP', 0) for r in runs]))
        avg_leo = round(np.mean([r[2].get('LEO', 0) for r in runs]))
        
        avg_gbs = avg_recovered - (avg_hap + avg_leo)
        
        resilience_results.append(avg_resilience)
        
        global_affected += sum(r[0] for r in runs)
        global_recovered += sum(r[0] * (r[1] / 100) for r in runs)
        
        print(f"{total_users:<10} | {avg_affected:<10} | {avg_recovered:<10} | {avg_hap:<6} | {avg_leo:<6} | {avg_gbs:<6} | {avg_resilience:.2f}%")

    weighted_avg = ((global_recovered / global_affected) * 100) if global_affected > 0 else 100.0
    print(f"TRUE WEIGHTED AVERAGE RESILIENCE : {weighted_avg:.2f}%\n")
    
    return resilience_results


def generate_report(total_users, res_10_rb, res_20_rb):
    plt.figure(figsize=(10, 6))
    plt.plot(total_users, res_10_rb, marker='o', linestyle='-', color='#1f77b4', label='10 RBs per Backup RU')
    plt.plot(total_users, res_20_rb, marker='s', linestyle='-', color='#ff7f0e', label='20 RBs per Backup RU')
    plt.title('Average Network Resilience vs. Total Users (7 GBS Topology)', fontsize=14, pad=15)
    plt.xlabel('Total Users in Network', fontsize=12)
    plt.ylabel('Network Resilience (%)', fontsize=12)
    plt.xlim(40, 210)
    plt.ylim(0, 105) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()
    
    
def evaluate_mc_scheme():
    """Function to run primary and MC evaluation"""
    print("\n" + "-"*60)
    print("RUNNING MULTI-CONNECTIVITY ")
    print("-"*60)
    
    np.random.seed(50)
    
    # Generate Topology
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)  
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=const.NUM_UE)
    
    # Run Primary connection
    primary_df = evaluate_primary_connection(
        ue_coords, bs_coords, hap_coord, leo_coord, get_all_link_budgets
    )
    
    # Debugging Block (Uncomment to inspect primary ground connections):
    # ------------------------------------------------------------------------------------------
    debug_records = []
    for _, row in primary_df[~primary_df['Is_NTN']].iterrows():
        d_jn = row['Dist_m']
        pl_jn = path_loss(d_jn, const.CARRIER_FREQ_GHZ)
        
        debug_records.append({
            'UE_ID': row['UE_ID'],
            'Primary_RU': row['Primary_RU'],
            'd_jn_m': d_jn,
            'PL_jn_dB': round(pl_jn, 2),
            'Gain_dBi': row['Gain_dBi'],
            '|w|^2': row['w_sq'],
            'h_sq_primary': f"{row['h_sq_jn']:.4e}",
            'Signal_W': f"{row['Signal_W']:.4e}",
            'Interf_W': f"{row['Interf_W']:.4e}",
            'Noise_W': f"{row['Noise_W']:.4e}",
            'SINR_dB': row['SINR_dB'],
            'Eps_Wireless': row['Eps_Wireless'],
            'a_jn': row['a_jn']
        })
    
    debug_df = pd.DataFrame(debug_records)
    print("\n" + "-"*100)
    print("      DEBUG: COMPONENTS FOR PRIMARY GROUND CONNECTIONS")
    print("      Equation: SINR = Signal_W / (Interf_W + Noise_W)")
    print("-"*100)
    print(debug_df.head(300).to_string(index=False)) 
    print("-"*100 + "\n")
    # --------------------------------------------------------------------------------------------
    
    # Apply Multi-Connectivity Scheme
    mc_df = apply_mc_scheme(primary_df)
    
    # Debugging block: (Uncomment to inspect secondary connections)
    # --------------------------------------------------------------------------------------------
    # debug_sec_records = []
    
    # for _, row in mc_df.iterrows():
    #     sec_node = row['Secondary_RU']
        
    #     # Skip if no secondary connection was established
    #     if sec_node == 'None' or pd.isna(sec_node):
    #         continue
            
    #     # FIX: Fetch the raw link data from the ORIGINAL primary_df using the matching UE_ID
    #     primary_row = primary_df[primary_df['UE_ID'] == row['UE_ID']].iloc[0]
        
    #     all_links = {link['name']: link for link in primary_row['All_Links']}
    #     sec_link = all_links.get(sec_node)
        
    #     if sec_link:
    #         d_j_prime = sec_link['dist']
    #         is_ntn = sec_link['is_ntn']
            
    #         # Recalculate Path Loss for display
    #         if is_ntn:
    #             pl_j_prime = free_space_path_loss(d_j_prime, const.CARRIER_FREQ_GHZ)
    #         else:
    #             pl_j_prime = path_loss(d_j_prime, const.CARRIER_FREQ_GHZ)
                
    #         # Calculate Interference: 0 for NTN, (Total GBS RX - Target GBS RX) for Terrestrial
    #         if is_ntn:
    #             interf_w = 0.0
    #         else:
    #             interf_w = sum(primary_row['GBS_Powers']) - sec_link['rx_w']
                
    #         noise_w = const.NOISE_SPECTRAL_DENSITY_W * const.BANDWIDTH_RB
            
    #         # Recalculate SINR and Error Probability for display
    #         sinr_lin = sinr(sec_link['p_tx'], sec_link['h_sq'], interf_w, 
    #                         const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_RB)
    #         eps_wireless = error_probability(sinr_lin, const.MODULATION_M)
            
    #         debug_sec_records.append({
    #             'UE_ID': row['UE_ID'],
    #             'Secondary_RU': sec_node,
    #             'd_j_prime_m': round(d_j_prime, 2),
    #             'PL_j_prime_dB': round(pl_j_prime, 2),
    #             'h_sq_sec': f"{sec_link['h_sq']:.4e}",
    #             'Signal_W': f"{sec_link['rx_w']:.4e}",
    #             'Interf_W': f"{interf_w:.4e}",
    #             'Noise_W': f"{noise_w:.4e}",
    #             'Sec_SINR_dB': round(10 * math.log10(sinr_lin), 2),
    #             'Sec_Eps': f"{eps_wireless:.2e}",
    #             'Sec_a_jn': round(row['a_j_prime_n'], 5)
    #         })
            
    # debug_sec_df = pd.DataFrame(debug_sec_records)
    # print("\n" + "-"*100)
    # print("      DEBUG: COMPONENTS FOR SECONDARY CONNECTIONS")
    # print("      Equation: SINR = Signal_W / (Interf_W + Noise_W)")
    # print("-"*100)
    # print(debug_sec_df.head(20).to_string(index=False)) 
    # print("-"*100 + "\n")
    # ----------------------------------------------------------------------------------------------
  
    
   # Print Summary
    print("\nNETWORK CONNECTION SUMMARY (MC SCHEME) ")
    
    # Calculate Primary Distribution
    pri_gbs = len(mc_df[~mc_df['Is_NTN']])
    pri_ntn = len(mc_df[mc_df['Is_NTN']])
    
    # Calculate Secondary Distribution
    sec_counts = mc_df['Secondary_RU'].value_counts()
    sec_hap = sec_counts.get('HAP', 0)
    sec_leo = sec_counts.get('LEO', 0)
    sec_gbs = sum(sec_counts.get(f'GBS_{i}', 0) for i in range(const.NUM_GBS))
    
    print(f"Primary   -> GBS: {pri_gbs} UEs | NTN (HAP/LEO): {pri_ntn} UEs")
    print(f"Secondary -> GBS: {sec_gbs} UEs | HAP: {sec_hap} UEs | LEO: {sec_leo} UEs")
    print("-" * 60)
        
    # Rename columns for crystal-clear transparency
    display_df = mc_df.rename(columns={
        'SINR_dB': 'Pri_SINR_dB',
        'a_jn': 'Pri_a_jn',
        'a_j_prime_n': 'Sec_a_jn',
        'a_n': 'Final_MC_a_n'
    })
    
    cols_to_print = [
        'UE_ID', 'Primary_RU', 'Pri_SINR_dB', 'Pri_a_jn', 
        'Secondary_RU', 'Sec_SINR_dB', 'Sec_a_jn', 'Final_MC_a_n'
    ]
    
    print("\n" + display_df[cols_to_print].head(100).to_string(index=False))
    
    
    # Visualize
    plot_topology(mc_df, bs_coords, hap_coord, leo_coord, ue_coords, title="Multi-Connectivity Path Allocation")


def evaluate_backup_scheme():
    """Function to visualize the Backup scheme network topology"""
    print("\n" + "-"*60)
    print("RUNNING BACKUP SCHEME VISUALIZATION ")
    print("-"*60)
    
    # Lock seed to 0 to match evaluate_mc_scheme exactly
    np.random.seed(50)
    
    # Generate Topology
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)  
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=const.NUM_UE)
    
    # Run Primary connection
    primary_df = evaluate_primary_connection(
        ue_coords, bs_coords, hap_coord, leo_coord, get_all_link_budgets
    )
    
    # Run Backup Allocation (Assuming GBS 4, 5, 6 fail)
    failed_gbs_list = [4, 5, 6]
    _, _, _, _, _, recovery_details = run_simulation(
        num_users=const.NUM_UE, 
        num_gbs=const.NUM_GBS, 
        num_failed_gbs=3, 
        fixed_backup_rbs=10, 
        seed_val=50, 
        verbose=False
    )
    
    # Prepare dataframe for plotting
    backup_plot_df = primary_df.copy()
    
    failed_names = [f'GBS_{i}' for i in failed_gbs_list]
    affected_mask = backup_plot_df['Primary_RU'].isin(failed_names)
    
    backup_plot_df.loc[affected_mask, 'Primary_RU'] = None
    backup_plot_df['RU_Pos'] = backup_plot_df.apply(
        lambda row: None if pd.isna(row['Primary_RU']) else row['RU_Pos'], axis=1
    )
    
    backup_plot_df['Secondary_RU'] = None
    backup_plot_df['Sec_Is_NTN'] = False
    
    # Inject the backup paths into the dataframe
    for detail in recovery_details:
        display_ue = detail['ue_id'] if isinstance(detail['ue_id'], str) else f"UE_{detail['ue_id']:02d}"
        idx = backup_plot_df.index[backup_plot_df['UE_ID'] == display_ue]
        
        if len(idx) > 0:
            backup_node = detail['backup_node']
            backup_plot_df.loc[idx, 'Secondary_RU'] = backup_node
            backup_plot_df.loc[idx, 'Sec_Is_NTN'] = backup_node in ['HAP', 'LEO']
            
    # Visualize using the updated plot_topology
    plot_topology(backup_plot_df, bs_coords, hap_coord, leo_coord, ue_coords, 
                  title="Backup Scheme Path Allocation (GBS 4, 5 & 6 Failed)")


    
if __name__ == "__main__":
    
    # debug_print_all_user_candidates()
    
    # Run the primary and MC scheme evaluation
    evaluate_mc_scheme()
    
    # Run the Backup scheme visualization
    evaluate_backup_scheme()
    
    # GENERATE AND PLOT THE ENSEMBLE SINR CURVE
    
    plot_deterministic_sinr_curve(min_dist=140.0, max_dist=800.0, num_points=200, target_gbs_idx=0)
    
    print("\n" + "-"*60)
    print(" RUNNING BACKUP ALLOCATION SIMULATIONS ")
    print("-"*60)
    
    #debug_single_ntn_links()
    
    # Generate the visual proof of risk-disjoint for the case of 100 UEs
    _, _, _, single_run_composition,_,_= run_simulation(num_users=100, num_gbs=7, fixed_backup_rbs=10, seed_val=42, verbose=False)
    np.random.seed(50)
    plot_risk_disjoint_proof(single_run_composition, num_users=100)
    
    # Run the main Monte Carlo simulation
    print("\n Running Full Monte Carlo Batch ")
    total_users = [50,100,150,200]
    
    print("\nExecuting Parameter Sweeps")
    
    #execute_3d_capacity_sweep(run_simulation)
    execute_design_contour_sweep(run_simulation, monte_carlo_runs=5)
    
    # 5. Comparison Sweep: Multi-Connectivity vs. Backup Connectivity
    sweep_users = np.arange(50, 201, 30)
    comparison_metrics = run_mc_vs_bc_sweep(
        user_range=sweep_users, 
        failed_gbs_list=(4, 5, 6), 
        fixed_backup_rbs=10, 
        seed_val=50
    )
    plot_mc_vs_bc_comparison(comparison_metrics)

    res_10 = run_monte_carlo_averaging(total_users, num_gbs=7, fixed_backup_rbs=10, runs_per_scenario=50)
    res_20 = run_monte_carlo_averaging(total_users, num_gbs=7, fixed_backup_rbs=20, runs_per_scenario=50)
        
    generate_report(total_users, res_10, res_20)