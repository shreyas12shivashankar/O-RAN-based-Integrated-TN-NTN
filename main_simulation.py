import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.system_model import (sinr, error_probability)
import src.constants as const
from src.risk_profiles import inject_heterogeneous_risks
from src.scheduler_final import allocate_backup_paths

from src.link_evaluator import get_all_link_budgets
from src.primary_path import evaluate_primary_connection
from src.mc_scheme import apply_mc_scheme
from src.visualization import plot_topology, plot_sinr_curve
from src.parameter_sweep import execute_design_contour_sweep, execute_resource_efficiency_benchmark


def evaluate_link(p_tx, h_sq, interference, dist, rho_wireless=1.0, rho_backhaul=1.0):
    """Calculates end-to-end availability, spectral efficiency, distance, SINR (dB), and error probability."""
    
    sinr_lin = sinr(p_tx, h_sq, interference, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_RB)
    
    safe_sinr_lin = max(sinr_lin, 1e-12)
    sinr_db = 10 * math.log10(safe_sinr_lin)
    capped_se = min(8.0, math.log2(1 + sinr_lin))
    
    # Calculate Reliability (psi_s) for each segment
    ep_wireless = error_probability(sinr_lin, const.MODULATION_M)
    psi_wireless = 1 - ep_wireless
    psi_backhaul = 1 - const.BACKHAUL_ERROR_PROB
     
    # Calculate E2E Availability 
    """For simplicty physical availablity (rho) is chosen to be unity (that means, 100% up-time) and 
    overall a_jn depends on reliability of segment (psi_s)"""
    # E2E Availability (a_jn) depends on 1 Wireless segment and 2 Backhaul segments between Core and UE
    a_jn = (psi_wireless * rho_wireless) * ((psi_backhaul * rho_backhaul) ** 2)
                
    return a_jn, capped_se, dist, sinr_db, ep_wireless


def run_simulation(num_users=const.NUM_UE, num_gbs=const.NUM_GBS, num_failed_gbs=1, fixed_backup_rbs=10, seed_val=None, verbose=False):
    """Executes a single simulation instance: network generation, failure injection, and backup path allocation."""
    if seed_val is not None:
        np.random.seed(seed_val)
    
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=num_gbs)
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=num_users, seed_val=seed_val)
    
    if num_failed_gbs == 3:
        failed_bs_indices = [4, 5, 6]
    elif num_failed_gbs == 1:
        failed_bs_indices = [0]
    else:
        failed_bs_indices = list(range(num_gbs - num_failed_gbs, num_gbs))
  
    active_nodes = ['HAP', 'LEO'] + [f'GBS_{i}' for i in range(num_gbs) if i not in failed_bs_indices]
    
    primary_df = evaluate_primary_connection(
        ue_coords, bs_coords, hap_coord, leo_coord, get_all_link_budgets
    )

    affected_users = inject_heterogeneous_risks(
        primary_df=primary_df,
        failed_bs_indices=failed_bs_indices,
        evaluate_link_func=evaluate_link,
        sinr_edge_threshold=5.0
    )
    
    recovered_count, allocated_loads, node_composition, assigned_scores, recovery_details = allocate_backup_paths(
        affected_users=affected_users, 
        active_nodes=active_nodes, 
        failed_bs_indices=failed_bs_indices,
        fixed_backup_rbs=fixed_backup_rbs,
        verbose=verbose
    )

    affected_count = len(affected_users)
    resilience = ((recovered_count / affected_count) * 100) if affected_count > 0 else 100.0
    
    return affected_users, resilience, allocated_loads, node_composition, assigned_scores, recovery_details


def plot_risk_disjoint_proof(node_composition):
    """A function to show the distrubution of users among the backup nodes"""
    backup_nodes = ['HAP', 'LEO']
    
    all_primaries = sorted({
        pri for node in backup_nodes if node in node_composition 
        for pri in node_composition[node].keys()
    })
    bottoms = np.zeros(len(backup_nodes))
    plt.figure(figsize=(7, 6))
    
    for primary in all_primaries:
        values = [node_composition.get(node, {}).get(primary, 0) for node in backup_nodes]
        plt.bar(backup_nodes, values, bottom=bottoms, label=f'Failed {primary}', width=0.4)
        bottoms += values

    plt.title(f'Risk-Disjoint Backup Allocation (51 Affected UEs)', fontsize=14, pad=15)
    plt.ylabel('Number of Allocated Users', fontsize=12)
    plt.yticks(range(0, int(max(bottoms)) + 5, 2))
    plt.legend(title="Injected Risk Type")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def run_monte_carlo_averaging(user_counts, num_gbs=7, fixed_backup_rbs=10, runs_per_scenario=50):
    """Runs Monte Carlo trials across varying user counts to determine average network resilience."""
    
    resilience_results = []
    global_affected = 0
    global_recovered = 0
    
    print(f"\nMonte Carlo Simulation ({runs_per_scenario} Runs/Point) | {num_gbs} GBS | {fixed_backup_rbs} RBs")
    print(f"{'Total UEs':<10} | {'Affected':<10} | {'Recovered':<10} | {'HAP_UEs':<8} | {'LEO_UEs':<8} | {'GBS_UEs':<8} | {'Network Resilience'}")
    
    for total_users in user_counts:
        runs = [
            run_simulation(num_users=total_users, num_gbs=num_gbs, fixed_backup_rbs=fixed_backup_rbs, seed_val=i, verbose=False)
            for i in range(runs_per_scenario)
        ]
            
        affected_counts = [len(r[0]) for r in runs]
        resilience_vals = [r[1] for r in runs]
        recovered_counts = [aff * (res / 100) for aff, res in zip(affected_counts, resilience_vals)]
        
        avg_affected = round(np.mean(affected_counts))
        avg_resilience = np.mean(resilience_vals)
        avg_recovered = round(np.mean(recovered_counts))
        
        avg_hap = round(np.mean([r[2].get('HAP', 0) for r in runs]))
        avg_leo = round(np.mean([r[2].get('LEO', 0) for r in runs]))
        avg_gbs = max(0, avg_recovered - (avg_hap + avg_leo))
        
        resilience_results.append(avg_resilience)
        global_affected += sum(affected_counts)
        global_recovered += sum(recovered_counts)
        
        print(f"{total_users:<10} | {avg_affected:<10} | {avg_recovered:<10} | {avg_hap:<8} | {avg_leo:<8} | {avg_gbs:<8} | {avg_resilience:.2f}%")
        
    weighted_avg = ((global_recovered / global_affected) * 100) if global_affected > 0 else 100.0
    print(f"TRUE WEIGHTED AVERAGE RESILIENCE : {weighted_avg:.2f}%\n")
    
    return resilience_results
    
    
def evaluate_mc_scheme():
    """Evaluates and visualizes primary and multi-connectivity path allocations."""  
    
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
    # debug_records = []
    # for _, row in primary_df[~primary_df['Is_NTN']].iterrows():
    #     d_jn = row['Dist_m']
    #     pl_jn = path_loss(d_jn, const.CARRIER_FREQ_GHZ)
        
    #     debug_records.append({
    #         'UE_ID': row['UE_ID'],
    #         'Primary_RU': row['Primary_RU'],
    #         'd_jn_m': d_jn,
    #         'PL_jn_dB': round(pl_jn, 2),
    #         'Gain_dBi': row['Gain_dBi'],
    #         '|w|^2': row['w_sq'],
    #         'h_sq_primary': f"{row['h_sq_jn']:.4e}",
    #         'Signal_W': f"{row['Signal_W']:.4e}",
    #         'Interf_W': f"{row['Interf_W']:.4e}",
    #         'Noise_W': f"{row['Noise_W']:.4e}",
    #         'SINR_dB': row['SINR_dB'],
    #         'Eps_Wireless': row['Eps_Wireless'],
    #         'a_jn': row['a_jn']
    #     })
    
    # debug_df = pd.DataFrame(debug_records)
    # print("\n" + "-"*100)
    # print("      DEBUG: COMPONENTS FOR PRIMARY GROUND CONNECTIONS")
    # print("      Equation: SINR = Signal_W / (Interf_W + Noise_W)")
    # print("-"*100)
    # print(debug_df.head(100).to_string(index=False)) 
    # print("-"*100 + "\n")
    # --------------------------------------------------------------------------------------------
    
    # Apply Multi-Connectivity Scheme
    mc_df = apply_mc_scheme(primary_df)
  
    # Print Summary
    print("\nNETWORK CONNECTION SUMMARY (MC SCHEME) ")
    pri_gbs = len(mc_df[~mc_df['Is_NTN']])
    pri_ntn = len(mc_df[mc_df['Is_NTN']])
    
    sec_counts = mc_df['Secondary_RU'].value_counts()
    sec_hap = sec_counts.get('HAP', 0)
    sec_leo = sec_counts.get('LEO', 0)
    sec_gbs = sum(sec_counts.get(f'GBS_{i}', 0) for i in range(const.NUM_GBS))
    
    print(f"Primary   -> GBS: {pri_gbs} UEs | NTN (HAP/LEO): {pri_ntn} UEs")
    print(f"Secondary -> GBS: {sec_gbs} UEs | HAP: {sec_hap} UEs | LEO: {sec_leo} UEs")
    print("-" * 60)
        
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
    
    # Primary connection plot   
    plot_topology(mc_df, bs_coords, hap_coord, leo_coord, ue_coords, 
                  title="Multi-Connectivity Path Allocation", plot_mode='primary')
    
    # Secondary connection plot              
    plot_topology(mc_df, bs_coords, hap_coord, leo_coord, ue_coords, 
                  title="Multi-Connectivity Path Allocation", plot_mode='secondary')


def evaluate_backup_scheme():
    """Function to visualize the Backup scheme network topology for failed GBSs"""
    np.random.seed(50)
    
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)  
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=const.NUM_UE)
    
    primary_df = evaluate_primary_connection(
        ue_coords, bs_coords, hap_coord, leo_coord, get_all_link_budgets
    )
    
    failed_gbs_list = [4, 5, 6] # The 3-GBS failure cluster
    _, _, _, _, _, recovery_details = run_simulation(
        num_users=const.NUM_UE, 
        num_gbs=const.NUM_GBS, 
        num_failed_gbs=3, 
        fixed_backup_rbs=10, 
        seed_val=50, 
        verbose=False
    )
    
    backup_plot_df = primary_df.copy()
    failed_names = [f'GBS_{i}' for i in failed_gbs_list]
    affected_mask = backup_plot_df['Primary_RU'].isin(failed_names)
    
    # Detach primary paths for failed nodes  
    backup_plot_df.loc[affected_mask, 'Primary_RU'] = None
    backup_plot_df['RU_Pos'] = backup_plot_df.apply(
        lambda row: None if pd.isna(row['Primary_RU']) else row['RU_Pos'], axis=1
    )
    
    backup_plot_df['Secondary_RU'] = None
    backup_plot_df['Sec_Is_NTN'] = False
    
    # Assign recovered backup routes
    for detail in recovery_details:
        display_ue = detail['ue_id'] if isinstance(detail['ue_id'], str) else f"UE_{detail['ue_id']:02d}"
        idx = backup_plot_df.index[backup_plot_df['UE_ID'] == display_ue]
        
        if len(idx) > 0:
            # Checks if this user's primary node was actually one of the failed GBSs
            if detail['primary_node'] in failed_names:
                backup_node = detail['backup_node']
                backup_plot_df.loc[idx, 'Secondary_RU'] = backup_node
                backup_plot_df.loc[idx, 'Sec_Is_NTN'] = backup_node in ['HAP', 'LEO']

    plot_topology(backup_plot_df, bs_coords, hap_coord, leo_coord, ue_coords, 
                  title="Primary TN Baseline (Catastrophic 3-GBS Failure)", plot_mode='primary')
                  
    plot_topology(backup_plot_df, bs_coords, hap_coord, leo_coord, ue_coords, 
                  title="Backup Connectivity Scheme (NTN Emergency Recovery)", plot_mode='secondary')


def print_backup_recovery_summary(recovery_details, affected_users):
    """Outputs recovery metrics, risk profile breakdown, and user-level allocation status."""
    total_affected = len(affected_users)
    total_recovered = sum(1 for u in recovery_details if u['is_recovered'])
    
    injected_counts = {}
    recovered_counts = {}
    
    for u in affected_users:
        rp = u['risk_profile']
        injected_counts[rp] = injected_counts.get(rp, 0) + 1
        
    for u in recovery_details:
        if u['is_recovered']:
            rp = u['risk_profile']
            recovered_counts[rp] = recovered_counts.get(rp, 0) + 1

    print("\n" + "="*85)
    print(" BACKUP CONNECTIVITY RECOVERED USERS SUMMARY ")
    print("="*85)
    print(f"Total Affected Users Injected : {total_affected}")
    print(f"Total Successfully Recovered  : {total_recovered}")
    print(f"Total Dropped (Outage)        : {total_affected - total_recovered}")
    print(f"Overall Recovery Ratio        : {(total_recovered/total_affected)*100:.2f}%\n")
    
    print(" BREAKDOWN BY RISK PROFILE ")
    print(f"{'Risk Profile':<15} | {'Injected':<8} | {'Recovered':<9} | {'Dropped':<7} | {'Recovery %'}")
    print("-" * 65)
    for rp, inj_cnt in sorted(injected_counts.items()):
        rec_cnt = recovered_counts.get(rp, 0)
        drop_cnt = inj_cnt - rec_cnt
        rec_pct = (rec_cnt / inj_cnt) * 100 if inj_cnt > 0 else 0.0
        print(f"{rp:<15} | {inj_cnt:<8} | {rec_cnt:<9} | {drop_cnt:<7} | {rec_pct:.1f}%")
    
    print("\n DETAILED AFFECTED USER LOG ")
    print(f"{'UE_ID':<8} | {'Assigned Risk':<15} | {'a_primary':<10} | {'Backup Node':<12} | {'a_backup':<10} | {'b_in (QoS)'}")
    print("-" * 85)
    for u in sorted(recovery_details, key=lambda x: x['ue_id']):
        ue_str = f"UE_{u['ue_id']:02d}" if isinstance(u['ue_id'], int) else str(u['ue_id'])
        a_prim_str = f"{u['a_primary']:.5f}"
        
        if u['is_recovered']:
            node_str = str(u['backup_node'])
            a_bck_str = f"{u['a_backup']:.5f}"
            b_in_str = f"{u['b_in']:.6f}"
        else:
            node_str = "-"
            a_bck_str = "-"
            b_in_str = "Not Recovered"
            
        print(f"{ue_str:<8} | {u['risk_profile']:<15} | {a_prim_str:<10} | {node_str:<12} | {a_bck_str:<10} | {b_in_str}")
    print("="*85 + "\n")
    
    
if __name__ == "__main__":
    
    evaluate_mc_scheme()
    evaluate_backup_scheme()
    plot_sinr_curve(min_dist=140.0, max_dist=800.0, num_points=200, target_gbs_idx=1)
    
    # Run the Monte Carlo simulation
    print("\n Running Full Monte Carlo Batch ")
    total_users = [70,100,140,210,280]
    
    execute_design_contour_sweep(run_simulation, monte_carlo_runs=5)
    
    res_10 = run_monte_carlo_averaging(total_users, num_gbs=7, fixed_backup_rbs=10, runs_per_scenario=50)
    res_20 = run_monte_carlo_averaging(total_users, num_gbs=7, fixed_backup_rbs=20, runs_per_scenario=50)
    
    affected_users_list, res, alloc_loads, node_comp, scores, details = run_simulation(
        num_users=100, num_gbs=7, fixed_backup_rbs=10, seed_val=50, verbose=False
    )
    print_backup_recovery_summary(details, affected_users_list)
    plot_risk_disjoint_proof(node_comp)
    
    # Run the Benchmark
    sweep_users = np.arange(50, 281, 30)
    execute_resource_efficiency_benchmark(user_range=sweep_users, num_gbs=7, seed_val=50)
        
