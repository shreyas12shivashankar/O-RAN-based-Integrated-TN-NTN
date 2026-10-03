import math
import numpy as np
import matplotlib.pyplot as plt

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.link_evaluator import get_all_link_budgets
from src.primary_path import evaluate_primary_connection
from src.risk_profiles import inject_heterogeneous_risks
from src.system_model import sinr, error_probability
import src.constants as const


def evaluate_link(p_tx, h_sq, interference, dist, rho_wireless=1.0, rho_backhaul=1.0):
   
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


def execute_design_contour_sweep(simulation_function, monte_carlo_runs=5):
    """Generates contour map sweeping user density against reserved backup RBs."""
    user_range = np.arange(70, 281, 35)
    rb_range = np.arange(5, 36, 5)
    
    X, Y = np.meshgrid(user_range, rb_range)
    Z = np.zeros_like(X, dtype=float)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            num_users = int(X[i, j])
            num_rbs = int(Y[i, j])
            
            point_scores = []
            
            for run_idx in range(monte_carlo_runs):
                returns = simulation_function(
                    num_users=num_users, 
                    num_gbs=7,
                    num_failed_gbs=3,
                    fixed_backup_rbs=num_rbs, 
                    seed_val=42 + run_idx, 
                    verbose=False
                )
                
                final_user_scores = returns[4]
                if isinstance(final_user_scores, dict) and len(final_user_scores) > 0:
                    point_scores.append(np.mean(list(final_user_scores.values())))
                else:
                    point_scores.append(0.0)
                    
            Z[i, j] = np.mean(point_scores)
                
    fig, ax = plt.subplots(figsize=(10, 7))
    cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1.0, 21), cmap='viridis', alpha=0.9)
    fig.colorbar(cf, ax=ax, label='Average E2E Availability ($b_{in}$)')
    
    target_thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]
    contours = ax.contour(X, Y, Z, levels=target_thresholds, colors=['black', 'black', 'red', 'orange', 'white'], linewidths=2.0)
    ax.clabel(contours, inline=True, fontsize=12, fmt='%1.2f')

    ax.set_xlabel('Total Users in Network', fontsize=12, weight='bold')
    ax.set_ylabel('Reserved Backup RBs per RU', fontsize=12, weight='bold')
    ax.set_title('Backup Capacity Dimensioning Map (3 GBS Failure)', fontsize=14, pad=15, weight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    
    
def execute_resource_efficiency_benchmark(user_range, num_gbs=7, seed_val=50):
    """Compares spectrum demand between Multi-Connectivity and Backup Connectivity across various users."""
    print("="*85)
    print(f"{'Total Users':<12} | {'MC Backup RBs':<15} | {'BC RBs (Case A)':<15} | {'BC RBs (Case B)':<15}")
    print("-" * 85)

    mc_backup_list = []
    bc_case_a_list = []
    bc_case_b_list = []

    for total_users in user_range:
        np.random.seed(seed_val)
        bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=num_gbs)
        hap_coord, leo_coord = get_ntn_nodes()
        ue_coords = get_random_users(n=total_users, seed_val=seed_val)

        mc_backup_demand = total_users 
        primary_df = evaluate_primary_connection(
            ue_coords, bs_coords, hap_coord, leo_coord, get_all_link_budgets
        )

        # Case A: 1 GBS Failure
        affected_a = inject_heterogeneous_risks(
            primary_df, failed_bs_indices=[0], evaluate_link_func=evaluate_link, sinr_edge_threshold=5.0
        )
        risk_counts_a = {}
        for u in affected_a:
            rp = u['risk_profile']
            risk_counts_a[rp] = risk_counts_a.get(rp, 0) + 1
        bc_demand_a = max(risk_counts_a.values()) if risk_counts_a else 0

        # Case B: 3 GBS Failure
        affected_b = inject_heterogeneous_risks(
            primary_df, failed_bs_indices=[4,5,6], evaluate_link_func=evaluate_link, sinr_edge_threshold=5.0
        )
        risk_counts_b = {}
        for u in affected_b:
            rp = u['risk_profile']
            risk_counts_b[rp] = risk_counts_b.get(rp, 0) + 1
        bc_demand_b = max(risk_counts_b.values()) if risk_counts_b else 0

        mc_backup_list.append(mc_backup_demand)
        bc_case_a_list.append(bc_demand_a)
        bc_case_b_list.append(bc_demand_b)

        print(f"{total_users:<12} | {mc_backup_demand:<15} | {bc_demand_a:<15} | {bc_demand_b:<15}")

    plt.figure(figsize=(11, 7))
    plt.plot(user_range, mc_backup_list, marker='o', linewidth=2, color='#1f77b4', label='Multi-Connectivity (MC)')
    plt.plot(user_range, bc_case_a_list, marker='s', linewidth=2.5, color='#2ca02c', label='BC Case A: Heterogeneous (1 GBS Fail)')
    plt.plot(user_range, bc_case_b_list, marker='^', linewidth=2.5, color='#d62728', linestyle='--', label='BC Case B: Catastrophic (3 GBS Fail)')
    plt.fill_between(user_range, mc_backup_list, bc_case_a_list, color='#2ca02c', alpha=0.15, label='Saved Spectrum (Case A)')

    
    plt.title('Resource Efficiency of MC vs BC', fontsize=15, pad=15)
    plt.xlabel('Total Network Users (N)', fontsize=12)
    plt.ylabel('Required MC secondary / Reserved backup RBs', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()