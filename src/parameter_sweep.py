# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm


# # 3D PLOT: Capacity Boundary (3 RU Failures)

# def execute_3d_capacity_sweep(simulation_function):
#     print("Running 3D Capacity Boundary Sweep...")
    
#     user_range = np.arange(50, 400, 50)  # X-axis
#     rb_range = np.arange(5, 55, 5)       # Y-axis
    
#     X, Y = np.meshgrid(user_range, rb_range)
#     Z = np.zeros_like(X, dtype=float)
    
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             num_users = int(X[i, j])
#             num_rbs = int(Y[i, j])
            
#             # Force 3 failed GBS for the absolute stress test
#             returns = simulation_function(
#                 num_users=num_users, 
#                 num_gbs=7,
#                 num_failed_gbs=3,
#                 fixed_backup_rbs=num_rbs, 
#                 seed_val=42, 
#                 verbose=False
#             )
            
#             b_in_scores = [0.0]
#             for item in returns:
#                 if isinstance(item, dict) and len(item) > 0:
#                     first_val = list(item.values())[0]
#                     if isinstance(first_val, float):
#                         b_in_scores = list(item.values())
#                         break
            
#             Z[i, j] = sum(b_in_scores) / len(b_in_scores) if b_in_scores else 0.0

#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='k', linewidth=0.5, alpha=0.9)
    
#     ax.set_xlabel('Number of Users', fontsize=12, labelpad=10)
#     ax.set_ylabel('Reserved NTN RBs', fontsize=12, labelpad=10)
#     ax.set_zlabel('Average Availability ($b_{in}$)', fontsize=12, labelpad=10)
#     ax.set_title('TN-NTN Capacity Boundary (3 GBS Failed)', fontsize=14, pad=15)
    
#     # Constrain Z-axis from 0.5 to 1.0 to highlight the degradation slope
#     ax.set_zlim(0.0, 1.0)
#     fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Average availability($b_{in})')
#     ax.view_init(elev=25, azim=-135)
    
#     plt.tight_layout()
#     plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.link_evaluator import get_all_link_budgets
from src.primary_path import evaluate_primary_connection
from src.mc_scheme import apply_mc_scheme
from src.risk_profiles import inject_bs_failure
from src.scheduler_final import allocate_backup_paths
from src.system_model import sinr, error_probability, check_transmission_success
import src.constants as const
from scipy.ndimage import gaussian_filter

def execute_design_contour_sweep(simulation_function, monte_carlo_runs=10):
    """
    Sweeps User Density vs. Reserved NTN Backup RBs.
    Constrained to 50-200 users to stay below the 225-user / 450-RB hard limit.
    """
    print("\n" + "="*60)
    print("RUNNING SYSTEM DESIGN CONTOUR SWEEP (50–200 Users)")
    print("="*60)

    user_range = np.arange(50, 201, 25)
    rb_range = np.arange(5, 41, 5)
    
    X, Y = np.meshgrid(user_range, rb_range)
    Z = np.zeros_like(X, dtype=float)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            num_users = int(X[i, j])
            num_rbs = int(Y[i, j])
            
            point_scores = []
            
            # 2. Monte Carlo Averaging to eliminate the "bounce"
            for run_idx in range(monte_carlo_runs):
                returns = simulation_function(
                    num_users=num_users, 
                    num_gbs=7,
                    num_failed_gbs=3,
                    fixed_backup_rbs=num_rbs, 
                    seed_val=42 + run_idx, # Vary the seed to average out spatial randomness
                    verbose=False
                )
                
                final_user_scores = returns[4]
                if isinstance(final_user_scores, dict) and len(final_user_scores) > 0:
                    point_scores.append(np.mean(list(final_user_scores.values())))
                else:
                    point_scores.append(0.0)
                    
            Z[i, j] = np.mean(point_scores)
                
    # Render the Contour Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 1. Draw the filled background heatmap (Viridis colormap)
    cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1.0, 21), cmap='viridis', alpha=0.9)
    fig.colorbar(cf, ax=ax, label='Average E2E Availability ($b_{in}$)')
    
    # 2. Draw the critical Design Boundary Lines (e.g., 0.80, 0.90, 0.95)
    target_thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]
    contours = ax.contour(X, Y, Z, levels=target_thresholds, colors=['black', 'black', 'red', 'orange', 'white'], linewidths=2.0)
    
    # Label the contour lines directly on the graph
    ax.clabel(contours, inline=True, fontsize=12, fmt='%1.2f')

    ax.set_xlabel('Total Users in Network', fontsize=12, weight='bold')
    ax.set_ylabel('Reserved NTN RBs (HAP/LEO)', fontsize=12, weight='bold')
    ax.set_title('O-RAN Backup Dimensioning Map (3 GBS Failed)', fontsize=14, pad=15)
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    

# 2. COMPARATIVE MC vs. BC SCHEME PERFORMANCE SWEEP

def sweep_evaluate_link(p_tx, h_sq, interference, dist):
    """Link budget evaluator enforcing PDCCH SINR threshold (-5 dB) and URLLC delay."""
    sinr_lin = sinr(p_tx, h_sq, interference, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_RB)
    sinr_db = 10 * np.log10(sinr_lin) if sinr_lin > 0 else -100.0
    
    # Control channel drop threshold
    if sinr_db < -5.0:
        return -1.0, 0.0, dist, sinr_db, 1.0

    se = min(np.log2(1 + sinr_lin), 8.0)
    eps = error_probability(sinr_lin, M=const.MODULATION_M)
    cap_mbps = (const.BANDWIDTH_RB * se) / 1e6
    
    is_success, _ = check_transmission_success(
        capacity_mbps=cap_mbps, distance_m=dist, error_probability=eps
    )
    
    psi_backhaul = 1.0 - const.BACKHAUL_ERROR_PROB
    a_jn = ((1.0 - eps) * 1.0) * (psi_backhaul * 1.0) if is_success else -1.0
    
    return a_jn, se, dist, sinr_db, eps


def run_mc_vs_bc_sweep(user_range, failed_gbs_list=(4, 5, 6), fixed_backup_rbs=10, seed_val=42):
    """
    Executes a direct head-to-head performance sweep between:
    - Multi-Connectivity (MC: Dual active paths with 2 RBs/user)
    - Backup Connectivity (BC: On-demand backup allocation via Algorithm 1)
    """
    print("\n" + "="*60)
    print("RUNNING MC vs. BC COMPARATIVE PARAMETER SWEEP")
    print("="*60)
    
    np.random.seed(seed_val)
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)
    hap_coord, leo_coord = get_ntn_nodes()
    
    active_nodes = ['HAP', 'LEO'] + [f'GBS_{i}' for i in range(const.NUM_GBS) if i not in failed_gbs_list]
    
    metrics = {
        'users': user_range,
        'mc_avail': [], 'bc_avail': [],
        'mc_outage': [], 'bc_outage': [],
        'mc_latencies': [], 'bc_latencies': []
    }

    for n_users in user_range:
        ue_coords = get_random_users(n=n_users, bs_coords=bs_coords, radius=const.CELL_RADIUS)
        
        # 1. Primary connection baseline
        primary_df = evaluate_primary_connection(
            ue_coords, bs_coords, hap_coord, leo_coord, get_all_link_budgets
        )
        
        # ----------------- Multi-Connectivity Scheme -----------------
        mc_df = apply_mc_scheme(primary_df)
        mc_avail = mc_df['a_n'].mean()
        mc_outage = np.mean(mc_df['a_n'] < const.RELIABILITY_THRESHOLD)
        
        metrics['mc_avail'].append(mc_avail)
        metrics['mc_outage'].append(mc_outage)
        metrics['mc_latencies'].extend(mc_df['Latency_ms'].dropna().tolist())
        
        # ----------------- Backup Connectivity Scheme ----------------
        affected_users = inject_bs_failure(
            primary_df=primary_df,
            failed_bs_indices=list(failed_gbs_list),
            evaluate_link_func=sweep_evaluate_link
        )
        
        _, _, _, final_user_scores, _ = allocate_backup_paths(
            affected_users=affected_users,
            active_nodes=active_nodes,
            failed_bs_indices=list(failed_gbs_list),
            fixed_backup_rbs=fixed_backup_rbs,
            verbose=False
        )
        
        # Determine total network availability under failure
        failed_tower_names = [f'GBS_{i}' for i in failed_gbs_list]
        bc_avail_scores = []
        bc_lats = []
        
        for _, row in primary_df.iterrows():
            ue_id = row['UE_Idx']
            p_node = row['Primary_RU']
            
            if p_node in failed_tower_names:
                score = final_user_scores.get(ue_id, 0.0)
                bc_avail_scores.append(score)
                if score > 0.0:
                    bc_lats.append(row['Latency_ms'] * 1.15)  # Switchover/fallback delay
            else:
                bc_avail_scores.append(row['a_jn'])
                bc_lats.append(row['Latency_ms'])
                
        bc_avail = np.mean(bc_avail_scores) if bc_avail_scores else 0.0
        bc_outage = np.mean(np.array(bc_avail_scores) < const.RELIABILITY_THRESHOLD) if bc_avail_scores else 1.0
        
        metrics['bc_avail'].append(bc_avail)
        metrics['bc_outage'].append(bc_outage)
        metrics['bc_latencies'].extend(bc_lats)

    return metrics


def plot_mc_vs_bc_comparison(metrics):
    """Generates the 3 primary performance comparison figures for URLLC evaluation."""
    user_range = metrics['users']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. System Availability vs User Density
    axes[0].plot(user_range, metrics['mc_avail'], marker='o', color='#1f77b4', lw=2.2, label='Multi-Connectivity (MC)')
    axes[0].plot(user_range, metrics['bc_avail'], marker='s', color='#d62728', lw=2.2, label='Backup Connectivity (BC)')
    axes[0].axhline(y=const.RELIABILITY_THRESHOLD, color='k', linestyle='--', alpha=0.8, 
                    label=f'Target ({const.RELIABILITY_THRESHOLD})')
    axes[0].set_xlabel('Number of Users ($N$)', fontsize=11)
    axes[0].set_ylabel('Mean Availability', fontsize=11)
    axes[0].set_title('Availability vs. User Density', fontsize=13)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower left', fontsize=10)
    
    # 2. Outage Probability (Logarithmic)
    axes[1].plot(user_range, metrics['mc_outage'], marker='o', color='#1f77b4', lw=2.2, label='MC Outage')
    axes[1].plot(user_range, metrics['bc_outage'], marker='s', color='#d62728', lw=2.2, label='BC Outage')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Number of Users ($N$)', fontsize=11)
    axes[1].set_ylabel('Outage Probability ($P_{out}$)', fontsize=11)
    axes[1].set_title('Outage Probability vs. User Density', fontsize=13)
    axes[1].grid(True, which="both", linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper left', fontsize=10)
    
    # 3. Latency Cumulative Distribution Function (CDF)
    mc_sorted = np.sort(metrics['mc_latencies'])
    bc_sorted = np.sort(metrics['bc_latencies'])
    p_mc = np.linspace(0, 1, len(mc_sorted))
    p_bc = np.linspace(0, 1, len(bc_sorted))
    
    axes[2].plot(mc_sorted, p_mc, color='#1f77b4', lw=2.2, label='MC Latency')
    axes[2].plot(bc_sorted, p_bc, color='#d62728', lw=2.2, label='BC Latency')
    axes[2].axvline(x=const.LATENCY_THRESHOLD * 1000.0, color='k', linestyle='--', alpha=0.8, 
                    label=f'Threshold ({int(const.LATENCY_THRESHOLD * 1000)} ms)')
    axes[2].set_xlabel('End-to-End Latency [ms]', fontsize=11)
    axes[2].set_ylabel('CDF', fontsize=11)
    axes[2].set_title('End-to-End Latency CDF', fontsize=13)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.show()