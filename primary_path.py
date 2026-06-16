import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes, draw_hexagon
from src.system_model import (
    distance_3D, path_loss, free_space_path_loss, channel_coefficient, sinr, rate, check_transmission_success,
    GAIN_GBS_DBI, GAIN_HAP_DBI, GAIN_LEO_DBI,
    K_UMA_DB_MEAN, K_UMA_DB_STD, K_HAP_STATIC, K_LEO_STATIC
)
import src.constants as const

def analyze_primary_paths():
    # 1. Setup Network 
    bs_coords = get_hexagonal_bs(radius=2000)
    hap_coord, leo_coord = get_ntn_nodes()
    
    # Create a dictionary to easily map RU names to their 3D coordinates for plotting
    node_coordinates = {
        'HAP': hap_coord,
        'LEO': leo_coord
    }
    for bs_id, bs_pos in enumerate(bs_coords):
        node_coordinates[f'GBS_{bs_id}'] = bs_pos

    np.random.seed(42) 
    ue_coords = get_random_users(n=100)

    # Power in linear Watts
    noise_density_w = 10 ** ((const.NOISE_SPECTRAL_DENSITY_DBM - 30) / 10)
    p_hap_w = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)
    p_leo_w = 10 ** ((const.TX_POWER_LEO - 30) / 10)
    p_gbs_w = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)

    results = []
    
    # 2. Calculate Link Budgets & Capacity 
    for ue_id, ue_pos in enumerate(ue_coords):
        link_profiles = {}

        # NTN 
        pl_hap_db = free_space_path_loss(distance_3D(hap_coord, ue_pos), const.CARRIER_FREQ_GHZ)
        h_hap_mag = channel_coefficient(antenna_gain_db=GAIN_HAP_DBI, path_loss_db=pl_hap_db, k_factor_db=K_HAP_STATIC)
        link_profiles['HAP'] = p_hap_w * (h_hap_mag**2)

        pl_leo_db = free_space_path_loss(distance_3D(leo_coord, ue_pos), const.CARRIER_FREQ_GHZ)
        h_leo_mag = channel_coefficient(antenna_gain_db=GAIN_LEO_DBI, path_loss_db=pl_leo_db, k_factor_db=K_LEO_STATIC)
        link_profiles['LEO'] = p_leo_w * (h_leo_mag**2)

        # Terrestrial (Dynamic UMa Fading)
        for bs_id, bs_pos in enumerate(bs_coords):
            pl_gbs_db = path_loss(distance_3D(bs_pos, ue_pos), const.CARRIER_FREQ_GHZ)
            
            # 1. Draw dynamic K in dB using Imported Constants
            k_db = np.random.normal(K_UMA_DB_MEAN, K_UMA_DB_STD)
            
            # 2. Evaluate terrestrial link using Imported GBS Gain Constant
            h_gbs_mag = channel_coefficient(antenna_gain_db=GAIN_GBS_DBI, path_loss_db=pl_gbs_db, k_factor_db=k_db)
            
            link_profiles[f'GBS_{bs_id}'] = p_gbs_w * (h_gbs_mag**2)
    
        # Determine Primary RU (Strongest Signal - Greedy Logic)
        primary_ru = max(link_profiles, key=link_profiles.get)
        max_rx_power_w = link_profiles[primary_ru]
        
        # Calculate SINR and Capacity
        if primary_ru in ['HAP', 'LEO']:
            sinr_linear = sinr(1.0, max_rx_power_w, 0, noise_density_w, const.BANDWIDTH_HZ)
        else:
            interference_w = sum(p for node, p in link_profiles.items() if node.startswith('GBS_') and node != primary_ru)
            sinr_linear = sinr(1.0, max_rx_power_w, interference_w, noise_density_w, const.BANDWIDTH_HZ)

        capacity_mbps = rate(const.BANDWIDTH_HZ, sinr_linear) / 1e6

        # Check Transmission Success (URLLC 10 ms threshold)
        dist_to_primary = distance_3D(node_coordinates[primary_ru], ue_pos)
        
        success, delay_ms = check_transmission_success(
            capacity_mbps=capacity_mbps, 
            distance_m=dist_to_primary, 
            packet_size_bytes=32, 
            max_latency_ms=10.0 
        )

        results.append({
            "UE_ID": f"UE_{ue_id:03d}",
            "Primary_RU": primary_ru,
            "Rx_Power_dBm": round(10 * np.log10(max_rx_power_w) + 30, 2),
            "Capacity_Mbps": round(capacity_mbps, 2),
            "Delay_ms": round(delay_ms, 4), # <--- Added delay to table
            "Tx_Success": "Pass" if success else "Drop" # <--- Added success status
        })

    # 3. Generate Tabular Output
    df = pd.DataFrame(results)
    
    # Force Pandas to print all 100 rows for the report
    pd.set_option('display.max_rows', None)
    
    print("USER EQUIPMENT PRIMARY PATH & CAPACITY REPORT")
    print(df.to_string(index=False)) 
        
    # --- Connection Summary ---
    hap_count = len(df[df['Primary_RU'] == 'HAP'])
    leo_count = len(df[df['Primary_RU'] == 'LEO'])
    gbs_count = len(df[df['Primary_RU'].str.startswith('GBS')])

    
    print("NETWORK CONNECTION SUMMARY")
    print(f"Total UEs connected to GBS (Terrestrial) : {gbs_count}")
    print(f"Total UEs connected to HAP (NTN)         : {hap_count}")
    print(f"Total UEs connected to LEO (NTN)         : {leo_count}")
    
    # Aggregate Sum Capacity
    sum_capacity = df['Capacity_Mbps'].sum()
    print(f"\n[SYSTEM METRIC] Overall Sum Capacity: {sum_capacity:.2f} Mbps")

    #  4. Generate Visual Topology Plot 
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw hexagonal cells
    for bs in bs_coords:
        draw_hexagon(ax, bs, radius=2000)

    # Plot the physical nodes
    ax.scatter(ue_coords[:,0], ue_coords[:,1], ue_coords[:,2], c='red', s=15, label='UE', zorder=5)
    ax.scatter(bs_coords[:,0], bs_coords[:,1], bs_coords[:,2], c='blue', marker='^', s=120, label='Ground BS', zorder=10)
    ax.scatter(hap_coord[0], hap_coord[1], hap_coord[2], c='black', marker='^', s=120, label='HAP', zorder=10)
    ax.scatter(leo_coord[0], leo_coord[1], leo_coord[2], c='green', marker='^', s=120, label='LEO', zorder=10)

    # Draw the Primary Paths (Lines connecting UE to RU)
    for ue_id, row in df.iterrows():
        ue_pos = ue_coords[ue_id]
        ru_name = row['Primary_RU']
        ru_pos = node_coordinates[ru_name]
        
        # Color code the lines based on connection type
        line_color = '#1f77b4' if 'GBS' in ru_name else '#ff7f0e'
        line_alpha = 0.85 if 'GBS' in ru_name else 0.85
        line_width = 1.2
        
        ax.plot([ue_pos[0], ru_pos[0]], 
                [ue_pos[1], ru_pos[1]], 
                [ue_pos[2], ru_pos[2]], 
                color=line_color, linestyle='-', linewidth=line_width, alpha=line_alpha)

    ax.set_box_aspect([1, 1, 0.6])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude Z (m)')
    ax.set_title('Network Topology: Primary Path Assignments')
    
    # Custom legend to include the connection lines
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#1f77b4', lw=2, alpha=0.85))
    labels.append('Terrestrial Link')
    handles.append(Line2D([0], [0], color='#ff7f0e', lw=2, alpha=0.85))
    labels.append('NTN Link')
    ax.legend(handles=handles, labels=labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_primary_paths()