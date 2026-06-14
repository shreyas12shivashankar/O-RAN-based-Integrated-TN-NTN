import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes, draw_hexagon
from src.system_model import distance_3D, path_loss, free_space_path_loss, channel_coefficient, sinr, rate
import src.constants as const

def analyze_primary_paths():
    # --- 1. Setup Network ---
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
    
    # --- 2. Calculate Link Budgets & Capacity ---
    for ue_id, ue_pos in enumerate(ue_coords):
        link_profiles = {}

        # NTN 
        pl_hap = free_space_path_loss(distance_3D(hap_coord, ue_pos), const.CARRIER_FREQ_GHZ)
        link_profiles['HAP'] = p_hap_w * np.abs(channel_coefficient(1.0, pl_hap, is_ntn=True))**2

        pl_leo = free_space_path_loss(distance_3D(leo_coord, ue_pos), const.CARRIER_FREQ_GHZ)
        link_profiles['LEO'] = p_leo_w * np.abs(channel_coefficient(1.0, pl_leo, is_ntn=True))**2

        # Terrestrial (Dynamic Fading)
        for bs_id, bs_pos in enumerate(bs_coords):
            pl_gbs = path_loss(distance_3D(bs_pos, ue_pos), const.CARRIER_FREQ_GHZ)
            k_linear = 10 ** (np.random.normal(9.0, 3.5) / 10)
            h_gbs = channel_coefficient(1.0, pl_gbs, rician_factor_k_linear=k_linear, is_ntn=False)
            link_profiles[f'GBS_{bs_id}'] = p_gbs_w * np.abs(h_gbs)**2

        # Determine Primary RU (Strongest Signal)
        primary_ru = max(link_profiles, key=link_profiles.get)
        max_rx_power_w = link_profiles[primary_ru]
        
        # Calculate SINR and Capacity
        if primary_ru in ['HAP', 'LEO']:
            sinr_linear = sinr(1.0, max_rx_power_w, 0, noise_density_w, const.BANDWIDTH_HZ)
        else:
            interference_w = sum(p for node, p in link_profiles.items() if node.startswith('GBS_') and node != primary_ru)
            sinr_linear = sinr(1.0, max_rx_power_w, interference_w, noise_density_w, const.BANDWIDTH_HZ)

        capacity_mbps = rate(const.BANDWIDTH_HZ, sinr_linear) / 1e6

        results.append({
            "UE_ID": f"UE_{ue_id:03d}",
            "Primary_RU": primary_ru,
            "Rx_Power_dBm": round(10 * np.log10(max_rx_power_w) + 30, 2),
            "Capacity_Mbps": round(capacity_mbps, 2)
        })

    # --- 3. Generate Tabular Output ---
    df = pd.DataFrame(results)
    
    # Force Pandas to print all 100 rows for the report
    pd.set_option('display.max_rows', None)
    
    print("\n" + "="*60)
    print("USER EQUIPMENT PRIMARY PATH & CAPACITY REPORT")
    print("="*60)
    print(df.to_string(index=False)) 
    print("="*60)
    
    # Aggregate Sum Capacity
    sum_capacity = df['Capacity_Mbps'].sum()
    print(f"\n[SYSTEM METRIC] Overall Sum Capacity: {sum_capacity:.2f} Mbps")
    print("="*60)

    # --- 4. Generate Visual Topology Plot ---
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
        line_color = 'brown' if 'GBS' in ru_name else 'gray'
        line_alpha = 0.4 if 'GBS' in ru_name else 0.2
        
        ax.plot([ue_pos[0], ru_pos[0]], 
                [ue_pos[1], ru_pos[1]], 
                [ue_pos[2], ru_pos[2]], 
                color=line_color, linestyle='-', linewidth=0.5, alpha=line_alpha)

    ax.set_box_aspect([1, 1, 0.6])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude Z (m)')
    ax.set_title('Network Topology: Primary Path Assignments')
    
    # Custom legend to include the connection lines
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='brown', lw=1, alpha=0.6))
    labels.append('Terrestrial Link')
    handles.append(Line2D([0], [0], color='gray', lw=1, alpha=0.4))
    labels.append('NTN Link')
    ax.legend(handles=handles, labels=labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_primary_paths()
