import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes, draw_hexagon
from src.system_model import (
    distance_3D, path_loss, free_space_path_loss, channel_coefficient, sinr, rate,
    GAIN_GBS_DBI, GAIN_HAP_DBI, GAIN_LEO_DBI,
    K_UMA_DB_MEAN, K_UMA_DB_STD, K_HAP_STATIC, K_LEO_STATIC
)
import src.constants as const

def analyze_primary_paths():
    
    # 1. Setup Network 
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)  
    hap_coord, leo_coord = get_ntn_nodes()
    
    np.random.seed(42) 
    ue_coords = get_random_users(n=const.NUM_UE)

    results = []
    
    # 2. Calculate Link Budgets & Capacity
    for ue_id in range(len(ue_coords)):
        ue_pos = ue_coords[ue_id]

        best_ru_name = None
        max_rx_power_w = 0.0
        best_is_ntn = False
        dist_to_primary = 0.0

        # Evaluate HAP Link
        dist_hap = distance_3D(hap_coord, ue_pos)
        pl_hap_db = free_space_path_loss(dist_hap, const.CARRIER_FREQ_GHZ)
        h_hap_mag = channel_coefficient(GAIN_HAP_DBI, pl_hap_db, K_HAP_STATIC)
        rx_hap_w = const.TX_POWER_HAP_W * (h_hap_mag**2)

        if rx_hap_w > max_rx_power_w:
            max_rx_power_w = rx_hap_w
            best_ru_name = 'HAP'
            best_is_ntn = True
            dist_to_primary = dist_hap

        # Evaluate LEO Link
        dist_leo = distance_3D(leo_coord, ue_pos)
        pl_leo_db = free_space_path_loss(dist_leo, const.CARRIER_FREQ_GHZ)
        h_leo_mag = channel_coefficient(GAIN_LEO_DBI, pl_leo_db, K_LEO_STATIC)
        rx_leo_w = const.TX_POWER_LEO_W * (h_leo_mag**2)

        if rx_leo_w > max_rx_power_w:
            max_rx_power_w = rx_leo_w
            best_ru_name = 'LEO'
            best_is_ntn = True
            dist_to_primary = dist_leo

        # Evaluate Terrestrial Links (Ground Base Stations)
        gbs_powers_w = [] # Store all terrestrial powers to calculate interference 
        
        for bs_id in range(len(bs_coords)):
            bs_pos = bs_coords[bs_id]
            dist_gbs = distance_3D(bs_pos, ue_pos)
            
            pl_gbs_db = path_loss(dist_gbs, const.CARRIER_FREQ_GHZ)
            k_db = np.random.normal(K_UMA_DB_MEAN, K_UMA_DB_STD)
            h_gbs_mag = channel_coefficient(GAIN_GBS_DBI, pl_gbs_db, k_db)
            
            rx_gbs_w = const.TX_POWER_GBS_W * (h_gbs_mag**2)
            gbs_powers_w.append(rx_gbs_w)

            # Greedy Logic
            if rx_gbs_w > max_rx_power_w:
                max_rx_power_w = rx_gbs_w
                best_ru_name = f'GBS_{bs_id}'
                best_is_ntn = False
                dist_to_primary = dist_gbs

        # Calculate SINR
        if best_is_ntn:
            # NTN links are modeled with zero terrestrial interference
            interference_w = 0.0 
        else:
            # Terrestrial interference is the sum of all GBS signals MINUS the one we connected to
            total_gbs_power = sum(gbs_powers_w)
            interference_w = total_gbs_power - max_rx_power_w

        sinr_linear = sinr(1.0, max_rx_power_w, interference_w, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_HZ)
        capacity_mbps = rate(const.BANDWIDTH_HZ, sinr_linear) / 1e6

        results.append({
            "UE_ID": f"UE_{ue_id:03d}",
            "Primary_RU": best_ru_name,
            "Rx_Power_dBm": round(10 * np.log10(max_rx_power_w) + 30, 2),
            "Capacity_Mbps": round(capacity_mbps, 2)
        })

    # 3. Generate Tabular Output
    df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    
    print("\nUSER EQUIPMENT PRIMARY PATH & CAPACITY")
    print(df.to_string(index=False)) 
        
    # Connection Summary 
    hap_count = len(df[df['Primary_RU'] == 'HAP'])
    leo_count = len(df[df['Primary_RU'] == 'LEO'])
    gbs_count = len(df[df['Primary_RU'].str.startswith('GBS')])
    
    print("\nNETWORK CONNECTION SUMMARY")
    print(f"Total UEs connected to GBS (Terrestrial) : {gbs_count}")
    print(f"Total UEs connected to HAP (NTN)         : {hap_count}")
    print(f"Total UEs connected to LEO (NTN)         : {leo_count}")
    
    sum_capacity = df['Capacity_Mbps'].sum()
    print(f"\n[SYSTEM METRIC] Overall Sum Capacity: {sum_capacity:.2f} Mbps")

    # 4. Generate Visual Topology Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw hexagonal cells
    for bs in bs_coords:
        draw_hexagon(ax, bs, radius=const.CELL_RADIUS)

    # Plot the physical nodes
    ax.scatter(ue_coords[:,0], ue_coords[:,1], ue_coords[:,2], c='red', s=15, label='UE', zorder=5)
    ax.scatter(bs_coords[:,0], bs_coords[:,1], bs_coords[:,2], c='blue', marker='^', s=120, label='Ground BS', zorder=10)
    ax.scatter(hap_coord[0], hap_coord[1], hap_coord[2], c='black', marker='^', s=120, label='HAP', zorder=10)
    ax.scatter(leo_coord[0], leo_coord[1], leo_coord[2], c='green', marker='^', s=120, label='LEO', zorder=10)

    # Draw the Primary Paths
    for ue_id, row in df.iterrows():
        ue_pos = ue_coords[ue_id]
        ru_name = row['Primary_RU']
        
        # Manually extract the coordinates of the chosen RU for the plot
        if ru_name == 'HAP':
            ru_pos = hap_coord
        elif ru_name == 'LEO':
            ru_pos = leo_coord
        else:
            bs_index = int(ru_name.split('_')[1])
            ru_pos = bs_coords[bs_index]
        
        line_color ='#1f77b4' if 'GBS' in ru_name else '#ff7f0e'
        line_alpha = 0.85 
        line_width = 1.2
        
        ax.plot([ue_pos[0], ru_pos[0]], 
                [ue_pos[1], ru_pos[1]], 
                [ue_pos[2], ru_pos[2]], 
                color=line_color, linestyle='-', linewidth=line_width, alpha=line_alpha)

    ax.set_box_aspect([1, 1, 0.6])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude Z (m)')
    ax.set_title('Network Topology: Primary Path Allocation')
    
    # Custom legend
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