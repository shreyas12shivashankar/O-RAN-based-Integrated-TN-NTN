import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from src.topology import draw_hexagon, get_hexagonal_bs
import src.constants as const
import math
from src.system_model import (
    path_loss, free_space_path_loss,distance_3D, gbs_3d_antenna_gain_db, channel_coefficient,
    GAIN_HAP_DBI, GAIN_LEO_DBI
)

def plot_topology(df, bs_coords, hap_coord, leo_coord, ue_coords, title='Network Topology'):
    """Handles all Matplotlib rendering separately from the logic."""
    fig = plt.figure(figsize=(12, 10)) 
    ax = fig.add_subplot(111, projection='3d')

    for bs in bs_coords: 
        draw_hexagon(ax, bs, radius=const.CELL_RADIUS)

    ax.scatter(ue_coords[:,0], ue_coords[:,1], ue_coords[:,2], c='red', s=15, label='UE')
    ax.scatter(bs_coords[:,0], bs_coords[:,1], bs_coords[:,2], c='blue', marker='^', s=120, label='Ground BS')
    ax.scatter(*hap_coord, c='black', marker='^', s=120, label='HAP')
    ax.scatter(*leo_coord, c='green', marker='^', s=120, label='LEO')
    
    for i, bs in enumerate(bs_coords):
        ax.text(bs[0], bs[1], bs[2] + 200, f'GBS_{i}', fontsize=10, weight='bold', color='darkblue')
        
    for i, ue in enumerate(ue_coords):
        ax.text(ue[0], ue[1], ue[2] + 100, f'UE_{i:02d}', fontsize=8, color='darkred')
    
    # Helper to determine line color based on the node name
    def get_color(ru_name):
        if pd.isna(ru_name): return '#1f77b4'
        if 'HAP' in ru_name: return '#8c564b'  # Brown
        if 'LEO' in ru_name: return '#2ca02c'  # Green
        return '#1f77b4'                       # Blue for GBS
        
    # Draw lines using stored coordinates in DataFrame
    for _, row in df.iterrows():
        u_pos = ue_coords[row['UE_Idx']]
        
        # Plot primary path (solid line) ONLY if it hasn't failed
        r_pos = row.get('RU_Pos')
        p_ru = row.get('Primary_RU')
        
        if pd.notna(p_ru) and r_pos is not None and isinstance(r_pos, (list, np.ndarray)):
            p_color = get_color(p_ru)
            ax.plot([u_pos[0], r_pos[0]], [u_pos[1], r_pos[1]], [u_pos[2], r_pos[2]], 
                    color=p_color, alpha=0.8, lw=1, linestyle='-')

        # Plot secondary path (dashed line)
        sec_ru = row.get('Secondary_RU')
        if pd.notna(sec_ru):
            sec_pos = None
            
            if sec_ru == 'HAP': sec_pos = hap_coord
            elif sec_ru == 'LEO': sec_pos = leo_coord
            elif sec_ru.startswith('GBS_'): sec_pos = bs_coords[int(sec_ru.split('_')[1])]
            
            if sec_pos is not None:
                sec_color = get_color(sec_ru)
                ax.plot([u_pos[0], sec_pos[0]], [u_pos[1], sec_pos[1]], [u_pos[2], sec_pos[2]], 
                        color=sec_color, alpha=0.5, lw=1, linestyle='--')

    ax.set_box_aspect([1, 1, 0.6])
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Altitude (m)', title=title)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Updated Legend for the 3 distinct connection colors
    custom_lines = [
        Line2D([0], [0], color='#1f77b4', lw=1.5, linestyle='-', label='Primary GBS'),
        Line2D([0], [0], color='#8c564b', lw=1.5, linestyle='-', label='Primary HAP'),
        Line2D([0], [0], color='#2ca02c', lw=1.5, linestyle='-', label='Primary LEO'),
        Line2D([0], [0], color='#1f77b4', lw=1.5, linestyle='--', alpha=0.5, label='Secondary GBS'),
        Line2D([0], [0], color='#8c564b', lw=1.5, linestyle='--', alpha=0.5, label='Secondary HAP'),
        Line2D([0], [0], color='#2ca02c', lw=1.5, linestyle='--', alpha=0.5, label='Secondary LEO')
    ]
    
    for line in custom_lines:
        by_label[line.get_label()] = line
        
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()
    

def plot_deterministic_sinr_curve(min_dist=10.0, max_dist=800.0, num_points=200, target_gbs_idx=0):
    """
    Generates a smooth, deterministic SINR vs Distance curve.
    Evaluates a UE walking exactly down the 30-degree main lobe of the target GBS.
    Dynamically includes co-channel interference based on the FR-1/FR-3 toggle.
    """
    import importlib
    importlib.reload(const)  # Force reload of constants to capture FR-1/FR-3 toggle
    
    azimuth_rad = math.radians(30.0)
    d_range = np.linspace(min_dist, max_dist, num_points)
    
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)
    noise_power = const.NOISE_SPECTRAL_DENSITY_W * const.BANDWIDTH_RB
    
    fr_label = f"FR-{const.FREQUENCY_REUSE}"  
   
    
    sinr_gbs_db, sinr_hap_db, sinr_leo_db = [], [], []
    sectors_pool = [30.0, 150.0, 270.0]
    
    # Dynamically fetch target base station coordinates and frequency band
    target_x, target_y = bs_coords[target_gbs_idx][0], bs_coords[target_gbs_idx][1]
    
    for d in d_range:
        ue_pos = np.array([
            target_x + d * math.cos(azimuth_rad), 
            target_y + d * math.sin(azimuth_rad), 
            1.5
        ])
        
        # --- SERVING LINK (DYNAMIC GBS) ---
        gbs_serving_pos = np.array([target_x, target_y, 25.0])
        pl_serving = path_loss(d, const.CARRIER_FREQ_GHZ)
        gain_serving_dbi = gbs_3d_antenna_gain_db(ue_pos, gbs_serving_pos, active_sector_deg=30.0)
        
        # w_sq = 1.0 isolates deterministic path loss (removes Rician variance)
        h_sq_serving = channel_coefficient(gain_serving_dbi, pl_serving, w_sq=1.0)
        rx_serving_w = const.TX_POWER_GBS_RB_W * h_sq_serving
        
        # --- INTERFERENCE CALCULATION ---
        interf_w = 0.0
        
        for i in range(len(bs_coords)):
            bs_pos = np.array([bs_coords[i][0], bs_coords[i][1], 25.0])
            d_interf = distance_3D(bs_pos, ue_pos)
            pl_interf = path_loss(d_interf, const.CARRIER_FREQ_GHZ)
                
            for sec in sectors_pool:
                # Skip the active serving sector panel itself
                if i == target_gbs_idx and sec == 30.0:
                    continue
                
                gain_interf_dbi = gbs_3d_antenna_gain_db(ue_pos, bs_pos, active_sector_deg=sec)
                h_sq_interf = channel_coefficient(gain_interf_dbi, pl_interf, w_sq=1.0)
                rx_interf = const.TX_POWER_GBS_RB_W * h_sq_interf
                
                # Apply interference based on the master toggle in constants.py
                if const.FREQUENCY_REUSE == 1:
                    interf_w += rx_interf
                elif const.FREQUENCY_REUSE == 3:
                    if const.FR3_CLUSTER_MAP[f'GBS_{i}'] == const.FR3_CLUSTER_MAP[f'GBS_{target_gbs_idx}']:
                        interf_w += rx_interf
                            

        # --- SINR CALCULATION ---
        sinr_gbs_lin = rx_serving_w / (interf_w + noise_power)
        sinr_gbs_db.append(10.0 * math.log10(sinr_gbs_lin))
        
        # --- NTN LINKS ---
        d_3d_hap = math.sqrt(d**2 + const.ALTITUDE_HAP**2)
        pl_hap = free_space_path_loss(d_3d_hap, const.CARRIER_FREQ_GHZ)
        h_sq_hap = channel_coefficient(GAIN_HAP_DBI, pl_hap, w_sq=1.0)
        sinr_hap_db.append(10.0 * math.log10((const.TX_POWER_HAP_RB_W * h_sq_hap) / noise_power))
        
        d_3d_leo = math.sqrt(d**2 + const.ALTITUDE_LEO**2)
        pl_leo = free_space_path_loss(d_3d_leo, const.CARRIER_FREQ_GHZ)
        h_sq_leo = channel_coefficient(GAIN_LEO_DBI, pl_leo, w_sq=1.0)
        sinr_leo_db.append(10.0 * math.log10((const.TX_POWER_LEO_RB_W * h_sq_leo) / noise_power))

    plt.figure(figsize=(9, 6))
    
    # Clean, professional labels for the legend
    plt.plot(d_range, sinr_gbs_db, color='#1f77b4', linewidth=3.0, label=f'Terrestrial GBS (Target: GBS_{target_gbs_idx})')
    plt.plot(d_range, sinr_hap_db, color='#8c564b', linestyle='--', linewidth=2.5, label='NTN HAP')
    plt.plot(d_range, sinr_leo_db, color='#2ca02c', linestyle='-.', linewidth=2.5, label='NTN LEO')

    plt.xlabel(f'Distance from Serving Base Station (GBS_{target_gbs_idx}) [m]', fontsize=12)
    plt.ylabel('Average SINR [dB]', fontsize=12)
    plt.title(f'Average SINR vs. Distance ({fr_label})', fontsize=14, pad=15)
    
    plt.xlim(min_dist, max_dist)
    
    # Dynamically scale Y-axis to guarantee lines are visible
    all_sinr = sinr_gbs_db + sinr_hap_db + sinr_leo_db
    plt.ylim(math.floor(min(all_sinr)/10)*10, math.ceil(max(all_sinr)/10)*10 + 5)
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()
    

# def plot_sinr_curve(min_dist=10.0, max_dist=800.0, num_points=200):
   
#     azimuth_rad = math.radians(30.0)
#     d_range = np.linspace(min_dist, max_dist, num_points)
    
#     bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)
#     noise_power = const.NOISE_SPECTRAL_DENSITY_W * const.BANDWIDTH_RB
    
#     # Determine if we are in FR-1 or FR-3 based on unique values in the dictionary
#     unique_bands = len(set(const.FR3_CLUSTER_MAP.values()))
#     fr_label = f"FR-{unique_bands}"
    
#     sinr_gbs_db, sinr_hap_db, sinr_leo_db = [], [], []
#     sectors_pool = [30.0, 150.0, 270.0]
    
#     for d in d_range:
#         ue_pos = np.array([d * math.cos(azimuth_rad), d * math.sin(azimuth_rad), 1.5])
        
#         # --- SERVING LINK (GBS 0) ---
#         gbs_0_pos = np.array([bs_coords[0][0], bs_coords[0][1], 25.0])
#         pl_serving = path_loss(d, const.CARRIER_FREQ_GHZ)
#         gain_serving_dbi = gbs_3d_antenna_gain_db(ue_pos, gbs_0_pos, active_sector_deg=30.0)
#         h_sq_serving = channel_coefficient(gain_serving_dbi, pl_serving, w_sq=1.0)
#         rx_serving_w = const.TX_POWER_GBS_RB_W * h_sq_serving
        
#         # --- INTERFERENCE CALCULATION ---
#         interf_w = 0.0
#         gbs_0_band = const.FR3_CLUSTER_MAP['GBS_0']
        
#         # Loop through neighbors (1 to 6)
#         for i in range(1, len(bs_coords)):
#             neighbor_key = f'GBS_{i}'
            
#             # ONLY add interference if this neighbor uses the same band as GBS_0
#             if const.FR3_CLUSTER_MAP[neighbor_key] == gbs_0_band:
#                 bs_pos = np.array([bs_coords[i][0], bs_coords[i][1], 25.0])
#                 d_interf = distance_3D(bs_pos, ue_pos)
#                 pl_interf = path_loss(d_interf, const.CARRIER_FREQ_GHZ)
                
#                 dx = ue_pos[0] - bs_pos[0]
#                 dy = ue_pos[1] - bs_pos[1]
#                 interf_azimuth_deg = math.degrees(math.atan2(dy, dx)) % 360.0
#                 active_sec = min(sectors_pool, key=lambda s: abs((interf_azimuth_deg - s + 180) % 360 - 180))
                
#                 gain_interf_dbi = gbs_3d_antenna_gain_db(ue_pos, bs_pos, active_sector_deg=active_sec)
#                 h_sq_interf = channel_coefficient(gain_interf_dbi, pl_interf, w_sq=1.0)
#                 interf_w += const.TX_POWER_GBS_RB_W * h_sq_interf

#         # --- SINR CALCULATION ---
#         sinr_gbs_lin = rx_serving_w / (interf_w + noise_power)
#         sinr_gbs_db.append(10.0 * math.log10(sinr_gbs_lin))
        
#         # --- NTN LINKS ---
#         d_3d_hap = math.sqrt(d**2 + const.ALTITUDE_HAP**2)
#         pl_hap = free_space_path_loss(d_3d_hap, const.CARRIER_FREQ_GHZ)
#         h_sq_hap = channel_coefficient(GAIN_HAP_DBI, pl_hap, w_sq=1.0)
#         sinr_hap_db.append(10.0 * math.log10((const.TX_POWER_HAP_RB_W * h_sq_hap) / noise_power))
        
#         d_3d_leo = math.sqrt(d**2 + const.ALTITUDE_LEO**2)
#         pl_leo = free_space_path_loss(d_3d_leo, const.CARRIER_FREQ_GHZ)
#         h_sq_leo = channel_coefficient(GAIN_LEO_DBI, pl_leo, w_sq=1.0)
#         sinr_leo_db.append(10.0 * math.log10((const.TX_POWER_LEO_RB_W * h_sq_leo) / noise_power))

#     plt.figure(figsize=(9, 6))
    
#     # Clean, professional labels for the legend
#     plt.plot(d_range, sinr_gbs_db, color='#1f77b4', linewidth=3.0, label='GBS')
#     plt.plot(d_range, sinr_hap_db, color='#8c564b', linestyle='--', linewidth=2.5, label='HAP')
#     plt.plot(d_range, sinr_leo_db, color='#2ca02c', linestyle='-.', linewidth=2.5, label='LEO')

#     # Minimalist Axis and Title Labels
#     plt.xlabel('Distance from Serving Base Station [m]', fontsize=12)
#     plt.ylabel('Average SINR [dB]', fontsize=12)
#     plt.title(f'Average SINR vs. Distance ({fr_label})', fontsize=14, pad=15)
    
#     plt.xlim(min_dist, max_dist)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend(loc='upper right', fontsize=11)
#     plt.tight_layout()
#     plt.show()
    


