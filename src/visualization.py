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


def plot_topology(df, bs_coords, hap_coord, leo_coord, ue_coords, title='Network Topology', plot_mode='both'):
    """Generates 3D visualization of network topology and active links for primary and backup modes."""    
    def get_color(ru_name):
        if pd.isna(ru_name): return '#1f77b4'
        if 'HAP' in ru_name: return '#8c564b'  
        if 'LEO' in ru_name: return '#2ca02c'  
        return '#1f77b4'                       

    if plot_mode in ['primary', 'secondary']:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        axes = [ax]
        sub_titles = [title] 
    else:
        fig = plt.figure(figsize=(18, 8)) 
        fig.suptitle(title, fontsize=16, weight='bold', y=0.95)
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        axes = [ax1, ax2]
        sub_titles = ["Primary Connections", "Secondary / Backup Connections"]

    for idx, ax in enumerate(axes):
        ax.set_title(sub_titles[idx], fontsize=13, pad=10)
        
        for bs in bs_coords: 
            draw_hexagon(ax, bs, radius=const.CELL_RADIUS)

        ax.scatter(ue_coords[:,0], ue_coords[:,1], ue_coords[:,2], c='red', s=15, alpha=0.6, label='UE')
        ax.scatter(bs_coords[:,0], bs_coords[:,1], bs_coords[:,2], c='blue', marker='^', s=100, label='Ground BS')
        ax.scatter(*hap_coord, c='black', marker='^', s=120, label='HAP')
        ax.scatter(*leo_coord, c='green', marker='^', s=120, label='LEO')
        
        for i, bs in enumerate(bs_coords):
            ax.text(bs[0], bs[1], bs[2] + 200, f'GBS_{i}', fontsize=9, weight='bold', color='darkblue')
            
        ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.15, 1.0))
        ax.set_box_aspect([1, 1, 0.6])
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Altitude (m)')

    for _, row in df.iterrows():
        u_pos = ue_coords[row['UE_Idx']]
        
        # Draw Primary Lines
        if plot_mode in ['primary', 'both']:
            r_pos = row.get('RU_Pos')
            p_ru = row.get('Primary_RU')
            target_ax = axes[0] if plot_mode == 'both' else axes[0]
            
            if pd.notna(p_ru) and r_pos is not None and isinstance(r_pos, (list, np.ndarray)):
                p_color = get_color(p_ru)
                target_ax.plot([u_pos[0], r_pos[0]], [u_pos[1], r_pos[1]], [u_pos[2], r_pos[2]], 
                               color=p_color, alpha=0.6, lw=1.2, linestyle='-')

        # Draw Secondary/Backup Lines
        if plot_mode in ['secondary', 'both']:
            sec_ru = row.get('Secondary_RU')
            target_ax = axes[1] if plot_mode == 'both' else axes[0]
            
            if pd.notna(sec_ru) and sec_ru != 'None':
                sec_pos = None
                if sec_ru == 'HAP': sec_pos = hap_coord
                elif sec_ru == 'LEO': sec_pos = leo_coord
                elif sec_ru.startswith('GBS_'): sec_pos = bs_coords[int(sec_ru.split('_')[1])]
                
                if sec_pos is not None:
                    sec_color = get_color(sec_ru)
                    target_ax.plot([u_pos[0], sec_pos[0]], [u_pos[1], sec_pos[1]], [u_pos[2], sec_pos[2]], 
                                   color=sec_color, alpha=0.6, lw=1.2, linestyle='--')

    custom_lines = [
        Line2D([0], [0], color='#1f77b4', lw=2, linestyle='-', label='TN Connection (GBS)'),
        Line2D([0], [0], color='#8c564b', lw=2, linestyle='-', label='NTN Connection (HAP)'),
        Line2D([0], [0], color='#2ca02c', lw=2, linestyle='-', label='NTN Connection (LEO)')
    ]
    
    fig.legend(handles=custom_lines, loc='lower center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    

def plot_sinr_curve(min_dist=10.0, max_dist=800.0, num_points=200, target_gbs_idx=0):
    """Plots comparative SINR vs Distance curve for GBS, HAP, and LEO nodes."""
    
    azimuth_rad = math.radians(30.0)
    d_range = np.linspace(min_dist, max_dist, num_points)
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)
    noise_power = const.NOISE_SPECTRAL_DENSITY_W * const.BANDWIDTH_RB
        
    sinr_gbs_db, sinr_hap_db, sinr_leo_db = [], [], []
    sectors_pool = [30.0, 150.0, 270.0]
    
    target_x, target_y = bs_coords[target_gbs_idx][0], bs_coords[target_gbs_idx][1]
    hap_coord = np.array([0.0, 0.0, const.ALTITUDE_HAP])
    leo_coord = np.array([0.0, 0.0, const.ALTITUDE_LEO])
    
    for d in d_range:
        ue_pos = np.array([target_x + d * math.cos(azimuth_rad), target_y + d * math.sin(azimuth_rad), 1.5])
        gbs_serving_pos = np.array([target_x, target_y, 25.0])

        pl_serving = path_loss(distance_3D(gbs_serving_pos, ue_pos), const.CARRIER_FREQ_GHZ)
        gain_serving_dbi = gbs_3d_antenna_gain_db(ue_pos, gbs_serving_pos, active_sector_deg=30.0)
        rx_serving_w = const.TX_POWER_GBS_RB_W * channel_coefficient(gain_serving_dbi, pl_serving, w_sq=1.0)

        # Full FR-1 co-channel interference accumulation
        interf_w = 0.0
        for i, bs in enumerate(bs_coords):
            bs_pos = np.array([bs[0], bs[1], 25.0])
            pl_interf = path_loss(distance_3D(bs_pos, ue_pos), const.CARRIER_FREQ_GHZ)
            for sec in sectors_pool:
                if i == target_gbs_idx and sec == 30.0:
                    continue
                gain_sec = gbs_3d_antenna_gain_db(ue_pos, bs_pos, active_sector_deg=sec)
                interf_w += const.TX_POWER_GBS_RB_W * channel_coefficient(gain_sec, pl_interf, w_sq=1.0)

        sinr_gbs_db.append(10.0 * math.log10(rx_serving_w / (interf_w + noise_power)))

        h_sq_hap = channel_coefficient(GAIN_HAP_DBI, free_space_path_loss(distance_3D(ue_pos, hap_coord), const.CARRIER_FREQ_GHZ), w_sq=1.0)
        sinr_hap_db.append(10.0 * math.log10((const.TX_POWER_HAP_RB_W * h_sq_hap) / noise_power))

        h_sq_leo = channel_coefficient(GAIN_LEO_DBI, free_space_path_loss(distance_3D(ue_pos, leo_coord), const.CARRIER_FREQ_GHZ), w_sq=1.0)
        sinr_leo_db.append(10.0 * math.log10((const.TX_POWER_LEO_RB_W * h_sq_leo) / noise_power))

    plt.figure(figsize=(9, 6))
    markers = list(range(0, num_points, 25)) + [num_points - 1]

    plt.plot(d_range, sinr_gbs_db, color='#1f77b4', linewidth=2.0, marker='o', markevery=markers, label='Terrestrial GBS (FR-1)')
    plt.plot(d_range, sinr_hap_db, color='#8c564b', linewidth=2.0, marker='o', markevery=markers, label='NTN HAP')
    plt.plot(d_range, sinr_leo_db, color='#2ca02c', linewidth=2.0, marker='o', markevery=markers, label='NTN LEO')

    plt.xlabel('Distance [m]', fontsize=12)
    plt.ylabel('Average SINR [dB]', fontsize=12)
    plt.title('Average SINR vs Distance', fontsize=14, pad=15)
    plt.xlim(min_dist - 20, max_dist + 20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='center right', fontsize=11)
    plt.tight_layout()
    plt.show()
    
                        

    