import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import src.constants as const
from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.system_model import (
    distance_3D, 
    path_loss, 
    free_space_path_loss,
    sinr, 
    channel_coefficient
)

def generate_visualization_data():
    bs_coords = get_hexagonal_bs(radius=1900)
    ue_coords = get_random_users(n=100, area_range=5000)
    
    ue_x, ue_y = ue_coords[:, 0], ue_coords[:, 1]
    rx_powers = []
    sinrs_db = []
    
    for ue_pos in ue_coords:
        bs_rx_powers_watts = []
        max_rx_power_dbm = -np.inf
        best_path_loss = 0
        best_bs_idx = 0
        
        # Calculate link profiles for all 7 Base Stations
        for bs_id, bs_pos in enumerate(bs_coords):
            dist = distance_3D(bs_pos, ue_pos)
            pl_db = path_loss(dist, const.CARRIER_FREQ_GHZ)
            rx_power_dbm = const.TX_POWER_GBS_HAP - pl_db
            
            rx_power_watts = 10 ** ((rx_power_dbm - 30) / 10)
            bs_rx_powers_watts.append(rx_power_watts)
            
            if rx_power_dbm > max_rx_power_dbm:
                max_rx_power_dbm = rx_power_dbm
                best_path_loss = pl_db
                best_bs_idx = bs_id
                
        # Channel coefficient and SINR calculation
        h_coef = channel_coefficient(antenna_gain=1.0, path_loss_db=best_path_loss, rician_factor_k=10)
        h_sq = np.abs(h_coef)**2
        p_watts = 10 ** ((const.TX_POWER_GBS_HAP - 30) / 10)
        
        interference_watts = sum(bs_rx_powers_watts) - bs_rx_powers_watts[best_bs_idx]
        noise_density_watts = 10 ** ((const.NOISE_SPECTRAL_DENSITY_DBM - 30) / 10)
        
        sinr_linear = sinr(p_watts, h_sq, interference_watts, noise_density_watts, const.BANDWIDTH_HZ)
        sinr_db = 10 * np.log10(sinr_linear) if sinr_linear > 0 else 0
        
        rx_powers.append(max_rx_power_dbm)
        sinrs_db.append(sinr_db)
        
    return bs_coords, ue_x, ue_y, np.array(rx_powers), np.array(sinrs_db)

if __name__ == "__main__":
    bs_coords, ue_x, ue_y, rx_powers, sinrs_db = generate_visualization_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # ------------------ PANEL 1: RECEIVED POWER HEATMAP ------------------
    sc1 = ax1.scatter(ue_x, ue_y, c=rx_powers, cmap='plasma', s=60, edgecolors='k', alpha=0.8)
    ax1.scatter(bs_coords[:,0], bs_coords[:,1], c='cyan', marker='^', s=150, edgecolor='black', label='Ground BS')
    fig.colorbar(sc1, ax=ax1, label='Primary Rx Power (dBm)')
    ax1.set_title("User Parameter Profile: Received Power Distribution")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_facecolor('#f7f7f7')
    ax1.legend()
    
    # ------------------ PANEL 2: SINR GRADIENT MAP ------------------
    sc2 = ax2.scatter(ue_x, ue_y, c=sinrs_db, cmap='viridis', s=60, edgecolors='k', alpha=0.8)
    ax2.scatter(bs_coords[:,0], bs_coords[:,1], c='red', marker='^', s=150, edgecolor='black', label='Ground BS')
    fig.colorbar(sc2, ax=ax2, label='Downlink SINR (dB)')
    ax2.set_title("User Parameter Profile: Downlink SINR & Interference Zones")
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_facecolor('#f7f7f7')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()