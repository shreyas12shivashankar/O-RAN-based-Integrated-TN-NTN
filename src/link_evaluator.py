import numpy as np
import math
from src.system_model import (
    distance_3D, path_loss, free_space_path_loss, channel_coefficient,
    gbs_3d_antenna_gain_db, GAIN_HAP_DBI, GAIN_LEO_DBI,
    K_UMA_DB_MEAN, K_HAP_STATIC, K_LEO_STATIC
)
import src.constants as const

def get_active_sector(bs_pos, ue_pos):
    """Determines the GBS sector panel pointing closest to the UE."""
    dx = ue_pos[0] - bs_pos[0]
    dy = ue_pos[1] - bs_pos[1]
    azimuth_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    sectors_pool = [30.0, 150.0, 270.0]

    def angular_difference(angle1, angle2):
        return abs((angle1 - angle2 + 180.0) % 360.0 - 180.0)

    active_sector = min(
        sectors_pool,
        key=lambda sector: angular_difference(azimuth_deg, sector)
    )
    return active_sector

def get_all_link_budgets(ue_pos, bs_coords, hap_coord, leo_coord, ue_idx=0, w_sq_gbs=None, w_sq_hap=1.0, w_sq_leo=1.0):
    """Calculates received power and link metrics for every node using per-RB transmit power."""
    links = []
    
    if w_sq_gbs is None:
        w_sq_gbs = [1.0] * len(bs_coords)

    # 1. NTN Links (HAP & LEO)
    d_hap = distance_3D(hap_coord, ue_pos)
    h_sq_hap = channel_coefficient(GAIN_HAP_DBI, free_space_path_loss(d_hap, const.CARRIER_FREQ_GHZ), w_sq_hap)
    rx_hap = const.TX_POWER_HAP_RB_W * h_sq_hap
    links.append({
        'name': 'HAP', 'rx_w': rx_hap, 'is_ntn': True, 'pos': hap_coord, 
        'dist': d_hap, 'p_tx': const.TX_POWER_HAP_RB_W, 'h_sq': h_sq_hap,
        'gain_dbi': GAIN_HAP_DBI, 'w_sq': w_sq_hap
    })
    
    d_leo = distance_3D(leo_coord, ue_pos)
    h_sq_leo = channel_coefficient(GAIN_LEO_DBI, free_space_path_loss(d_leo, const.CARRIER_FREQ_GHZ), w_sq_leo)
    rx_leo = const.TX_POWER_LEO_RB_W * h_sq_leo
    links.append({
        'name': 'LEO', 'rx_w': rx_leo, 'is_ntn': True, 'pos': leo_coord, 
        'dist': d_leo, 'p_tx': const.TX_POWER_LEO_RB_W, 'h_sq': h_sq_leo,
        'gain_dbi': GAIN_LEO_DBI, 'w_sq': w_sq_leo
    })

    # 2. Terrestrial Links (GBS) with Sectorization (r=1, s=3)
    gbs_sector_powers = [] 
    sectors_pool = [30.0, 150.0, 270.0]
    
    for i, bs_pos in enumerate(bs_coords):
        d_gbs = distance_3D(bs_pos, ue_pos)
        pl_gbs = path_loss(d_gbs, const.CARRIER_FREQ_GHZ)
            
        best_sector = get_active_sector(bs_pos=bs_pos, ue_pos=ue_pos)
        sector_rx_dict = {}
        best_serving_gain = -999.0
        best_h_sq = 0.0
        best_rx = 0.0
        
        # Calculate received power radiating from all 3 fixed sector panels
        for sec in sectors_pool:
            gain_dbi = gbs_3d_antenna_gain_db(ue_pos=ue_pos, bs_pos=bs_pos, active_sector_deg=sec)
            h_sq = channel_coefficient(gain_dbi, pl_gbs, w_sq_gbs[i])
            rx_w = const.TX_POWER_GBS_RB_W * h_sq
            
            sector_rx_dict[sec] = rx_w
            
            if sec == best_sector:
                best_serving_gain = gain_dbi
                best_h_sq = h_sq
                best_rx = rx_w
                
        gbs_sector_powers.append(sector_rx_dict)
        
        links.append({
            'name': f'GBS_{i}', 
            'rx_w': best_rx,       
            'is_ntn': False, 
            'pos': bs_pos, 
            'dist': d_gbs, 
            'p_tx': const.TX_POWER_GBS_RB_W, 
            'h_sq': best_h_sq,
            'gain_dbi': best_serving_gain,
            'w_sq': w_sq_gbs[i],
            'sector_deg': best_sector
        })
        
    return links, gbs_sector_powers