import numpy as np
import pandas as pd
from src.system_model import (
    sinr, error_probability, check_transmission_success,
    compute_spatial_cholesky, apply_rician_fading,
    K_UMA_DB_MEAN, K_HAP_STATIC, K_LEO_STATIC
)
import src.constants as const

def evaluate_primary_connection(ue_coords, bs_coords, hap_coord, leo_coord, link_evaluator_func):
    RHO_PHYSICAL = 1.0 
    PSI_BACKHAUL = 1.0 - const.BACKHAUL_ERROR_PROB
    
    node_capacity = {f'GBS_{i}': const.PRIMARY_RBS_GBS for i in range(const.NUM_GBS)}
    node_capacity['HAP'] = const.TOTAL_RBS_GBS
    node_capacity['LEO'] = const.TOTAL_RBS_GBS
    
    ue_positions_array = np.array(ue_coords)
    spatial_L = compute_spatial_cholesky(ue_positions_array, const.CARRIER_FREQ_GHZ)
    
    gbs_fading_maps = [apply_rician_fading(spatial_L, K_UMA_DB_MEAN) for _ in range(const.NUM_GBS)]
    hap_fading_map = apply_rician_fading(spatial_L, K_HAP_STATIC)
    leo_fading_map = apply_rician_fading(spatial_L, K_LEO_STATIC)
    
    user_data_list = []
    for ue_id, ue_pos in enumerate(ue_coords):
        w_sq_gbs = [gbs_fading_maps[i][ue_id] for i in range(const.NUM_GBS)]
        w_sq_hap = hap_fading_map[ue_id]
        w_sq_leo = leo_fading_map[ue_id]
        
        links, gbs_powers = link_evaluator_func(
            ue_pos, bs_coords, hap_coord, leo_coord, 
            ue_idx=ue_id, w_sq_gbs=w_sq_gbs, w_sq_hap=w_sq_hap, w_sq_leo=w_sq_leo
        )
        gbs_links = [l for l in links if not l['is_ntn']]
        max_gbs_rx = max([l['rx_w'] for l in gbs_links]) if gbs_links else 0.0
        
        user_data_list.append({
            'ue_id': ue_id,
            'ue_pos': ue_pos,
            'links': links,
            'gbs_powers': gbs_powers,
            'max_gbs_rx': max_gbs_rx
        })
        
    user_data_list.sort(key=lambda x: x['max_gbs_rx'], reverse=True)
    results = []

    for user_data in user_data_list:
        ue_id = user_data['ue_id']
        links = user_data['links']
        gbs_powers = user_data['gbs_powers']

        # RESTORED: Strictly enforce the local geographic boundary
        valid_links = [l for l in links if l['is_ntn'] or l['dist'] <= const.MAX_ATTACH_DIST]
        sorted_links = sorted(valid_links, key=lambda x: x['rx_w'], reverse=True)
        
        best_link = None
        best_metrics = None

        for cand in sorted_links:
            if node_capacity[cand['name']] <= 0:
                continue

            if not cand['is_ntn']:
                cand_name = cand['name']
                cand_sector = cand['sector_deg']
                interf_w = 0.0
                
                for i in range(const.NUM_GBS):
                    for sec in (30.0, 150.0, 270.0):
                        if f'GBS_{i}' == cand_name and sec == cand_sector:
                            continue 
                        
                        if const.FREQUENCY_REUSE == 1:
                            interf_w += gbs_powers[i][sec]
                        elif const.FREQUENCY_REUSE == 3:
                            if const.FR3_CLUSTER_MAP[f'GBS_{i}'] == const.FR3_CLUSTER_MAP[cand_name]:
                                interf_w += gbs_powers[i][sec]
            else:
                interf_w = 0.0

            sinr_lin = sinr(
                p_jn=cand['p_tx'], 
                h_sq=cand['h_sq'], 
                interference_power=interf_w,
                noise_density=const.NOISE_SPECTRAL_DENSITY_W, 
                bandwidth=const.BANDWIDTH_RB
            )

            # Keep the realistic control channel drop for safety
            sinr_db = 10 * np.log10(sinr_lin)
            if sinr_db < -5.0:
                continue  

            spectral_efficiency = min(np.log2(1 + sinr_lin), 8.0)
            cap_mbps = (const.BANDWIDTH_RB * spectral_efficiency) / 1e6
            eps_wireless = error_probability(sinr_lin, M=const.MODULATION_M)

            is_success, d_total_ms = check_transmission_success(
                capacity_mbps=cap_mbps, 
                distance_m=cand['dist'],
                packet_size_bytes=64,
                max_latency_ms=30.0,
                error_probability=eps_wireless
            )

            if not is_success:
                continue

            best_link = cand
            best_metrics = {
                'sinr_lin': sinr_lin,
                'cap_mbps': cap_mbps,
                'eps_wireless': eps_wireless,
                'interf_w': interf_w,
                'latency_ms': d_total_ms
            }
            node_capacity[cand['name']] -= 1
            break

        # NTN Offloading Fallback
        if best_link is None:
            ntn_cands = [l for l in sorted_links if l['is_ntn'] and node_capacity[l['name']] > 0]
            
            if ntn_cands:
                best_link = ntn_cands[0]
                sinr_lin = sinr(
                    p_jn=best_link['p_tx'], h_sq=best_link['h_sq'], interference_power=0.0,
                    noise_density=const.NOISE_SPECTRAL_DENSITY_W, bandwidth=const.BANDWIDTH_RB
                )
                
                if 10 * np.log10(sinr_lin) < -5.0:
                    continue

                se = min(np.log2(1 + sinr_lin), 8.0)
                cap_mbps = (const.BANDWIDTH_RB * se) / 1e6
                eps_wireless = error_probability(sinr_lin, M=const.MODULATION_M)
                
                _, d_total_ms = check_transmission_success(
                    capacity_mbps=cap_mbps, distance_m=best_link['dist'],
                    packet_size_bytes=64, max_latency_ms=30.0, error_probability=eps_wireless
                )
                
                best_metrics = {
                    'sinr_lin': sinr_lin, 'cap_mbps': cap_mbps,
                    'eps_wireless': eps_wireless, 'interf_w': 0.0, 'latency_ms': d_total_ms
                }
                node_capacity[best_link['name']] -= 1
            else:
                continue
        
        noise_w = const.NOISE_SPECTRAL_DENSITY_W * const.BANDWIDTH_RB
        a_jn = ((1.0 - eps_wireless) * RHO_PHYSICAL) * (PSI_BACKHAUL * RHO_PHYSICAL)

        results.append({
            "UE_Idx": ue_id, "UE_ID": f"UE_{ue_id:02d}", "Primary_RU": best_link['name'],
            "Is_NTN": best_link['is_ntn'], "RU_Pos": best_link['pos'], "Dist_m": round(best_link['dist'], 2),
            "P_tx_W": best_link['p_tx'], "h_sq_jn": best_link['h_sq'], "Signal_W": best_link['rx_w'],
            "Interf_W": best_metrics['interf_w'], "Noise_W": noise_w,
            "Rx_Power_dBm": round(10 * np.log10(best_link['rx_w']) + 30, 2),
            "SINR_dB": round(10 * np.log10(best_metrics['sinr_lin']), 2),
            "Eps_Wireless": f"{best_metrics['eps_wireless']:.2e}", "a_jn": round(a_jn, 5),
            "Latency_ms": round(best_metrics['latency_ms'], 2), "Capacity_Mbps": round(best_metrics['cap_mbps'], 2),
            "Gain_dBi": round(best_link.get('gain_dbi', 0.0), 2),
            "w_sq": round(best_link.get('w_sq', 1.0), 4),
            "All_Links": links, "GBS_Powers": gbs_powers
        })
        
    results.sort(key=lambda x: x['UE_Idx'])
    return pd.DataFrame(results)