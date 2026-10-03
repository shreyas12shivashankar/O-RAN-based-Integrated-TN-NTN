import numpy as np
import pandas as pd
import src.constants as const
from src.system_model import (
    K_HAP_STATIC,
    K_LEO_STATIC,
    K_UMA_DB_MEAN,
    apply_rician_fading,
    check_transmission_success,
    compute_spatial_cholesky,
    error_probability,
    sinr,
)

def evaluate_primary_connection(ue_coords, bs_coords, hap_coord, leo_coord, link_evaluator_func):
    """Establishes primary associations for all UEs."""
    rho_physical = 1.0
    psi_backhaul = 1.0 - const.BACKHAUL_ERROR_PROB

    node_capacity = {f'GBS_{i}': const.PRIMARY_RBS_GBS for i in range(const.NUM_GBS)}
    node_capacity['HAP'] = const.TOTAL_RBS_GBS
    node_capacity['LEO'] = const.TOTAL_RBS_GBS

    ue_positions_array = np.array(ue_coords)
    spatial_l = compute_spatial_cholesky(ue_positions_array, const.CARRIER_FREQ_GHZ)

    gbs_fading_maps = [apply_rician_fading(spatial_l, K_UMA_DB_MEAN) for _ in range(const.NUM_GBS)]
    hap_fading_map = apply_rician_fading(spatial_l, K_HAP_STATIC)
    leo_fading_map = apply_rician_fading(spatial_l, K_LEO_STATIC)

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
            'links': links,
            'gbs_powers': gbs_powers,
            'max_gbs_rx': max_gbs_rx
        })

    # Prioritize scheduling users with strongest maximum received power
    user_data_list.sort(key=lambda x: x['max_gbs_rx'], reverse=True)
    results = []

    for user_data in user_data_list:
        ue_id = user_data['ue_id']
        links = user_data['links']
        gbs_powers = user_data['gbs_powers']

        viable_gbs = []
        viable_ntn = []

        for cand in links:
            if node_capacity[cand['name']] <= 0:
                continue

            # Co-channel interference: aggregate all non-serving sector powers
            interf_w = 0.0
            if not cand['is_ntn']:
                cand_name = cand['name']
                cand_sector = cand['sector_deg']
                for i in range(const.NUM_GBS):
                    for sec in (30.0, 150.0, 270.0):
                        if f'GBS_{i}' == cand_name and sec == cand_sector:
                            continue
                        interf_w += gbs_powers[i][sec]

            sinr_lin = sinr(cand['p_tx'], cand['h_sq'], interf_w, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_RB)
            spectral_efficiency = min(np.log2(1 + sinr_lin), 8.0)
            cap_mbps = (const.BANDWIDTH_RB * spectral_efficiency) / 1e6
            eps_wireless = error_probability(sinr_lin, M=const.MODULATION_M)

            is_success, d_total_ms = check_transmission_success(
                capacity_mbps=cap_mbps, distance_m=cand['dist'],
                packet_size_bytes=64, max_latency_ms=30.0, error_probability=eps_wireless
            )

            if not is_success:
                continue

            cand_metrics = {
                'sinr_lin': sinr_lin, 'cap_mbps': cap_mbps,
                'eps_wireless': eps_wireless, 'interf_w': interf_w,
                'latency_ms': d_total_ms
            }

            if not cand['is_ntn']:
                viable_gbs.append((cand, cand_metrics))
            else:
                viable_ntn.append((cand, cand_metrics))

        viable_gbs.sort(key=lambda x: x[0]['rx_w'], reverse=True)
        viable_ntn.sort(key=lambda x: x[0]['rx_w'], reverse=True)

        best_link, best_metrics = (viable_gbs[0] if viable_gbs else (viable_ntn[0] if viable_ntn else (None, None)))
        noise_w = const.NOISE_SPECTRAL_DENSITY_W * const.BANDWIDTH_RB

        if best_link:
            node_capacity[best_link['name']] -= 1
            a_jn = ((1.0 - best_metrics['eps_wireless']) * rho_physical) * ((psi_backhaul * rho_physical) ** 2)

            results.append({
                "UE_Idx": ue_id, "UE_ID": f"UE_{ue_id:02d}", "Primary_RU": best_link['name'],
                "Is_NTN": best_link['is_ntn'], "RU_Pos": best_link['pos'], "Dist_m": round(best_link['dist'], 2),
                "P_tx_W": best_link['p_tx'], "h_sq_jn": best_link['h_sq'], "Signal_W": best_link['rx_w'],
                "Interf_W": best_metrics['interf_w'], "Noise_W": noise_w,
                "Rx_Power_dBm": round(10 * np.log10(max(best_link['rx_w'], 1e-12)) + 30, 2),
                "SINR_dB": round(10 * np.log10(max(best_metrics['sinr_lin'], 1e-12)), 2),
                "Eps_Wireless": f"{best_metrics['eps_wireless']:.2e}", "a_jn": round(a_jn, 5),
                "Latency_ms": round(best_metrics['latency_ms'], 2), "Capacity_Mbps": round(best_metrics['cap_mbps'], 2),
                "Gain_dBi": round(best_link.get('gain_dbi', 0.0), 2),
                "w_sq": round(best_link.get('w_sq', 1.0), 4),
                "All_Links": links, "GBS_Powers": gbs_powers
            })
        else:
            results.append({
                "UE_Idx": ue_id, "UE_ID": f"UE_{ue_id:02d}", "Primary_RU": None,
                "Is_NTN": False, "RU_Pos": None, "Dist_m": None,
                "P_tx_W": None, "h_sq_jn": None, "Signal_W": None,
                "Interf_W": None, "Noise_W": None,
                "Rx_Power_dBm": None, "SINR_dB": None,
                "Eps_Wireless": None, "a_jn": 0.0,
                "Latency_ms": None, "Capacity_Mbps": None,
                "Gain_dBi": None, "w_sq": None,
                "All_Links": links, "GBS_Powers": gbs_powers
            })

    results.sort(key=lambda x: x['UE_Idx'])
    return pd.DataFrame(results)