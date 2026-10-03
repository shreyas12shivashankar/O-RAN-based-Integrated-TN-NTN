import pandas as pd
import src.constants as const

def inject_heterogeneous_risks(primary_df, failed_bs_indices=[0], evaluate_link_func=None, sinr_edge_threshold=5.0):
    """
    Simulates and injects three mutually disjoint risk types :
      1. GBS Failure: Complete hardware failure of the designated ground base station.
      2. Low_SINR: Cell-edge degradation in one geographic cluster.
      3. Weather: Environmental degradation in a separate geographic cluster.
    """
    if failed_bs_indices is None:
        failed_bs_indices = [0]
        
    affected_users = []
    
    # Partitioning the network to ensure the physical risks remain disjoint
    low_sinr_cluster = {1, 2, 6}
    weather_cluster = {3, 4, 5}
    
    for _, row in primary_df.iterrows():
        ue_id = row['UE_Idx']
        primary_name = row['Primary_RU']
        is_ntn = row['Is_NTN']
        sinr_db = row['SINR_dB']
        links = row['All_Links']
        gbs_powers = row['GBS_Powers']
        
        if pd.isna(primary_name) or is_ntn:
            continue

        primary_bs_idx = int(primary_name.split('_')[1])
        risk_profile = None
        
        if primary_bs_idx in failed_bs_indices:
            risk_profile = "BS_Failure"
        elif sinr_db < sinr_edge_threshold:
            if primary_bs_idx in low_sinr_cluster:
                risk_profile = "Low_SINR"
            elif primary_bs_idx in weather_cluster:
                risk_profile = "Weather"
                
        if not risk_profile:
            continue
            
        user_links = {
            "ue_id": ue_id,
            "primary_node": primary_name,
            "risk_profile": risk_profile,
            "primary_availability": 0.0 if risk_profile == "BS_Failure" else row['a_jn'],
            "candidate_links": {},
            "spectral_efficiencies": {},
            "distances": {},
            "eps_values": {},
            "sinrs": {}
        }

        for link in links:
            name = link['name']
            interference = 0.0
            
            if not link['is_ntn']:
                bs_id = int(name.split('_')[1])
                if bs_id in failed_bs_indices:
                    continue
                
                cand_sector = link.get('sector_deg', 30.0)
                                
                for idx in range(len(gbs_powers)):
                    if idx in failed_bs_indices:
                        continue
                        
                    for sec in (30.0, 150.0, 270.0):
                        if idx == bs_id and sec == cand_sector:
                            continue
                        interference += gbs_powers[idx][sec]
            
            a_jn, capped_se, dist, sec_sinr_db, eps = evaluate_link_func(
                link['p_tx'], link['h_sq'], interference, link['dist']
            )
            
            if a_jn >= 0.0:
                user_links["candidate_links"][name] = a_jn
                user_links["spectral_efficiencies"][name] = capped_se
                user_links["distances"][name] = dist
                user_links["eps_values"][name] = eps
                user_links["sinrs"][name] = sec_sinr_db
                
        affected_users.append(user_links)
        
    return affected_users

