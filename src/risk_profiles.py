from src.link_evaluator import get_all_link_budgets
import src.constants as const
 
def inject_bs_failure(primary_df, failed_bs_indices, evaluate_link_func):
    """ Simulates Ground Base Station failure risk """
    affected_users = []
    
    for _, row in primary_df.iterrows():
        ue_id = row['UE_Idx']
        primary_name = row['Primary_RU']
        is_ntn = row['Is_NTN']
        links = row['All_Links']
        gbs_powers = row['GBS_Powers']
        
        if is_ntn:
            continue

        primary_bs_idx = int(primary_name.split('_')[1])
        if primary_bs_idx not in failed_bs_indices:
            continue
        
        # User is affected. 
        user_links = {
            "ue_id": ue_id, 
            "primary_node": primary_name,
            "primary_availability": 0.0, 
            "candidate_links": {},
            "spectral_efficiencies": {},  
            "distances": {},
            "eps_values": {} 
        }

        # Calculate backup candidates under degraded network conditions
        for link in links:
            name = link['name']
            
            if not link['is_ntn'] and link['dist'] > const.MAX_ATTACH_DIST:
                continue 
            
            if not link['is_ntn']:
                bs_id = int(name.split('_')[1])
                
                # Skip failed nodes
                if bs_id in failed_bs_indices:
                    continue
                
                cand_sector = link.get('sector_deg', 30.0)
                interference = 0.0
                                
                for idx in range(len(gbs_powers)):
                    if idx in failed_bs_indices:
                        continue # Failed towers emit no interference
                        
                    for sec in (30.0, 150.0, 270.0):
                        if idx == bs_id and sec == cand_sector:
                            continue # Skip the serving sector itself
                            
                        if const.FREQUENCY_REUSE == 1:
                            interference += gbs_powers[idx][sec]
                        elif const.FREQUENCY_REUSE == 3:
                            if const.FR3_CLUSTER_MAP[f'GBS_{idx}'] == const.FR3_CLUSTER_MAP[name]:
                                interference += gbs_powers[idx][sec]
                                
            else: 
                interference = 0.0
            
            # Evaluate the degraded link and unpack the physical metrics
            a_jn, capped_se, dist, sinr_db, eps = evaluate_link_func(
                link['p_tx'], link['h_sq'], interference, link['dist']
            )
            
            if a_jn >= 0.0:
                user_links["candidate_links"][name] = a_jn
                user_links["spectral_efficiencies"][name] = capped_se
                user_links["distances"][name] = dist
                user_links["eps_values"][name] = eps
                
                if "sinrs" not in user_links:
                    user_links["sinrs"] = {}
                user_links["sinrs"][name] = sinr_db
                
        affected_users.append(user_links)
        
    return affected_users

