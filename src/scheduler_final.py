import math
from src.system_model import check_transmission_success
import src.constants as const


def risk_disjoint_groups(affected_users):
    """ Form risk-disjoint user groups based on heterogeneous risks."""
    if not affected_users: 
        return []
        
    risk_map = {}
    for user in affected_users:
        r_prof = user.get("risk_profile", "Unknown")
        risk_map.setdefault(r_prof, []).append(user)

    disjoint_groups = []
    while any(risk_map.values()):
        current_group = []
        for r_prof in list(risk_map.keys()):
            if risk_map[r_prof]:
                current_group.append(risk_map[r_prof].pop(0))
        if current_group:
            disjoint_groups.append(current_group)
            
    return disjoint_groups


def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices=None, fixed_backup_rbs=10, verbose=False):
    """ User grouping and backup path allocation with dynamic available RBs."""
    if not affected_users: 
        return 0, {}, {}, {}, []

    disjoint_groups = risk_disjoint_groups(affected_users)
    
    allocated_loads = {node: 0 for node in active_nodes}
    remaining_rbs = {node: fixed_backup_rbs for node in active_nodes}
    node_composition = {node: {} for node in active_nodes}
    final_user_scores = {}
    recovery_details = []
    recovered_count = 0

    def get_best_terrestrial_backup(user_dict):
        gbs_links = [a for node, a in user_dict["candidate_links"].items() if "GBS" in node]
        return max(gbs_links) if gbs_links else 0.0

    for group in disjoint_groups:
        contending_counts = {node: 0 for node in active_nodes}
        for user in group:
            for node in user["candidate_links"].keys():
                if node in contending_counts:
                    contending_counts[node] += 1
        
        group_phi = {}
        for node in active_nodes:
            N_i = contending_counts[node]
            r_avail = remaining_rbs[node]
            if N_i > 0 and r_avail > 0:
                group_phi[node] = min(1.0, r_avail / N_i)
            else:
                group_phi[node] = 0.0  

        rbs_consumed_this_group = set()
        sorted_users = sorted(group, key=get_best_terrestrial_backup)

        for user in sorted_users:
            ue_id = user["ue_id"]
            p_node = user.get("primary_node", "Unknown")
            r_prof = user.get("risk_profile", "Unknown")
            a_primary = user.get("primary_availability", 0.0)
            
            best_node = None
            best_b_in = 0.0 
            best_phi_val = 0.0
            best_a_backup_val = 0.0

            for node, a_backup_raw in user["candidate_links"].items():
                if node not in active_nodes or remaining_rbs[node] <= 0:
                    continue
                
                phi_i = group_phi[node]
                if phi_i <= 0.0:
                    continue
                
                se_bps_hz = user["spectral_efficiencies"][node]
                distance_m = user["distances"][node]
                eps_val = user["eps_values"].get(node, 1e-5)

                effective_cap_mbps = (const.BANDWIDTH_RB * se_bps_hz * phi_i) / 1e6
                
                is_success, _ = check_transmission_success(
                    capacity_mbps = effective_cap_mbps,
                    distance_m = distance_m,
                    error_probability = eps_val
                )
                
                if not is_success:
                    continue
                
                b_in = a_primary + (1.0 - a_primary) * a_backup_raw * phi_i
                
                if b_in > best_b_in:
                    best_b_in = b_in
                    best_node = node
                    best_phi_val = phi_i
                    best_a_backup_val = a_backup_raw

            is_recovered = best_node is not None
            if is_recovered:
                allocated_loads[best_node] += 1
                if best_node not in rbs_consumed_this_group:
                    remaining_rbs[best_node] = max(0, remaining_rbs[best_node] - 1)
                    rbs_consumed_this_group.add(best_node)

                node_composition[best_node][r_prof] = node_composition[best_node].get(r_prof, 0) + 1
                final_user_scores[ue_id] = best_b_in
                recovered_count += 1
            else:
                final_user_scores[ue_id] = 0.0
            
            sinr_val = user.get("sinrs", {}).get(best_node, 0.0) if best_node else 0.0
            recovery_details.append({
                "ue_id": ue_id,
                "primary_node": p_node,
                "risk_profile": r_prof,
                "a_primary": a_primary,
                "backup_node": best_node if is_recovered else "None",
                "sinr": sinr_val,
                "a_backup": best_a_backup_val,
                "phi_i": best_phi_val,
                "b_in": best_b_in,
                "is_recovered": is_recovered
            })

    return recovered_count, allocated_loads, node_composition, final_user_scores, recovery_details