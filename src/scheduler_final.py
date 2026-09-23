import math
from src.system_model import check_transmission_success
import src.constants as const

def risk_disjoint_groups(affected_users):
    """
    Algorithm 1 - Form risk-disjoint user groups.
    """
    if not affected_users: 
        return []
        
    risk_map = {}
    for user in affected_users:
        p_node = user.get("primary_node", "Unknown")
        if p_node not in risk_map: 
            risk_map[p_node] = []
        risk_map[p_node].append(user)
    
    disjoint_groups = []
    while any(risk_map.values()):
        current_group = []
        for p_node in list(risk_map.keys()):
            if risk_map[p_node]: 
                current_group.append(risk_map[p_node].pop(0))
        if current_group: 
            disjoint_groups.append(current_group)
            
    return disjoint_groups


def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices=None, fixed_backup_rbs=10, verbose=False):
    """
    Algorithm 1: User grouping and backup path allocation with dynamic available RBs.
    """
    if not affected_users: 
        return 0, {}, {}, {}, []

    disjoint_groups = risk_disjoint_groups(affected_users)
    
    # Track allocated users and remaining available backup RBs per node
    allocated_loads = {node: 0 for node in active_nodes}
    remaining_rbs = {node: fixed_backup_rbs for node in active_nodes}
    node_composition = {node: {} for node in active_nodes}
    final_user_scores = {}
    recovery_details = []
    recovered_count = 0

    def get_best_terrestrial_backup(user_dict):
        gbs_links = [
            a_raw for node, a_raw in user_dict["candidate_links"].items()
            if "GBS" in node
        ]
        return max(gbs_links) if gbs_links else 0.0

    for group in disjoint_groups:
        
        # Count potential contenders in this group
        contending_counts = {node: 0 for node in active_nodes}
        for user in group:
            for node in user["candidate_links"].keys():
                if node in contending_counts:
                    contending_counts[node] += 1
        
        # Compute shareability using current available backup RBs
        group_phi = {}
        for node in active_nodes:
            N_i = contending_counts[node]
            r_avail = remaining_rbs[node]
            if N_i > 0 and r_avail > 0:
                group_phi[node] = min(1.0, r_avail / N_i)
            else:
                group_phi[node] = 0.0  # Exhausted resources yield zero shareability

        sorted_users = sorted(group, key=get_best_terrestrial_backup)

        # Allocate backup path according to Eq.14
        for user in sorted_users:
            ue_id = user["ue_id"]
            p_node = user.get("primary_node", "Unknown")
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
                
                # Eq.13
                b_in = a_primary + (1.0 - a_primary) * a_backup_raw * phi_i
                
                # Eq.14
                if b_in > best_b_in:
                    best_b_in = b_in
                    best_node = node
                    best_phi_val = phi_i
                    best_a_backup_val = a_backup_raw

            if best_node:
                allocated_loads[best_node] += 1
                remaining_rbs[best_node] = max(0, remaining_rbs[best_node] - 1)
                
                if p_node not in node_composition[best_node]:
                    node_composition[best_node][p_node] = 0
                node_composition[best_node][p_node] += 1
                
                final_user_scores[ue_id] = best_b_in
                recovered_count += 1
                
                sinr_val = user.get("sinrs", {}).get(best_node, 0.0)
                recovery_details.append({
                    "ue_id": ue_id,
                    "primary_node": p_node,
                    "backup_node": best_node,
                    "sinr": sinr_val,
                    "a_in": best_a_backup_val,
                    "phi_i": best_phi_val,
                    "b_in": best_b_in
                })
            else:
                final_user_scores[ue_id] = 0.0

    return recovered_count, allocated_loads, node_composition, final_user_scores, recovery_details


