import numpy as np

def allocate_backup_paths(affected_users, active_nodes, total_backup_rbs_per_node):
    
    # Initialize the backup pool for EVERY active node
    backup_paths = {
        node_name: {"r_i": total_backup_rbs_per_node, "users": [], "failed_bs_in_path": []}
        for node_name in active_nodes
    }
    
    recovered_users = 0
    dropped_users = 0

    for user in affected_users:
        user_risk = user["primary_gbs"]  

        best_path_name = None
        best_b_in = -1

        for path_name, path_data in backup_paths.items():
            
            # Fetch raw availability. If 0, the link failed the physical 0.99999 check.
            a_in = user.get(f"a_in_{path_name}", 0) 
            if a_in == 0:
                continue 
                
            # Evaluate proposed capacity
            proposed_load_Ni = len(path_data["users"]) + 1
            
            # Hard Capacity Enforcer: Discard path if node has exhausted all RBs
            if proposed_load_Ni > path_data["r_i"]:
                continue
                
            # Eq 12: Shareability multiplier
            phi_i = path_data["r_i"] / proposed_load_Ni
            shareability_factor = min(1.0, phi_i) 
            
            # Eq 13: Calculate E2E Shared Availability
            b_in = a_in * shareability_factor
            
            # Eq 14: Select the backup path that maximizes E2E availability
            if b_in > best_b_in:
                best_b_in = b_in
                best_path_name = path_name
                 
        if best_path_name is not None:
            # Commit allocation to the optimal path
            backup_paths[best_path_name]["users"].append(user["ue_id"])
            backup_paths[best_path_name]["failed_bs_in_path"].append(user_risk)
            user["final_connection"] = f"{best_path_name}_Shared_Pool"
            recovered_users += 1
        else:
            # User drops if all paths are physically invalid or at max RB capacity
            user["final_connection"] = "Dropped"
            dropped_users += 1

    total_affected = len(affected_users)
    resilience_percentage = (recovered_users / total_affected) * 100 if total_affected > 0 else 100

    return resilience_percentage, backup_paths, affected_users