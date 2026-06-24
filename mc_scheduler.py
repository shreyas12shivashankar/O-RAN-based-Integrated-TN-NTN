# import numpy as np

# def allocate_backup_paths(affected_users, active_nodes, rb_distribution=None):
    
#     # Initialize the backup pool for every active node
#     backup_paths = {
#         node_name: {"r_i": rb_distribution, "users": [], "failed_bs_in_path": []}
#         for node_name in active_nodes
#     }
    
#     recovered_users = 0
#     dropped_users = 0

#     for user in affected_users:
#         user_risk = user["primary_gbs"]  

#         best_path_name = None
#         best_b_in = -1

#         for path_name, path_data in backup_paths.items():
            
#             # Fetch raw availability. If 0, the link failed the physical 0.99999 check.
#             a_in = user.get(f"a_in_{path_name}", 0) 
#             if a_in == 0:
#                 continue 
                
#             # Evaluate proposed capacity
#             proposed_load_Ni = len(path_data["users"]) + 1  #How many users are already assigned  to this backup node + current useer
            
#             # Hard Capacity enforcement : Discard path if node has exhausted all RBs
#             if proposed_load_Ni > path_data["r_i"]:
#                 continue
                
#             # Eq 12: Shareability factor
#             phi_i = path_data["r_i"] / proposed_load_Ni #  calculates shareability factor
#             shareability_factor = min(1.0, phi_i)  # Capping shareability to 1.0
            
#             # Eq 13: Calculate E2E Shared Availability
#             b_in = a_in * shareability_factor
            
#             # Eq 14: Select the backup path that maximizes E2E availability
#             if b_in > best_b_in:
#                 best_b_in = b_in
#                 best_path_name = path_name
                 
#         if best_path_name is not None:
#             # Commit allocation to the optimal path
#             backup_paths[best_path_name]["users"].append(user["ue_id"])
#             backup_paths[best_path_name]["failed_bs_in_path"].append(user_risk)
#             user["final_connection"] = f"{best_path_name}_Shared_Pool"
#             recovered_users += 1
#         else:
#             # User drops if all paths are physically invalid or at max RB capacity
#             user["final_connection"] = "Dropped"
#             dropped_users += 1

#     total_affected = len(affected_users)
#     resilience_percentage = (recovered_users / total_affected) * 100 if total_affected > 0 else 100

#     return resilience_percentage, backup_paths, affected_users


import numpy as np

def allocate_backup_paths(affected_users, active_nodes, rb_distribution=None):
    """
    Matrix-based Risk-Aware Backup Path Allocator with Strict Hard Capacity limits.
    """
    if rb_distribution is None:
        rb_distribution = {node: 10 for node in active_nodes}
        
    # Initialize the backup pool trackers
    backup_paths = {
        node_name: {
            "r_i": rb_distribution.get(node_name, 0), 
            "users": [], 
            "failed_bs_in_path": []
        }
        for node_name in active_nodes
    }
    
    # Build the Availability Matrix (Preference Ranking)
    user_preferences = {}
    for user in affected_users:
        prefs = []
        for node in active_nodes:
            a_in = user.get(f"a_in_{node}", 0)
            if a_in > 0: 
                prefs.append((node, a_in))
                
        # Sort highest availability first
        prefs.sort(key=lambda x: x[1], reverse=True)
        user_preferences[user["ue_id"]] = prefs

    recovered_users = 0
    dropped_users = 0

    # Strict Capacity-Aware Allocation
    for user in affected_users:
        ue_id = user["ue_id"]
        prefs = user_preferences.get(ue_id, [])
        
        allocated = False
        for target_node, a_in in prefs:
            current_load = len(backup_paths[target_node]["users"])
            max_rbs = backup_paths[target_node]["r_i"]
            
            # STRICT HARD CAP: Only allocate if there is an empty RB available
            if current_load < max_rbs:
                backup_paths[target_node]["users"].append(ue_id)
                backup_paths[target_node]["failed_bs_in_path"].append(user.get("primary_gbs", "Unknown"))
                
                # Commit the connection
                user["final_connection"] = f"{target_node}_Shared_Pool"
                user["shareability"] = 1.0  # Since 1 user takes exactly 1 RB, phi remains 1.0
                user["b_in"] = a_in
                
                allocated = True
                recovered_users += 1
                break
        
        # If all preferred nodes were completely full, the user is immediately dropped
        if not allocated:
            user["final_connection"] = "Dropped"
            user["shareability"] = 0.0
            dropped_users += 1

    total_affected = len(affected_users)
    resilience_percentage = (recovered_users / total_affected) * 100 if total_affected > 0 else 100

    return resilience_percentage, backup_paths, affected_users