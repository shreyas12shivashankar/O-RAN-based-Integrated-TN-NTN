
# import numpy as np

# def allocate_backup_paths(affected_users, active_nodes, rb_distribution=None):
#     """
#     Matrix-based Risk-Aware Backup Path Allocator with Strict Hard Capacity limits.
#     """
#     if rb_distribution is None:
#         rb_distribution = {node: 10 for node in active_nodes}
        
#     # Initialize the backup pool trackers
#     backup_paths = {
#         node_name: {
#             "r_i": rb_distribution.get(node_name, 0), 
#             "users": [], 
#             "failed_bs_in_path": []
#         }
#         for node_name in active_nodes
#     }
    
#     # Build the Availability Matrix (Preference Ranking)
#     user_preferences = {}
#     for user in affected_users:
#         prefs = []
#         for node in active_nodes:
#             a_in = user.get(f"a_in_{node}", 0)
#             if a_in > 0: 
#                 prefs.append((node, a_in))
                
#         # Sort highest availability first
#         prefs.sort(key=lambda x: x[1], reverse=True)
#         user_preferences[user["ue_id"]] = prefs

#     recovered_users = 0
#     dropped_users = 0

#     # Strict Capacity-Aware Allocation
#     for user in affected_users:
#         ue_id = user["ue_id"]
#         prefs = user_preferences.get(ue_id, [])
        
#         allocated = False
#         for target_node, a_in in prefs:
#             current_load = len(backup_paths[target_node]["users"])
#             max_rbs = backup_paths[target_node]["r_i"]
            
#             # STRICT HARD CAP: Only allocate if there is an empty RB available
#             if current_load < max_rbs:
#                 backup_paths[target_node]["users"].append(ue_id)
#                 backup_paths[target_node]["failed_bs_in_path"].append(user.get("primary_gbs", "Unknown"))
                
#                 # Commit the connection
#                 user["final_connection"] = f"{target_node}_Shared_Pool"
#                 user["shareability"] = 1.0  # Since 1 user takes exactly 1 RB, phi remains 1.0
#                 user["b_in"] = a_in
                
#                 allocated = True
#                 recovered_users += 1
#                 break
        
#         # If all preferred nodes were completely full, the user is immediately dropped
#         if not allocated:
#             user["final_connection"] = "Dropped"
#             user["shareability"] = 0.0
#             dropped_users += 1

#     total_affected = len(affected_users)
#     resilience_percentage = (recovered_users / total_affected) * 100 if total_affected > 0 else 100

#     return resilience_percentage, backup_paths, affected_users

def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices, fixed_rb_value=10, verbose=False):
    """
    Executes the Risk-Aware Backup Path Allocation (Algorithm 1).
    Groups users by risk, calculates shareability, and dynamically offloads users.
    """
    if not affected_users:
        return 0, {}

    # 1. Risk-Disjoint Grouping (Group by the specific GBS that failed)
    groups = {bs_idx: [] for bs_idx in failed_bs_indices}
    for user in affected_users:
        if user["primary_gbs"] in groups:
            groups[user["primary_gbs"]].append(user)
            
    allocated_loads = {node: 0 for node in active_nodes}
    recovered_count = 0

    # 2. Shareability & Offload Allocation Loop
    for bs_idx, group_users in groups.items():
        Ni_group_size = len(group_users)
        
        if Ni_group_size == 0:
            continue
            
        for user in group_users:
            best_node = None
            best_b_in = -1.0
            
            for node_name in active_nodes:
                if node_name in user: # If the physical link exists
                    a_in = user[node_name]
                    
                    # Predict load to calculate dynamic shareability penalty
                    proposed_load = allocated_loads[node_name] + 1
                    phi = min(1.0, fixed_rb_value / proposed_load) 
                    
                    # Calculate Shared E2E Availability (b_in)
                    b_in = a_in * phi 
                    
                    if b_in > best_b_in:
                        best_b_in = b_in
                        best_node = node_name
            
            # Tuning drop threshold set to 0.70 based on paper charts
            if best_node is not None and best_b_in >= 0.70: 
                allocated_loads[best_node] += 1
                recovered_count += 1
                if verbose:
                    print(f"UE {user['ue_id']:<3} -> Assigned to {best_node} (Score: {best_b_in:.3f})")
            else:
                if verbose:
                    print(f"UE {user['ue_id']:<3} -> DROPPED")

    return recovered_count, allocated_loads