import numpy as np
import pandas as pd

def allocate_backup_paths(affected_users, total_backup_rbs=10):
    
    # Allocates affected users to a pool of backup RBs.
    # Users from different failed Ground BS can share an RB by Risk-disjoint grouping.
    
    # System Requirements from Paper Baseline
    TARGET_AVAILABILITY = 0.99999      # URLLC 5-nines threshold
    BACKUP_LINK_RELIABILITY = 0.999999 # Assumed inherent reliability of the NTN path
    
    # Initialize allocation pool structure
    backup_rbs = [
        {"rb_id": i, "users": [], "failed_towers_in_rb": []} 
        for i in range(total_backup_rbs)
    ]
    
    recovered_users = 0
    dropped_users = 0

    for user in affected_users:
        allocated = False
        user_risk = user["primary_gbs"]  # The specific GBS that failed

        for rb in backup_rbs:
            # CONDITION 1: Risk-Disjoint Rule
            # Two users from the EXACT SAME failed tower cannot share an RB
            if user_risk in rb["failed_towers_in_rb"]:
                continue 
                
            # Constraint 2: Dynamic E2E Availability Check
            proposed_load = len(rb["users"]) + 1
            
            # Eq 12: Shareability penalty for this specific RB pool (r_i = 1)
            shareability_penalty = 1.0 / proposed_load
            
            # Eq 13: E2E Availability (Assuming primary path a_jn = 0)
            e2e_availability = BACKUP_LINK_RELIABILITY * shareability_penalty
            
            # Reject allocation if sharing drops availability below URLLC threshold
            if e2e_availability < TARGET_AVAILABILITY:
                continue
                
            # Commit allocation
            rb["users"].append(user["ue_id"])
            rb["failed_towers_in_rb"].append(user_risk)
            allocated = True
            recovered_users += 1
            break
                 
        if not allocated:
            dropped_users += 1

    total_affected = len(affected_users)
    resilience_percentage = (recovered_users / total_affected) * 100 if total_affected > 0 else 100

    print(f"MC Scheduler Results")
    print(f"Shared Backup RBs: {total_backup_rbs}")
    print(f"Users Successfully Recovered: {recovered_users}")
    print(f"Users Dropped due to Capacity limit: {dropped_users}")
    print(f"Network Resilience: {resilience_percentage:.2f}%")
    
    return resilience_percentage, backup_rbs

if __name__ == "__main__":
    from src.risk_profiles import generate_bs_failure_scenario
    
    # Run the corrected 7-BS simulation setup
    stranded_users = generate_bs_failure_scenario(total_users=100)
    
    # Test with exactly 10 Global Backup RBs
    allocate_backup_paths(stranded_users, total_backup_rbs=10)
    
# import numpy as np

# def allocate_backup_paths(affected_users, total_backup_rbs=10):
#     """
#     Executes Risk-Disjoint resource scheduling. 
#     Enforces fault isolation and physical channel multiplexing limits.
#     """
#     MAX_USERS_PER_RB = 3
    
#     # Initialize allocation pool structure
#     backup_rbs = [
#         {"rb_id": i, "users": [], "fault_domains": []} 
#         for i in range(total_backup_rbs)
#     ]
    
#     recovered_count = 0
#     dropped_count = 0

#     for user in affected_users:
#         allocated = False
#         domain = user["primary_gbs"]

#         for rb in backup_rbs:
#             # Constraint 1: Risk-Disjoint Rule (Fault Isolation)
#             if domain in rb["fault_domains"]:
#                 continue
                
#             # Constraint 2: Physical Channel Capacity Limit
#             if len(rb["users"]) >= MAX_USERS_PER_RB:
#                 continue
                
#             # Commit allocation
#             rb["users"].append(user["ue_id"])
#             rb["fault_domains"].append(domain)
#             allocated = True
#             recovered_count += 1
#             break
            
#         if not allocated:
#             dropped_count += 1

#     total_affected = len(affected_users)
#     resilience_idx = (recovered_count / total_affected) * 100 if total_affected > 0 else 100.0

#     print(" [Scheduler Telemetry]")
#     print(f" Allocated channels : {total_backup_rbs} RBs")
#     print(f" Recovered targets  : {recovered_count} UEs")
#     print(f" Exhausted/Dropped  : {dropped_count} UEs")
#     print(f" Fleet Resilience   : {resilience_idx:.2f}%")
    
#     return resilience_idx, backup_rbs

# if __name__ == "__main__":
#     from src.risk_profiles import generate_bs_failure_scenario
    
#     # Execute standalone test evaluation
#     test_users = generate_bs_failure_scenario(total_users=100, total_gbs=7)
#     _, _ = allocate_backup_paths(test_users, total_backup_rbs=10)