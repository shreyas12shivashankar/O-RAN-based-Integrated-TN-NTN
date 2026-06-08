import numpy as np
import pandas as pd
    
def generate_bs_failure_scenario(total_users=100, total_gbs=7):
    # Generates a network topology with 50% UEs affected by BS failures.
    failed_gbs = ["GBS_0", "GBS_1", "GBS_2"] 
    active_gbs = ["GBS_3", "GBS_4", "GBS_5", "GBS_6"]
    
    users = []

    # Force exactly 50 users into the failed and 50 into the active groups.
    for i in range(total_users):
        if i < 50:
            primary_gbs = failed_gbs[i % len(failed_gbs)] 
        else:
            primary_gbs = active_gbs[i % len(active_gbs)] 
            
        users.append({
            "ue_id": i,
            "primary_gbs": primary_gbs,
            "status": "Healthy"
        })
    
    affected_users = []

    for u in users:
       if u["primary_gbs"] in failed_gbs:
           affected_users.append(u)
    
    for user in affected_users:
        user["status"] = f"Stranded {user['primary_gbs']} Failed"
            
    return affected_users