# import numpy as np
# import pandas as pd

# # Risk configuration

# FAILED_BS_ID = 2
# WEATHER_PENALTY_DB = 25.0
# SINR_OUTAGE_THRESHOLD_DB = 5.0
# MAX_USERS_PER_CELL = 10 

# def apply_hardware_and_weather_risks(active_risk, bs_id, ue_pos, pl_db):
#     """Applies pre-association path loss penalties based on active risks."""
#     if active_risk == "GBS_Failure" and bs_id == FAILED_BS_ID:
#          # Infinite path loss = total signal blackout
#         return np.inf 
        
#     elif active_risk == "Weather_Hazard":
#         # 2km x 2km localized storm cell
#         if 1000 < ue_pos[0] < 3000 and 1000 < ue_pos[1] < 3000:
#             return pl_db + WEATHER_PENALTY_DB
            
#     return pl_db

# def apply_capacity_congestion(active_risk, df):
#     """Applies post-association network scheduler drops for limited RBs."""
#     if active_risk != "Limited_RBs":
#         return df

#     # Isolate users connected to terrestrial towers
#     gbs_users = df[df['Connected_To'].str.startswith('BS')].copy()
    
#     # Sort by distance so closest users get the RBs first
#     gbs_users = gbs_users.sort_values(by=['Connected_To', 'Distance (m)'])
    
#     # Group by Base Station and apply the capacity cutoff
#     gbs_users['Capacity_Rank'] = gbs_users.groupby('Connected_To').cumcount()
    
#     # Identify users who got dropped by the scheduler
#     dropped_mask = gbs_users['Capacity_Rank'] >= MAX_USERS_PER_CELL
#     dropped_ue_ids = gbs_users[dropped_mask]['UE_ID']
    
#     # Re-route the dropped users to the HAP backup
#     df.loc[df['UE_ID'].isin(dropped_ue_ids), 'Connected_To'] = 'HAP_0'
#     df.loc[df['UE_ID'].isin(dropped_ue_ids), 'Status'] = 'Rescued (Limited_RBs)'
    
#     return df



import numpy as np
import random
import pandas as pd

# ==========================================
# RISK CONFIGURATION: BS FAILURE (50% RATE)
# ==========================================
BS_FAILURE_RATE = 0.5 

def generate_failed_towers(total_bs_count):
    """
    Calculates exactly how many towers equal 50% of the network 
    and randomly selects their IDs to fail for the current simulation run.
    """
    all_bs_ids = list(range(total_bs_count))
    num_failing_towers = int(total_bs_count * BS_FAILURE_RATE)
    failed_bs_ids = random.sample(all_bs_ids, num_failing_towers)
    return failed_bs_ids

def apply_bs_failure_risk(bs_id, failed_bs_ids, pl_db):
    """
    Checks if the current Base Station is on the failed list.
    If yes, applies an infinite path loss (0.0 availability).
    """
    if bs_id in failed_bs_ids:
        return np.inf 
    return pl_db

# ==========================================
# NTN CAPACITY SCHEDULER (ALGORITHM 1)
# ==========================================
def apply_risk_disjoint_grouping(df, total_hap_rbs=10):
    """
    Implements Algorithm 1: Risk-Disjoint User Grouping.
    Users can only share a HAP RB if their primary terrestrial risks are independent.
    """
    # Isolate the users requesting a HAP connection
    hap_users = df[df['Connected_To'].str.startswith('HAP')].copy()
    
    # Initialize the HAP Resource Blocks
    hap_rbs = {rb_id: {'ue_ids': [], 'active_risks': set()} for rb_id in range(total_hap_rbs)}
    
    successful_rescues = []
    dropped_ues = []
    
    for index, row in hap_users.iterrows():
        ue_id = row['UE_ID']
        primary_risk = row['Primary_BS_Risk'] 
        assigned = False
        
        # Try to pack the user into an existing HAP RB safely
        for rb_id, rb_data in hap_rbs.items():
            
            # THE DISJOINT CONDITION: Is this RB safe from simultaneous collision?
            if primary_risk not in rb_data['active_risks']:
                rb_data['ue_ids'].append(ue_id)
                rb_data['active_risks'].add(primary_risk)
                
                successful_rescues.append(ue_id)
                assigned = True
                break 
                
        # If all 10 RBs already contain someone with the same risk, drop the user
        if not assigned:
            dropped_ues.append(ue_id)
            
    return successful_rescues, dropped_ues, hap_rbs