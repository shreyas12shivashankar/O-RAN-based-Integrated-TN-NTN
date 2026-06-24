import numpy as np
from src.topology import get_hexagonal_bs, get_random_users

def generate_bs_failure_scenario(total_users, total_bs, failed_bs_list):
    """
    Identifies which users are geographically affected by the failing base stations.
    """
    # Generate the exact same deterministic topology as the main simulation
    bs_coords = get_hexagonal_bs(radius=2000)
    
    # Use the same seed as the main simulation to ensure user coordinates match exactly
    np.random.seed(42) 
    ue_coords = get_random_users(n=total_users)
    
    affected_users = []
    
    for ue_id, ue_pos in enumerate(ue_coords):
        # Find the geographically closest Ground Base Station
        distances = [np.linalg.norm(ue_pos - bs_pos) for bs_pos in bs_coords]
        closest_bs_idx = np.argmin(distances)
        primary_gbs = f"GBS_{closest_bs_idx}"
        
        # If their primary GBS is in the failure list, add them to the affected pool
        if primary_gbs in failed_bs_list:
            affected_users.append({
                "ue_id": ue_id,
                "primary_gbs": primary_gbs
            })
            
    return affected_users