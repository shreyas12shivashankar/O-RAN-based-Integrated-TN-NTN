import numpy as np
import matplotlib.pyplot as plt

def calculate_theoretical_availability():
    
    # Varying the physical availability of a single segment (rho_s) from 0.8 to 1.0
    # This represents the complement of the failure probability (failure = 1 - rho_s)
    rho_s = np.linspace(0.80, 1.00, 100)
    
    # An End-to-End path consists of segments (Core -> Edge -> BS -> UE)
    num_segments = 3 
    
    # Network Load Parameters for Dynamic Shareability
    total_users = 100

    # Calculating availability for each case:
    # Case 1: Single Path (Series of segments)
    a_single = rho_s ** num_segments
    a_backup_path = rho_s ** num_segments

    # Case 2: Multi-Connectivity
    a_mc = 1 - ((1 - a_single) * (1 - a_backup_path))

    # Defining Shareability 
    # The probability of primary failure for any given user
    p_primary_fail = 1 - a_single
    
    # Expected number of users thrown into the backup pool at this failure rate
    n_affected = total_users * p_primary_fail
    
    # Create empty arrays to store the shareability for both RB scenarios
    phi_10_rbs = np.zeros(len(n_affected))
    phi_20_rbs = np.zeros(len(n_affected))
    
    # Standard readable loop to calculate shareability at every point on the graph
    for i in range(len(n_affected)):
        users = n_affected[i]
        
        if users > 0:
            # If affected users exceed the available RBs, they share the capacity
            phi_10_rbs[i] = min(1.0, 10 / users)
            phi_20_rbs[i] = min(1.0, 20 / users)
        else:
            # If no users are affected, shareability is perfect
            phi_10_rbs[i] = 1.0
            phi_20_rbs[i] = 1.0

    # Scenario 3: Backup Path Availabilities
    a_backup_10 = a_single + ((1 - a_single) * a_backup_path * phi_10_rbs)
    a_backup_20 = a_single + ((1 - a_single) * a_backup_path * phi_20_rbs)

    # Plotting
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(rho_s, a_single, label='Single Path', color='#1f77b4', linewidth=2)  
    plt.plot(rho_s, a_backup_10, label='Backup Path (10 RBs)', color='#ff7f0e', linewidth=2)
    plt.plot(rho_s, a_backup_20, label='Backup Path (20 RBs)', color='#ff7f0e', linewidth=2)
    plt.plot(rho_s, a_mc, label='Multi-connectivity', color='#2ca02c', linewidth=2, linestyle='--')
    
    
    # Formatting the plot
    plt.title('Theoretical Availability vs. Segment Physical Availability ($\\rho_s$)')
    plt.xlabel('Segment Physical Availability ($\\rho_s$)')
    plt.ylabel('End-to-End Availability')
    plt.xlim(0.80, 1.00)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # Add a visual marker for where congestion begins
    # Find the first point where 10 RB pool gets congested
    congestion_10_idx = np.where(phi_10_rbs < 1.0)[0][-1] if np.any(phi_10_rbs < 1.0) else None
    if congestion_10_idx is not None:
        rho_10 = rho_s[congestion_10_idx]
        plt.axvline(x=rho_10, color='#ff7f0e', linestyle=':', alpha=0.6)
        plt.text(rho_10 - 0.003, 0.4, '← 10 RB Pool Exhausted', rotation=90, color='#ff7f0e', verticalalignment='center')

    # Find the first point where 20 RB pool gets congested
    congestion_20_idx = np.where(phi_20_rbs < 1.0)[0][-1] if np.any(phi_20_rbs < 1.0) else None
    if congestion_20_idx is not None:
        rho_20 = rho_s[congestion_20_idx]
        plt.axvline(x=rho_20, color='#1f77b4', linestyle=':', alpha=0.6)
        plt.text(rho_20 - 0.003, 0.4, '← 20 RB Pool Exhausted', rotation=90, color='#1f77b4', verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    calculate_theoretical_availability()
