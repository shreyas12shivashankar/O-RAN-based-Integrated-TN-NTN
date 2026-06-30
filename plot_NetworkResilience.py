import matplotlib.pyplot as plt

def generate_report(total_users, res_10_rb, res_20_rb):
        
    plt.figure(figsize=(10, 6))
    
    plt.plot(total_users, res_10_rb, marker='o', linestyle='-', color='#1f77b4', label='10 RBs per Backup Node')
    plt.plot(total_users, res_20_rb, marker='s', linestyle='-', color='#ff7f0e', label='20 RBs per Backup Node')
    
    plt.title('Average Network Resilience vs. Total Users (7 GBS Topology)', fontsize=14, pad=15)
    plt.xlabel('Total Users in Network', fontsize=12)
    plt.ylabel('Network Resilience (%)', fontsize=12)
    
    plt.xlim(100, 1000)
    plt.ylim(0, 105) 
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
   
    plt.show()
