import pandas as pd
import matplotlib.pyplot as plt

def generate_report():
    try:
        df = pd.read_csv("simulation_results.csv")
    except FileNotFoundError:
        print("Error: simulation_results.csv not found. Please run main_simulation.py first.")
        return
        
    plt.figure(figsize=(10, 6))
    
    plt.plot(df["Total_UEs"], df["Resilience_10_RB"], marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8, label='10 RBs per Backup Node')
    plt.plot(df["Total_UEs"], df["Resilience_20_RB"], marker='s', linestyle='-', color='#ff7f0e', linewidth=2, markersize=8, label='20 RBs per Backup Node')
    
    plt.title('Average Network Resilience vs. Total Users (7 GBS Topology)', fontsize=14, pad=15)
    plt.xlabel('Total Users in Network', fontsize=12)
    plt.ylabel('Network Resilience (%)', fontsize=12)
    
    plt.xlim(100, 1000)
    plt.ylim(0, 105) 
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig("Final_Resilience_Report.png", dpi=300)
    print("Plot successfully rendered and saved as Final_Resilience_Report.png")
    
    plt.show()

if __name__ == "__main__":
    generate_report()