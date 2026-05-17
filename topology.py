import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import src.constants as const

def get_hexagonal_bs(radius=1000):
    """Places BS centers such that hexagons touch perfectly."""
    bs_pos = [np.array([0, 0, 0])]  # Center BS
    for i in range(6):
        # Angle for neighbor centers: 0, 60, 120, 180, 240, 300 degrees
        angle = np.radians(60 * i) 
        # Distance between centers in a perfect honeycomb is sqrt(3) * radius
        dist = np.sqrt(3) * radius
        x = dist * np.cos(angle)
        y = dist * np.sin(angle)
        bs_pos.append(np.array([x, y, 0]))
    return np.array(bs_pos)

def draw_hexagon(ax, center, radius):
    """Draws a hexagon with a 30-degree offset to align flat sides."""
    # Vertices at 30, 90, 150... degrees make the hexagons snap together
    angles = np.linspace(0, 2*np.pi, 7) + np.pi/6 
    x_hex = center[0] + radius * np.cos(angles)
    y_hex = center[1] + radius * np.sin(angles)
    z_hex = np.zeros(7) 
    ax.plot(x_hex, y_hex, z_hex, color='blue', linestyle='-', linewidth=1.5, alpha=0.9)

def get_random_users(n=100, area_range=3500):
    """
    Uniformly distributes UEs in the service area.
    area_range=3500 keeps them mostly inside the 7-cell cluster.
    """
    x = np.random.uniform(-area_range, area_range, n)
    y = np.random.uniform(-area_range, area_range, n)
    z = np.zeros(n) # Ground users
    return np.vstack((x, y, z)).T

def get_ntn_nodes():
    """Returns coordinates for HAP and LEO from constants."""
    hap = np.array([0, 0, const.ALTITUDE_HAP])
    leo = np.array([0, 0, const.ALTITUDE_LEO])
    return hap, leo

if __name__ == "__main__":
    # --- Parameters ---
    CELL_RADIUS = 1000 
    
    # --- Generate Data ---
    bs_coords = get_hexagonal_bs(radius=CELL_RADIUS)
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=100)

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Draw Honeycomb
    for bs in bs_coords:
        draw_hexagon(ax, bs, radius=CELL_RADIUS)

    # 2. Plot Nodes
    ax.scatter(ue_coords[:,0], ue_coords[:,1], ue_coords[:,2], c='red', s=8, label='UE', alpha=0.6)
    ax.scatter(bs_coords[:,0], bs_coords[:,1], bs_coords[:,2], c='blue', marker='^', s=80, label='BS')
    ax.scatter(hap_coord[0], hap_coord[1], hap_coord[2], c='black', marker='s', s=100, label='HAP')
    ax.scatter(leo_coord[0], leo_coord[1], leo_coord[2], c='green', marker='D', s=100, label='LEO')

    # Scaling and Labels
    ax.set_box_aspect([1, 1, 0.7]) 
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Perfectly Aligned NTN-ORAN Topology')
    ax.legend()
    plt.show()