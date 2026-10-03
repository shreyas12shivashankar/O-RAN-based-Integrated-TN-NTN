import numpy as np
import matplotlib.pyplot as plt
import src.constants as const

def get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS):
    
    bs_pos = [np.array([0, 0, 0])]  # Center Base station
    
    for i in range(min(6,num_gbs-1)):
        # Angle for neighbor centers: 0, 60, 120, 180, 240, 300 degrees
        angle = np.radians(60 * i) 
        # Distance between one cell to another cell centers
        dist = np.sqrt(3) * radius
        x = dist * np.cos(angle)
        y = dist * np.sin(angle)
        bs_pos.append(np.array([x, y, 0]))
    return np.array(bs_pos)

def draw_hexagon(ax, center, radius):
    # Draws a hexagon on the 3D axis.
    angles = np.linspace(0, 2*np.pi, 7) + np.pi/6   # Rotated by 30 degrees for honeycomb pattern
    x_hex = center[0] + radius * np.cos(angles)
    y_hex = center[1] + radius * np.sin(angles)
    z_hex = np.zeros(7) 
    ax.plot(x_hex, y_hex, z_hex, color='blue', linestyle='-', linewidth=1.5)


def get_random_users(n=const.NUM_UE, bs_coords=None, radius=const.CELL_RADIUS, seed_val=None):
    if seed_val is not None:
        np.random.seed(seed_val)
        
    if bs_coords is None:
        bs_coords = get_hexagonal_bs(radius)
        
    num_cells = len(bs_coords)
    users_per_cell = n // num_cells
    remainder = n % num_cells
    
    ue_pos = []
    
    for i, bs in enumerate(bs_coords):
        cell_users = users_per_cell + (remainder if i == 0 else 0)
        
        # Use square root on the uniform distribution to prevent clustering at the direct center of the tower
        r = radius * np.sqrt(np.random.uniform(0.05, 1.0, cell_users))
        theta = np.random.uniform(0, 2 * np.pi, cell_users)
        
        x = bs[0] + r * np.cos(theta)
        y = bs[1] + r * np.sin(theta)
        z = np.zeros(cell_users)
        
        ue_pos.extend(np.vstack((x, y, z)).T)
        
    return np.array(ue_pos)


def get_ntn_nodes():
    # Returns coordinates for HAP and LEO from constants.
    hap = np.array([0, 0, const.ALTITUDE_HAP]) # Positioned at centre of geographic area
    leo = np.array([0, 0, const.ALTITUDE_LEO])
    return hap, leo

if __name__ == "__main__":
    
    # Generate Data
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS)
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=const.NUM_UE, area_range=const.AREA_RANGE, radius=const.CELL_RADIUS) 

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw Honeycomb shaped hexagons
    for bs in bs_coords:
        draw_hexagon(ax, bs, radius=const.CELL_RADIUS)

    # Plot Nodes
    ax.scatter(ue_coords[:,0], ue_coords[:,1], ue_coords[:,2], c='red', s=10, label='UE')
    ax.scatter(bs_coords[:,0], bs_coords[:,1], bs_coords[:,2], c='blue', marker='^', s=100, label='BS')
    ax.scatter(hap_coord[0], hap_coord[1], hap_coord[2], c='black', marker='^', s=100, label='HAP')
    ax.scatter(leo_coord[0], leo_coord[1], leo_coord[2], c='green', marker='^', s=100, label='LEO')

    ax.set_box_aspect([1, 1, 0.7]) # Z axis is scaled down for better visualization
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('TN-NTN Topology')
    ax.legend()
    plt.show()
    
