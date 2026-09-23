# Simulation parameters

# Backhaul error probability 
BACKHAUL_ERROR_PROB = 1e-6

# Reliaility threshold
RELIABILITY_THRESHOLD = 0.99999

# Latency threshold 
LATENCY_THRESHOLD = 0.030

# Noise spectral density in dBm
NOISE_SPECTRAL_DENSITY_DBM = -174

# Carrier frequency in GHz
CARRIER_FREQ_GHZ = 2

# System bandwidth (10 MHz) chosen from Table I
SYS_BANDWIDTH_HZ = 10e6

# Single resource block (RB) bandwidth r_jn = 180 kHz (Considering LTE numerology)
BANDWIDTH_RB = 180e3 

# Transmit power in dBm
TX_POWER_GBS_HAP = 46   # Ground base station / HAP Tx power
TX_POWER_LEO = 50       # LEO satellite Tx power

# Convert dBm to linear scale
TX_POWER_GBS_W = 10 ** ((TX_POWER_GBS_HAP - 30) / 10)
TX_POWER_HAP_W = 10 ** ((TX_POWER_GBS_HAP - 30) / 10)
TX_POWER_LEO_W = 10 ** ((TX_POWER_LEO - 30) / 10)
NOISE_SPECTRAL_DENSITY_W = 10 ** ((NOISE_SPECTRAL_DENSITY_DBM - 30) / 10)

# Total RBs per RU, considering 10MHz system bandwidth with Sub-carrier spacing of 15kHz with 12 subcarriers (180kHz)
TOTAL_RBS_GBS = 50
PRIMARY_RBS_GBS = 40

# Transmit power per RB in watts
TX_POWER_GBS_RB_W = TX_POWER_GBS_W / TOTAL_RBS_GBS
TX_POWER_HAP_RB_W = TX_POWER_HAP_W / TOTAL_RBS_GBS
TX_POWER_LEO_RB_W = TX_POWER_LEO_W / TOTAL_RBS_GBS

# Modulation order (16-QAM)
MODULATION_M = 16

# Network toplogy parameters
NUM_GBS = 7           # Number of ground base stations
CELL_RADIUS = 700     # Calculated to cover entire 10 sq.km ground area
AREA_RANGE = 1550    
NUM_UE = 100
MAX_ATTACH_DIST = 1200.0

ALTITUDE_HAP = 20000    # 20 Km
ALTITUDE_LEO = 110000   # 110 Km

# Frequency reuse factor across 7 GBSs
FREQUENCY_REUSE = 1    # Set to 1 for FR-1 (Full interference), 3 for FR-3 (Clustered)

FR3_CLUSTER_MAP = {
    'GBS_0': 'F_A',  
    'GBS_1': 'F_B',
    'GBS_2': 'F_C',
    'GBS_3': 'F_B',
    'GBS_4': 'F_C',
    'GBS_5': 'F_B',
    'GBS_6': 'F_C',
}