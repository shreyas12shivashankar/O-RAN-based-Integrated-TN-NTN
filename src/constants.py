# Table I: Simulation Parameters 

# Backhaul error probability (epsilon_b)
BACKHAUL_ERROR_PROB = 1e-6 

# Reliability threshold (1 - epsilon_th)
RELIABILITY_THRESHOLD = 0.99999

# Latency threshold 
LATENCY_THRESHOLD_S = 0.03

# Noise spectral density (N0) in dBm
NOISE_DENSITY_DBM = -174 

# Carrier frequency (fc) in GHz
CARRIER_FREQ_GHZ = 2 

# Bandwidth in Hz (10-15 MHz)
BANDWIDTH_HZ = 10e6 

# Transmit Powers in dBm
TX_POWER_AV = 23
TX_POWER_GBS_HAP = 46
TX_POWER_LEO = 50 

# Topology Parameters
# Altitudes in meters
ALTITUDE_LEO = 110000  # 110 km 
ALTITUDE_HAP = 20000   # 20 km 

# Network Scale
NUM_BS = 7             # Number of ground base stations 
AREA_SQ_KM = 10        # Total network service area  