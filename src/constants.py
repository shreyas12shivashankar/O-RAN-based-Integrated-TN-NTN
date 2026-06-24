# Table I : Simulation parameters

# Backhaul error probability 
BACKHAUL_ERROR_PROB = 10e-6

# Reliaility threshold
RELIABILITY_THRESHOLD = 0.99999

# Latency threshold 
LATENCY_THRESHOLD = 0.030


# Noise spectral density in dBm
NOISE_SPECTRAL_DENSITY_DBM = -174

# Carrier frequency in GHz
CARRIER_FREQ_GHZ = 2

# Bandwidth (10-15 MHz)
BANDWIDTH_HZ = 10e6

# Transmit power in dBm
TX_POWER_AV = 23        # Aerial vehicle Tx power
TX_POWER_GBS_HAP = 46   # Ground base station / HAP Tx power
TX_POWER_LEO = 50       # LEO satellite Tx power

# Modulation factor (16-QAM)
MODULATION_M = 16

# Network toplogy parameters
NUM_GBS = 7           # Number of ground base stations
AREA_SQ_KM = 100        # Total network service area in square kilometers

CELL_RADIUS = 2000   # Calcualted to cover entire 100 sq.km
AREA_RANGE = 5000    
NUM_UE = 100       

ALTITUDE_HAP = 20000    # 20 Km
ALTITUDE_LEO = 110000   # 110 Km

# Convert dBm to linear
NOISE_SPECTRAL_DENSITY_W = 10 ** ((NOISE_SPECTRAL_DENSITY_DBM - 30) / 10)
TX_POWER_GBS_W = 10 ** ((TX_POWER_GBS_HAP - 30) / 10)
TX_POWER_HAP_W = 10 ** ((TX_POWER_GBS_HAP - 30) / 10)
TX_POWER_LEO_W = 10 ** ((TX_POWER_LEO - 30) / 10)
