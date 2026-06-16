import numpy as np
import scipy.stats as stats
import scipy.special as sp
from scipy.special import erfc


# Antenna gains in dBi (Assuming UE with omni-directional antenna of 0 dBi)
GAIN_GBS_DBI = 16.84    
GAIN_HAP_DBI = 32.0
GAIN_LEO_DBI = 38.0

# Ricean K-factors in dB for different links
K_UMA_DB_MEAN = 9.0   # Average K-factor for Urban Macro (UMA) terrestrial links as per 3GPP TR 38.901
K_UMA_DB_STD = 3.5
K_HAP_STATIC = 15.0   # Static K-factor for HAP links in S-band (12~15 dB)
K_LEO_STATIC = 15.0   # Static K-factor for LEO satellite links in S-band (12~15 dB)


# Channel Model Functions

def distance_3D(pos_j, pos_n):
    # Eq (1): Euclidean 3D distance between RU j and UE n.
    return np.linalg.norm(np.array(pos_j) - np.array(pos_n))

def path_loss(d_jn, fc_ghz):
    # Eq (2): Path loss in dB between Terrestrial RU j and UE n.
    return 28 + 22 * np.log10(d_jn) + 20 * np.log10(fc_ghz)

def free_space_path_loss(d_jn, fc_ghz):
    # Eq (3): Free space path loss in dB between NTN RU j and UE n.
    return 32.45 + 20 * np.log10(d_jn) + 20 * np.log10(fc_ghz)

def get_rician_fading_and_pdf(k_db):
    # Convert K from dB to linear scale
    k_lin = 10 ** (k_db / 10.0)
    
    # Calculate LoS amplitude (rho) and scattered NLoS (sigma)
    rho = np.sqrt(k_lin / (k_lin + 1))
    sigma = np.sqrt(1 / (2 * (k_lin + 1)))
    
    # Draw small scale fading magnitude |w_jn| directly using SciPy
    w_jn_mag = stats.rice.rvs(rho / sigma, scale=sigma)
    
    # Calculate exact PDF density (Eq 13)
    bessel_term = sp.i0((w_jn_mag * rho) / (sigma**2))
    exponential_term = np.exp(-(w_jn_mag**2 + rho**2) / (2 * sigma**2))
    pdf_density = (w_jn_mag / sigma**2) * exponential_term * bessel_term
    
    return w_jn_mag, pdf_density

def channel_coefficient(antenna_gain_db, path_loss_db, k_factor_db):
    # Eq (4): Channel coefficient between RU j and UE n
    # Convert gains and losses from dB to linear scale inside the function
    g_jn_linear = 10 ** (antenna_gain_db / 10.0)
    path_loss_linear = 10 ** (path_loss_db / 10.0)
    
    # Both Terrestrial and NTN links use the exact mathematical Rician distribution.
    # NTN links will pass a high static K-factor (e.g., 15 dB), 
    # which natively converges the fading magnitude (w_jn) to ~1.0.
    w_jn, _ = get_rician_fading_and_pdf(k_factor_db)
           
    # Return absolute channel magnitude |h_jn|
    return np.sqrt(g_jn_linear / path_loss_linear) * w_jn

def sinr(p_jn, h_sq, interference_power, noise_density, bandwidth):
    # Eq (5): SINR at UE n from RU j
    noise_power = noise_density * bandwidth
    signal_power = p_jn * h_sq
    return signal_power / (interference_power + noise_power)

def rate(bandwidth, sinr):
    # Eq (6): Achievable rate from RU j to UE n
    return bandwidth * np.log2(1 + sinr)

def error_probability(sinr, M=16):
    # Eq (7): Error probability for M-QAM modulation
    x = np.sqrt((3 * sinr * np.log2(M)) / (M - 1))
    q_function = 0.5 * erfc(x / np.sqrt(2))
    return (4 / np.log2(M)) * q_function

def check_transmission_success(capacity_mbps, distance_m, packet_size_bytes=32, max_latency_ms=10.0):
    """ Calculates E2E delay and checks if it meets the URLLC latency threshold. 
    Based on parameters from Salehi et al. Table I.
    """
    # 1. Convert units
    packet_size_bits = packet_size_bytes * 8
    capacity_bps = capacity_mbps * 1e6
    speed_of_light = 3e8 
    
    # 2. Calculate delay components in milliseconds
    d_trans_ms = (packet_size_bits / capacity_bps) * 1000
    d_prop_ms = (distance_m / speed_of_light) * 1000
    
    # From Table I: Queueing delay bound is 0.3 ms
    d_queue_ms = 0.3 
    
    # Total one-way delay
    d_total_ms = d_trans_ms + d_prop_ms + d_queue_ms
    
    # 3. Evaluate Success
    is_successful = d_total_ms <= max_latency_ms
    
    return is_successful, d_total_ms