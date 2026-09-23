import numpy as np
import math
import scipy.stats as stats
import scipy.special as sp
from scipy.special import erfc

# Antenna gains in dBi
GAIN_HAP_DBI = 32.0
GAIN_LEO_DBI = 38.0

# Rician K-factors in dB for different links
K_UMA_DB_MEAN = 9.0   # Average K-factor for Urban Macro (UMA) terrestrial links as per 3GPP TR 38.901
K_UMA_DB_SD = 3.5
K_HAP_STATIC = 15.0   # Static K-factor for HAP links in S-band
K_LEO_STATIC = 15.0   # Static K-factor for LEO satellite links in S-band

def distance_3D(pos_j, pos_n):
    """Euclidean 3D distance between RU j and UE n."""
    return np.linalg.norm(np.array(pos_j) - np.array(pos_n))

def path_loss(d_jn, fc_ghz):
    """Path loss in dB between Terrestrial RU j and UE n."""
    return 28.0 + 22.0 * np.log10(max(d_jn, 1.0)) + 20.0 * np.log10(fc_ghz)

def free_space_path_loss(d_jn, fc_ghz):
    """Free space path loss in dB between NTN RU j and UE n."""
    return 32.45 + 20.0 * np.log10(max(d_jn, 1.0)) + 20.0 * np.log10(fc_ghz)

# def get_rician_fading_and_pdf(k_db, seed_val=None):
#     # Convert K from dB to linear scale
#     k_lin = 10 ** (k_db / 10.0)
    
#     # Calculate LoS amplitude (rho) and scattered NLoS (sigma)
#     rho = np.sqrt(k_lin / (k_lin + 1))
#     sigma = np.sqrt(1 / (2 * (k_lin + 1)))
    
#     # Draw small scale fading magnitude |w_jn| directly using SciPy
#     if seed_val is not None:
#         rng = np.random.RandomState(seed_val)
#         w_jn_mag = stats.rice.rvs(b= rho / sigma, scale=sigma, random_state=rng)
#     else:
#         w_jn_mag = stats.rice.rvs(b= rho / sigma, scale=sigma)
        
#     # Calculate exact PDF density (Eq 13)
#     z = (w_jn_mag * rho) / (sigma**2)
#     exponential_adjusted = np.exp(-((w_jn_mag - rho)**2) / (2 * sigma**2))
#     bessel_scaled = sp.ive(0, z)
#     pdf = (w_jn_mag / sigma**2) * exponential_adjusted * bessel_scaled
    
#     return w_jn_mag, pdf


def compute_spatial_cholesky(ue_positions: np.ndarray, carrier_freq_ghz: float) -> np.ndarray:
    """Computes the Eigenvalue Decomposition matrix 'L' ONCE for the user spatial distribution."""
    lambda_c = 3e8 / (carrier_freq_ghz * 1e9)
    diff = ue_positions[:, np.newaxis, :] - ue_positions[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    
    corr_matrix = sp.j0((2.0 * np.pi * dist_matrix) / lambda_c)
    vals, vecs = np.linalg.eigh(corr_matrix)
    vals[vals < 0] = 0.0  
    
    return vecs @ np.diag(np.sqrt(vals))

def apply_rician_fading(L: np.ndarray, k_factor_db: float, seed: int = None) -> np.ndarray:
    """Applies random variables to the pre-computed spatial matrix 'L'."""
    rng = np.random.default_rng(seed)
    n_users = L.shape[0]
    
    k_lin = 10.0 ** (k_factor_db / 10.0)
    rho = math.sqrt(k_lin / (k_lin + 1.0))      
    sigma = math.sqrt(1.0 / (2.0 * (k_lin + 1.0))) 
    
    z_real = rng.standard_normal(n_users)
    z_imag = rng.standard_normal(n_users)
    
    x_corr = L @ z_real
    y_corr = L @ z_imag
    
    w_real = rho + (sigma * x_corr)
    w_imag = sigma * y_corr
    
    return w_real**2 + w_imag**2

def gbs_3d_antenna_gain_db(ue_pos, bs_pos, h_gbs=25.0, h_ue=1.5, n_elements=8, tilt_deg=93.0, g_e_max_dbi=8.0, active_sector_deg=None):
    """Computes 3D antenna gain using standard math module for massive loop speedup."""
    dx = ue_pos[0] - bs_pos[0]
    dy = ue_pos[1] - bs_pos[1]
    
    d_2d_safe = max(math.sqrt(dx**2 + dy**2), 1e-6)
    delta_h = h_gbs - h_ue
    
    theta_deg = 90.0 + math.degrees(math.atan(delta_h / d_2d_safe))
    phi_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    
    a_v = -min(12.0 * ((theta_deg - tilt_deg) / 65.0)**2, 30.0)
    
    theta_rad = math.radians(theta_deg)
    tilt_rad = math.radians(tilt_deg)
    cos_diff = math.cos(theta_rad) - math.cos(tilt_rad)
    
    if abs(cos_diff) < 1e-9:
        g_a = float(n_elements)
    else:
        num = math.sin(n_elements * math.pi * cos_diff / 2.0)**2
        den = n_elements * math.sin(math.pi * cos_diff / 2.0)**2
        g_a = num / max(den, 1e-12)
        
    sectors = (30.0, 150.0, 270.0) if active_sector_deg is None else (active_sector_deg,)
    best_gain_linear = 0.0
    
    for sector_azi in sectors:
        phi_offset = (phi_deg - sector_azi + 180.0) % 360.0 - 180.0
        a_h = -min(12.0 * (phi_offset / 65.0)**2, 30.0)
        a_3d = -min(-(a_v + a_h), 30.0)
        
        g_e_linear = 10.0 ** ((g_e_max_dbi + a_3d) / 10.0)
        g_total_linear = g_e_linear * g_a
        
        if g_total_linear > best_gain_linear:
            best_gain_linear = g_total_linear
            
    return 10.0 * math.log10(max(best_gain_linear, 1e-12))

def channel_coefficient(antenna_gain_db, path_loss_db, w_sq):
    """
    Computes instantaneous channel magnitude squared (|h_jn|^2) 
    using deterministic geometry multiplied by the spatially correlated Rician power.
    """
    g_jn_linear = 10.0 ** (antenna_gain_db / 10.0)
    path_loss_linear = 10.0 ** (path_loss_db / 10.0)
    h_sq = (g_jn_linear / path_loss_linear) * w_sq
    return h_sq

def sinr(p_jn, h_sq, interference_power, noise_density, bandwidth):
    """Computes SINR at UE n from RU j."""
    noise_power = noise_density * bandwidth
    signal_power = p_jn * h_sq
    return signal_power / (interference_power + noise_power)

def rate(bandwidth, sinr):
    """Achievable rate from RU j to UE n."""
    return bandwidth * np.log2(1 + sinr)

def error_probability(sinr, M=16):
    """Error probability for M-QAM modulation."""
    x = np.sqrt((3 * sinr * np.log2(M)) / (M - 1))
    q_function = 0.5 * erfc(x / np.sqrt(2))
    return (4 / np.log2(M)) * q_function

def check_transmission_success(capacity_mbps, distance_m, packet_size_bytes=64, max_latency_ms=30.0, error_probability=1e-5):
    """Calculates E2E delay and checks against URLLC latency threshold."""
    packet_size_bits = packet_size_bytes * 8
    capacity_bps = capacity_mbps * 1e6
    speed_of_light = 3e8 
    
    d_trans_ms = (packet_size_bits / capacity_bps) * 1000.0
    d_prop_ms = (distance_m / speed_of_light) * 1000.0
    d_queue_ms = 0.3 
    d_backhaul_ms = 1.0
    
    eps = max(1e-12, min(0.99, error_probability))
    arq_scaling_factor = 1.0 / (1.0 - eps)
    
    d_trans_scaled_ms = d_trans_ms * arq_scaling_factor
    d_prop_scaled_ms = d_prop_ms * arq_scaling_factor
        
    d_total_ms = d_trans_scaled_ms + d_prop_scaled_ms + d_queue_ms + d_backhaul_ms
    is_successful = d_total_ms <= max_latency_ms
    
    return is_successful, d_total_ms