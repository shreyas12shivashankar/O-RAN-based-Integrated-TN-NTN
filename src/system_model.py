import numpy as np
from scipy.special import erfc

def distance_3D(pos_j, pos_n):
    # Eq(1): Eucledian 3D distance between RU j and UE n.
    return np.linalg.norm(pos_j - pos_n)

def path_loss(d_jn, fc_ghz):
    # Eq (2): Path loss in dB between Terrestrial RU j and UE n.
    # d_jn in meters and fc_ghz in GHz.
    return 28 + 22 * np.log10(d_jn) + 20 * np.log10(fc_ghz)

def free_space_path_loss(d_jn, fc_ghz):
    # Eq (3): Free space path loss in dB between NTN RU j and UE n.
    # d_jn in meters and fc_ghz in GHz.
    return 32.45 + 20 * np.log10(d_jn) + 20 * np.log10(fc_ghz)

def channel_coefficient(antenna_gain, path_loss_db, rician_factor_k_linear=None, is_ntn=False):
    # Eq (4): Channel coefficient between RU j and UE n.
    path_loss_linear = 10 ** (path_loss_db / 10)
    
    if is_ntn:
        # AWGN Channel Assumption for NTN (HAP/LEO)
        # Perfect Line-of-Sight, no multipath scattering
        w_jn = 1.0
    else:
        # Small-scale Rician fading for Terrestrial Networks
        # rician_factor_k_linear is the K-factor in linear scale (not dB)
        w_jn = np.sqrt(rician_factor_k_linear / (rician_factor_k_linear + 1)) + \
               np.sqrt(1 / (rician_factor_k_linear + 1)) * (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
               
    return np.sqrt(antenna_gain / path_loss_linear) * w_jn

def sinr(p_jn, h_sq, interference_power, noise_density, bandwidth):
    # Eq (5): SINR at UE n from RU j.
    noise_power = noise_density * bandwidth
    signal_power = p_jn * h_sq
    return signal_power / (interference_power + noise_power)

def rate(bandwidth, sinr):
    # Eq(6): Achievable rate from RU j to UE n (Shannon Capacity).
    return bandwidth * np.log2(1 + sinr)

def error_probability(sinr, M=16):
    # Eq (7): Error probability for M-QAM modulation scheme.
    # Q function is expressed in terms of erfc (complementary error function).
    x = np.sqrt((3 * sinr * np.log2(M)) / (M - 1))
    q_function = 0.5 * erfc(x / np.sqrt(2))
    return (4 / np.log2(M)) * q_function