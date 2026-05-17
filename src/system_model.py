import numpy as np

def calculate_3D_distance(pos_j, pos_n):
    """Equation (1): Euclidean 3D distance between RU j and UE n."""
    # pos_j: [x, y, h] of RU, pos_n: [x, y, h] of UE
    return np.linalg.norm(pos_j - pos_n)


def calculate_path_loss(d_jn, fc_ghz):
    """Equation (2): Path loss in dB."""
    # d_jn: distance in meters, fc_ghz: carrier frequency in GHz
    return 28 + 22 * np.log10(d_jn) + 20 * np.log10(fc_ghz) 

def calculate_free_space_path_loss(d_jn, fc_ghz):
    """Equation (3): Free-space path loss in dB."""
    # d_jn: distance in meters, fc_ghz: carrier frequency in GHz
    return 32.45 + 20 * np.log10(d_jn) + 20 * np.log10(fc_ghz) 

def calculate_channel_coefficient(antenna_gain, path_loss_db, rician_factor_k):
    """Equation (4): Channel coefficient h_jn."""
    path_loss_linear = 10**(path_loss_db / 10)
    # Omega captures Rician fading [cite: 95]
    # Simplified representation; full implementation requires Rician distribution sampling
    omega = np.sqrt(rician_factor_k / (rician_factor_k + 1)) + \
            np.sqrt(1 / (rician_factor_k + 1)) * (np.random.randn() + 1j*np.random.randn())/np.sqrt(2)
    
    return np.sqrt(antenna_gain / path_loss_linear) * omega

def calculate_sinr(p_watts, h_sq, interference_watts, noise_density, bandwidth):
    """Equation (5): SINR gamma_jn."""
    noise_power = noise_density * bandwidth
    return (p_watts * h_sq) / (interference_watts + noise_power)

from scipy.special import erfc

def calculate_rate(bandwidth_rb, sinr):
    """Equation (6): Achievable rate R_jn."""
    return bandwidth_rb * np.log2(1 + sinr)

def calculate_error_prob(sinr, M=16):
    """Equation (7): 16-QAM error probability epsilon_jn."""
    # Q(x) = 0.5 * erfc(x / sqrt(2))
    term = np.sqrt((3 * sinr * np.log2(M)) / (M - 1))
    q_func = 0.5 * erfc(term / np.sqrt(2))
    return (4 / np.log2(M)) * q_func