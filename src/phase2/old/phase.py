import numpy as np
from scipy.signal import unwrap

def process_complex_csi(complex_csi_data):
    """
    Converts a time-series of complex CSI data into two real-valued arrays: 
    Magnitude and Sanitzed Phase.

    Args:
        complex_csi_data (np.ndarray): A 1D or 2D NumPy array where the 
                                       elements are complex numbers (a + bj). 
                                       (e.g., one subcarrier's data over time).

    Returns:
        tuple: A tuple containing (magnitude_data, sanitized_phase_data).
    """
    
    # 1. Calculate Magnitude (r) and Raw Phase (theta)
    
    # The magnitude preserves signal strength information.
    # np.abs() calculates |z| = sqrt(a^2 + b^2)
    magnitude = np.abs(complex_csi_data)
    
    # The raw phase captures the angle theta using atan2.
    # np.angle() uses atan2(imag(z), real(z)), resolving quadrant ambiguity.
    raw_phase = np.angle(complex_csi_data)
    
    # 2. Phase Unwrapping
    
    # This step removes the 2*pi discontinuities inherent in the atan2 function 
    # (jumps from pi to -pi or vice-versa), ensuring a continuous phase signal.
    unwrapped_phase = unwrap(raw_phase)
    
    # 3. Phase Sanitization (Remove Linear Offset)
    
    # Wi-Fi hardware offsets (CFO, SFO) introduce a large, typically linear, 
    # phase component that masks the small motion-induced phase changes. 
    # We remove this linear component using a simple Least Squares fit.
    
    # Create the x-axis for the linear fit (time steps or subcarrier index)
    N = unwrapped_phase.shape[0]
    time_index = np.arange(N)
    
    # Fit a 1st-degree polynomial (linear fit: y = mx + c)
    # The polynomial is unwrapped_phase ~ p[0]*time_index + p[1]
    p = np.polyfit(time_index, unwrapped_phase, 1)
    
    # Calculate the linear component (the fitted line)
    linear_trend = np.polyval(p, time_index)
    
    # Subtract the linear trend to isolate the motion-induced phase
    sanitized_phase = unwrapped_phase - linear_trend
    
    return magnitude, sanitized_phase

# --- Example Usage ---
# 1. Create a dummy complex CSI signal (e.g., 100 time steps)
time_steps = 100
# Simulate a signal where the phase slowly increases (movement) 
# and then gets wrapped near pi (3.14)
simulated_phase = np.linspace(1, 5, time_steps) + np.sin(np.linspace(0, 10, time_steps))
simulated_magnitude = np.ones(time_steps) * 10 

# Convert magnitude and simulated continuous phase back to a complex number 
# to mimic raw CSI data. This will include the wrapping.
raw_complex_data = simulated_magnitude * np.exp(1j * simulated_phase)

# 2. Run the preprocessing method
magnitude_out, phase_out = process_complex_csi(raw_complex_data)

# 3. Print the results (first 5 samples)
print("--- Complex CSI Preprocessing Results ---")
print(f"Input Complex Data (first 5): {raw_complex_data[:5]}")
print("---")
print(f"Output Magnitude (first 5): {magnitude_out[:5]}")
print(f"Output Sanitized Phase (first 5): {phase_out[:5]}")