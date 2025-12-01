import cmath # Standard library for complex math operations
import math  # Standard library for basic math (not strictly needed here, but good practice)

def get_polar_coordinates(z):
    """
    Calculates the magnitude (r) and phase (theta) of a single complex number.

    Args:
        z (complex): The complex number in the form a + bj.

    Returns:
        tuple: A tuple containing (magnitude, phase_in_radians).
    """
    # 1. Calculate Magnitude (r)
    # cmath.polar(z) returns (r, theta). The r value is the magnitude.
    # Alternatively, you could use abs(z)
    magnitude = abs(z)
    
    # 2. Calculate Phase (theta)
    # cmath.phase(z) is equivalent to math.atan2(imag(z), real(z)).
    # It correctly returns the angle in the range (-pi, pi], resolving quadrant ambiguity.
    phase_in_radians = cmath.phase(z)
    
    # Alternatively, you can decompose the complex number and use math.atan2
    # real_part = z.real
    # imag_part = z.imag
    # phase_in_radians = math.atan2(imag_part, real_part)

    return magnitude, phase_in_radians

# --- Example Usage ---

# Example 1: z1 = 3 + 4j (Quadrant I)
z1 = 3 + 4j
r1, theta1 = get_polar_coordinates(z1)
print(f"z1 = {z1}: Magnitude (r) = {r1:.2f}, Phase (theta) = {theta1:.2f} radians")
print(f"{r1 * theta1}")

# Example 2: z2 = -3 - 4j (Quadrant III) - Same magnitude, different phase
z2 = -3 - 4j
r2, theta2 = get_polar_coordinates(z2)
print(f"z2 = {z2}: Magnitude (r) = {r2:.2f}, Phase (theta) = {theta2:.2f} radians")
print(f"{r2 * theta2}")

# Example 3: z3 = 0 - 5j (Negative Imaginary Axis)
z3 = 0 - 5j
r3, theta3 = get_polar_coordinates(z3)
print(f"z3 = {z3}: Magnitude (r) = {r3:.2f}, Phase (theta) = {theta3:.2f} radians")
print(f"{r3 * theta3}")

import re
def extract_S_C_numbers( filename="E1_S01_C03_A03_T01.csv"):
    """
    Extract subject (Sxx) and class (C03) numbers from filename.
    Example: 'E1_S01_C03_A03_T01.csv' -> (1, 3)
    """
     
    match = re.search(r'S(\d+).*C(\d+).*A(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None
a, b, c =extract_S_C_numbers()
print(a,b, c)

   