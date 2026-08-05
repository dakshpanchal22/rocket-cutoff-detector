import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# ------------------------------------------------------------
# 0. Paths and folders
# ------------------------------------------------------------
# We run this script from the project root (rocket_cutoff_detector)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # up two levels: src/python -> src -> root

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')
PLOTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
# ------------------------------------------------------------
# 1. Global parameters
# ------------------------------------------------------------
NORMAL_COUNT   = 2000      # number of normal burn curves
ANOMALY_COUNT  = 500       # total anomalous curves
SEED           = 42        # for reproducibility

# Engine / physics constants
BASE_PRESSURE     = 150.0   # nominal chamber pressure [bar]
THRUST_PER_BAR    = 0.8     # thrust = pressure * 0.8 [kN]
FUEL_FLOW_PER_BAR = 0.05    # fuel flow = pressure * 0.05 [kg/s]
NOISE_LEVEL       = 1.5     # standard deviation of Gaussian noise on pressure

# Startup / shutdown parameters
STARTUP_TIME      = 10      # time steps to go from 0 to full thrust
STEADY_MIN        = 60      # minimum steady-state duration
STEADY_MAX        = 100     # maximum steady-state duration
SHUTDOWN_TIME     = 8       # normal shutdown ramp-down time

# Anomaly probabilities (must sum to 1)
ANOMALY_TYPES = {
    'cutoff':         0.4,   # sudden drop to zero
    'gradual_decay':  0.3,   # linear decrease over several steps
    'oscillation':    0.2,   # high-frequency oscillations
    'stuck_valve':    0.1    # never shuts down (values stay high)
}

np.random.seed(SEED)

# ------------------------------------------------------------
# 2. Normal pressure profile generator
# ------------------------------------------------------------
def generate_normal_pressure():
    """
    Returns a 1D numpy array of chamber pressure for a normal burn:
    - Startup ramp
    - Steady state (random duration)
    - Shutdown ramp
    All with realistic noise + occasional spikes.
    """
    # Random total length (steady duration varies)
    steady_len = np.random.randint(STEADY_MIN, STEADY_MAX + 1)
    total_len = STARTUP_TIME + steady_len + SHUTDOWN_TIME
    
    # Create time axis
    t = np.arange(total_len, dtype=float)
    
    # Ideal (noise-free) pressure profile
    pressure_ideal = np.zeros(total_len, dtype=float)
    
    # Startup: linear ramp from 0 to BASE_PRESSURE
    startup_ramp = np.linspace(0, BASE_PRESSURE, STARTUP_TIME, endpoint=False)
    pressure_ideal[:STARTUP_TIME] = startup_ramp
    
    # Steady state: constant BASE_PRESSURE
    pressure_ideal[STARTUP_TIME:STARTUP_TIME+steady_len] = BASE_PRESSURE
    
    # Shutdown: linear ramp from BASE_PRESSURE to 0
    shutdown_ramp = np.linspace(BASE_PRESSURE, 0, SHUTDOWN_TIME, endpoint=True)
    pressure_ideal[STARTUP_TIME+steady_len:] = shutdown_ramp
    
    # Add Gaussian noise
    noise = np.random.normal(0, NOISE_LEVEL, total_len)
    pressure = pressure_ideal + noise
    
    # Add rare random spikes (1% probability per point)
    spike_mask = np.random.rand(total_len) < 0.01
    spike_amplitude = np.random.normal(0, 8.0, total_len)  # larger spikes
    pressure[spike_mask] += spike_amplitude[spike_mask]
    
    # Add slow sensor drift (random walk)
    drift = np.cumsum(np.random.normal(0, 0.05, total_len))
    pressure += drift
    
    # Ensure no negative pressure (unphysical)
    pressure = np.maximum(pressure, 0.0)
    
    return pressure, total_len

# ------------------------------------------------------------
# 3. Anomalous profile generators
# ------------------------------------------------------------
def anomaly_cutoff(pressure_ideal, total_len):
    """Sudden drop to zero at a random point after startup."""
    # Pick a random cutoff point in the steady region (startup+10 to total-10)
    cutoff_start = np.random.randint(STARTUP_TIME+10, total_len-10)
    pressure_ideal[cutoff_start:] = 0.0
    return pressure_ideal

def anomaly_gradual_decay(pressure_ideal, total_len):
    """Linear decay from a random point to zero over several steps."""
    decay_start = np.random.randint(STARTUP_TIME+10, total_len-20)
    decay_len   = np.random.randint(15, 25)   # length of decay
    decay_end   = min(decay_start + decay_len, total_len)
    # Linear ramp from current value to 0
    start_val = pressure_ideal[decay_start]
    for i in range(decay_start, decay_end):
        frac = (i - decay_start) / max(decay_len-1, 1)
        pressure_ideal[i] = start_val * (1 - frac)
    # Set the rest to zero
    pressure_ideal[decay_end:] = 0.0
    return pressure_ideal

def anomaly_oscillation(pressure_ideal, total_len):
    """High-frequency oscillation added on top of steady state."""
    osc_start = np.random.randint(STARTUP_TIME+10, total_len-30)
    osc_len   = np.random.randint(30, 60)   # oscillate for 30-60 steps
    osc_end   = min(osc_start + osc_len, total_len)
    t_osc = np.arange(osc_end - osc_start)
    # Sinusoidal oscillation with amplitude ~20% of base pressure
    oscillation = 0.2 * BASE_PRESSURE * np.sin(2.0 * np.pi * t_osc / 3.0)
    pressure_ideal[osc_start:osc_end] += oscillation
    # After oscillation, go back to normal (or continue) – we'll keep as is.
    return pressure_ideal

def anomaly_stuck_valve(pressure_ideal, total_len):
    """Valve fails open: pressure stays high and never shuts down."""
    # Remove the shutdown ramp: set the last part to BASE_PRESSURE
    shutdown_start = total_len - SHUTDOWN_TIME
    pressure_ideal[shutdown_start:] = BASE_PRESSURE
    return pressure_ideal

# ------------------------------------------------------------
# 4. Main generation function
# ------------------------------------------------------------
def generate_curve(is_anomaly=False):
    """
    Returns:
        thrust, pressure, fuel_flow : 1D arrays of same length
        anomaly_type : string ('normal' or one of the anomaly keys)
    """
    # Generate base normal pressure profile
    pressure, length = generate_normal_pressure()
    anomaly_type = 'normal'
    
    if is_anomaly:
        # Choose anomaly type according to probabilities
        types = list(ANOMALY_TYPES.keys())
        probs = list(ANOMALY_TYPES.values())
        anomaly_type = np.random.choice(types, p=probs)
        
        # Build the ideal profile for the anomaly function to modify
        # We need a clean ideal (noise-free) shape to apply anomaly logic cleanly.
        # Reconstruct the ideal from the generated pressure? Better: we have the function
        # generate_normal_pressure but we used noise inside. For anomaly logic, we'll
        # create a temporary ideal profile without noise.
        steady_len = length - STARTUP_TIME - SHUTDOWN_TIME
        ideal = np.zeros(length)
        ideal[:STARTUP_TIME] = np.linspace(0, BASE_PRESSURE, STARTUP_TIME, endpoint=False)
        ideal[STARTUP_TIME:STARTUP_TIME+steady_len] = BASE_PRESSURE
        ideal[STARTUP_TIME+steady_len:] = np.linspace(BASE_PRESSURE, 0, SHUTDOWN_TIME, endpoint=True)
        
        # Apply anomaly modification
        if anomaly_type == 'cutoff':
            ideal = anomaly_cutoff(ideal, length)
        elif anomaly_type == 'gradual_decay':
            ideal = anomaly_gradual_decay(ideal, length)
        elif anomaly_type == 'oscillation':
            ideal = anomaly_oscillation(ideal, length)
        elif anomaly_type == 'stuck_valve':
            ideal = anomaly_stuck_valve(ideal, length)
        
        # Add noise, spikes, drift to the modified ideal
        pressure = ideal + np.random.normal(0, NOISE_LEVEL, length)
        # Spikes
        spike_mask = np.random.rand(length) < 0.01
        pressure[spike_mask] += np.random.normal(0, 8.0, length)[spike_mask]
        # Drift
        pressure += np.cumsum(np.random.normal(0, 0.05, length))
        pressure = np.maximum(pressure, 0.0)
    
    # Derive thrust and fuel flow from pressure (with independent small noise)
    thrust = THRUST_PER_BAR * pressure + np.random.normal(0, 0.2, len(pressure))
    fuel_flow = FUEL_FLOW_PER_BAR * pressure + np.random.normal(0, 0.01, len(pressure))
    
    # Clip to physical limits (non-negative)
    thrust = np.maximum(thrust, 0.0)
    fuel_flow = np.maximum(fuel_flow, 0.0)
    
    return thrust, pressure, fuel_flow, anomaly_type

    # ------------------------------------------------------------
# 5. Save a single curve to CSV
# ------------------------------------------------------------
def save_curve_csv(filename, thrust, pressure, fuel_flow):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'thrust', 'pressure', 'fuel_flow'])
        for i in range(len(thrust)):
            writer.writerow([i, thrust[i], pressure[i], fuel_flow[i]])

# ------------------------------------------------------------
# 6. Generate all data
# ------------------------------------------------------------
all_data = []   # will store tuples: (thrust, pressure, fuel_flow, label)

# Normal samples
print(f"Generating {NORMAL_COUNT} normal curves...")
for i in range(NORMAL_COUNT):
    t, p, ff, _ = generate_curve(is_anomaly=False)
    save_curve_csv(f'normal_{i+1:04d}.csv', t, p, ff)
    all_data.append((t, p, ff, 0))   # label 0 = normal

# Anomalous samples
print(f"Generating {ANOMALY_COUNT} anomalous curves...")
for i in range(ANOMALY_COUNT):
    t, p, ff, atype = generate_curve(is_anomaly=True)
    save_curve_csv(f'anomaly_{i+1:04d}.csv', t, p, ff)
    all_data.append((t, p, ff, 1))   # label 1 = anomaly

print("Data generation complete.")

# ------------------------------------------------------------
# 7. Save combined dataset as NPZ (with variable lengths)
# ------------------------------------------------------------
# --- ADD THESE FOUR LINES BELOW ---
thrust_list    = [item[0] for item in all_data]
pressure_list  = [item[1] for item in all_data]
fuel_flow_list = [item[2] for item in all_data]
labels_list    = [item[3] for item in all_data]
# --- END ADDITION ---

# Convert lists of variable-length arrays to object arrays
thrust_arr    = np.array(thrust_list, dtype=object)
pressure_arr  = np.array(pressure_list, dtype=object)
fuel_flow_arr = np.array(fuel_flow_list, dtype=object)
labels_arr    = np.array(labels_list)   # scalars are fine

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'rocket_data.npz'),
    thrust=thrust_arr,
    pressure=pressure_arr,
    fuel_flow=fuel_flow_arr,
    labels=labels_arr
)
print("Saved combined NPZ file.")

# ------------------------------------------------------------
# 8. Plot examples
# ------------------------------------------------------------
num_examples = 6
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# Select random indices from normal and anomalous
normal_indices = np.random.choice(range(NORMAL_COUNT), size=3, replace=False)
anom_indices   = np.random.choice(range(NORMAL_COUNT, NORMAL_COUNT+ANOMALY_COUNT), size=3, replace=False)

for ax, idx in zip(axes[:3], normal_indices):
    t = thrust_list[idx]
    ax.plot(t, label='Thrust', linewidth=0.8)
    ax.set_title(f'Normal (sample {idx})')
    ax.legend()

for ax, idx in zip(axes[3:], anom_indices):
    t = thrust_list[idx]
    ax.plot(t, label='Thrust', linewidth=0.8, color='red')
    ax.set_title(f'Anomaly (sample {idx})')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rocket_data_examples.png'), dpi=150)
plt.show()

