import numpy as np
import os
import matplotlib.pyplot as plt

def read_combined_amp_phase_file(filename):
    """
    Reads combined amplitude–phase file and returns:
    amplitude (first row), Fy_amp, phase_fy, M_amp, phase_mx
    """
    data = []

    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith(("Amplitude", "=", "U_reduced", "-")):
                continue
            parts = line.split()
            data.append([float(p) for p in parts])

    data = np.array(data)

    # first row ONLY (as requested)
    first = data[0]

    return {
        "Amplitude": first[1],
        "Fy_amp": first[2],
        "Phase_Fy": first[3],
        "Moment_amp": first[4],
        "Phase_Mx": first[5],
    }


def read_critical_velocity_file(filename):
    """
    Reads Critical_Velocity_Mode.txt and extracts the FIRST (minimum) critical velocity.
    Returns dict with: Critical_Velocity, Reduced_Velocity, Frequency, Damping
    """
    critical_velocities = []
    
    with open(filename, "r") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for lines starting with "Critical Velocity Analysis for Mode"
        if line.startswith("Critical Velocity Analysis for Mode"):
            # Next 4 lines contain the data
            if i + 4 < len(lines):
                try:
                    crit_vel = float(lines[i + 1].split(":")[1].strip().split()[0])
                    red_vel = float(lines[i + 2].split(":")[1].strip().split()[0])
                    freq = float(lines[i + 3].split(":")[1].strip().split()[0])
                    damp = float(lines[i + 4].split(":")[1].strip().split()[0])
                    
                    critical_velocities.append({
                        "Critical_Velocity": crit_vel,
                        "Reduced_Velocity": red_vel,
                        "Frequency": freq,
                        "Damping": damp
                    })
                except (IndexError, ValueError):
                    pass
            i += 5
        else:
            i += 1
    
    # Return the first (minimum) critical velocity
    if critical_velocities:
        min_crit_vel = min(critical_velocities, key=lambda x: x["Critical_Velocity"])
        return min_crit_vel
    else:
        return None


def make_plot(x, y, xlabel, ylabel, title, save_path, phase_window=None):
    """
    phase_window: float or None
        If not None → y-limits set to ±phase_window around first y value
    """
    plt.figure()
    plt.plot(x, y, 'o-', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if phase_window is not None:
        y_ref = y[0]
        plt.ylim(y_ref - phase_window, y_ref + phase_window)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def write_critical_velocity_table(motion_path, motion, combinations, crit_vels):
    """
    Writes a tabular text file with:
    Combination | Critical Velocity | Variance
    Variance is calculated as (value - reference) where reference is first combination
    """
    output_file = os.path.join(motion_path, f"{motion}_Critical_Velocity_Summary.txt")
    
    reference_value = crit_vels[2]
    
    with open(output_file, "w") as f:
        # Write header
        f.write("=" * 70 + "\n")
        f.write(f"Critical Velocity Analysis Summary - {motion}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Reference Value (Combination 1): {reference_value:.6f} m/s\n\n")
        
        # Write table header
        f.write(f"{'Combination':<15} {'Critical Velocity':<25} {'Variance':<20}\n")
        f.write(f"{'':=<15} {'':=<25} {'':=<20}\n")
        
        # Write data rows
        for i, (comb, crit_vel) in enumerate(zip(combinations, crit_vels)):
            variance = crit_vel - reference_value
            f.write(f"{comb:<15} {crit_vel:<25.6f} {variance:<20.6f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Critical velocity table saved: {output_file}")


# Main script
base_path = os.getcwd()
motions = ["Heave", "Pitch"]
n_cases = range(0, 7)

results = {
    "Heave": [],
    "Pitch": []
}

critical_velocity_results = {
    "Heave": [],
    "Pitch": []
}

# Extract amplitude/phase data and critical velocity data
for n in n_cases:
    amp_path = os.path.join(base_path, f"{n}_ConstantAmplitude", "Amplitude_and_Phase")
    crit_vel_path = os.path.join(base_path, f"{n}_ConstantAmplitude", "Critical_Velocity_Results")

    for motion in motions:
        # Amplitude and Phase
        file_path = os.path.join(amp_path, f"{n}_{motion}_Amplitude_Phase.txt")
        vals = read_combined_amp_phase_file(file_path)
        results[motion].append(vals)
        
        # Critical Velocity
        crit_vel_file = os.path.join(crit_vel_path, "Critical_Velocity_Mode.txt")
        crit_vel_data = read_critical_velocity_file(crit_vel_file)
        if crit_vel_data:
            critical_velocity_results[motion].append(crit_vel_data)


# Create plots for amplitude/phase in Nonlinearity_Study
output_nonlinearity = os.path.join(base_path, "Combined_Results", "Nonlinearity_Study")

for motion in motions:
    motion_path = os.path.join(output_nonlinearity, motion)
    os.makedirs(motion_path, exist_ok=True)

    amps = np.array([d["Amplitude"] for d in results[motion]])
    fy_amp = np.array([d["Fy_amp"] for d in results[motion]])
    phase_fy = np.array([d["Phase_Fy"] for d in results[motion]])
    m_amp = np.array([d["Moment_amp"] for d in results[motion]])
    phase_mx = np.array([d["Phase_Mx"] for d in results[motion]])

    make_plot(
        amps, fy_amp,
        f"{motion} Amplitude",
        "Fy Amplitude",
        f"{motion}: Fy vs Amplitude",
        os.path.join(motion_path, "Fy_vs_Amplitude.png")
    )

    make_plot(
        amps, phase_fy,
        f"{motion} Amplitude",
        "Phase Fy [rad]",
        f"{motion}: Phase Fy vs Amplitude",
        os.path.join(motion_path, "PhaseFy_vs_Amplitude.png"),
        phase_window=0.5
    )

    make_plot(
        amps, m_amp,
        f"{motion} Amplitude",
        "Moment Amplitude",
        f"{motion}: Moment vs Amplitude",
        os.path.join(motion_path, "Moment_vs_Amplitude.png")
    )

    make_plot(
        amps, phase_mx,
        f"{motion} Amplitude",
        "Phase Mx [rad]",
        f"{motion}: Phase Mx vs Amplitude",
        os.path.join(motion_path, "PhaseMx_vs_Amplitude.png"),
        phase_window=0.5
    )


# Create critical velocity plots in Critical_Velocity folder
output_critical = os.path.join(base_path, "Combined_Results", "Critical_Velocity")

for motion in motions:
    motion_path = os.path.join(output_critical, motion)
    os.makedirs(motion_path, exist_ok=True)
    
    if critical_velocity_results[motion]:
        amps = np.array([d["Amplitude"] for d in results[motion][:len(critical_velocity_results[motion])]])
        crit_vels = np.array([d["Critical_Velocity"] for d in critical_velocity_results[motion]])
        red_vels = np.array([d["Reduced_Velocity"] for d in critical_velocity_results[motion]])
        freqs = np.array([d["Frequency"] for d in critical_velocity_results[motion]])
        damps = np.array([d["Damping"] for d in critical_velocity_results[motion]])
        
        # Generate tabular summary file
        write_critical_velocity_table(motion_path, motion, amps, crit_vels)

        make_plot(
            amps, crit_vels,
            f"{motion} Amplitude",
            "Critical Velocity [m/s]",
            f" Critical Velocity vs Amplitude",
            os.path.join(motion_path, "CriticalVelocity_vs_Amplitude.png")
        )
        
        make_plot(
            amps, red_vels,
            f"{motion} Amplitude",
            "Reduced Velocity",
            f"{motion}: Reduced Velocity vs Amplitude",
            os.path.join(motion_path, "ReducedVelocity_vs_Amplitude.png")
        )
        
        make_plot(
            amps, freqs,
            f"{motion} Amplitude",
            "Frequency [rad/s]",
            f"{motion}: Frequency vs Amplitude",
            os.path.join(motion_path, "Frequency_vs_Amplitude.png")
        )
        
        make_plot(
            amps, damps,
            f"{motion} Amplitude",
            "Damping [rad/s]",
            f"{motion}: Damping vs Amplitude",
            os.path.join(motion_path, "Damping_vs_Amplitude.png")
        )

print("Nonlinearity plots created for Heave and Pitch.")
print("Critical velocity plots created for Heave and Pitch in Combined_Results/Critical_Velocity/")