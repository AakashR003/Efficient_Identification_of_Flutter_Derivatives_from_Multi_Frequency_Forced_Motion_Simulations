import numpy as np
import os
import matplotlib.pyplot as plt

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
        else:
            i += 1
    
    # Return the first (minimum) critical velocity
    if critical_velocities:
        min_crit_vel = min(critical_velocities, key=lambda x: x["Critical_Velocity"])
        return min_crit_vel
    else:
        return None

def make_plot(x, y, xlabel, ylabel, title, save_path, bridge_labels=None):
    """
    bridge_labels: list of str or None
    If provided, plot multiple bridges with different markers
    """
    plt.figure(figsize=(10, 6))
    
    if bridge_labels is None:
        # Single bridge plot
        plt.plot(x, y, 'o-', linewidth=2)
    else:
        # Multiple bridges plot
        markers = ['o', 's', '^', 'd', 'v']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (bridge_name, y_vals) in enumerate(y.items()):
            plt.plot(x, y_vals, marker=markers[i], linestyle='-', 
                    linewidth=2, label=bridge_name, color=colors[i], markersize=8)
        plt.legend(loc='best', fontsize=10)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def write_critical_velocity_table(motion_path, motion, combinations, crit_vels_dict):
    """
    Writes a tabular text file with all bridges combined:
    Combination | Bridge1 | Bridge2 | ... | Bridge5
    """
    output_file = os.path.join(motion_path, f"{motion}_Critical_Velocity_Summary.txt")
    bridge_names = list(crit_vels_dict.keys())
    
    with open(output_file, "w") as f:
        # Write header
        f.write("=" * 100 + "\n")
        f.write(f"Critical Velocity Analysis Summary - {motion}\n")
        f.write("=" * 100 + "\n\n")
        
        # Write table header
        header = f"{'Combination':<15}"
        for bridge in bridge_names:
            header += f"{bridge:<20}"
        f.write(header + "\n")
        f.write("=" * 100 + "\n")
        
        # Write data rows
        for i, comb in enumerate(combinations):
            row = f"{comb:<15}"
            for bridge in bridge_names:
                if i < len(crit_vels_dict[bridge]):
                    row += f"{crit_vels_dict[bridge][i]:<20.6f}"
                else:
                    row += f"{'N/A':<20}"
            f.write(row + "\n")
        
        # Write variance section for each bridge
        f.write("\n" + "=" * 100 + "\n")
        f.write("VARIANCE ANALYSIS (Variance = Value - Reference[Combination 1])\n")
        f.write("=" * 100 + "\n\n")
        
        for bridge in bridge_names:
            f.write(f"\n{bridge}:\n")
            f.write(f"{'Combination':<15} {'Critical Velocity':<25} {'Variance':<20}\n")
            f.write(f"{'':=<15} {'':=<25} {'':=<20}\n")
            
            reference_value = crit_vels_dict[bridge][0]
            for i, (comb, crit_vel) in enumerate(zip(combinations, crit_vels_dict[bridge])):
                variance = crit_vel - reference_value
                f.write(f"{comb:<15} {crit_vel:<25.6f} {variance:<20.6f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
    
    print(f"Critical velocity table saved: {output_file}")

# Bridge names
BRIDGE_NAMES = ["Diana_2020", "Great_Belt", "Great_Belt_v2", "Humen", "Tacoma_Narrows"]

# Main script
base_path = os.getcwd()
motions = ["Heave", "Pitch"]
n_cases = range(0, 7)

# Initialize results for all bridges
critical_velocity_results = {motion: {bridge: [] for bridge in BRIDGE_NAMES} for motion in motions}

# Extract critical velocity data for all bridges
for bridge_name in BRIDGE_NAMES:
    print(f"\nProcessing data for {bridge_name}...")
    
    for n in n_cases:
        crit_vel_path = os.path.join(base_path, f"{n}_ConstantAmplitude", 
                                     "Critical_Velocity_Results", bridge_name)
        
        for motion in motions:
            # Critical Velocity
            crit_vel_file = os.path.join(crit_vel_path, "Critical_Velocity_Mode.txt")
            if os.path.exists(crit_vel_file):
                crit_vel_data = read_critical_velocity_file(crit_vel_file)
                if crit_vel_data:
                    critical_velocity_results[motion][bridge_name].append(crit_vel_data)

# Create combined critical velocity plots
output_critical = os.path.join(base_path, "Combined_Results", "Critical_Velocity")

for motion in motions:
    motion_path = os.path.join(output_critical, motion)
    os.makedirs(motion_path, exist_ok=True)
    
    combinations = [0, 1, 2, 3, 4, 5, 6]  # Hard Coded
    
    # Collect data for all bridges
    crit_vels_all = {}
    red_vels_all = {}
    freqs_all = {}
    damps_all = {}
    
    for bridge_name in BRIDGE_NAMES:
        if critical_velocity_results[motion][bridge_name]:
            crit_vels_all[bridge_name] = np.array([d["Critical_Velocity"] for d in critical_velocity_results[motion][bridge_name]])
            red_vels_all[bridge_name] = np.array([d["Reduced_Velocity"] for d in critical_velocity_results[motion][bridge_name]])
            freqs_all[bridge_name] = np.array([d["Frequency"] for d in critical_velocity_results[motion][bridge_name]])
            damps_all[bridge_name] = np.array([d["Damping"] for d in critical_velocity_results[motion][bridge_name]])
    
    if crit_vels_all:
        # Generate combined tabular summary file
        write_critical_velocity_table(motion_path, motion, combinations, crit_vels_all)
        
        # Create combined plots
        make_plot(
            combinations, crit_vels_all,
            f"Number of Combinations", "Critical Velocity [m/s]",
            f"Critical Velocity vs Number of Combinations - All Bridges",
            os.path.join(motion_path, "CriticalVelocity_vs_Number_of_Combinations.png"),
            bridge_labels=BRIDGE_NAMES
        )
        
        make_plot(
            combinations, red_vels_all,
            f"Number of Combinations", "Reduced Velocity",
            f"Reduced Velocity Vs Number of Combinations - All Bridges",
            os.path.join(motion_path, "ReducedVelocity_Vs_Number_of_Combinations.png"),
            bridge_labels=BRIDGE_NAMES
        )
        
        make_plot(
            combinations, freqs_all,
            f"Number of Combinations", "Frequency [rad/s]",
            f"Frequency Vs Number of Combinations - All Bridges",
            os.path.join(motion_path, "Frequency_Vs_Number_of_Combinations.png"),
            bridge_labels=BRIDGE_NAMES
        )
        
        make_plot(
            combinations, damps_all,
            f"Number of Combinations", "Damping [rad/s]",
            f"Damping Vs Number of Combinations - All Bridges",
            os.path.join(motion_path, "Damping_Vs_Number_of_Combinations.png"),
            bridge_labels=BRIDGE_NAMES
        )

print("\n" + "="*60)
print("Critical velocity plots created for all bridges in Combined_Results/Critical_Velocity/")
print("Critical velocity summary tables created with all bridges combined.")
print("="*60)