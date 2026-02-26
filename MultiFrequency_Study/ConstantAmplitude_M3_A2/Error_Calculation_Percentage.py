import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

def compute_error_percent(data, ref):
    """
    Percentage error w.r.t reference with linearization:
    - Linearize Fy_amp by dividing by Heave/Pitch amplitude
    - Linearize M_amp by dividing by Heave/Pitch amplitude
    - Calculate percentage error for amplitudes: (linearized_data - linearized_ref) / linearized_ref * 100
    - Phase errors are absolute differences in radians (NOT percentage)
    """
    error = np.zeros_like(data)
    error[:, 0] = data[:, 0]  # U_reduced
    error[:, 1] = data[:, 1]  # Amplitude (Heave or Pitch)
    
    # Linearized Fy amplitude error (%)
    fy_linearized_data = data[:, 2] / data[:, 1]
    fy_linearized_ref = ref[:, 2] / ref[:, 1]
    error[:, 2] = (fy_linearized_data - fy_linearized_ref) / fy_linearized_ref * 100.0
    
    # Phase Fy - absolute difference (Radians)
    error[:, 3] = data[:, 3] - ref[:, 3]
    
    # Linearized Moment amplitude error (%)
    m_linearized_data = data[:, 4] / data[:, 1]
    m_linearized_ref = ref[:, 4] / ref[:, 1]
    error[:, 4] = (m_linearized_data - m_linearized_ref) / m_linearized_ref * 100.0
    
    # Phase Mx - absolute difference (Radians)
    error[:, 5] = data[:, 5] - ref[:, 5]
    
    return error

def read_full_amp_phase_table(filename):
    """Returns numpy array: [U, Amp, Fy_amp, Phase_Fy, M_amp, Phase_Mx]"""
    rows = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith(("Amplitude", "=", "U_reduced", "-")):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows)

def read_flutter_derivative_file(filename):
    """Returns numpy array: [Reduced Velocity, Derivative]"""
    rows = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith(("Flutter", "=", "Reduced", "-")):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows)

# ───────────────────────────────────────────────
# Main execution
# ───────────────────────────────────────────────

motions = ["Heave", "Pitch"]
reference_n = 1
max_n = 4
base = os.getcwd()

all_errors = {"Heave": {}, "Pitch": {}}

# Process Amplitude and Phase errors
for motion in motions:
    ref_file = os.path.join(
        base, f"{reference_n}_ConstantAmplitude", "Amplitude_and_Phase",
        f"{reference_n}_{motion}_Amplitude_Phase.txt"
    )
    ref_data = read_full_amp_phase_table(ref_file)
    
    for n in range(2, max_n + 1):
        file_n = os.path.join(
            base, f"{n}_ConstantAmplitude", "Amplitude_and_Phase",
            f"{n}_{motion}_Amplitude_Phase.txt"
        )
        data_n = read_full_amp_phase_table(file_n)
        
        error_percent = compute_error_percent(data_n, ref_data)
        all_errors[motion][n] = error_percent
        
        out_file = os.path.join(
            base, f"{n}_ConstantAmplitude", "Amplitude_and_Phase",
            f"{n}_{motion}_Amplitude_Phase_ErrorPercent_wrt_1.txt"
        )
        # You can add write_error_file(out_file, error_percent) here if still needed
        
        print(f"Processed error for {motion} - Combination {n}")

# Process Flutter Derivatives errors (kept as is - only plotting part shown below)
flutter_derivatives = ["H1", "H2", "H3", "H4", "A1", "A2", "A3", "A4"]
all_flutter_errors = {deriv: {} for deriv in flutter_derivatives}

# ... (your flutter derivative reading & error calculation code remains unchanged)

# ───────────────────────────────────────────────
# Plotting preparation
# ───────────────────────────────────────────────

path_combined_results = os.path.join(base, "Combined_Results")
path_error_plots = os.path.join(path_combined_results, "Error_Plots")
os.makedirs(path_error_plots, exist_ok=True)

error_columns = {
    2: "Err Fy Amp (%)",
    3: "Err Phase Fy",
    4: "Err M Amp (%)",
    5: "Err Phase Mx"
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # enough for n=2,3,4

line_styles = {'Heave': '-', 'Pitch': '--'}
markers = {'Fy': 'o', 'M': 's'}
marker_sizes = {'Fy': 6, 'M': 6}

# ───────────────────────────────────────────────
# INDIVIDUAL PLOTS - Amplitude & Phase
# ───────────────────────────────────────────────

for motion in motions:
    for col_idx, col_label in error_columns.items():
        plt.figure(figsize=(10, 6))
        
        for i, n in enumerate(sorted(all_errors[motion].keys())):
            error_data = all_errors[motion][n]
            U = error_data[:, 0]
            values = error_data[:, col_idx]
            plt.plot(U, values, marker='o', color=colors[i], label=f'Combination {n}')
        
        plt.xlabel('Reduced Velocity', fontsize=12)
        
        # Correct y-label depending on whether it's phase or amplitude
        if "Phase" in col_label:
            plt.ylabel('Phase Error (Radians)', fontsize=12)
        else:
            plt.ylabel('Error (%)', fontsize=12)
            
        plt.title(f'{motion} – {col_label}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        fname = f'{motion}_{col_label.replace(" ", "_").replace("%", "pct").replace("(", "").replace(")", "")}.png'
        plot_filename = os.path.join(path_error_plots, fname)
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"Created: {plot_filename}")

# ───────────────────────────────────────────────
# COMBINED PLOT 1 – Amplitude Errors
# ───────────────────────────────────────────────

plt.figure(figsize=(14, 8))

for motion in motions:
    for i, n in enumerate(sorted(all_errors[motion].keys())):
        d = all_errors[motion][n]
        U = d[:, 0]
        plt.plot(U, d[:, 2], linestyle=line_styles[motion], marker=markers['Fy'],
                 markersize=marker_sizes['Fy'], color=colors[i], alpha=0.9)
        plt.plot(U, d[:, 4], linestyle=line_styles[motion], marker=markers['M'],
                 markersize=marker_sizes['M'], color=colors[i], alpha=0.9)

plt.xlabel('Reduced Velocity', fontsize=12)
plt.ylabel('Error (%)', fontsize=12)
plt.title('Combined Amplitude Errors – Heave & Pitch (Fy & M)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Legend proxies
legend_elements = []
for i, n in enumerate(sorted({n for m in all_errors for n in all_errors[m]})):
    legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, label=f'Combination {n}'))

legend_elements += [
    Line2D([0], [0], color='gray', linestyle='-', lw=2, label='Heave (solid)'),
    Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Pitch (dashed)'),
    Line2D([0], [0], color='gray', marker='o', linestyle='None', markersize=7, label='Fy amplitude'),
    Line2D([0], [0], color='gray', marker='s', linestyle='None', markersize=7, label='M amplitude'),
]

plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=True)
plt.tight_layout()
plot_filename = os.path.join(path_error_plots, 'Combined_Amplitude_Errors.png')
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Created: {plot_filename}")

# ───────────────────────────────────────────────
# COMBINED PLOT 2 – Phase Errors
# ───────────────────────────────────────────────

plt.figure(figsize=(14, 8))

for motion in motions:
    for i, n in enumerate(sorted(all_errors[motion].keys())):
        d = all_errors[motion][n]
        U = d[:, 0]
        plt.plot(U, d[:, 3], linestyle=line_styles[motion], marker=markers['Fy'],
                 markersize=marker_sizes['Fy'], color=colors[i], alpha=0.9)
        plt.plot(U, d[:, 5], linestyle=line_styles[motion], marker=markers['M'],
                 markersize=marker_sizes['M'], color=colors[i], alpha=0.9)

plt.xlabel('Reduced Velocity', fontsize=12)
plt.ylabel('Phase Error (Radians)', fontsize=12)
plt.title('Combined Phase Errors – Heave & Pitch (Fy & Mx)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Reuse same legend style (just change last two labels)
legend_elements_phase = legend_elements[:-2] + [
    Line2D([0], [0], color='gray', marker='o', linestyle='None', markersize=7, label='Fy phase'),
    Line2D([0], [0], color='gray', marker='s', linestyle='None', markersize=7, label='Mx phase'),
]

plt.legend(handles=legend_elements_phase, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=True)
plt.tight_layout()
plot_filename = os.path.join(path_error_plots, 'Combined_Phase_Errors.png')
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Created: {plot_filename}")

print("\nAll plots saved in:", path_error_plots)