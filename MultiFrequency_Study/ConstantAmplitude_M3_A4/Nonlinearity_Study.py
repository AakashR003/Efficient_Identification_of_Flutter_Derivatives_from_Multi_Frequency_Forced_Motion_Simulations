import numpy as np
import os
import matplotlib.pyplot as plt

def read_combined_amp_phase_file(filename):
    """
    Reads combined amplitude–phase file and returns all rows:
    U_reduced, amplitude, Fy_amp, phase_fy, M_amp, phase_mx for each row
    """
    data = []

    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith(("Amplitude", "=", "U_reduced", "-")):
                continue
            parts = line.split()
            data.append([float(p) for p in parts])

    data = np.array(data)
    
    # Return all rows including U_reduced
    results = []
    for row in data:
        results.append({
            "U_reduced": row[0],
            "Amplitude": row[1],
            "Fy_amp": row[2],
            "Phase_Fy": row[3],
            "Moment_amp": row[4],
            "Phase_Mx": row[5],
        })
    
    return results

def make_plot(x, y_data, xlabel, ylabel, title, save_path, phase_window=None, labels=None):
    """
    y_data: list of arrays, one for each row
    phase_window: float or None
        If not None → y-limits set to ±phase_window around first y value of first row
    """
    plt.figure(figsize=(10, 6))
    
    for i, y in enumerate(y_data):
        label = labels[i] if labels else f"Row {i+1}"
        plt.plot(x, y, 'o-', linewidth=2, label=label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Set x-axis to show only integers
    plt.xticks(x)

    if phase_window is not None:
        y_ref = y_data[0][0]  # First value of first row
        plt.ylim(y_ref - phase_window, y_ref + phase_window)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Main script
base_path = os.getcwd()
motions = ["Heave", "Pitch"]
n_cases = range(1, 5)

results = {
    "Heave": [],
    "Pitch": []
}

# Extract amplitude/phase data
for n in n_cases:
    amp_path = os.path.join(base_path, f"{n}_ConstantAmplitude", "Amplitude_and_Phase")

    for motion in motions:
        # Amplitude and Phase - now returns all rows
        file_path = os.path.join(amp_path, f"{n}_{motion}_Amplitude_Phase.txt")
        all_rows = read_combined_amp_phase_file(file_path)
        results[motion].append(all_rows)

# Create plots for amplitude/phase in Nonlinearity_Study
output_nonlinearity = os.path.join(base_path, "Combined_Results", "Nonlinearity_Study")

for motion in motions:
    motion_path = os.path.join(output_nonlinearity, motion)
    os.makedirs(motion_path, exist_ok=True)

    # Combinations (x-axis)
    combinations = np.array([1, 2, 3, 4])
    
    # Prepare data for each of the 8 rows
    n_rows = 8
    fy_normalized_rows = []
    phase_fy_rows = []
    m_normalized_rows = []
    phase_mx_rows = []
    
    # Get U_reduced values from the first combination (they should be the same across all combinations)
    u_reduced_values = [results[motion][0][row_idx]["U_reduced"] for row_idx in range(n_rows)]
    
    for row_idx in range(n_rows):
        # Extract data for this row across all combinations
        amps = np.array([results[motion][case_idx][row_idx]["Amplitude"] for case_idx in range(len(n_cases))])
        fy_amp = np.array([results[motion][case_idx][row_idx]["Fy_amp"] for case_idx in range(len(n_cases))])
        phase_fy = np.array([results[motion][case_idx][row_idx]["Phase_Fy"] for case_idx in range(len(n_cases))])
        m_amp = np.array([results[motion][case_idx][row_idx]["Moment_amp"] for case_idx in range(len(n_cases))])
        phase_mx = np.array([results[motion][case_idx][row_idx]["Phase_Mx"] for case_idx in range(len(n_cases))])
        
        # Normalized force and moment amplitudes
        fy_normalized_rows.append(fy_amp / amps)
        phase_fy_rows.append(phase_fy)
        m_normalized_rows.append(m_amp / amps)
        phase_mx_rows.append(phase_mx)
    
    # Create labels with U_reduced values rounded to 2 decimals
    row_labels = [f"U_r = {u_val:.2f}" for u_val in u_reduced_values]

    make_plot(
        combinations, fy_normalized_rows,
        "Number of Combinations",
        "Normalized Fy Amplitude (Fy/Amplitude)",
        f"{motion}: Normalized Fy vs Number of Combinations",
        os.path.join(motion_path, "NormalizedFy_vs_Combinations.png"),
        #phase_window=0.2,
        labels=row_labels
    )

    make_plot(
        combinations, phase_fy_rows,
        "Number of Combinations",
        "Phase Fy [rad]",
        f"{motion}: Phase Fy vs Number of Combinations",
        os.path.join(motion_path, "PhaseFy_vs_Combinations.png"),
        #phase_window=0.5,
        labels=row_labels
    )

    make_plot(
        combinations, m_normalized_rows,
        "Number of Combinations",
        "Normalized Moment Amplitude (M/Amplitude)",
        f"{motion}: Normalized Moment vs Number of Combinations",
        os.path.join(motion_path, "NormalizedMoment_vs_Combinations.png"),
        #phase_window=0.02,
        labels=row_labels
    )

    make_plot(
        combinations, phase_mx_rows,
        "Number of Combinations",
        "Phase Mx [rad]",
        f"{motion}: Phase Mx vs Number of Combinations",
        os.path.join(motion_path, "PhaseMx_vs_Combinations.png"),
        #phase_window=0.5,
        labels=row_labels
    )

    # ========== NEW: Normalized to First Combination Plots ==========
    # Normalize each row by its first combination value
    fy_normalized_to_first = []
    phase_fy_normalized_to_first = []
    m_normalized_to_first = []
    phase_mx_normalized_to_first = []
    
    for row_idx in range(n_rows):
        # Normalize by dividing by the first value
        fy_normalized_to_first.append(fy_normalized_rows[row_idx] / fy_normalized_rows[row_idx][0])
        phase_fy_normalized_to_first.append(phase_fy_rows[row_idx] / phase_fy_rows[row_idx][0])
        m_normalized_to_first.append(m_normalized_rows[row_idx] / m_normalized_rows[row_idx][0])
        phase_mx_normalized_to_first.append(phase_mx_rows[row_idx] / phase_mx_rows[row_idx][0])
    
    make_plot(
        combinations, fy_normalized_to_first,
        "Number of Combinations",
        "Normalized Fy (relative to Combination 1)",
        f"{motion}: Normalized Fy (First Comb = 1)",
        os.path.join(motion_path, "NormalizedFy_vs_Combinations_RelativeToFirst.png"),
        labels=row_labels
    )

    make_plot(
        combinations, phase_fy_normalized_to_first,
        "Number of Combinations",
        "Phase Fy (relative to Combination 1)",
        f"{motion}: Phase Fy (First Comb = 1)",
        os.path.join(motion_path, "PhaseFy_vs_Combinations_RelativeToFirst.png"),
        labels=row_labels
    )

    make_plot(
        combinations, m_normalized_to_first,
        "Number of Combinations",
        "Normalized Moment (relative to Combination 1)",
        f"{motion}: Normalized Moment (First Comb = 1)",
        os.path.join(motion_path, "NormalizedMoment_vs_Combinations_RelativeToFirst.png"),
        labels=row_labels
    )

    make_plot(
        combinations, phase_mx_normalized_to_first,
        "Number of Combinations",
        "Phase Mx (relative to Combination 1)",
        f"{motion}: Phase Mx (First Comb = 1)",
        os.path.join(motion_path, "PhaseMx_vs_Combinations_RelativeToFirst.png"),
        labels=row_labels
    )

print("Nonlinearity plots created for Heave and Pitch.")
print("Additional normalized-to-first plots also created.")