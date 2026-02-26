import numpy as np

def compute_error_percent(data, ref):
    """
    Percentage error w.r.t reference:
    (data - ref) / ref * 100
    """

    error = np.zeros_like(data)

    error[:, 0] = data[:, 0]  # U_reduced
    error[:, 1] = data[:, 1]  # Amplitude

    for col in range(2, data.shape[1]):
        error[:, col] = (data[:, col] - ref[:, col]) / ref[:, col] * 100.0

    return error


def read_full_amp_phase_table(filename):
    """
    Returns numpy array with columns:
    [U, Amp, Fy_amp, Phase_Fy, M_amp, Phase_Mx]
    """
    rows = []

    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith(("Amplitude", "=", "U_reduced", "-")):
                continue
            rows.append([float(x) for x in line.split()])

    return np.array(rows)

def compute_error(data, ref):
    """
    data, ref: numpy arrays with same shape
    """
    error = np.zeros_like(data)

    error[:, 0] = data[:, 0]  # U_reduced (copied)
    error[:, 1] = data[:, 1]  # Amplitude (copied)

    # Fy amplitude → relative error
    error[:, 2] = (data[:, 2] - ref[:, 2]) / ref[:, 2]

    # Phase Fy → absolute error
    error[:, 3] = data[:, 3] - ref[:, 3]

    # Moment amplitude → relative error
    error[:, 4] = (data[:, 4] - ref[:, 4]) / ref[:, 4]

    # Phase Mx → absolute error
    error[:, 5] = data[:, 5] - ref[:, 5]

    return error

def write_error_file(filename, error_data):
    with open(filename, "w") as f:
        f.write("Amplitude and Phase Error (%)\n")
        f.write("Reference: 1_ConstantAmplitude\n")
        f.write("============================================================\n\n")

        f.write(
            f"{'U_reduced':<15}"
            f"{'Amplitude':>18}"
            f"{'Err Fy Amp (%)':>20}"
            f"{'Err Phase Fy (%)':>22}"
            f"{'Err M Amp (%)':>20}"
            f"{'Err Phase Mx (%)':>22}\n"
        )
        f.write("-" * 120 + "\n")

        for row in error_data:
            f.write(
                f"{row[0]:<15.8f}"
                f"{row[1]:>18.6e}"
                f"{row[2]:>20.6e}"
                f"{row[3]:>22.6e}"
                f"{row[4]:>20.6e}"
                f"{row[5]:>22.6e}\n"
            )


import os
import numpy as np

motions = ["Heave", "Pitch"]
reference_n = 1
max_n = 4

base = os.getcwd()

for motion in motions:

    ref_file = os.path.join(
        base,
        f"{reference_n}_ConstantAmplitude",
        "Amplitude_and_Phase",
        f"{reference_n}_{motion}_Amplitude_Phase.txt"
    )

    ref_data = read_full_amp_phase_table(ref_file)

    for n in range(2, max_n + 1):

        file_n = os.path.join(
            base,
            f"{n}_ConstantAmplitude",
            "Amplitude_and_Phase",
            f"{n}_{motion}_Amplitude_Phase.txt"
        )

        data_n = read_full_amp_phase_table(file_n)

        error_percent = compute_error_percent(data_n, ref_data)

        out_file = os.path.join(
            base,
            f"{n}_ConstantAmplitude",
            "Amplitude_and_Phase",
            f"{n}_{motion}_Amplitude_Phase_ErrorPercent_wrt_1.txt"
        )

        write_error_file(out_file, error_percent)

        print(f"Created: {out_file}")
