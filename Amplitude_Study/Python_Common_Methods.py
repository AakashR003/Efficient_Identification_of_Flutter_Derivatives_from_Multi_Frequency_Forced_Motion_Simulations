import os

def read_flutter_derivatives(filename):

    """
    {
        0.2: {
            "U_reduced": 13.66120218579235,
            "derivatives": {}
        }
    }
    """

    data = {}
    current_freq = None
    current_derivatives = {}
    current_U = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Frequency"):
                current_freq = float(line.split(":")[1])
                current_derivatives = {}
                current_U = None

            elif line.startswith("Reduced Velocity"):
                current_U = float(line.split(":")[1])

            elif line.startswith(("H1", "H2", "H3", "H4", "A1", "A2", "A3", "A4")):
                key, val = line.split(":")
                current_derivatives[key.strip()] = float(val)

            elif line == "" and current_freq is not None:
                data[current_freq] = {"U_reduced": current_U, "derivatives": current_derivatives}
                current_freq = None

        if current_freq is not None:
            data[current_freq] = {"U_reduced": current_U, "derivatives": current_derivatives}

    return data


def read_amplitude_phase_data(filename):

    """
    Example Output

    {
        0.2: {
            "U_reduced": 13.66120218579235,
            "values": {
                "Tita Amplitude": 0.017,
                "Fy Amplitude": 0.015045949079503516,
                "Phase Fy": -2.882562370349097,
                "Moment Amplitude": 0.00113767611835333,
                "Phase Mx": 0.16895426470026642
            }
        }
    }

    """
    data = {}
    current_freq = None
    current_vals = {}
    current_U = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Frequency"):
                current_freq = float(line.split(":")[1])
                current_vals = {}
                current_U = None

            elif line.startswith("Reduced Velocity"):
                current_U = float(line.split(":")[1])

            elif any(line.startswith(k) for k in ["Tita Amplitude","Heave Amplitude","Fy Amplitude","Phase Fy","Moment Amplitude","Phase Mx"]):
                key, val = line.split(":")
                current_vals[key.strip()] = float(val)

            elif line == "" and current_freq is not None:
                data[current_freq] = {"U_reduced": current_U, "values": current_vals}
                current_freq = None

        if current_freq is not None:
            data[current_freq] = {"U_reduced": current_U, "values": current_vals}

    return data


def write_amplitude_phase_file(path, filename, data_dict, columns):

    velocities = sorted(data_dict.keys(), reverse=True)

    with open(os.path.join(path, filename), "w") as f:
        f.write("Amplitude and Phase (Combined)\n")
        f.write("==============================================\n\n")

        header = f"{'U_reduced':<15}"
        for col in columns:
            header += f"{col:>22}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for v in velocities:
            line = f"{v:<15}"
            for col in columns:
                val = data_dict[v].get(col, float("nan"))
                line += f"{val:>22.6e}"
            f.write(line + "\n")