import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy.linalg import eig

from custom_path import *


# ------------------------- STRUCTURAL PARAMETERS -------------------------

"""
breadth = 0.2  # m

# Heave
mass = 2.36775  # kg
heave_damping_ratio = 0.0016
heave_frequency = 3.9  # Hz

# Pitch
inertia = 0.0075  # kg*m^2
pitch_damping_ratio = 0.0055
pitch_frequency = 7.7  # Hz

# Aerodynamic parameters
air_density = 1.2  # kg/m^3

# Calcuated Structural Properties
heave_stiffness = (2 * np.pi * heave_frequency) ** 2 * mass
pitch_stiffness = (2 * np.pi * pitch_frequency) ** 2 * inertia
heave_damping_coefficient = ( 2 * mass * (2 * np.pi * heave_frequency) * heave_damping_ratio)
pitch_damping_coefficient = ( 2 * inertia * (2 * np.pi * pitch_frequency) * pitch_damping_ratio )
"""

polynomial_degree = 3
flutter_derivatives = ["H1", "H2", "H3", "H4", "A1", "A2", "A3", "A4"]

# ------------------------- STRUCTURAL MATRICES -------------------------


def Structural_Properties_Matrix(structural_properties):
    
    mass, inertia, heave_damping_coefficient, pitch_damping_coefficient, heave_stiffness, pitch_stiffness = structural_properties
    
    M = np.array([[mass, 0], [0, inertia]])

    C = 1j * np.array([[heave_damping_coefficient, 0], [0, pitch_damping_coefficient]])

    K = np.array([[heave_stiffness, 0], [0, pitch_stiffness]])

    return M, C, K


# ------------------------- AERODYNAMIC MATRICES -------------------------


def Aerodynamic_Properties_Matrix(aerodynamic_properties):

    flutter_derivatives, air_density, breadth = aerodynamic_properties

    H1, H2, H3, H4 = flutter_derivatives["H1"], flutter_derivatives["H2"], flutter_derivatives["H3"],flutter_derivatives["H4"]
    
    A1, A2, A3, A4 = flutter_derivatives["A1"], flutter_derivatives["A2"], flutter_derivatives["A3"], flutter_derivatives["A4"]

    # Aerodynamic damping matrix
    C = 1j * 0.5 * air_density * breadth**2 * np.array([
                [H1, H2 * breadth],
                [A1 * breadth, A2 * breadth**2]])

    # Aerodynamic stiffness matrix
    K = 0.5 * air_density * breadth**2 * np.array([
                [H4, H3 * breadth],
                [A4 * breadth, A3 * breadth**2]])

    return C, K


# ------------------------- TOTAL MATRICES -------------------------


def Total_Properties_Matrix(structural_properties, aerodynamic_properties):

    M_struct, C_struct, K_struct = Structural_Properties_Matrix(structural_properties)
    C_aero, K_aero = Aerodynamic_Properties_Matrix(aerodynamic_properties)

    M = -M_struct - C_aero - K_aero

    C = C_struct

    K = K_struct

    return M, C, K


# ------------------------- SOLVE QUADRATIC EIGEN VALUE -------------------------


def Linearse_Eigen_Value_Problem(structural_properties, aerodynamic_properties):
    M, C, K = Total_Properties_Matrix(structural_properties, aerodynamic_properties)

    zeros = np.zeros_like(M)

    A = np.concatenate(
        [np.concatenate([zeros, K], axis=1), 
         np.concatenate([K, C], axis=1)], axis=0
    )

    B = np.concatenate(
        [np.concatenate([K, zeros], axis=1), 
         np.concatenate([zeros, -M], axis=1)],
        axis=0,
    )

    return A, B


def Solve_Quadratic_Eigen_Value(structural_properties, aerodynamic_properties):
    A, B = Linearse_Eigen_Value_Problem(structural_properties, aerodynamic_properties)
    eigenvalues, eigenvectors = eig(A, B)

    # system_size = M.shape[0]
    # eigenvectors = eigenvectors[:system_size,:]

    omegas = eigenvalues
    mode_shapes = eigenvectors
    omegas = eigenvalues[np.real(eigenvalues) > 0]
    mode_shapes = eigenvectors[:, np.real(eigenvalues) > 0]

    return omegas, mode_shapes


# ------------------------- EXTRACT FLUTTER DERIVATIVES -------------------------


def Extract_Flutter_Derivatives(path):


    flutter_derivatives_points = {"H1": [], "H2": [], "H3": [], "H4": [], "A1": [], "A2": [], "A3": [], "A4": []}
    flutter_derivatives_curve = {"H1": [], "H2": [], "H3": [], "H4": [], "A1": [], "A2": [], "A3": [], "A4": []}

    for derivative in flutter_derivatives:
        with open( path + str(derivative) + ".txt") as input_file:

            data = [[], []]  # [U_red, Derivative]

            for i, line in enumerate(input_file):
                if i < 5:      
                    continue

                line = line.strip()
                if not line:   
                    continue

                line_data = line.split()   # <-- FIX
                data[0].append(float(line_data[0]))
                data[1].append(float(line_data[1]))

            flutter_derivatives_points[derivative] = data


        # Fitting polynomial
        coef = np.polyfit(data[0], data[1], polynomial_degree)
        #coef = np.polyfit(data[0] + empty, data[1] + empty, poly_degree)
        flutter_derivatives_curve[derivative] = coef

    return flutter_derivatives_points, flutter_derivatives_curve


# ------------------------- FIND CRITICAL VELOCITY -------------------------


def UReduced_Vs_Omega(structural_properties, air_density, breadth, flutter_derivatives_curve, U_reduced_list, Theodorsen=False):

    omega_list = []

    flutter_derivative_at_Ureduced = {"H1": 0, "H2": 0, "H3": 0, "H4": 0, "A1": 0, "A2": 0, "A3": 0, "A4": 0}
    for i in range(0, len(U_reduced_list)):
        
        for derivative in flutter_derivatives:
            if Theodorsen == True:
                flutter_derivative_at_Ureduced[derivative] = flutter_derivatives_curve[derivative][i]
            else:
                flutter_derivative_at_Ureduced[derivative] = np.polyval( flutter_derivatives_curve[derivative], U_reduced_list[i])
        
        aerodynamic_properties = (flutter_derivative_at_Ureduced, air_density, breadth)
        omega, eigen_mode = Solve_Quadratic_Eigen_Value(structural_properties, aerodynamic_properties)

        # Order solutions consistently
        if i == 0:
            pass
        else:
            
            max_score = 0
            prev_omegas = np.array(omega_list)[i - 1]  # omegas = omega_list
            for omega_perm, mode_perm in zip(itertools.permutations(omega), itertools.permutations(eigen_mode)):
                
                omega_perm = np.array(omega_perm)
                diff = np.abs(prev_omegas - omega_perm)
                score = 1.0 / np.sqrt(diff).sum()
                if score > max_score:
                    max_score = score
                    omega = omega_perm
                    eigen_mode = np.array(mode_perm)
        omega_list.append(omega)

    return U_reduced_list, omega_list


def Critical_Velocity(structural_properties, air_density, breadth, flutter_derivatives_curve, U_reduced_list, path= None, Theodorsen=False):
    critical_velocity_dict = {}
    omega_previous = [0, 0]  # Mode1, Mode2

    flutter_derivative_at_Ureduced = {"H1": 0, "H2": 0, "H3": 0, "H4": 0, "A1": 0, "A2": 0, "A3": 0, "A4": 0}

    if path is not None:    
        with open( path + "Critical_Velocity_Mode.txt", "w") as output_file:
            output_file.write(f"Critical Velocity Analysis \n\n\n")

    for j in range(0, len(U_reduced_list)):
        
        U_reduced = U_reduced_list[j]
        for derivative in flutter_derivatives:
            if Theodorsen == True:
                flutter_derivative_at_Ureduced[derivative] = flutter_derivatives_curve[derivative][j]
            else:
                flutter_derivative_at_Ureduced[derivative] = np.polyval( flutter_derivatives_curve[derivative], U_reduced)
        
        aerodynamic_properties = (flutter_derivative_at_Ureduced, air_density, breadth)
        omegas, eigen_modes = Solve_Quadratic_Eigen_Value(structural_properties, aerodynamic_properties)

        for i in range(0, len(omegas)):

            #check imaginery value of omega change from possitive to negative
            if omega_previous[i].imag > 0 and omegas[i].imag < 0:

                print(f"\nCritical velocity:")
                print( "Mode =", i + 1, "frequency =", omegas[i].real, "damping =", omegas[i].imag)
                critical_velocity = U_reduced * omegas[i].real / 2 / np.pi * breadth 
                print("Reduced_Velocity =", U_reduced, "Critical_Velocity =", critical_velocity)

                if path is not None:
                    with open( path + "Critical_Velocity_Mode.txt", "a") as output_file:
                        output_file.write(f"Critical Velocity Analysis for Mode {i+1}\n")
                        output_file.write(f"Critical Velocity: {critical_velocity} m/s\n")
                        output_file.write(f"Reduced Velocity: {U_reduced}\n")
                        output_file.write(f"Frequency: {omegas[i].real} rad/s\n")
                        output_file.write(f"Damping: {omegas[i].imag} rad/s\n\n")

                eigen_modes = eigen_modes[:2, :]  # Select first 2 DOFs
                mode1 = eigen_modes[:, 0]
                mode2 = eigen_modes[:, 1]
                mag1 = np.abs(mode1)  # [abs(x1), abs(x2)]
                mag2 = np.abs(mode2)  # [abs(y1), abs(y2)]
                phase1 = np.angle(mode2)[0]  # [phase(x1), phase(x2)]
                phase2 = np.angle(mode2)[1]  # [phase(y1), phase(y2)]

                # For DOF (choose 0 or 1)
                Dof = i 
                phase_diff = phase2 - phase1
                ratio = mag1[Dof] / mag2[Dof]

                ratio = mag2[0] / mag2[1]

                critical_velocity_dict[critical_velocity] = {
                    "Mode_number": i + 1,
                    "U_reduced": U_reduced,
                    "Frequency": omegas[i].real,
                    "Damping": omegas[i].imag,
                    "Phase_difference": phase_diff,
                    "Magnitude_ratio": ratio
                }

            omega_previous[i] = omegas[i]

    return critical_velocity_dict


# ------------------------- PLOTTING -------------------------


def Plot_Flutter_Derivatives(flutter_derivatives_points, flutter_derivatives_curve, path = "_", additional_data=None):
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(4, 2, wspace=0.3, hspace=0.4)

    U_reduced = np.arange(0.1, 25, 0.1)  # adjust range as needed - Dont use if theodorsen derivatives

    color = "tab:blue"

    if additional_data is None:
        additional_data = {}

    for column, letter in enumerate("HA"):
        for row in range(4):
            deriv_ax = fig.add_subplot(gs[row, column])
            key = f"{letter}{row + 1}"

            # Plot polynomial curve
            polynomial_coef = flutter_derivatives_curve[key]
            fitted_polynomial = polynomial_coef

            # Plot points
            # flutter_derivatives_points is not needed for theodorsen derivatives as it a analytical function
            if flutter_derivatives_points is not None:
                data = flutter_derivatives_points[key]
                deriv_ax.scatter(data[0], data[1], marker="+", color=color)
                fitted_polynomial = np.polyval(polynomial_coef, U_reduced)

            deriv_ax.plot(U_reduced, fitted_polynomial, color=color)

            # Additional points for comparision
            if key in additional_data:
                for (xpt, ypt) in additional_data[key]:
                    deriv_ax.scatter(
                        xpt, ypt,
                        color="red",
                        s=20,
                        marker="o",
                        #label="Added point" if "Added point" not in deriv_ax.get_legend_handles_labels()[1] else ""
                    )

            deriv_ax.set_ylabel(f"${letter}_{row + 1}$")
            deriv_ax.set_xlim([min(U_reduced), max(U_reduced)])
            #deriv_ax.legend(loc="lower left", fontsize=8)

        deriv_ax.set_xlabel(r"$U_{red}$")
    fig.savefig(path + "Flutter_Derivatives_Plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def Plot_Critical_Frequency(U_reduced_list, omega_list, path="_"):
    # Find the maximum number of eigenvalues per U
    max_modes = max(len(np.atleast_1d(w)) for w in omega_list)

    # Convert to 2D array with NaNs for missing values
    omega_array = np.full((len(omega_list), max_modes), np.nan, dtype=complex)
    for i, w in enumerate(omega_list):
        w = np.atleast_1d(w)
        omega_array[i, : len(w)] = w

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 4))

    custom_colors = ["red", "green"]

    # Plot each eigenmode branch
    for mode in range(max_modes):
        omega_real = np.real(omega_array[:, mode])
        omega_imag = np.imag(omega_array[:, mode])

        if np.all(np.isnan(omega_real)):
            continue

        color = custom_colors[mode % len(custom_colors)]

        # Real part
        ax1.plot(U_reduced_list, omega_real, color=color, linewidth=1.2, label=f"Mode {mode + 1}")

    ax1.set_ylabel("Oscillating frequency R(ω)")
    # Imaginary part
    ax2 = ax1.twinx()
    ax2.axhline(0, color="black", linewidth=1, linestyle="--")

    for mode in range(max_modes):
        omega_imag = np.imag(omega_array[:, mode])
        if np.all(np.isnan(omega_imag)):
            continue

        color = custom_colors[mode % len(custom_colors)]

        ax2.plot(U_reduced_list, omega_imag, "--", color=color, linewidth=1.2)

    ax2.set_ylabel("Damping Ratio Im(ω)")
    ax1.set_xlabel("Reduced Velocity")
    ax2.tick_params(axis="y", labelcolor="black")

    # Merge legends
    custom_lines = [Line2D([0], [0], color='black', linewidth=1.2),
                Line2D([0], [0], color='black', linewidth=1.2, linestyle='--')]
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1 + custom_lines, labels_1 + ['Real part', 'Imaginary part'], loc="upper right", fontsize=8)

    plt.title("Eigen Frequency vs Reduced Velocity")
    plt.tight_layout()
    fig.savefig(path + "Critical_Velocity.png", dpi=300, bbox_inches="tight")
    plt.show()


def Plot_Damping_ratio(U_reduced_list, omega_list, breadth, path = "_"):
    # Find maximum eigenvalues per U
    max_modes = max(len(np.atleast_1d(w)) for w in omega_list)

    # Convert to 2D array with NaNs for missing values
    omega_array = np.full((len(omega_list), max_modes), np.nan, dtype=complex)
    for i, w in enumerate(omega_list):
        w = np.atleast_1d(w)
        omega_array[i, : len(w)] = w

    custom_colors = ["red", "blue", "green", "purple"]

    fig, axes = plt.subplots(max_modes, 1, figsize=(8, 2.5 * max_modes), sharex=True)

    # If only one mode, axes is not a list → make it a list
    if max_modes == 1:
        axes = [axes]

    for mode in range(max_modes):
        ax = axes[mode]

        omega_real = np.real(omega_array[:, mode])
        omega_imag = np.imag(omega_array[:, mode])

        if np.all(np.isnan(omega_real)) or np.all(np.isnan(omega_imag)):
            continue

        # Compute new X-axis:
        freq_hz = omega_real / (2 * np.pi)
        U_actual = U_reduced_list * freq_hz * breadth

        color = custom_colors[mode % len(custom_colors)]

        ax.plot(U_actual, omega_imag,
                color=color,
                linewidth=1.2,
                label=f"Mode {mode+1} Im(ω)")

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("Damping Ratio Im(ω)")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Wind Velocity")

    fig.suptitle("Damping Ratio vs Wind Velocity", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(path + "Damping_ratio.png", dpi=300, bbox_inches='tight')
    plt.show()


def Plot_Frequency(U_reduced_list, omega_list, breadth, path = "_"):
    # Find maximum eigenvalues per U
    max_modes = max(len(np.atleast_1d(w)) for w in omega_list)

    # Convert to 2D array s
    omega_array = np.full((len(omega_list), max_modes), np.nan, dtype=complex)
    for i, w in enumerate(omega_list):
        w = np.atleast_1d(w)
        omega_array[i, : len(w)] = w

    custom_colors = ["red", "blue", "green", "purple"]

    fig, axes = plt.subplots(max_modes, 1, figsize=(8, 2.5 * max_modes), sharex=True)

    # If only one mode, axes is not a list → make it a list
    if max_modes == 1:
        axes = [axes]

    for mode in range(max_modes):
        ax = axes[mode]

        omega_real = np.real(omega_array[:, mode]) 
        omega_imag = np.imag(omega_array[:, mode])

        if np.all(np.isnan(omega_real)) or np.all(np.isnan(omega_real)):
            continue

        # Compute new X-axis:
        freq_hz = omega_real / (2 * np.pi)
        U_actual = U_reduced_list * freq_hz * breadth

        color = custom_colors[mode % len(custom_colors)]

        ax.plot(U_actual, freq_hz,
                color=color,
                linewidth=1.2,
                label=f"Mode {mode+1} R(ω)")

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("Frequency R(ω)")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Wind Velocity")

    fig.suptitle("Frequency vs Wind Velocity", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(path + "Frequency.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_Eigen_Vector(f1, f2, amp_ratio, phase_diff, path = "_", t_end=2, fs=1000):
    """
    f1, f2: frequencies in Hz
    amp_ratio: amplitude of second wave relative to first (first = 1)
    phase_diff: phase of second wave relative to first in radians
    t_end: total time to plot (seconds)
    fs: sampling frequency
    """
    t = np.linspace(0, t_end, int(t_end*fs))
    y1 = np.sin(2 * np.pi * f1 * t)
    y2_phase = amp_ratio * np.sin(2 * np.pi * f2 * t + phase_diff)
    phase_diff = np.degrees(phase_diff) 
    
    plt.figure(figsize=(8,4))
    plt.plot(t, y1, label="Heave")
    plt.plot(t, y2_phase, label=f"Pitch (phase={phase_diff:.2f} defgrees)")
    plt.title("Displacement")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path + "Eigen_Vector.png", dpi=300, bbox_inches='tight')
    plt.show()