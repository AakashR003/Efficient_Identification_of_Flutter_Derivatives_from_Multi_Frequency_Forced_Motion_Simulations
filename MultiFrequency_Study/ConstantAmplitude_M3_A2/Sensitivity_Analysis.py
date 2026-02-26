import os
import numpy as np
import matplotlib.pyplot as plt
from Critical_Velocity_Methods import *

# Define all bridge configurations
BRIDGE_CONFIGS = {
    "Diana_2020": {
        "breadth": 31,
        "mass": 22740,
        "heave_damping_ratio": 0.003,
        "heave_frequency": 0.10,
        "inertia": 2.47 * 1e6,
        "pitch_damping_ratio": 0.003,
        "pitch_frequency": 0.278,
        "air_density": 1.22,
    },
    "Great_Belt": {
        "breadth": 18.3,
        "mass": 12800,
        "heave_damping_ratio": 0.003,
        "heave_frequency": 0.1408,
        "inertia": 428400,
        "pitch_damping_ratio": 0.003,
        "pitch_frequency": 0.3602,
        "air_density": 1.2,
    },
    "Great_Belt_v2": {
        "breadth": 18.3,
        "mass": 23687,
        "heave_damping_ratio": 0.003,
        "heave_frequency": 0.097,
        "inertia": 2.501 * 1e6,
        "pitch_damping_ratio": 0.003,
        "pitch_frequency": 0.27,
        "air_density": 1.2,
    },
    "Humen": {
        "breadth": 35.6,
        "mass": 18300,
        "heave_damping_ratio": 0.005,
        "heave_frequency": 0.1117,
        "inertia": 2.09 * 1e6,
        "pitch_damping_ratio": 0.005,
        "pitch_frequency": 0.3612,
        "air_density": 1.2,
    },
    "Tacoma_Narrows": {
        "breadth": 11.9,
        "mass": 4250,
        "heave_damping_ratio": 0.005,
        "heave_frequency": 0.13,
        "inertia": 177730,
        "pitch_damping_ratio": 0.005,
        "pitch_frequency": 0.2,
        "air_density": 1.2,
    },
}


def calculate_structural_properties(config):
    """Calculate structural properties from bridge configuration."""
    mass = config["mass"]
    inertia = config["inertia"]
    heave_frequency = config["heave_frequency"]
    pitch_frequency = config["pitch_frequency"]
    heave_damping_ratio = config["heave_damping_ratio"]
    pitch_damping_ratio = config["pitch_damping_ratio"]
    
    heave_stiffness = (2 * np.pi * heave_frequency) ** 2 * mass
    pitch_stiffness = (2 * np.pi * pitch_frequency) ** 2 * inertia
    heave_damping_coefficient = (
        2 * mass * (2 * np.pi * heave_frequency) * heave_damping_ratio
    )
    pitch_damping_coefficient = (
        2 * inertia * (2 * np.pi * pitch_frequency) * pitch_damping_ratio
    )
    
    return (
        mass,
        inertia,
        heave_damping_coefficient,
        pitch_damping_coefficient,
        heave_stiffness,
        pitch_stiffness,
    )


def _Critical_Velocity(flutter_derivatives_curve, sensitivity_derivative, delta, 
                       structural_properties, air_density, breadth):
    """
    Calculate critical velocity with a small perturbation in a flutter derivative.
    """
    critical_velocity_dict = {}
    omega_previous = [0, 0]  # Mode1, Mode2 

    for U_reduced in range(10, 10000):
        U_reduced = U_reduced / 1000
        flutter_derivative_at_Ureduced = {
            "H1": 0, "H2": 0, "H3": 0, "H4": 0, "A1": 0, "A2": 0, "A3": 0, "A4": 0,
        }
        
        for derivative in flutter_derivatives:
            flutter_derivative_at_Ureduced[derivative] = np.polyval(
                flutter_derivatives_curve[derivative], U_reduced
            )
        
        # Add perturbation to the sensitivity derivative
        flutter_derivative_at_Ureduced[sensitivity_derivative] += delta
        
        aerodynamic_properties = flutter_derivative_at_Ureduced, air_density, breadth
        omegas, eigen_modes = Solve_Quadratic_Eigen_Value(
            structural_properties, aerodynamic_properties
        )

        for i in range(0, len(omegas)):
            if omega_previous[i].imag > 0 and omegas[i].imag < 0:
                critical_velocity = omegas[i].real * breadth / 2 / np.pi * U_reduced
                critical_velocity_dict[critical_velocity] = {
                    "Mode_number": i + 1,
                    "U_reduced": U_reduced,
                    "Frequency": omegas[i].real,
                    "Damping": omegas[i].imag
                }
            omega_previous[i] = omegas[i]
            
    return critical_velocity_dict


def calculate_sensitivity(bridge_name, config, flutter_derivatives_curve, delta=0.00000001):
    """
    Calculate sensitivity of critical velocity to flutter derivatives for a bridge.
    """
    print(f"\nCalculating sensitivity for {bridge_name}...")
    
    # Calculate structural properties
    structural_properties = calculate_structural_properties(config)
    air_density = config["air_density"]
    breadth = config["breadth"]
    
    # Calculate baseline critical velocity
    U_reduced_list = np.arange(0.1, 25, 0.1)
    critical_velocity = min(Critical_Velocity(structural_properties, air_density, breadth, flutter_derivatives_curve, U_reduced_list, Theodorsen=False))
    
    print(f"Baseline Critical Velocity: {critical_velocity:.4f} m/s")
    
    # Calculate sensitivity for each flutter derivative
    sensitivity_list = []
    
    for derivative in flutter_derivatives:
        _critical_velocity = min(_Critical_Velocity(
            flutter_derivatives_curve, derivative, delta,
            structural_properties, air_density, breadth
        ))
        
        sensitivity = (critical_velocity - _critical_velocity) / delta
        sensitivity_list.append(sensitivity)
        
        print(f"{derivative}: {sensitivity:.6e}")
    
    return sensitivity_list, critical_velocity


def plot_sensitivity(sensitivity_data, save_path):
    """
    Create sensitivity plot for all bridges.
    """
    plt.figure(figsize=(12, 7))
    
    markers = ['o', 's', '^', 'd', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    x_positions = np.arange(len(flutter_derivatives))
    width = 0.15
    
    for i, (bridge_name, sensitivity_values) in enumerate(sensitivity_data.items()):
        offset = (i - 2) * width  # Center the bars
        plt.bar(x_positions + offset, sensitivity_values, width, 
                label=bridge_name, color=colors[i], alpha=0.8)
    
    plt.xlabel("Flutter Derivatives", fontsize=12)
    plt.ylabel("Sensitivity", fontsize=12)
    plt.title("Sensitivity of Critical Velocity to Flutter Derivatives - All Bridges", fontsize=13)
    plt.xticks(x_positions, flutter_derivatives)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sensitivity plot saved: {save_path}")


def plot_sensitivity_scatter(sensitivity_data, save_path):
    """
    Create scatter plot for sensitivity analysis.
    """
    plt.figure(figsize=(12, 7))
    
    markers = ['o', 's', '^', 'd', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    x_positions = np.arange(len(flutter_derivatives))
    
    for i, (bridge_name, sensitivity_values) in enumerate(sensitivity_data.items()):
        plt.scatter(x_positions, sensitivity_values, s=100, alpha=0.7,
                   marker=markers[i], color=colors[i], label=bridge_name)
    
    plt.xlabel("Flutter Derivatives", fontsize=12)
    plt.ylabel("Sensitivity", fontsize=12)
    plt.title("Sensitivity Analysis - Scatter Plot - All Bridges", fontsize=13)
    plt.xticks(x_positions, flutter_derivatives)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sensitivity scatter plot saved: {save_path}")


def save_sensitivity_table(sensitivity_data, critical_velocities, save_path):
    """
    Save sensitivity data to a text file.
    """
    with open(save_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("SENSITIVITY ANALYSIS - CRITICAL VELOCITY TO FLUTTER DERIVATIVES\n")
        f.write("=" * 120 + "\n\n")
        
        # Write critical velocities
        f.write("Baseline Critical Velocities (m/s):\n")
        f.write("-" * 60 + "\n")
        for bridge, crit_vel in critical_velocities.items():
            f.write(f"{bridge:<20}: {crit_vel:>15.6f} m/s\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("SENSITIVITY VALUES\n")
        f.write("=" * 120 + "\n\n")
        
        # Write table header
        header = f"{'Derivative':<15}"
        for bridge in sensitivity_data.keys():
            header += f"{bridge:<20}"
        f.write(header + "\n")
        f.write("=" * 120 + "\n")
        
        # Write sensitivity values
        for i, derivative in enumerate(flutter_derivatives):
            row = f"{derivative:<15}"
            for bridge_name in sensitivity_data.keys():
                row += f"{sensitivity_data[bridge_name][i]:<20.6e}"
            f.write(row + "\n")
        
        f.write("\n" + "=" * 120 + "\n")
    
    print(f"Sensitivity table saved: {save_path}")


# Main execution
if __name__ == "__main__":
    
    # Extract flutter derivatives (assuming same for all cases)
    path = os.path.join(project_path, "1" + "_ConstantAmplitude", "Flutter_Derivatives", "1" + "_")
    flutter_derivatives_points, flutter_derivatives_curve = Extract_Flutter_Derivatives(path)
    
    # Create output directory
    output_path = os.path.join(os.getcwd(), "Combined_Results", "Sensitivity")
    os.makedirs(output_path, exist_ok=True)
    
    # Store sensitivity data for all bridges
    sensitivity_data = {}
    critical_velocities = {}
    
    # Calculate sensitivity for each bridge
    for bridge_name, config in BRIDGE_CONFIGS.items():
        sensitivity_list, crit_vel = calculate_sensitivity(
            bridge_name, config, flutter_derivatives_curve
        )
        sensitivity_data[bridge_name] = sensitivity_list
        critical_velocities[bridge_name] = crit_vel
    
    # Create combined plots
    plot_sensitivity(
        sensitivity_data,
        os.path.join(output_path, "Sensitivity_Bar_Plot.png")
    )
    
    plot_sensitivity_scatter(
        sensitivity_data,
        os.path.join(output_path, "Sensitivity_Scatter_Plot.png")
    )
    
    # Save sensitivity table
    save_sensitivity_table(
        sensitivity_data,
        critical_velocities,
        os.path.join(output_path, "Sensitivity_Analysis_Summary.txt")
    )
    
    print("\n" + "=" * 60)
    print("Sensitivity analysis completed for all bridges!")
    print(f"Results saved in: {output_path}")
    print("=" * 60)