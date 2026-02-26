
import shutil
from Critical_Velocity_Methods import *
#from theodorsen_derivatives import *

import shutil
import os
import numpy as np
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

U_reduced_list = np.arange(0.1, 25, 0.1)

# Loop through each bridge configuration
for bridge_name, bridge_config in BRIDGE_CONFIGS.items():
    
    print(f"\n{'='*60}")
    print(f"Processing Bridge: {bridge_name}")
    print(f"{'='*60}\n")
    
    # Extract bridge parameters
    breadth = bridge_config["breadth"]
    mass = bridge_config["mass"]
    heave_damping_ratio = bridge_config["heave_damping_ratio"]
    heave_frequency = bridge_config["heave_frequency"]
    inertia = bridge_config["inertia"]
    pitch_damping_ratio = bridge_config["pitch_damping_ratio"]
    pitch_frequency = bridge_config["pitch_frequency"]
    air_density = bridge_config["air_density"]
    
    # Calculated Structural Properties
    heave_stiffness = (2 * np.pi * heave_frequency) ** 2 * mass
    pitch_stiffness = (2 * np.pi * pitch_frequency) ** 2 * inertia
    heave_damping_coefficient = (
        2 * mass * (2 * np.pi * heave_frequency) * heave_damping_ratio)
    pitch_damping_coefficient = (
        2 * inertia * (2 * np.pi * pitch_frequency) * pitch_damping_ratio
    )
    
    structural_properties = mass, inertia, heave_damping_coefficient, pitch_damping_coefficient, heave_stiffness, pitch_stiffness
    
    # Loop through cases
    for i in range(1, 5):
        name = str(i)
        path = os.path.join(project_path, name + "_ConstantAmplitude", "Flutter_Derivatives", name + "_")
        flutter_derivatives_points, flutter_derivatives_curve = Extract_Flutter_Derivatives(path)
        
        # Modified path to include bridge name
        path_storage = os.path.join(project_path, name + "_ConstantAmplitude", "Critical_Velocity_Results", bridge_name, "")
        os.makedirs(path_storage, exist_ok=True)
        
        Plot_Flutter_Derivatives(flutter_derivatives_points, flutter_derivatives_curve, path=path_storage)
        
        critical_velocity = Critical_Velocity(structural_properties, air_density, breadth, flutter_derivatives_curve, U_reduced_list, path=path_storage, Theodorsen=False)
        print(critical_velocity)
        
        U_reduced_list, omega_list = UReduced_Vs_Omega(structural_properties, air_density, breadth, flutter_derivatives_curve, U_reduced_list, Theodorsen=False)
        
        Plot_Critical_Frequency(U_reduced_list, omega_list, path=path_storage)
        Plot_Damping_ratio(U_reduced_list, omega_list, breadth, path=path_storage)
        Plot_Frequency(U_reduced_list, omega_list, breadth, path=path_storage)
        
        f1 = list(critical_velocity.values())[0]["Frequency"]
        f2 = list(critical_velocity.values())[0]["Frequency"]
        amp_ratio = list(critical_velocity.values())[0]["Magnitude_ratio"]
        phase_diff = list(critical_velocity.values())[0]["Phase_difference"]
        
        plot_Eigen_Vector(f1, f2, amp_ratio, phase_diff, path=path_storage)