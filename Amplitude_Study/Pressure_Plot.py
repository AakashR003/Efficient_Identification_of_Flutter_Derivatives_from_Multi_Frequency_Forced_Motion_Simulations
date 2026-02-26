import numpy as np
import os
import shutil
import re

from MultiFrequency_Data import *
from Python_Common_Methods import read_flutter_derivatives

def read_amplitudes_phases(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    data = {}
    # Find frequency and reduced velocity
    freq_match = re.search(r'Frequency : ([\d\.]+)', text)
    vel_match = re.search(r'Reduced Velocity: ([\d\.]+)', text)
    if freq_match and vel_match:
        freq = float(freq_match.group(1))
        U_reduced = float(vel_match.group(1))
    else:
        raise ValueError("No frequency or velocity found")
    # Find amplitudes
    amp_section = re.search(r'Amplitude and Phase Data of Signals\s*(.*)', text, re.DOTALL)
    if amp_section:
        amp_line = amp_section.group(1).strip()
        matches = re.findall(r'([a-zA-Z ]+?) : ([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)', amp_line)
        amplitudes = {}
        for name, val in matches:
            amplitudes[name.strip()] = float(val)
    else:
        raise ValueError("No amplitude section")
    data[freq] = {"U_reduced": U_reduced, "amplitudes": amplitudes}
    return data

frequency = [frequency_set_0,
             frequency_set_1, 
             frequency_set_2, 
             frequency_set_3,
             frequency_set_4,
             frequency_set_5,
             frequency_set_6
             ]

path_base = "_ConstantAmplitude"

iteration_variable2 = 0
for i in frequency:
    derivative_dict = {}
    amplitude_dict = {}
    iteration_variable1 = 1

    for frequency_list in i:
        
        for prescribed_motion in ["Heave", "Pitch"]:
        
            current_path = os.path.join(str(iteration_variable2) + path_base, f"{len(frequency_list)}_{iteration_variable1}", prescribed_motion)

            path_results = os.path.join(current_path, "Results")
            path_flutter = os.path.join(path_results, "Flutter_Derivatives.txt")
            data = read_flutter_derivatives(path_flutter)
            report_data = []  
            for freq, values in data.items():

                derivative = values["derivatives"]
                U_reduced = values["U_reduced"]
                derivative_dict.setdefault(U_reduced, {}).update(derivative)

            # Add amplitude and phase extraction
            path_force = os.path.join(path_results, "Force Results", f"{prescribed_motion} Results.txt")
            data_amp = read_amplitudes_phases(path_force)
            for freq, values in data_amp.items():
                U_reduced = values["U_reduced"]
                amps = values["amplitudes"]
                motion_amp_key = f"{prescribed_motion} Amplitude"
                amp_data = {
                    "motion_amp": amps[motion_amp_key],
                    "fy_amp": amps["Fy Amplitude"],
                    "fy_phase": amps["Phase Fy"],
                    "mom_amp": amps["Moment Amplitude"],
                    "mom_phase": amps["Phase Mx"]
                }
                amplitude_dict.setdefault(prescribed_motion, {})[U_reduced] = amp_data

        iteration_variable1 += 1

    path_flutter_combine = os.path.join(str(iteration_variable2) + path_base,"Flutter_Derivatives")

    velocities = sorted(derivative_dict.keys(), reverse=True)
    shutil.rmtree(path_flutter_combine)
    os.makedirs(path_flutter_combine, exist_ok=True)

    derivative_keys = ['A1', 'A2', 'A3', 'A4', 'H1', 'H2', 'H3', 'H4']
    for key in derivative_keys:
        filename = os.path.join(path_flutter_combine, f"{iteration_variable2}_{key}.txt")
        with open(filename, 'w') as f:
            f.write("Flutter Derivatives\n")
            f.write("================================\n\n")
            f.write(f"{'Reduced Velocity':<15}{'Derivative':>25}\n")
            f.write("-" * 50 + "\n")

            for v in velocities:
                deriv_value = derivative_dict[v][key]
                f.write(f"{f'{v}':<15}{f'{deriv_value}':>30}\n")
                #f.write(f"{v} {deriv_value}\n")
        print(f"Created: {filename}")

    # Add writing for amplitudes and phases
    path_force_combine = os.path.join(str(iteration_variable2) + path_base, "Force_Results")
    shutil.rmtree(path_force_combine)
    os.makedirs(path_force_combine, exist_ok=True)

    for motion in ["Heave", "Pitch"]:
        filename = os.path.join(path_force_combine, f"{iteration_variable2}_{motion}.txt")
        with open(filename, 'w') as f:
            # Minimal header to match user example style, but with clarity
            for v in velocities:
                if motion in amplitude_dict and v in amplitude_dict[motion]:
                    vals = amplitude_dict[motion][v]
                    f.write(f"{v} {vals['motion_amp']} {vals['fy_amp']} {vals['fy_phase']} {vals['mom_amp']} {vals['mom_phase']}\n")
        print(f"Created: {filename}")
    iteration_variable2 += 1