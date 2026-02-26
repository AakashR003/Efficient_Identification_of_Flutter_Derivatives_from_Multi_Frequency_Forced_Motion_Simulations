
import numpy as np

import os
import shutil

from MultiFrequency_Data import *
from Python_Common_Methods import read_flutter_derivatives, read_amplitude_phase_data, write_amplitude_phase_file

frequency = [
             frequency_set_1, 
             frequency_set_2, 
             frequency_set_3,
             frequency_set_4,
             ]

path_base = "_ConstantAmplitude"


for i in frequency:
    derivative_dict = {}
    iteration_variable1 = 1

    for frequency_list in i:
        
        for prescribed_motion in ["Heave", "Pitch"]:
        
            current_path = os.path.join(str(len(frequency_list)) + path_base, f"{len(frequency_list)}_{iteration_variable1}", prescribed_motion)

            path_results = os.path.join(current_path, "Results")
            path_flutter = os.path.join(path_results, "Flutter_Derivatives.txt")
            data = read_flutter_derivatives(path_flutter)
            report_data = []  
            for freq, values in data.items():

                derivative = values["derivatives"]
                U_reduced = values["U_reduced"]
                derivative_dict.setdefault(U_reduced, {}).update(derivative)
        iteration_variable1 += 1

    path_flutter_combine = os.path.join(str(len(frequency_list)) + path_base,"Flutter_Derivatives")


    velocities = sorted(derivative_dict.keys(), reverse=True)
    shutil.rmtree(path_flutter_combine)
    os.makedirs(path_flutter_combine, exist_ok=True)

    derivative_keys = ['A1', 'A2', 'A3', 'A4', 'H1', 'H2', 'H3', 'H4']
    for key in derivative_keys:
        filename = os.path.join(path_flutter_combine, f"{len(frequency_list)}_{key}.txt")
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


# ------------------------------------------------- Amplitude and Phase Files ------------------------------------------------- #


pitch_dict = {}
heave_dict = {}

for i in frequency:
    iteration_variable1 = 1

    for frequency_list in i:
        for prescribed_motion in ["Heave", "Pitch"]:

            current_path = os.path.join(str(len(frequency_list)) + path_base, f"{len(frequency_list)}_{iteration_variable1}",prescribed_motion)
            path_results = os.path.join(current_path, "Results")
            path_amp_phase = os.path.join(path_results, "Amplitude_and_Phase.txt")
            data = read_amplitude_phase_data(path_amp_phase)

            for _, values in data.items():
                U = values["U_reduced"]
                vals = values["values"]
                if prescribed_motion == "Pitch":
                    pitch_dict.setdefault(U, {}).update(vals)
                else:
                    heave_dict.setdefault(U, {}).update(vals)
        iteration_variable1 += 1

    path_output = os.path.join(str(len(frequency_list)) + path_base,"Amplitude_and_Phase")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)
    os.makedirs(path_output, exist_ok=True)

    # ---------- Pitch ----------
    pitch_columns = ["Tita Amplitude","Fy Amplitude","Phase Fy","Moment Amplitude","Phase Mx"]
    write_amplitude_phase_file(path_output,f"{len(frequency_list)}_Pitch_Amplitude_Phase.txt",pitch_dict,pitch_columns)

    # ---------- Heave ----------
    heave_columns = ["Heave Amplitude","Fy Amplitude","Phase Fy","Moment Amplitude","Phase Mx"]
    write_amplitude_phase_file(path_output,f"{len(frequency_list)}_Heave_Amplitude_Phase.txt",heave_dict,heave_columns)
    print(f"Created {path_output}")
