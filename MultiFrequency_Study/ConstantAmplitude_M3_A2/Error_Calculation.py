
import numpy as np

import os
#import shutil

from MultiFrequency_Data import *
from Python_Common_Methods import read_flutter_derivatives


def compute_percentage_error(derivative, derivative_ref):
    error = {}

    for key in derivative_ref.keys():
        ref = derivative_ref[key]
        val = derivative[key]

        if ref == 0.0:
            error[key] = float("nan")  
        else:
            error[key] = (val - ref) / abs(ref) * 100.0

    return error


def write_error_report(path, report_data):

    with open(path, "w") as f:
        f.write("Flutter Derivatives Error Report\n")
        f.write("================================\n\n")

        for block in report_data:
            f.write(f"Frequency       : {block['frequency']}\n")
            f.write(f"Reduced Velocity: {block['U_reduced']}\n")

            f.write(f"{'Derivative':<10}{'Reference':>15}{'Computed':>15}{'Error (%)':>15}\n")
            f.write("-" * 55 + "\n")

            for key in block["ref"].keys():
                f.write(f"{key:<10}"f"{block['ref'][key]:>15.6f}"f"{block['val'][key]:>15.6f}"f"{block['error'][key]:>15.3f}\n")

            f.write("\n\n") 


frequency = [ 
             frequency_set_1,
             frequency_set_2, 
             frequency_set_3,
             frequency_set_4,
             #frequency_set_8
             ]

path_base = "_ConstantAmplitude"
derivative_ref = {}

for freq in frequency:
    derivative_dict = {}
    iteration_variable1 = 1

    for prescribed_motion in ["Heave", "Pitch"]:
        iteration_variable1 = 1
        
        for frequency_list in frequency_set_1:
            current_path_ref = os.path.join(str(len(frequency_list)) + path_base, f"{len(frequency_list)}_{iteration_variable1}", prescribed_motion, "Results")

            path_ref = os.path.join(current_path_ref, "Flutter_Derivatives.txt")

            ref_data = read_flutter_derivatives(path_ref)

            for frequ, values in ref_data.items():
                derivative_ref[(frequ, prescribed_motion)] = values["derivatives"]

            iteration_variable1 += 1


    for prescribed_motion in ["Heave", "Pitch"]:
        iteration_variable1 = 1

        for frequency_list in freq:

            current_path = os.path.join(str(len(frequency_list)) + path_base, f"{len(frequency_list)}_{iteration_variable1}", prescribed_motion, "Results")

            path_flutter = os.path.join(current_path, "Flutter_Derivatives.txt")
            data = read_flutter_derivatives(path_flutter)
            report_data = []  

            for frequ, values in data.items():

                derivative = values["derivatives"]
                U_reduced = values["U_reduced"]
                ref = derivative_ref[(frequ, prescribed_motion)]
                error = compute_percentage_error(derivative, ref)
                report_data.append({"frequency": frequ, "U_reduced": U_reduced, "ref": ref, "val": derivative, "error": error })

            path_error_report = os.path.join(current_path, "Error_Report_Flutter_Derivatives.txt")
            write_error_report(path_error_report, report_data)
            iteration_variable1 += 1
