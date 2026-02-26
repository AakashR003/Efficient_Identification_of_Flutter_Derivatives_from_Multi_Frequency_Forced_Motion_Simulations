# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:57:18 2025

@author: aakas
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

import os
import shutil

from MultiFrequency_Data import *


def decompose_signal(signal, sample_rate, top_count, threshold=0.00):
    """
    Decompose signal into constituent sine waves using FFT,
    returning only positive frequency components (including zero frequency).
    Returns a list of components with frequency, amplitude, phase, and waveform.
    """
    fft_result = fft(signal)
    
    frequencies = fftfreq(len(signal), 1 / sample_rate)
    
    sine_components = []
    
    for i in range(len(fft_result)):
        frequency = frequencies[i]
        # Only consider positive frequencies (>= 0)
        if frequency >= 0:
            amplitude = abs(fft_result[i]) / len(signal) * 2
            if amplitude > threshold:
                phase = np.angle(fft_result[i])
                
                t = np.arange(len(signal)) / sample_rate
                sine_wave = amplitude * np.cos(2 * np.pi * frequency * t + phase)
                sine_components.append({
                    'frequency': frequency,
                    'amplitude': amplitude,
                    'phase': phase,
                    'waveform': sine_wave
                })
    top_components = sorted(sine_components, key=lambda x: x['amplitude'], reverse=True)[:top_count]
    return top_components

def plot_signal_and_components(signal, components, sample_rate, path_signal_components):

    t = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(12, 8))
    
    # Plot original input signal
    plt.subplot(len(components) + 1, 1, 1)
    plt.plot(t, signal)
    plt.title('Input Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot each separated sine wave component
    for i, comp in enumerate(components, start=2):
        plt.subplot(len(components) + 1, 1, i)
        plt.plot(t, comp['waveform'])
        plt.title(f'Sine Wave Component at {comp["frequency"]:.1f} Hz')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(path_signal_components, dpi=300, bbox_inches='tight')
    plt.close()  
    print(f"__________Signal Plot saved successfully")
    
def plot_amplitude_vs_frequency(components, path_DFT):
    """
    Plot amplitude spectrum (amplitude vs frequency) from FFT components.
    """
    frequencies = [comp['frequency'] for comp in components]
    amplitudes = [comp['amplitude'] for comp in components]

    plt.figure(figsize=(10, 5))
    plt.stem(frequencies, amplitudes, basefmt=" ")
    plt.title('Discrete Fourier Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(path_DFT, dpi=300, bbox_inches='tight')
    plt.close()  
    print(f"__________DFT Plot saved successfully")


def extract_peak_terms(components, top_count, returndict = False):
    top_components = sorted(components, key=lambda x: x['amplitude'], reverse=True)[:top_count]
    frequ = []
    
    for i, comp in enumerate(top_components, start=1):
        frequ.append(comp['frequency']) 
    frequ = [float(x) for x in frequ] 
    frequ.sort()
    
    freq_to_amp = {truncate(c['frequency'], 2): c['amplitude'] for c in components} ##Change decimal places as needed
    amplitudes = [freq_to_amp[f] for f in frequ if f in freq_to_amp]
    
    if returndict == True:
        return freq_to_amp
    return amplitudes


def extract_amplitude_phase(components, target_frequency, tolerance=1e-2): ##Change tolerance as needed
    """
    Extract amplitude and phase for a given target frequency from the list of components.
    Returns a dictionary with amplitude and phase, or None if not found.
    """
    for component in components:
        if abs(component['frequency'] - target_frequency) < tolerance:
            return {
                'amplitude': component['amplitude'],
                'phase': component['phase']
            }
    return None 

def truncate(number, decimals):
    multiplier = 10 ** decimals
    return math.trunc(number * multiplier) / multiplier
    

path_base = "_ConstantAmplitude"
wind_velocity = 1
n = 4074 # number of lines you want to remove from top
sample_frequency = 2500

frequency = [frequency_set_1, frequency_set_2, frequency_set_3]

for i in frequency:
  
    iteration_variable1 = 1
    prescribed_motions = ["Heave", "Pitch"]

    for prescribed_motion in prescribed_motions:
    
        for j in i:

            current_path = str(len(j)) + path_base
            current_path = os.path.join(current_path, str(len(j)) + "_" +str(iteration_variable1))
            path_CFD_results = os.path.join(current_path, "CFD_Results")
            path_results = os.path.join(current_path ,"Results")
            shutil.rmtree(path_results)
            os.makedirs(path_results, exist_ok=True)

            aerodynamic_force_path = os.path.join(path_CFD_results , "Aerodynamic_Forces_" + prescribed_motion +".dat")
            aerodynamic_motion_path = os.path.join(path_CFD_results , "Input_Motion.dat")

            path_force_components = os.path.join(path_results, current_path + "_Fy_and_Components.png")
            path_moment_components = os.path.join(path_results, current_path + "_Mx_and_Components.png")
            path_motion_components = os.path.join(path_results, current_path + "_Motion_and_Components.png")
            path_DFT_Fy = os.path.join(path_results, current_path + "_DFT_Fy.png")
            path_DFT_moment = os.path.join(path_results, current_path + "_DFT_moment.png")
            path_fitted_signal = os.path.join(path_results, current_path + "_Fitted_Signal.png")       
            print(current_path)

            #Data Processing
            data_force = np.loadtxt(aerodynamic_force_path, skiprows=2)  # Read data
            data_force = data_force[n:len(data_force)]
            time, fx, fy, moment = data_force[:, 0], data_force[:, 1], data_force[:, 2], data_force[:, 6]  # Forces
            data_motion = np.loadtxt(aerodynamic_motion_path, skiprows = 2)
            data_motion = data_motion[n:len(data_motion)]
            u, v, theta = data_motion[:, 1], data_motion[:, 2], data_motion[:, 3]  # Displacement
            print("__________Length of list", len(fy), len(v))


            #FFT
            force_components = decompose_signal(fy, sample_frequency, top_count=10)
            moment_components = decompose_signal(moment, sample_frequency, top_count=10)
            motion_components = decompose_signal(v, sample_frequency, top_count=2)

            plot_signal_and_components(fy,decompose_signal(fy, sample_frequency, top_count=2),sample_frequency, path_force_components)
            plot_signal_and_components(moment,decompose_signal(moment, sample_frequency, top_count=2),sample_frequency, path_moment_components)
            plot_signal_and_components(v, motion_components,sample_frequency, path_motion_components)
            freq_and_force_amp = extract_peak_terms(force_components, top_count=10, returndict=True)
            freq_and_moment_amp = extract_peak_terms(moment_components, top_count=10, returndict=True)
            plot_amplitude_vs_frequency(decompose_signal(fy, sample_frequency, top_count=500), path_DFT_Fy)
            plot_amplitude_vs_frequency(decompose_signal(moment, sample_frequency, top_count=500), path_DFT_moment)
            

            for k in j:
                #Extracting phase and amplitude
                frequency = k
                phi1 = extract_amplitude_phase(force_components, target_frequency=frequency)["phase"]
                phi2 = extract_amplitude_phase(motion_components, target_frequency=frequency)["phase"]
                phase_diff = ((phi2 - phi1) + np.pi) % (2 * np.pi) - np.pi
                print("__________Relative phase difference (fy w.r.t v):", phase_diff)

                F = freq_and_force_amp[frequency] 
                F_cos = F * np.cos(phase_diff)
                F_sine = -F * np.sin(phase_diff)

                t = time
                x_recon = F_cos * np.cos(2*np.pi*frequency*t) + F_sine * np.sin(2*np.pi*frequency*t)
                plt.figure(figsize=(8,4))
                plt.plot(t, fy, label="fy")
                plt.plot(t, x_recon, '--', label="Reconstructed Fy (relative phase)")
                plt.legend()
                plt.xlabel("Time [s]")
                plt.ylabel("Amplitude")
                plt.title(" Fy relative to V")
                plt.savefig(path_fitted_signal, dpi=300, bbox_inches='tight')
                plt.close()  # Close figure to free memory
                print(f"__________Fitted Signal Plot saved successfully")

                
                #Extracting phase and amplitude
                phi1 = extract_amplitude_phase(moment_components, target_frequency=frequency)["phase"]
                phi2 = extract_amplitude_phase(motion_components, target_frequency=frequency)["phase"]
                phase_diff = ((phi2 - phi1) + np.pi) % (2 * np.pi) - np.pi
                print("__________Relative phase difference (fy w.r.t v):", phase_diff)

                N = len(fy)
                fs = sample_frequency
                freqs = np.fft.fftfreq(N, 1/fs)
                
                M = freq_and_moment_amp[frequency] 
                M_cos = M * np.cos(phase_diff)
                M_sine = -M * np.sin(phase_diff)

                t = time
                x_recon = M_cos * np.cos(2*np.pi*frequency*t) + M_sine * np.sin(2*np.pi*frequency*t)
                plt.figure(figsize=(8,4))
                plt.plot(t, fy, label="Original fy")
                plt.plot(t, x_recon, '--', label="Reconstructed moment (relative phase)")
                plt.legend()
                plt.xlabel("Time [s]")
                plt.ylabel("Amplitude")
                plt.title(" Moment relative to V")
                plt.savefig(path_fitted_signal, dpi=300, bbox_inches='tight')
                plt.close()  # Close figure to free memory
                print(f"__________Fitted Signal Plot saved successfully")


                #Finding derivatives
                row = 1.225
                omega = 2 * np.pi * frequency  
                B = 0.366
                h = 0.0183
                U_reduced = wind_velocity / (frequency * B)

                if prescribed_motion == "Heave":
                    H1 = 2 * F_cos / (row*omega*omega*B*B*h)
                    H4 = 2 * F_sine / (row*omega*omega*B*B*h)
                    A1 = 2 * M_cos / (row*omega*omega*B*B*h)
                    A4 = 2 * M_sine / (row*omega*omega*B*B*h)

                    print("__________Frequency",frequency ,"hz      H1*", H1)
                    print("__________Frequency",frequency ,"hz      H4*", H4)
                    print("__________Frequency",frequency ,"hz      A1*", A1)
                    print("__________Frequency",frequency ,"hz      A4*", A4)



                    # Write to file
                    path_flutter_derivatives = os.path.join(path_results, current_path + "_"+ str(k) + "_Flutter_Derivatives.txt")
                    with open(path_flutter_derivatives, "w") as f:
                        f.write("Flutter Derivatives Results\n")
                        f.write("===========================\n")
                        f.write(f"Frequency       : {frequency}\n")
                        f.write(f"Reduced Velocity: {U_reduced}\n")
                        f.write(f"H1          : {H1}\n")
                        f.write(f"H4          : {H4}\n")
                        f.write(f"A1          : {A1}\n")
                        f.write(f"A4          : {A4}\n")

                elif prescribed_motion == "Pitch":
                    H2 = 2 * F_cos / (row*omega*omega*B*B*B*h)
                    H3 = 2 * F_sine / (row*omega*omega*B*B*B*h)
                    A2 = 2 * M_cos / (row*omega*omega*B*B*B*h)
                    A3 = 2 * M_sine / (row*omega*omega*B*B*B*h)

                    print("__________Frequency",frequency ,"hz      H1*", H1)
                    print("__________Frequency",frequency ,"hz      H4*", H4)
                    print("__________Frequency",frequency ,"hz      A1*", A1)
                    print("__________Frequency",frequency ,"hz      A4*", A4)



                    # Write to file
                    path_flutter_derivatives = os.path.join(path_results, current_path + "_"+ str(k) + "_Flutter_Derivatives.txt")
                    with open(path_flutter_derivatives, "a") as f:
                        f.write(f"H2          : {H2}\n") 
                        f.write(f"H3          : {H3}\n") 
                        f.write(f"A2          : {A2}\n") 
                        f.write(f"A3          : {A3}\n")

                    print(f"__________Flutter Derivatives successfully written")
    iteration_variable1 += 1
