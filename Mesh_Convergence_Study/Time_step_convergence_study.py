# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:57:18 2025

@author: aakas
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from fractions import Fraction

import os


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


def plot_time_interval_convergence(x, y, y_label, title, save_path):
    """
    Generic plot function with auto y-limits (+/-20% beyond min/max).
    """
    y_min, y_max = min(y), max(y)
    margin = 1 * (y_max - y_min)  # 20% margin
    margin = 0.1

    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.xlabel("Time Interval")
    plt.ylabel(y_label)
    plt.title(title)
    plt.ylim(y_min - margin, y_max + margin)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def time_intervals(A, C):
    """
    Returns all possible integer x values for the equation:
        (A / x) * y = C
    where y can be any integer in y_range (default: -10 to 10, excluding 0).
    """
    y_range = range(0, 11)
    # Convert C to a rational number for exact computation
    C = Fraction(C).limit_denominator()
    
    # denominator cannot be zero
    if C == 0:
        raise ValueError("C cannot be zero, because equation becomes A*y/x = 0.")
    
    time_interval = set()
    
    for y in y_range:
        if y == 0:
            continue  # skip y=0 because it makes the equation trivial
        
        # x = A*y / C
        x = Fraction(A * y, C)
        
        # we accept only integer x
        if x.denominator == 1 and x.numerator != 0:
            time_interval.add(x.numerator)
    
    return sorted(time_interval)


def truncate(number, decimals):
    multiplier = 10 ** decimals
    return math.trunc(number * multiplier) / multiplier
    

path_base = "Model_"

"""
#Model 1
n = 850 + 25 + 1000 # number of lines you want to remove from top
frequency = 0.25
sample_frequency = 250
# """


# """
#Model 2,3,4
n = 7320 # number of lines you want to remove from top
frequency = 0.25
sample_frequency = 500
file = 2
# """


"""
#Model 5
n = 8500 - 415# number of lines you want to remove from top
frequency = 0.25
sample_frequency = 500
# """

time_interval_list = time_intervals(sample_frequency, frequency)
H1_list = []
H4_list = []
A1_list = []
A4_list = []
new_time_interval_list = []

for time_interval in time_interval_list:

    current_path = path_base + str(file)
    path_CFD_results = os.path.join(current_path, "CFD_Results")
    path_results = os.path.join(current_path ,"Results")
    os.makedirs(path_results, exist_ok=True)

    aerodynamic_force_path = os.path.join(path_CFD_results , "Aerodynamic_Forces.dat")
    aerodynamic_motion_path = os.path.join(path_CFD_results , "Input_Motion.dat")
    print(current_path)

    #Data Processing
    data_force = np.loadtxt(aerodynamic_force_path, skiprows=2)  # Read data
    if len(data_force) < n + time_interval:
        print(f"__________STOP {time_interval} due to insufficient data length.")
        break
    new_time_interval_list.append(time_interval)
    data_force = data_force[n: n + time_interval]
    time, fx, fy, moment = data_force[:, 0], data_force[:, 1], data_force[:, 2], data_force[:, 6]  # Forces
    data_motion = np.loadtxt(aerodynamic_motion_path, skiprows = 2)
    data_motion = data_motion[n:n + time_interval]
    v, w, theta = data_motion[:, 1], data_motion[:, 2], data_motion[:, 3]  # Displacement
    print("__________Length of list", len(fy), len(w))


    #FFT
    force_components = decompose_signal(fy, sample_frequency, top_count=10)
    moment_components = decompose_signal(moment, sample_frequency, top_count=10)
    motion_components = decompose_signal(w, sample_frequency, top_count=2)

    freq_and_force_amp = extract_peak_terms(force_components, top_count=10, returndict=True)
    freq_and_moment_amp = extract_peak_terms(moment_components, top_count=10, returndict=True)
    

    #Extracting phase and amplitude
    phi1 = extract_amplitude_phase(force_components, target_frequency=frequency)["phase"]
    phi2 = extract_amplitude_phase(motion_components, target_frequency=frequency)["phase"]
    phase_diff = ((phi1 - phi2) + np.pi) % (2 * np.pi) - np.pi
    print("__________Relative phase difference (fy w.r.t w):", phase_diff)

    F = freq_and_force_amp[frequency] 
    F_sine = F * np.cos(phase_diff)
    F_cos = F * np.sin(phase_diff)

    
    #Extracting phase and amplitude
    phi1 = extract_amplitude_phase(moment_components, target_frequency=frequency)["phase"]
    phi2 = extract_amplitude_phase(motion_components, target_frequency=frequency)["phase"]
    phase_diff = ((phi2 - phi1) + np.pi) % (2 * np.pi) - np.pi
    print("__________Relative phase difference (fy w.r.t w):", phase_diff)
    
    M = freq_and_moment_amp[frequency] 
    M_sine = M * np.cos(phase_diff)
    M_cos = -M * np.sin(phase_diff)


    #Finding derivatives
    row = 1.225
    omega = 2 * np.pi * frequency 
    B = 0.366
    h = 0.0183

    H1 = 2 * F_cos / (row*omega*omega*B*B*h)
    H4 = 2 * F_sine / (row*omega*omega*B*B*h)
    A1 = 2 * M_cos / (row*omega*omega*B*B*B*h)
    A4 = 2 * M_sine / (row*omega*omega*B*B*B*h)
    H1_list.append(H1)
    H4_list.append(H4)
    A1_list.append(A1)
    A4_list.append(A4)

    print("__________Frequency",frequency,"hz      H1*", H1)
    print("__________Frequency",frequency,"hz      H4*", H4)
    print("__________Frequency",frequency,"hz      A1*", A1)
    print("__________Frequency",frequency,"hz      A4*", A4)


path_time_step_convergence = os.path.join(path_results, "Time_Step_Convergence")
os.makedirs(path_time_step_convergence, exist_ok=True)

Derivatives = [
    (H1_list, "H1*", "H1* Convergence Study", "H1_TimeStep_Convergence.png"),
    (H4_list, "H4*", "H4* Convergence Study", "H4_TimeStep_Convergence.png"),
    (A1_list, "A1*", "A1* Convergence Study", "A1_TimeStep_Convergence.png"),
    (A4_list, "A4*", "A4* Convergence Study", "A4_TimeStep_Convergence.png"),
]

for y_list, y_label, title, filename in Derivatives:
    save_path = os.path.join(path_time_step_convergence, current_path + filename)
    plot_time_interval_convergence(new_time_interval_list, y_list, y_label, title, save_path)