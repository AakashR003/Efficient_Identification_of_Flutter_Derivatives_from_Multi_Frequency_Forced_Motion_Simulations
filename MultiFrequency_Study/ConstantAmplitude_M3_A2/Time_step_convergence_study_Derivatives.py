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
import shutil

from MultiFrequency_Data import *


def decompose_signal(signal, sample_rate, top_count, threshold=0.00):
    """
    Decompose signal into constituent sine waves using FFT,
    returning only positive frequency components (including zero frequency).
    Returns a list of components with frequency, amplitude, phase, and waveform.
    """
    N = len(signal)
    fft_result = fft(signal)
    frequencies = fftfreq(N, 1 / sample_rate)
    
    # Vectorized operations - process all positive frequencies at once
    positive_mask = frequencies >= 0
    pos_freqs = frequencies[positive_mask]
    pos_fft = fft_result[positive_mask]
    
    amplitudes = np.abs(pos_fft) / N * 2
    above_threshold = amplitudes > threshold
    
    # Filter before creating components
    filtered_freqs = pos_freqs[above_threshold]
    filtered_amps = amplitudes[above_threshold]
    filtered_phases = np.angle(pos_fft[above_threshold])
    
    # Get top components BEFORE generating waveforms
    top_indices = np.argsort(filtered_amps)[-top_count:][::-1]
    
    t = np.arange(N) / sample_rate
    sine_components = []
    
    # Only generate waveforms for top components
    for idx in top_indices:
        freq = filtered_freqs[idx]
        amp = filtered_amps[idx]
        phase = filtered_phases[idx]
        sine_wave = amp * np.cos(2 * np.pi * freq * t + phase)
        sine_components.append({
            'frequency': freq,
            'amplitude': amp,
            'phase': phase,
            'waveform': sine_wave
        })
    
    return sine_components


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
    parts = os.path.normpath(path_DFT).split(os.sep)
    plt.title(f"Ampltide Spectrum {parts[-3]} {os.path.basename(path_DFT).replace('DFT_','').replace('.png','')} {parts[-4]}")

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(-1, 10)
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
    
    freq_to_amp = {truncate(c['frequency'], 3): c['amplitude'] for c in components} ##Change decimal places as needed
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








import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from fractions import Fraction
from MultiFrequency_Data import *
#from Python_Common_Methods import decompose_signal, extract_peak_terms, extract_amplitude_phase

# Configuration
path_base = "_ConstantAmplitude"
n = 7320  # Skip initial data points
sample_frequency = 500
row = 1.225  # Density
B = 0.366   # Reference length

frequency_sets = [frequency_set_1, frequency_set_2, frequency_set_3, frequency_set_4]

# Store reference derivatives (computed once with full time interval)
ref_derivatives = {}

# Store errors for combined plotting
combined_errors = {
    'Heave': {'H1': {}, 'H4': {}, 'A1': {}, 'A4': {}},
    'Pitch': {'H2': {}, 'H3': {}, 'A2': {}, 'A3': {}}
}


def get_time_intervals_for_frequencies(frequency_list, sample_frequency, max_time_interval=40000):
    """
    Generate time intervals that ensure all frequencies in the list fall on FFT bins.
    
    For multiple frequencies, we need time intervals where:
    - Each frequency falls exactly on an FFT bin
    - FFT bin resolution = sample_frequency / time_interval
    - A frequency falls on a bin if: frequency = k * (sample_frequency / time_interval)
    - Rearranging: time_interval = k * sample_frequency / frequency
    
    We find common time intervals that work for ALL frequencies in the list,
    up to max_time_interval.
    """
    epsilon = 1e-10

    # List to store valid L
    valid_lengths = []

    for L in range(1, max_time_interval + 1):
        is_valid = True
        for f in frequency_list:
            n = f[0] * L / sample_frequency
            # Check if n is close to an integer
            if abs(n - round(n)) > epsilon:
                is_valid = False
                break
        if is_valid:
            valid_lengths.append(L)
    
    return sorted(list(valid_lengths))


def compute_derivatives(force_path, motion_path, time_interval, frequency, h_amp, tita_amp, prescribed_motion, freq_list_len):
    """Compute flutter derivatives from CFD data."""
    
    # Load data
    data_force = np.loadtxt(force_path, skiprows=2)
    if len(data_force) < n + time_interval:
        return None
    
    data_force = data_force[n:n + time_interval]
    fy = data_force[:, 2]
    moment = data_force[:, 6]
    
    data_motion = np.loadtxt(motion_path, skiprows=2)
    data_motion = data_motion[n:n + time_interval]
    w = data_motion[:, 2]
    theta = data_motion[:, 3]
    
    # FFT decomposition
    force_comp = decompose_signal(fy, sample_frequency, top_count=10)
    moment_comp = decompose_signal(moment, sample_frequency, top_count=10)
    
    if prescribed_motion == "Heave":
        motion_comp = decompose_signal(w, sample_frequency, top_count=freq_list_len)
    else:
        motion_comp = decompose_signal(theta, sample_frequency, top_count=freq_list_len)
    
    # Extract amplitudes
    freq_force_amp = extract_peak_terms(force_comp, top_count=10, returndict=True)
    freq_moment_amp = extract_peak_terms(moment_comp, top_count=10, returndict=True)
    
    if frequency not in freq_force_amp or frequency not in freq_moment_amp:
        return None
    
    # Phase differences
    phi1_f = extract_amplitude_phase(force_comp, target_frequency=frequency)["phase"]
    phi2 = extract_amplitude_phase(motion_comp, target_frequency=frequency)["phase"]
    phase_diff_f = ((phi2 - phi1_f) + np.pi) % (2 * np.pi) - np.pi
    
    phi1_m = extract_amplitude_phase(moment_comp, target_frequency=frequency)["phase"]
    phase_diff_m = ((phi2 - phi1_m) + np.pi) % (2 * np.pi) - np.pi
    
    # Force and moment components
    F = freq_force_amp[frequency]
    F_sine = F * np.cos(phase_diff_f)
    F_cos = -F * np.sin(phase_diff_f)
    
    M = freq_moment_amp[frequency]
    M_sine = M * np.cos(phase_diff_m)
    M_cos = -M * np.sin(phase_diff_m)
    
    # Compute derivatives
    omega = 2 * np.pi * frequency
    
    if prescribed_motion == "Heave":
        return {
            'H1': 2 * F_cos / (row * omega**2 * B**2 * h_amp),
            'H4': 2 * F_sine / (row * omega**2 * B**2 * h_amp),
            'A1': 2 * M_cos / (row * omega**2 * B**3 * h_amp),
            'A4': 2 * M_sine / (row * omega**2 * B**3 * h_amp)
        }
    else:  # Pitch
        return {
            'H2': 2 * F_cos / (row * omega**2 * B**3 * tita_amp),
            'H3': 2 * F_sine / (row * omega**2 * B**3 * tita_amp),
            'A2': 2 * M_cos / (row * omega**2 * B**4 * tita_amp),
            'A3': 2 * M_sine / (row * omega**2 * B**4 * tita_amp)
        }


def write_derivatives_table(save_path, time_intervals, derivatives, ref_derivatives, prescribed_motion):
    """Write derivatives, reference values, and errors to Excel and text files."""
    
    # Determine derivative names
    if prescribed_motion == "Heave":
        deriv_names = ['H1', 'H4', 'A1', 'A4']
    else:
        deriv_names = ['H2', 'H3', 'A2', 'A3']
    
    # Prepare data for table
    table_data = []
    
    for i, time_int in enumerate(time_intervals):
        row = {'Time_Interval': time_int}
        
        for deriv_name in deriv_names:
            ref_val = ref_derivatives[deriv_name]
            current_val = derivatives[deriv_name][i]
            error = (current_val - ref_val) / ref_val * 100 if ref_val != 0 else float('nan')
            
            row[f'{deriv_name}_Reference'] = ref_val
            row[f'{deriv_name}_Current'] = current_val
            row[f'{deriv_name}_Error_%'] = error
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save to Excel
    excel_path = save_path.replace('.txt', '.xlsx')
    df.to_excel(excel_path, index=False, float_format='%.6f')
    
    # Save to text file (formatted)
    with open(save_path, 'w') as f:
        f.write(f"Flutter Derivatives Convergence Analysis - {prescribed_motion}\n")
        f.write("=" * 150 + "\n\n")
        
        # Write header
        f.write(f"{'Time Interval':>15}")
        for deriv_name in deriv_names:
            f.write(f"{deriv_name + '_Ref':>15}{deriv_name + '_Current':>15}{deriv_name + '_Error%':>15}")
        f.write("\n")
        f.write("-" * 150 + "\n")
        
        # Write data rows
        for i, time_int in enumerate(time_intervals):
            f.write(f"{time_int:>15}")
            
            for deriv_name in deriv_names:
                ref_val = ref_derivatives[deriv_name]
                current_val = derivatives[deriv_name][i]
                error = (current_val - ref_val) / ref_val * 100 if ref_val != 0 else float('nan')
                
                f.write(f"{ref_val:>15.6f}{current_val:>15.6f}{error:>15.3f}")
            
            f.write("\n")
    
    print(f"  Saved table: {excel_path}")
    print(f"  Saved table: {save_path}")


def plot_convergence(time_intervals, values, ylabel, title, save_path):
    """Plot convergence study."""
    plt.figure(figsize=(10, 6))
    plt.plot(time_intervals, values, marker='o', linewidth=2)
    plt.xlabel('Time Interval', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Main processing loop
for set_idx, freq_set in enumerate(frequency_sets):
    
    for prescribed_motion in ["Heave", "Pitch"]:
        
        for freq_idx, frequency_list in enumerate(freq_set, 1):
            
            for tuple_idx, freq_tuple in enumerate(frequency_list, 1):
                
                freq_actual, h_amp_actual, tita_amp_actual, number = freq_tuple
                freq_ref, h_amp_ref, tita_amp_ref, _ = frequency_set_1[number - 1][0]
                
                # Compute reference derivatives once
                ref_key = (prescribed_motion, number)
                
                if ref_key not in ref_derivatives:
                    print(f"Computing reference for {prescribed_motion}, frequency index {number}")
                    
                    # Get time intervals for single frequency (reference)
                    time_interval_list = get_time_intervals_for_frequencies(
                        frequency_set_1[number - 1], sample_frequency, max_time_interval=40000
                    )
                    
                    ref_path = os.path.join("1" + path_base, f"1_{number}", prescribed_motion, "CFD_Results")
                    force_path = os.path.join(ref_path, "Aerodynamic_Forces.dat")
                    motion_path = os.path.join(ref_path, "Aerodynamic_Input_Motion.dat")
                    
                    # Check available data length first
                    try:
                        data_force = np.loadtxt(force_path, skiprows=2)
                        available_length = len(data_force) - n
                        print(f"  Available data length: {available_length}")
                    except:
                        print(f"ERROR: Could not read reference file for {prescribed_motion}, frequency index {number}")
                        continue
                    
                    # Filter time intervals to only those with sufficient data
                    valid_intervals = [t for t in time_interval_list if t <= available_length]
                    
                    if not valid_intervals:
                        print(f"ERROR: No valid time intervals for {prescribed_motion}, frequency index {number}")
                        continue
                    
                    # Use the largest valid interval
                    time_interval_full = valid_intervals[-1]
                    print(f"  Using time interval: {time_interval_full}")
                    
                    ref_deriv = compute_derivatives(force_path, motion_path, time_interval_full, 
                                                   freq_ref, h_amp_ref, tita_amp_ref, 
                                                   prescribed_motion, 1)
                    
                    if ref_deriv:
                        ref_derivatives[ref_key] = ref_deriv
                        print(f"Reference: {ref_deriv}")
                    else:
                        print(f"ERROR: Could not compute reference for {prescribed_motion}, frequency index {number}")
                        continue
                
                # Get time intervals for current frequency list
                # This ensures all frequencies in the list fall on FFT bins
                time_interval_list = get_time_intervals_for_frequencies(
                    frequency_list, sample_frequency, max_time_interval=40000
                )
                
                # Check if reference exists before proceeding
                if ref_key not in ref_derivatives:
                    print(f"  Skipping - no reference available for {prescribed_motion}, frequency index {number}")
                    continue
                
                print(f"\nProcessing Set {set_idx + 1}, {prescribed_motion}, Freq List {freq_idx}, Freq {freq_actual} Hz")
                print(f"Valid time intervals: {time_interval_list}")
                
                # Process current configuration
                errors = {k: [] for k in ref_derivatives[ref_key].keys()}
                values = {k: [] for k in ref_derivatives[ref_key].keys()}
                valid_time_intervals = []
                
                # Check available data length for current configuration
                current_path_check = os.path.join(f"{len(frequency_list)}{path_base}", 
                                          f"{len(frequency_list)}_{freq_idx}", 
                                          prescribed_motion, "CFD_Results")
                force_path_check = os.path.join(current_path_check, "Aerodynamic_Forces.dat")
                
                try:
                    data_check = np.loadtxt(force_path_check, skiprows=2)
                    available_length = len(data_check) - n
                    print(f"  Available data length: {available_length}")
                    
                    # Filter time intervals based on available data
                    time_interval_list = [t for t in time_interval_list if t <= available_length]
                    print(f"  Filtered time intervals: {time_interval_list}")
                except Exception as e:
                    print(f"  ERROR reading data file: {e}")
                    continue
                
                for time_interval in time_interval_list:
                    
                    current_path = os.path.join(f"{len(frequency_list)}{path_base}", 
                                              f"{len(frequency_list)}_{freq_idx}", 
                                              prescribed_motion, "CFD_Results")
                    force_path = os.path.join(current_path, "Aerodynamic_Forces.dat")
                    motion_path = os.path.join(current_path, "Aerodynamic_Input_Motion.dat")
                    
                    deriv = compute_derivatives(force_path, motion_path, time_interval,
                                              freq_actual, h_amp_actual, tita_amp_actual,
                                              prescribed_motion, len(frequency_list))
                    
                    if deriv is None:
                        print(f"  Skipping time interval {time_interval} - insufficient data or frequency not found")
                        break
                    
                    valid_time_intervals.append(time_interval)
                    ref_vals = ref_derivatives[ref_key]
                    
                    for key in deriv.keys():
                        values[key].append(deriv[key])
                        error = (deriv[key] - ref_vals[key]) / ref_vals[key] * 100
                        errors[key].append(error)
                
                if not valid_time_intervals:
                    print(f"  No valid time intervals for this configuration")
                    continue
                
                # Save individual plots
                result_path = os.path.join(f"{len(frequency_list)}{path_base}", 
                                          f"{len(frequency_list)}_{freq_idx}", 
                                          prescribed_motion, "Time_Step_Convergence")
                os.makedirs(result_path, exist_ok=True)
                
                # Write derivatives table
                table_path = os.path.join(result_path, f"{tuple_idx}_Derivatives_Table.txt")
                write_derivatives_table(table_path, valid_time_intervals, values, 
                                      ref_derivatives[ref_key], prescribed_motion)
                
                for key in errors.keys():
                    # Value convergence
                    plot_convergence(valid_time_intervals, values[key], key, 
                                   f"{key} Convergence Study", 
                                   os.path.join(result_path, f"{tuple_idx}{key}_TimeStep_Convergence.png"))
                    
                    # Error convergence
                    plot_convergence(valid_time_intervals, errors[key], f"{key} Error (%)", 
                                   f"{key} Error Convergence Study", 
                                   os.path.join(result_path, f"{tuple_idx}{key}_Error_TimeStep_Convergence.png"))
                    
                    # Store for combined plotting
                    if freq_actual not in combined_errors[prescribed_motion][key]:
                        combined_errors[prescribed_motion][key][freq_actual] = []
                    
                    combined_errors[prescribed_motion][key][freq_actual].append(
                        (valid_time_intervals.copy(), errors[key].copy(), set_idx + 2)
                    )


# Create combined plots and tables
print("\nCreating combined plots and tables...")
base_path = "global_results"

# Clean and recreate global_results directory
if os.path.exists(base_path):
    shutil.rmtree(base_path)
os.makedirs(base_path, exist_ok=True)

for prescribed_motion in ['Heave', 'Pitch']:
    motion_path = os.path.join(base_path, prescribed_motion)
    os.makedirs(motion_path, exist_ok=True)
    
    deriv_names = ['H1', 'H4', 'A1', 'A4'] if prescribed_motion == 'Heave' else ['H2', 'H3', 'A2', 'A3']
    
    for deriv_name in deriv_names:
        for freq_value in sorted(combined_errors[prescribed_motion][deriv_name].keys()):
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            for time_intervals, errors, set_num in combined_errors[prescribed_motion][deriv_name][freq_value]:
                plt.plot(time_intervals, errors, marker='o', label=f"Combination {set_num-1}", linewidth=2)
            
            plt.xlabel('Time Interval', fontsize=12)
            plt.ylabel(f'{deriv_name} Error (%)', fontsize=12)
            plt.title(f'{deriv_name} Error vs Time Interval - Frequency {freq_value} Hz', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(motion_path, f"{deriv_name}_Freq_{freq_value}_Combined_Error.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot: {plot_path}")

# Create single combined Excel file for all data
print("\nCreating combined Excel file...")

for prescribed_motion in ['Heave', 'Pitch']:
    motion_path = os.path.join(base_path, prescribed_motion)
    deriv_names = ['H1', 'H4', 'A1', 'A4'] if prescribed_motion == 'Heave' else ['H2', 'H3', 'A2', 'A3']
    
    excel_path = os.path.join(motion_path, f"{prescribed_motion}_Combined_Errors.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Create a sheet for each derivative
        for deriv_name in deriv_names:
            
            # Collect data for this derivative across all frequencies
            freq_data = {}
            
            for freq_value in sorted(combined_errors[prescribed_motion][deriv_name].keys()):
                
                # Organize data by time interval
                time_interval_data = {}
                
                for time_intervals, errors, set_num in combined_errors[prescribed_motion][deriv_name][freq_value]:
                    for time_int, error in zip(time_intervals, errors):
                        if time_int not in time_interval_data:
                            time_interval_data[time_int] = {}
                        time_interval_data[time_int][f'Set_{set_num}'] = error
                
                # Convert to DataFrame with Time_Interval as index
                if time_interval_data:
                    df_freq = pd.DataFrame.from_dict(time_interval_data, orient='index')
                    df_freq.index.name = 'Time_Interval'
                    df_freq = df_freq.sort_index()
                    
                    # Add frequency identifier to column names
                    df_freq.columns = [f'Freq_{freq_value}_Hz_{col}' for col in df_freq.columns]
                    
                    freq_data[freq_value] = df_freq
            
            # Combine all frequencies for this derivative
            if freq_data:
                df_combined = pd.concat(freq_data.values(), axis=1)
                df_combined = df_combined.sort_index()
                
                # Write to sheet
                df_combined.to_excel(writer, sheet_name=deriv_name, float_format='%.6f')
        
        # Create a summary sheet with all data in long format
        all_data = []
        for deriv_name in deriv_names:
            for freq_value in sorted(combined_errors[prescribed_motion][deriv_name].keys()):
                for time_intervals, errors, set_num in combined_errors[prescribed_motion][deriv_name][freq_value]:
                    for time_int, error in zip(time_intervals, errors):
                        all_data.append({
                            'Derivative': deriv_name,
                            'Frequency_Hz': freq_value,
                            'Time_Interval': time_int,
                            'Frequency_Set': set_num,
                            'Error_%': error
                        })
        
        if all_data:
            df_all = pd.DataFrame(all_data)
            df_all.to_excel(writer, sheet_name='All_Data', index=False, float_format='%.6f')
    
    print(f"  Saved combined Excel: {excel_path}")

print("\nAll processing complete!")















"""


path_base = "_ConstantAmplitude"
wind_velocity = 1
n = 7320 #number of initial data points to skip
sample_frequency = 500

frequency = [
             frequency_set_1, 
             frequency_set_2, 
             frequency_set_3,
             frequency_set_4,
             ]

iteration_variable2 = 0

ref_frequency_set = frequency_set_1

# Dictionary to store reference derivatives (computed once with full time interval)
ref_derivatives = {}

for freq in frequency:

    prescribed_motions = ["Heave", "Pitch"]

    for prescribed_motion in prescribed_motions:
    
        iteration_variable1 = 1
        for frequency_list in freq:

            iteration_variable3 = 1
            for frequency_tuple in frequency_list:

                frequency_actual, h_amplitude_actual, tita_amplitude_actual, number = frequency_tuple
                frequency_ref, h_amplitude_ref, tita_amplitude_ref, _ = frequency_set_1[number - 1][0]

                time_interval_list = time_intervals(sample_frequency, frequency_set_1[number - 1][0][0])
                
                # Key for storing/retrieving reference derivatives
                ref_key = (prescribed_motion, number)
                
                # Compute reference derivatives ONCE if not already computed
                if ref_key not in ref_derivatives:
                    print(f"__________Computing GLOBAL REFERENCE for {prescribed_motion}, frequency index {number}")
                    
                    # Use FULL time interval (last one in the list)
                    time_interval_full = time_interval_list[-1]
                    
                    frequency, h_amplitude, tita_amplitude = frequency_ref, h_amplitude_ref, tita_amplitude_ref
                    
                    # Path for reference frequency set (single frequency cases, so use "1")
                    current_path_ref = "1" + path_base
                    current_path_ref = os.path.join(current_path_ref, "1_" + str(number), prescribed_motion)
                    path_CFD_results_ref = os.path.join(current_path_ref, "CFD_Results")
                    
                    aerodynamic_force_path_ref = os.path.join(path_CFD_results_ref, "Aerodynamic_Forces.dat")
                    aerodynamic_motion_path_ref = os.path.join(path_CFD_results_ref, "Aerodynamic_Input_Motion.dat")
                    
                    # Data Processing for reference
                    data_force_ref = np.loadtxt(aerodynamic_force_path_ref, skiprows=2)
                    data_force_ref = data_force_ref[n: n + time_interval_full]
                    time_ref, fx_ref, fy_ref, moment_ref = data_force_ref[:, 0], data_force_ref[:, 1], data_force_ref[:, 2], data_force_ref[:, 6]
                    
                    data_motion_ref = np.loadtxt(aerodynamic_motion_path_ref, skiprows=2)
                    data_motion_ref = data_motion_ref[n:n + time_interval_full]
                    v_ref, w_ref, theta_ref = data_motion_ref[:, 1], data_motion_ref[:, 2], data_motion_ref[:, 3]
                    
                    # FFT for reference
                    force_components_full_ref = decompose_signal(fy_ref, sample_frequency, top_count=500)
                    moment_components_full_ref = decompose_signal(moment_ref, sample_frequency, top_count=500)
                    
                    if prescribed_motion == "Heave":
                        motion_components_ref = decompose_signal(w_ref, sample_frequency, top_count=len(ref_frequency_set))
                    elif prescribed_motion == "Pitch":
                        motion_components_ref = decompose_signal(theta_ref, sample_frequency, top_count=len(ref_frequency_set))
                    
                    force_components_ref = force_components_full_ref[:10]
                    moment_components_ref = moment_components_full_ref[:10]
                    
                    freq_and_force_amp_ref = extract_peak_terms(force_components_ref, top_count=10, returndict=True)
                    freq_and_moment_amp_ref = extract_peak_terms(moment_components_ref, top_count=10, returndict=True)
                    
                    # Extracting phase and amplitude for reference
                    phi1_ref = extract_amplitude_phase(force_components_ref, target_frequency=frequency)["phase"]
                    phi2_ref = extract_amplitude_phase(motion_components_ref, target_frequency=frequency)["phase"]
                    phase_diff_Fy_ref = ((phi2_ref - phi1_ref) + np.pi) % (2 * np.pi) - np.pi
                    
                    F_ref = freq_and_force_amp_ref[frequency]
                    F_sine_ref = F_ref * np.cos(phase_diff_Fy_ref)
                    F_cos_ref = -F_ref * np.sin(phase_diff_Fy_ref)
                    
                    phi1_ref = extract_amplitude_phase(moment_components_ref, target_frequency=frequency)["phase"]
                    phi2_ref = extract_amplitude_phase(motion_components_ref, target_frequency=frequency)["phase"]
                    phase_diff_Mx_ref = ((phi2_ref - phi1_ref) + np.pi) % (2 * np.pi) - np.pi
                    
                    M_ref = freq_and_moment_amp_ref[frequency]
                    M_sine_ref = M_ref * np.cos(phase_diff_Mx_ref)
                    M_cos_ref = -M_ref * np.sin(phase_diff_Mx_ref)
                    
                    # Finding reference derivatives
                    row = 1.225
                    omega = 2 * np.pi * frequency
                    B = 0.366
                    
                    if prescribed_motion == "Heave":
                        H1_ref = 2 * F_cos_ref / (row*omega*omega*B*B*h_amplitude)
                        H4_ref = 2 * F_sine_ref / (row*omega*omega*B*B*h_amplitude)
                        A1_ref = 2 * M_cos_ref / (row*omega*omega*B*B*B*h_amplitude)
                        A4_ref = 2 * M_sine_ref / (row*omega*omega*B*B*B*h_amplitude)
                        
                        ref_derivatives[ref_key] = {
                            'H1': H1_ref,
                            'H4': H4_ref,
                            'A1': A1_ref,
                            'A4': A4_ref
                        }
                        print(f"__________Reference H1*: {H1_ref}, H4*: {H4_ref}, A1*: {A1_ref}, A4*: {A4_ref}")
                        
                    elif prescribed_motion == "Pitch":
                        H2_ref = 2 * F_cos_ref / (row*omega*omega*B*B*B*tita_amplitude)
                        H3_ref = 2 * F_sine_ref / (row*omega*omega*B*B*B*tita_amplitude)
                        A2_ref = 2 * M_cos_ref / (row*omega*omega*B*B*B*B*tita_amplitude)
                        A3_ref = 2 * M_sine_ref / (row*omega*omega*B*B*B*B*tita_amplitude)
                        
                        ref_derivatives[ref_key] = {
                            'H2': H2_ref,
                            'H3': H3_ref,
                            'A2': A2_ref,
                            'A3': A3_ref
                        }
                        print(f"__________Reference H2*: {H2_ref}, H3*: {H3_ref}, A2*: {A2_ref}, A3*: {A3_ref}")
      
                H1_list = []
                H4_list = []
                A1_list = []
                A4_list = []

                H2_list = []
                H3_list = []
                A2_list = []
                A3_list = []

                H1_list_error = []
                H4_list_error = []
                A1_list_error = []
                A4_list_error = []

                H2_list_error = []
                H3_list_error = []
                A2_list_error = []
                A3_list_error = []

                new_time_interval_list = []
                print("__________Total time_interval", time_interval_list)

                for time_interval in time_interval_list:
                    
                    H1_list_ref = []
                    H4_list_ref = []
                    A1_list_ref = []
                    A4_list_ref = []

                    H2_list_ref = []
                    H3_list_ref = []
                    A2_list_ref = []
                    A3_list_ref = []

                    for ref_or_not in range(1,3):

                        if ref_or_not == 1:
                            frequency, h_amplitude, tita_amplitude = frequency_ref, h_amplitude_ref, tita_amplitude_ref
                        if ref_or_not == 2:
                            frequency, h_amplitude, tita_amplitude = frequency_actual, h_amplitude_actual, tita_amplitude_actual

                        current_path = str(len(frequency_list)) + path_base
                        current_path = os.path.join(current_path, str(len(frequency_list)) + "_" +str(iteration_variable1), prescribed_motion)
                        path_CFD_results = os.path.join(current_path, "CFD_Results")
                        path_results = os.path.join(current_path)
                        os.makedirs(path_results, exist_ok=True)

                        aerodynamic_force_path = os.path.join(path_CFD_results , "Aerodynamic_Forces" +".dat")
                        aerodynamic_motion_path = os.path.join(path_CFD_results , "Aerodynamic_Input_Motion.dat")
                        print(current_path)
                        print("__________Current Time Interval:", time_interval)

                        #Data Processing
                        data_force = np.loadtxt(aerodynamic_force_path, skiprows=2)  # Read data
                        if len(data_force) < n + time_interval:
                            print(f"__________STOP {time_interval} due to insufficient data length.")
                            break
                        
                        data_force = data_force[n: n + time_interval]
                        time, fx, fy, moment = data_force[:, 0], data_force[:, 1], data_force[:, 2], data_force[:, 6]  # Forces
                        data_motion = np.loadtxt(aerodynamic_motion_path, skiprows = 2)
                        data_motion = data_motion[n:n + time_interval]
                        v, w, theta = data_motion[:, 1], data_motion[:, 2], data_motion[:, 3]  # Displacement
                        print("__________Length of list", len(fy), len(w))


                        #FFT
                        force_components_full = decompose_signal(fy, sample_frequency, top_count=500)
                        moment_components_full = decompose_signal(moment, sample_frequency, top_count=500)

                        if prescribed_motion == "Heave":
                            motion_components = decompose_signal(w, sample_frequency, top_count=len(frequency_list))
                        elif prescribed_motion == "Pitch":
                            motion_components = decompose_signal(theta, sample_frequency, top_count=len(frequency_list))

                        # Reuse computed components - just slice the list
                        force_components = force_components_full[:10]  # Top 10 for analysis
                        moment_components = moment_components_full[:10]
                        
                        freq_and_force_amp = extract_peak_terms(force_components, top_count=10, returndict=True)
                        freq_and_moment_amp = extract_peak_terms(moment_components, top_count=10, returndict=True)

                        if frequency not in freq_and_force_amp:
                            print(f"__________Frequency {frequency} Hz not found in force components, skipping...")
                            continue

                        #Extracting phase and amplitude
                        phi1 = extract_amplitude_phase(force_components, target_frequency=frequency)["phase"]
                        phi2 = extract_amplitude_phase(motion_components, target_frequency=frequency)["phase"]
                        phase_diff_Fy = ((phi2 - phi1) + np.pi) % (2 * np.pi) - np.pi
                        #print("__________Relative phase difference (fy w.r.t v):", phase_diff_Fy)

                        F = freq_and_force_amp[frequency] 
                        F_sine = F * np.cos(phase_diff_Fy)
                        F_cos = -F * np.sin(phase_diff_Fy)

                        
                        #Extracting phase and amplitude
                        phi1 = extract_amplitude_phase(moment_components, target_frequency=frequency)["phase"]
                        phi2 = extract_amplitude_phase(motion_components, target_frequency=frequency)["phase"]
                        phase_diff_Mx = ((phi2 - phi1) + np.pi) % (2 * np.pi) - np.pi
                        #print("__________Relative phase difference (Mx w.r.t v):", phase_diff_Mx)

                        N = len(fy)
                        fs = sample_frequency
                        freqs = np.fft.fftfreq(N, 1/fs)

                        if frequency not in freq_and_moment_amp:
                            print(f"__________Frequency {frequency} Hz not found in moment components, skipping...")
                            continue
                        if ref_or_not == 2:
                            new_time_interval_list.append(time_interval)
                        
                        M = freq_and_moment_amp[frequency] 
                        M_sine = M * np.cos(phase_diff_Mx)
                        M_cos = -M * np.sin(phase_diff_Mx)

                        #Finding derivatives
                        row = 1.225
                        omega = 2 * np.pi * frequency 
                        B = 0.366
                        U_reduced = wind_velocity / (frequency * B)

                        if prescribed_motion == "Heave":
                            H1 = 2 * F_cos / (row*omega*omega*B*B*h_amplitude)
                            H4 = 2 * F_sine / (row*omega*omega*B*B*h_amplitude)
                            A1 = 2 * M_cos / (row*omega*omega*B*B*B*h_amplitude)
                            A4 = 2 * M_sine / (row*omega*omega*B*B*B*h_amplitude)
                            
                            if ref_or_not == 1:
                                H1_list_ref.append(H1)
                                H4_list_ref.append(H4)
                                A1_list_ref.append(A1)
                                A4_list_ref.append(A4)

                            if ref_or_not == 2:
                                H1_list.append(H1)
                                H4_list.append(H4)
                                A1_list.append(A1)
                                A4_list.append(A4)

                                # Get global reference values
                                ref_vals = ref_derivatives[ref_key]
                                H1_ref_global = ref_vals['H1']
                                H4_ref_global = ref_vals['H4']
                                A1_ref_global = ref_vals['A1']
                                A4_ref_global = ref_vals['A4']
                                
                                # Calculate error using global reference
                                H1_list_error.append((H1 - H1_ref_global) / H1_ref_global * 100)
                                H4_list_error.append((H4 - H4_ref_global) / H4_ref_global * 100)
                                A1_list_error.append((A1 - A1_ref_global) / A1_ref_global * 100)
                                A4_list_error.append((A4 - A4_ref_global) / A4_ref_global * 100)
                                print("__________H1 Error:", H1_list_error[-1], "H1:", H1, "H1_ref_global:", H1_ref_global)
                            # print("__________Frequency",frequency ,"hz      H1*", H1)
                            # print("__________Frequency",frequency ,"hz      H4*", H4)
                            # print("__________Frequency",frequency ,"hz      A1*", A1)
                            # print("__________Frequency",frequency ,"hz      A4*", A4)

                        elif prescribed_motion == "Pitch":
                            H2 = 2 * F_cos / (row*omega*omega*B*B*B*tita_amplitude)
                            H3 = 2 * F_sine / (row*omega*omega*B*B*B*tita_amplitude)
                            A2 = 2 * M_cos / (row*omega*omega*B*B*B*B*tita_amplitude)
                            A3 = 2 * M_sine / (row*omega*omega*B*B*B*B*tita_amplitude)

                            if ref_or_not == 1:
                                H2_list_ref.append(H2)
                                H3_list_ref.append(H3)
                                A2_list_ref.append(A2)
                                A3_list_ref.append(A3)

                            if ref_or_not == 2:
                                H2_list.append(H2)
                                H3_list.append(H3)
                                A2_list.append(A2)
                                A3_list.append(A3)

                                # Get global reference values
                                ref_vals = ref_derivatives[ref_key]
                                H2_ref_global = ref_vals['H2']
                                H3_ref_global = ref_vals['H3']
                                A2_ref_global = ref_vals['A2']
                                A3_ref_global = ref_vals['A3']
                                
                                # Calculate error using global reference
                                H2_list_error.append((H2 - H2_ref_global) / H2_ref_global * 100)
                                H3_list_error.append((H3 - H3_ref_global) / H3_ref_global * 100)
                                A2_list_error.append((A2 - A2_ref_global) / A2_ref_global * 100)
                                A3_list_error.append((A3 - A3_ref_global) / A3_ref_global * 100)

                            # print("__________Frequency",frequency ,"hz      H2*", H2)
                            # print("__________Frequency",frequency ,"hz      H3*", H3)
                            # print("__________Frequency",frequency ,"hz      A2*", A2)
                            # print("__________Frequency",frequency ,"hz      A3*", A3)

                    # Store or compute reference derivatives from ref_frequency_set
                    path_time_step_convergence = os.path.join(path_results, "Time_Step_Convergence")
                    os.makedirs(path_time_step_convergence, exist_ok=True)

                    if prescribed_motion == "Heave":
                        Derivatives = [
                            (H1_list, "H1*", "H1* Convergence Study", str(iteration_variable3) + "H1_TimeStep_Convergence.png"),
                            (H4_list, "H4*", "H4* Convergence Study", str(iteration_variable3) + "H4_TimeStep_Convergence.png"),
                            (A1_list, "A1*", "A1* Convergence Study", str(iteration_variable3) + "A1_TimeStep_Convergence.png"),
                            (A4_list, "A4*", "A4* Convergence Study", str(iteration_variable3) + "A4_TimeStep_Convergence.png"),
                        ]
                    elif prescribed_motion == "Pitch":
                        Derivatives = [
                            (H2_list, "H2*", "H2* Convergence Study", str(iteration_variable3) + "H2_TimeStep_Convergence.png"),
                            (H3_list, "H3*", "H3* Convergence Study", str(iteration_variable3) + "H3_TimeStep_Convergence.png"),
                            (A2_list, "A2*", "A2* Convergence Study", str(iteration_variable3) + "A2_TimeStep_Convergence.png"),
                            (A3_list, "A3*", "A3* Convergence Study", str(iteration_variable3) + "A3_TimeStep_Convergence.png"),
                        ]


                    if prescribed_motion == "Heave":
                        Errors = [
                            (H1_list_error, "H1* Error (%)", "H1* Error Convergence Study", str(iteration_variable3) + "H1_Error_TimeStep_Convergence.png"),
                            (H4_list_error, "H4* Error (%)", "H4* Error Convergence Study", str(iteration_variable3) + "H4_Error_TimeStep_Convergence.png"),
                            (A1_list_error, "A1* Error (%)", "A1* Error Convergence Study", str(iteration_variable3) + "A1_Error_TimeStep_Convergence.png"),
                            (A4_list_error, "A4* Error (%)", "A4* Error Convergence Study", str(iteration_variable3) + "A4_Error_TimeStep_Convergence.png"),
                        ]
                    elif prescribed_motion == "Pitch":
                        Errors = [
                            (H2_list_error, "H2* Error (%)", "H2* Error Convergence Study", str(iteration_variable3) + "H2_Error_TimeStep_Convergence.png"),
                            (H3_list_error, "H3* Error (%)", "H3* Error Convergence Study", str(iteration_variable3) + "H3_Error_TimeStep_Convergence.png"),
                            (A2_list_error, "A2* Error (%)", "A2* Error Convergence Study", str(iteration_variable3) + "A2_Error_TimeStep_Convergence.png"),
                            (A3_list_error, "A3* Error (%)", "A3* Error Convergence Study", str(iteration_variable3) + "A3_Error_TimeStep_Convergence.png"),
                        ]
                
                for y_list, y_label, title, filename in Derivatives:
                    save_path = os.path.join(path_time_step_convergence, filename)
                    plot_time_interval_convergence(new_time_interval_list, y_list, y_label, title, save_path)

                for y_list, y_label, title, filename in Errors:
                    save_path = os.path.join(path_time_step_convergence, filename)
                    plot_time_interval_convergence(new_time_interval_list, y_list, y_label, title, save_path)
                iteration_variable3 += 1
            iteration_variable1 += 1
        iteration_variable2 += 1

        
"""





"""

path_base = "_ConstantAmplitude"
wind_velocity = 1
n = 7320 #number of initial data points to skip
sample_frequency = 500

frequency = [
             #frequency_set_1, 
             frequency_set_2, 
             frequency_set_3,
             frequency_set_4,
             ]

iteration_variable2 = 0

ref_frequency_set = frequency_set_1

# Dictionary to store reference derivatives
ref_derivatives = {}

for freq in frequency:

    prescribed_motions = ["Heave", "Pitch"]

    for prescribed_motion in prescribed_motions:
    
        iteration_variable1 = 1
        for frequency_list in freq:

            iteration_variable3 = 1
            for frequency_tuple in frequency_list:

                frequency_actual, h_amplitude_actual, tita_amplitude_actual, number = frequency_tuple
                frequency_ref, h_amplitude_ref, tita_amplitude_ref, _ = frequency_set_1[number - 1][0]

                time_interval_list = time_intervals(sample_frequency, frequency_set_1[number - 1][0][0]) 
      
                H1_list = []
                H4_list = []
                A1_list = []
                A4_list = []

                H2_list = []
                H3_list = []
                A2_list = []
                A3_list = []

                H1_list_error = []
                H4_list_error = []
                A1_list_error = []
                A4_list_error = []

                H2_list_error = []
                H3_list_error = []
                A2_list_error = []
                A3_list_error = []

                new_time_interval_list = []
                print("__________Total time_interval", time_interval_list)

                for time_interval in time_interval_list:
                    
                    H1_list_ref = []
                    H4_list_ref = []
                    A1_list_ref = []
                    A4_list_ref = []

                    H2_list_ref = []
                    H3_list_ref = []
                    A2_list_ref = []
                    A3_list_ref = []

                    for ref_or_not in range(1,3):

                        if ref_or_not == 1:
                            frequency, h_amplitude, tita_amplitude = frequency_ref, h_amplitude_ref, tita_amplitude_ref
                        if ref_or_not == 2:
                            frequency, h_amplitude, tita_amplitude = frequency_actual, h_amplitude_actual, tita_amplitude_actual

                        current_path = str(len(frequency_list)) + path_base
                        current_path = os.path.join(current_path, str(len(frequency_list)) + "_" +str(iteration_variable1), prescribed_motion)
                        path_CFD_results = os.path.join(current_path, "CFD_Results")
                        path_results = os.path.join(current_path)
                        os.makedirs(path_results, exist_ok=True)

                        aerodynamic_force_path = os.path.join(path_CFD_results , "Aerodynamic_Forces" +".dat")
                        aerodynamic_motion_path = os.path.join(path_CFD_results , "Aerodynamic_Input_Motion.dat")
                        print(current_path)
                        print("__________Current Time Interval:", time_interval)

                        #Data Processing
                        data_force = np.loadtxt(aerodynamic_force_path, skiprows=2)  # Read data
                        if len(data_force) < n + time_interval:
                            print(f"__________STOP {time_interval} due to insufficient data length.")
                            break
                        
                        data_force = data_force[n: n + time_interval]
                        time, fx, fy, moment = data_force[:, 0], data_force[:, 1], data_force[:, 2], data_force[:, 6]  # Forces
                        data_motion = np.loadtxt(aerodynamic_motion_path, skiprows = 2)
                        data_motion = data_motion[n:n + time_interval]
                        v, w, theta = data_motion[:, 1], data_motion[:, 2], data_motion[:, 3]  # Displacement
                        print("__________Length of list", len(fy), len(w))


                        #FFT
                        force_components_full = decompose_signal(fy, sample_frequency, top_count=500)
                        moment_components_full = decompose_signal(moment, sample_frequency, top_count=500)

                        if prescribed_motion == "Heave":
                            motion_components = decompose_signal(w, sample_frequency, top_count=len(frequency_list))
                        elif prescribed_motion == "Pitch":
                            motion_components = decompose_signal(theta, sample_frequency, top_count=len(frequency_list))

                        # Reuse computed components - just slice the list
                        force_components = force_components_full[:10]  # Top 10 for analysis
                        moment_components = moment_components_full[:10]
                        
                        freq_and_force_amp = extract_peak_terms(force_components, top_count=10, returndict=True)
                        freq_and_moment_amp = extract_peak_terms(moment_components, top_count=10, returndict=True)

                        if frequency not in freq_and_force_amp:
                            print(f"__________Frequency {frequency} Hz not found in force components, skipping...")
                            continue

                        #Extracting phase and amplitude
                        phi1 = extract_amplitude_phase(force_components, target_frequency=frequency)["phase"]
                        phi2 = extract_amplitude_phase(motion_components, target_frequency=frequency)["phase"]
                        phase_diff_Fy = ((phi2 - phi1) + np.pi) % (2 * np.pi) - np.pi
                        #print("__________Relative phase difference (fy w.r.t v):", phase_diff_Fy)

                        F = freq_and_force_amp[frequency] 
                        F_sine = F * np.cos(phase_diff_Fy)
                        F_cos = -F * np.sin(phase_diff_Fy)

                        
                        #Extracting phase and amplitude
                        phi1 = extract_amplitude_phase(moment_components, target_frequency=frequency)["phase"]
                        phi2 = extract_amplitude_phase(motion_components, target_frequency=frequency)["phase"]
                        phase_diff_Mx = ((phi2 - phi1) + np.pi) % (2 * np.pi) - np.pi
                        #print("__________Relative phase difference (Mx w.r.t v):", phase_diff_Mx)

                        N = len(fy)
                        fs = sample_frequency
                        freqs = np.fft.fftfreq(N, 1/fs)

                        if frequency not in freq_and_moment_amp:
                            print(f"__________Frequency {frequency} Hz not found in moment components, skipping...")
                            continue
                        if ref_or_not == 2:
                            new_time_interval_list.append(time_interval)
                        
                        M = freq_and_moment_amp[frequency] 
                        M_sine = M * np.cos(phase_diff_Mx)
                        M_cos = -M * np.sin(phase_diff_Mx)

                        #Finding derivatives
                        row = 1.225
                        omega = 2 * np.pi * frequency 
                        B = 0.366
                        U_reduced = wind_velocity / (frequency * B)

                        if prescribed_motion == "Heave":
                            H1 = 2 * F_cos / (row*omega*omega*B*B*h_amplitude)
                            H4 = 2 * F_sine / (row*omega*omega*B*B*h_amplitude)
                            A1 = 2 * M_cos / (row*omega*omega*B*B*B*h_amplitude)
                            A4 = 2 * M_sine / (row*omega*omega*B*B*B*h_amplitude)
                            
                            if ref_or_not == 1:
                                H1_list_ref.append(H1)
                                H4_list_ref.append(H4)
                                A1_list_ref.append(A1)
                                A4_list_ref.append(A4)

                            if ref_or_not == 2:
                                H1_list.append(H1)
                                H4_list.append(H4)
                                A1_list.append(A1)
                                A4_list.append(A4)

                                H1_list_error.append((H1 - H1_list_ref[0]) / H1_list_ref[0] * 100)
                                H4_list_error.append((H4 - H4_list_ref[0]) / H4_list_ref[0] * 100)
                                A1_list_error.append((A1 - A1_list_ref[0]) / A1_list_ref[0] * 100)
                                A4_list_error.append((A4 - A4_list_ref[0]) / A4_list_ref[0] * 100)
                                print(H1_list_error, H1, H1_list_ref[0], H1 - H1_list_ref[0])
                            # print("__________Frequency",frequency ,"hz      H1*", H1)
                            # print("__________Frequency",frequency ,"hz      H4*", H4)
                            # print("__________Frequency",frequency ,"hz      A1*", A1)
                            # print("__________Frequency",frequency ,"hz      A4*", A4)

                        elif prescribed_motion == "Pitch":
                            H2 = 2 * F_cos / (row*omega*omega*B*B*B*tita_amplitude)
                            H3 = 2 * F_sine / (row*omega*omega*B*B*B*tita_amplitude)
                            A2 = 2 * M_cos / (row*omega*omega*B*B*B*B*tita_amplitude)
                            A3 = 2 * M_sine / (row*omega*omega*B*B*B*B*tita_amplitude)

                            if ref_or_not == 1:
                                H2_list_ref.append(H2)
                                H3_list_ref.append(H3)
                                A2_list_ref.append(A2)
                                A3_list_ref.append(A3)

                            if ref_or_not == 2:
                                H2_list.append(H2)
                                H3_list.append(H3)
                                A2_list.append(A2)
                                A3_list.append(A3)

                                H2_list_error.append((H2 - H2_list_ref[0]) / H2_list_ref[0] * 100)
                                H3_list_error.append((H3 - H3_list_ref[0]) / H3_list_ref[0] * 100)
                                A2_list_error.append((A2 - A2_list_ref[0]) / A2_list_ref[0] * 100)
                                A3_list_error.append((A3 - A3_list_ref[0]) / A3_list_ref[0] * 100)

                            # print("__________Frequency",frequency ,"hz      H2*", H2)
                            # print("__________Frequency",frequency ,"hz      H3*", H3)
                            # print("__________Frequency",frequency ,"hz      A2*", A2)
                            # print("__________Frequency",frequency ,"hz      A3*", A3)

                    # Store or compute reference derivatives from ref_frequency_set
                    path_time_step_convergence = os.path.join(path_results, "Time_Step_Convergence")
                    os.makedirs(path_time_step_convergence, exist_ok=True)

                    if prescribed_motion == "Heave":
                        Derivatives = [
                            (H1_list, "H1*", "H1* Convergence Study", str(iteration_variable3) + "H1_TimeStep_Convergence.png"),
                            (H4_list, "H4*", "H4* Convergence Study", str(iteration_variable3) + "H4_TimeStep_Convergence.png"),
                            (A1_list, "A1*", "A1* Convergence Study", str(iteration_variable3) + "A1_TimeStep_Convergence.png"),
                            (A4_list, "A4*", "A4* Convergence Study", str(iteration_variable3) + "A4_TimeStep_Convergence.png"),
                        ]
                    elif prescribed_motion == "Pitch":
                        Derivatives = [
                            (H2_list, "H2*", "H2* Convergence Study", str(iteration_variable3) + "H2_TimeStep_Convergence.png"),
                            (H3_list, "H3*", "H3* Convergence Study", str(iteration_variable3) + "H3_TimeStep_Convergence.png"),
                            (A2_list, "A2*", "A2* Convergence Study", str(iteration_variable3) + "A2_TimeStep_Convergence.png"),
                            (A3_list, "A3*", "A3* Convergence Study", str(iteration_variable3) + "A3_TimeStep_Convergence.png"),
                        ]


                    if prescribed_motion == "Heave":
                        Errors = [
                            (H1_list_error, "H1*", "H1* Error Convergence Study", str(iteration_variable3) + "H1_Error_TimeStep_Convergence.png"),
                            (H4_list_error, "H4*", "H4* Error Convergence Study", str(iteration_variable3) + "H4_Error_TimeStep_Convergence.png"),
                            (A1_list_error, "A1*", "A1* Error Convergence Study", str(iteration_variable3) + "A1_Error_TimeStep_Convergence.png"),
                            (A4_list_error, "A4*", "A4* Error Convergence Study", str(iteration_variable3) + "A4_Error_TimeStep_Convergence.png"),
                        ]
                    elif prescribed_motion == "Pitch":
                        Errors = [
                            (H2_list_error, "H2*", "H2* Error Convergence Study", str(iteration_variable3) + "H2_Error_TimeStep_Convergence.png"),
                            (H3_list_error, "H3*", "H3* Error Convergence Study", str(iteration_variable3) + "H3_Error_TimeStep_Convergence.png"),
                            (A2_list_error, "A2*", "A2* Error Convergence Study", str(iteration_variable3) + "A2_Error_TimeStep_Convergence.png"),
                            (A3_list_error, "A3*", "A3* Error Convergence Study", str(iteration_variable3) + "A3_Error_TimeStep_Convergence.png"),
                        ]
                
                for y_list, y_label, title, filename in Derivatives:
                    save_path = os.path.join(path_time_step_convergence, filename)
                    plot_time_interval_convergence(new_time_interval_list, y_list, y_label, title, save_path)

                for y_list, y_label, title, filename in Errors:
                    save_path = os.path.join(path_time_step_convergence, filename)
                    plot_time_interval_convergence(new_time_interval_list, y_list, y_label, title, save_path)
                iteration_variable3 += 1
            iteration_variable1 += 1
        iteration_variable2 += 1
"""