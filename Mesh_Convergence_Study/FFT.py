# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:57:18 2025

@author: aakas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter


def decompose_signal(signal, sample_rate, threshold=0.00, top_count=2):
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

def plot_signal_and_components(signal, components, sample_rate):
    t = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(12, 8))
    
    # Plot original input signal
    plt.subplot(len(components) + 1, 1, 1)
    plt.plot(t, signal, label='Original Signal')
    plt.title('Original Input Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Plot each separated sine wave component
    for i, comp in enumerate(components, start=2):
        plt.subplot(len(components) + 1, 1, i)
        plt.plot(t, comp['waveform'], label=f'Component: {comp["frequency"]:.1f} Hz')
        plt.title(f'Separated Sine Wave Component at {comp["frequency"]:.1f} Hz')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()
    
def plot_amplitude_vs_frequency(components):
    """
    Plot amplitude spectrum (amplitude vs frequency) from FFT components.
    """
    frequencies = [comp['frequency'] for comp in components]
    amplitudes = [comp['amplitude'] for comp in components]

    plt.figure(figsize=(10, 5))
    plt.stem(frequencies, amplitudes, basefmt=" ")
    plt.title('Amplitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def extract_peak_terms(components, top_count, returndict = False):
    top_components = sorted(components, key=lambda x: x['amplitude'], reverse=True)[:top_count]
    frequ = []
    
    for i, comp in enumerate(top_components, start=1):
        frequ.append(comp['frequency'])
    frequ = [float(x) for x in frequ] 
    frequ.sort()
    
    freq_to_amp = {c['frequency']: c['amplitude'] for c in components}
    # Extract amplitudes for the desired frequencies
    amplitudes = [freq_to_amp[f] for f in frequ if f in freq_to_amp]
    
    if returndict == True:
        freq_to_amp = {f: freq_to_amp[f] for f in frequ if f in freq_to_amp}
        #print("Frequency and amplitude",freq_to_amp)
        freq_to_amp = dict(sorted(freq_to_amp.items()))
        return freq_to_amp
    return amplitudes


def extract_amplitude_phase(components, target_frequency, tolerance=1e-6):
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
    return None  # If no matching frequency is found
    

file_name_force = "Global_results_box1_forces.dat"
file_name_motion = "Global_results_box1_motion.dat"

n = 4074 # number of lines you want to remove from top
frequency = 2.34375
sample_frequency = 2500


#Data Processing
data_force = np.loadtxt(file_name_force, skiprows=2)  # Read data
data_force = data_force[n:len(data_force)]
time, fx, fy, moment = data_force[:, 0], data_force[:, 1], data_force[:, 2], data_force[:, 6]  # Forces
data_motion = np.loadtxt(file_name_motion, skiprows = 2)
data_motion = data_motion[n:len(data_motion)]
u, v, theta = data_motion[:, 1], data_motion[:, 2], data_motion[:, 3]  # Displacement
print("length of list", len(fy), len(v))


#FFT
Force_components = decompose_signal(fy, sample_frequency)
Displacement_components = decompose_signal(v, sample_frequency)

plot_signal_and_components(fy,Force_components,sample_frequency)
freq_and_force_amp = extract_peak_terms(Force_components, top_count=10, returndict=True)
#plot_amplitude_vs_frequency(Force_components)
print("Frequency and Force amplitude",freq_and_force_amp)

plot_signal_and_components(v,Displacement_components,sample_frequency)
freq_and_displacement_amp = extract_peak_terms(Displacement_components, top_count=10, returndict=True)
#plot_amplitude_vs_frequency(Displacement_components)
print("Frequency and Displacement amplitude",freq_and_displacement_amp)


#Extracting phase and amplitude
phi1 = extract_amplitude_phase(Force_components, target_frequency=frequency)["phase"]
phi2 = extract_amplitude_phase(Displacement_components, target_frequency=frequency)["phase"]
phase_diff = 0 #phi1 #((phi2 - phi1) + np.pi) % (2 * np.pi) - np.pi
print(np.degrees(phi1), np.degrees(phi2))
print("Relative phase difference (fy w.r.t v):", phase_diff)

N = len(fy)
fs = sample_frequency
freqs = np.fft.fftfreq(N, 1/fs)
f_dom = frequency 
A1 = freq_and_force_amp[f_dom] 
A_cos = A1 * np.cos(phase_diff)
A_sine = -A1 * np.sin(phase_diff)
print(f"A_cos (cos term) = {A_cos:.4f}")
print(f"A_sine (sin term) = {A_sine:.4f}")

t = time
x_recon = A_cos * np.cos(2*np.pi*f_dom*t) + A_sine * np.sin(2*np.pi*f_dom*t)
plt.figure(figsize=(8,4))
plt.plot(t, fy, label="Original fy")
plt.plot(t, x_recon, '--', label="Reconstructed fy (relative phase)")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Cosine + Sine reconstruction of fy relative to reference v")
plt.show()


#Finding derivatives
row = 1.225
omega = 2 * np.pi * f_dom 
B = 0.366
h = 0.0183

H1_Star = 2 * A_cos / (row*omega*omega*B*B*h)
H4_Star = 2 * A_sine / (row*omega*omega*B*B*h)

print("Frequency",f_dom,"hz      H1*", H1_Star)
print("Frequency",f_dom,"hz      H4*", H4_Star )







