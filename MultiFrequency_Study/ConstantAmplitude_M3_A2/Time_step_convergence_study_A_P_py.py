# -*- coding: utf-8 -*-
"""
Modified version: Instead of flutter derivatives, we now extract, save, and plot
AMPLITUDES and RELATIVE PHASES of lift (fy), moment, and motion at the target frequency.
Reference values are taken from the single-frequency long-window runs.
Convergence is shown as:
- Amplitude convergence (absolute values + % error vs reference)
- Relative phase convergence (in radians + absolute difference vs reference)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd
import shutil
from fractions import Fraction

from MultiFrequency_Data import *   # contains frequency_set_1, _2, _3, _4

# ========================== KEEP THESE HELPERS ==========================
def decompose_signal(signal, sample_rate, top_count, threshold=0.00):
    N = len(signal)
    fft_result = fft(signal)
    frequencies = fftfreq(N, 1 / sample_rate)
    positive_mask = frequencies >= 0
    pos_freqs = frequencies[positive_mask]
    pos_fft = fft_result[positive_mask]
    amplitudes = np.abs(pos_fft) / N * 2
    above_threshold = amplitudes > threshold
    filtered_freqs = pos_freqs[above_threshold]
    filtered_amps = amplitudes[above_threshold]
    filtered_phases = np.angle(pos_fft[above_threshold])
    top_indices = np.argsort(filtered_amps)[-top_count:][::-1]
    t = np.arange(N) / sample_rate
    sine_components = []
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

def extract_amplitude_phase(components, target_frequency, tolerance=1e-2):
    for component in components:
        if abs(component['frequency'] - target_frequency) < tolerance:
            return {
                'amplitude': component['amplitude'],
                'phase': component['phase']
            }
    return None

def get_time_intervals_for_frequencies(frequency_list, sample_frequency, max_time_interval=40000):
    epsilon = 1e-10
    valid_lengths = []
    for L in range(1, max_time_interval + 1):
        is_valid = True
        for f in frequency_list:
            n = f[0] * L / sample_frequency
            if abs(n - round(n)) > epsilon:
                is_valid = False
                break
        if is_valid:
            valid_lengths.append(L)
    return sorted(valid_lengths)

def plot_convergence(time_intervals, values, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(time_intervals, values, marker='o', linewidth=2)
    plt.xlabel('Time Interval', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def write_amplitude_phase_table(save_path, time_intervals, values, ref_values, motion_type):
    # values and ref_values are dicts with keys: amp_motion, amp_fy, amp_moment, phase_fy_rel, phase_moment_rel
    table_data = []
    for i, t_int in enumerate(time_intervals):
        row = {'Time_Interval': t_int}
        for k in ['amp_fy', 'amp_moment', 'phase_fy_rel', 'phase_moment_rel']:
            ref = ref_values[k]
            cur = values[k][i]
            if k.startswith('amp'):
                err = (cur - ref) / ref * 100 if ref != 0 else np.nan
                row[f'{k}_Ref'] = ref
                row[f'{k}_Cur'] = cur
                row[f'{k}_Error_%'] = err
            else:  # phase
                err = ((cur - ref + np.pi) % (2*np.pi) - np.pi) * 180 / np.pi   # error in degrees
                row[f'{k}_Ref_rad'] = ref
                row[f'{k}_Cur_rad'] = cur
                row[f'{k}_Error_deg'] = err
        table_data.append(row)

    df = pd.DataFrame(table_data)
    excel_path = save_path.replace('.txt', '.xlsx')
    df.to_excel(excel_path, index=False, float_format='%.6f')

    with open(save_path, 'w') as f:
        f.write(f"Amplitude & Phase Convergence - {motion_type}\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"{'Time':>8} {'Fy_Amp_Ref':>12} {'Fy_Amp':>12} {'Fy_Err_%':>10} "
                f"{'Mom_Amp_Ref':>12} {'Mom_Amp':>12} {'Mom_Err_%':>10} "
                f"{'Fy_Ph_Ref':>10} {'Fy_Ph':>10} {'Fy_Ph_Err°':>10} "
                f"{'Mom_Ph_Ref':>10} {'Mom_Ph':>10} {'Mom_Ph_Err°':>10}\n")
        f.write("-" * 120 + "\n")
        for row in table_data:
            f.write(f"{row['Time_Interval']:8d} "
                    f"{row['amp_fy_Ref']:12.6f} {row['amp_fy_Cur']:12.6f} {row['amp_fy_Error_%']:10.3f} "
                    f"{row['amp_moment_Ref']:12.6f} {row['amp_moment_Cur']:12.6f} {row['amp_moment_Error_%']:10.3f} "
                    f"{row['phase_fy_rel_Ref_rad']:10.4f} {row['phase_fy_rel_Cur_rad']:10.4f} {row['phase_fy_rel_Error_deg']:10.3f} "
                    f"{row['phase_moment_rel_Ref_rad']:10.4f} {row['phase_moment_rel_Cur_rad']:10.4f} {row['phase_moment_rel_Error_deg']:10.3f}\n")
    print(f"  Saved table: {excel_path} and {save_path}")

# ========================== CONFIG ==========================
path_base = "_ConstantAmplitude"
n = 7320
sample_frequency = 500

frequency_sets = [frequency_set_1, frequency_set_2, frequency_set_3, frequency_set_4]

# Global reference storage: key = (prescribed_motion, number) → dict of amp/phase
ref_signal_info = {}

# For combined error plots
combined_errors = {
    'Heave': {
        'Fy_Amp_%': {}, 'Moment_Amp_%': {}, 'Fy_Phase_deg': {}, 'Moment_Phase_deg': {}
    },
    'Pitch': {
        'Fy_Amp_%': {}, 'Moment_Amp_%': {}, 'Fy_Phase_deg': {}, 'Moment_Phase_deg': {}
    }
}

# ========================== MAIN LOOP ==========================
for set_idx, freq_set in enumerate(frequency_sets):
    for prescribed_motion in ["Heave", "Pitch"]:
        for freq_idx, frequency_list in enumerate(freq_set, 1):
            for tuple_idx, freq_tuple in enumerate(frequency_list, 1):
                freq_actual, h_amp_actual, tita_amp_actual, number = freq_tuple
                freq_ref, h_amp_ref, tita_amp_ref, _ = frequency_set_1[number - 1][0]

                ref_key = (prescribed_motion, number)

                # ------------------- COMPUTE REFERENCE (single-freq long window) -------------------
                if ref_key not in ref_signal_info:
                    print(f"Computing reference for {prescribed_motion}, freq index {number}")
                    time_interval_list_ref = get_time_intervals_for_frequencies(
                        frequency_set_1[number - 1], sample_frequency, max_time_interval=40000
                    )
                    ref_path = os.path.join("1" + path_base, f"1_{number}", prescribed_motion, "CFD_Results")
                    force_path = os.path.join(ref_path, "Aerodynamic_Forces.dat")
                    motion_path = os.path.join(ref_path, "Aerodynamic_Input_Motion.dat")

                    data_force = np.loadtxt(force_path, skiprows=2)
                    available = len(data_force) - n
                    valid_intervals = [t for t in time_interval_list_ref if t <= available]
                    if not valid_intervals:
                        print("  No valid reference interval")
                        continue
                    t_full = valid_intervals[-1]

                    # Extract components for reference
                    data_force = data_force[n:n + t_full]
                    fy = data_force[:, 2]
                    moment = data_force[:, 6]
                    data_motion = np.loadtxt(motion_path, skiprows=2)[n:n + t_full]
                    motion_sig = data_motion[:, 2] if prescribed_motion == "Heave" else data_motion[:, 3]

                    fy_comp = decompose_signal(fy, sample_frequency, top_count=10)
                    mom_comp = decompose_signal(moment, sample_frequency, top_count=10)
                    mot_comp = decompose_signal(motion_sig, sample_frequency, top_count=1)

                    mot_info = extract_amplitude_phase(mot_comp, freq_ref)
                    fy_info = extract_amplitude_phase(fy_comp, freq_ref)
                    mom_info = extract_amplitude_phase(mom_comp, freq_ref)

                    if mot_info is None or fy_info is None or mom_info is None:
                        print("  Could not extract reference components")
                        continue

                    phase_fy = ((fy_info['phase'] - mot_info['phase'] + np.pi) % (2 * np.pi) - np.pi)
                    phase_mom = ((mom_info['phase'] - mot_info['phase'] + np.pi) % (2 * np.pi) - np.pi)

                    ref_signal_info[ref_key] = {
                        'amp_motion': mot_info['amplitude'],
                        'amp_fy': fy_info['amplitude'],
                        'amp_moment': mom_info['amplitude'],
                        'phase_fy_rel': phase_fy,
                        'phase_moment_rel': phase_mom
                    }
                    print(f"  Reference: Fy amp={fy_info['amplitude']:.5f}, Mom amp={mom_info['amplitude']:.5f}, "
                          f"Phase Fy={phase_fy:.4f} rad")

                # ------------------- CURRENT MULTI-FREQUENCY CASE -------------------
                time_interval_list = get_time_intervals_for_frequencies(
                    frequency_list, sample_frequency, max_time_interval=40000
                )

                current_path_check = os.path.join(f"{len(frequency_list)}{path_base}",
                                                  f"{len(frequency_list)}_{freq_idx}",
                                                  prescribed_motion, "CFD_Results")
                try:
                    data_check = np.loadtxt(os.path.join(current_path_check, "Aerodynamic_Forces.dat"), skiprows=2)
                    avail = len(data_check) - n
                    time_interval_list = [t for t in time_interval_list if t <= avail]
                except Exception as e:
                    print(f"  Could not read data: {e}")
                    continue

                print(f"\nProcessing Set {set_idx+1}, {prescribed_motion}, FreqList {freq_idx}, Freq {freq_actual:.3f} Hz")
                print(f"  Valid intervals: {time_interval_list}")

                # Storage for this case
                values = {k: [] for k in ['amp_fy', 'amp_moment', 'phase_fy_rel', 'phase_moment_rel']}
                valid_time_intervals = []

                for t_int in time_interval_list:
                    current_path = os.path.join(f"{len(frequency_list)}{path_base}",
                                                f"{len(frequency_list)}_{freq_idx}",
                                                prescribed_motion, "CFD_Results")
                    force_path = os.path.join(current_path, "Aerodynamic_Forces.dat")
                    motion_path = os.path.join(current_path, "Aerodynamic_Input_Motion.dat")

                    data_force = np.loadtxt(force_path, skiprows=2)
                    if len(data_force) < n + t_int:
                        break
                    data_force = data_force[n:n + t_int]
                    fy = data_force[:, 2]
                    moment = data_force[:, 6]
                    data_motion = np.loadtxt(motion_path, skiprows=2)[n:n + t_int]
                    motion_sig = data_motion[:, 2] if prescribed_motion == "Heave" else data_motion[:, 3]

                    fy_comp = decompose_signal(fy, sample_frequency, top_count=10)
                    mom_comp = decompose_signal(moment, sample_frequency, top_count=10)
                    mot_comp = decompose_signal(motion_sig, sample_frequency, top_count=len(frequency_list))

                    mot_info = extract_amplitude_phase(mot_comp, freq_actual)
                    fy_info = extract_amplitude_phase(fy_comp, freq_actual)
                    mom_info = extract_amplitude_phase(mom_comp, freq_actual)

                    if any(x is None for x in (mot_info, fy_info, mom_info)):
                        print(f"  Freq {freq_actual:.3f} not found at T={t_int}")
                        continue

                    phase_fy_rel = ((fy_info['phase'] - mot_info['phase'] + np.pi) % (2 * np.pi) - np.pi)
                    phase_mom_rel = ((mom_info['phase'] - mot_info['phase'] + np.pi) % (2 * np.pi) - np.pi)

                    valid_time_intervals.append(t_int)
                    values['amp_fy'].append(fy_info['amplitude'])
                    values['amp_moment'].append(mom_info['amplitude'])
                    values['phase_fy_rel'].append(phase_fy_rel)
                    values['phase_moment_rel'].append(phase_mom_rel)

                if not valid_time_intervals:
                    continue

                # ------------------- SAVE TABLES & INDIVIDUAL PLOTS -------------------
                result_path = os.path.join(f"{len(frequency_list)}{path_base}",
                                           f"{len(frequency_list)}_{freq_idx}",
                                           prescribed_motion, "Time_Step_Convergence")
                os.makedirs(result_path, exist_ok=True)

                table_path = os.path.join(result_path, f"{tuple_idx}_AmpPhase_Table.txt")
                write_amplitude_phase_table(table_path, valid_time_intervals, values,
                                            ref_signal_info[ref_key], prescribed_motion)

                # Individual convergence plots
                for key, label in [('amp_fy', 'Lift Amplitude'), ('amp_moment', 'Moment Amplitude'),
                                   ('phase_fy_rel', 'Lift Phase rel. Motion [rad]'),
                                   ('phase_moment_rel', 'Moment Phase rel. Motion [rad]')]:
                    plot_convergence(valid_time_intervals, values[key],
                                     label, f"{label} – {freq_actual:.3f} Hz",
                                     os.path.join(result_path, f"{tuple_idx}_{key}_Convergence.png"))

                # Store for combined plots (error)
                ref = ref_signal_info[ref_key]
                for k, comb_key in [
                    ('amp_fy', 'Fy_Amp_%'),
                    ('amp_moment', 'Moment_Amp_%'),
                    ('phase_fy_rel', 'Fy_Phase_deg'),
                    ('phase_moment_rel', 'Moment_Phase_deg')
                ]:
                    if k.startswith('amp'):
                        errs = [(v - ref[k]) / ref[k] * 100 if ref[k] != 0 else np.nan for v in values[k]]
                    else:
                        errs = [((v - ref[k] + np.pi) % (2*np.pi) - np.pi) * 180 / np.pi for v in values[k]]

                    if freq_actual not in combined_errors[prescribed_motion][comb_key]:
                        combined_errors[prescribed_motion][comb_key][freq_actual] = []
                    combined_errors[prescribed_motion][comb_key][freq_actual].append(
                        (valid_time_intervals.copy(), errs, set_idx + 2)
                    )

# ========================== COMBINED PLOTS & EXCEL ==========================
base_path = "global_results_amp_phase"
if os.path.exists(base_path):
    shutil.rmtree(base_path)
os.makedirs(base_path, exist_ok=True)

for motion in ['Heave', 'Pitch']:
    motion_dir = os.path.join(base_path, motion)
    os.makedirs(motion_dir, exist_ok=True)

    for key, title_prefix in [
        ('Fy_Amp_%', 'Lift Amplitude Error %'),
        ('Moment_Amp_%', 'Moment Amplitude Error %'),
        ('Fy_Phase_deg', 'Lift Phase Error [°]'),
        ('Moment_Phase_deg', 'Moment Phase Error [°]')
    ]:
        for freq in sorted(combined_errors[motion][key].keys()):
            plt.figure(figsize=(10, 6))
            for tints, errs, setnum in combined_errors[motion][key][freq]:
                plt.plot(tints, errs, marker='o', label=f"Combination {setnum-1}")
            plt.xlabel('Time Interval')
            plt.ylabel(title_prefix)
            plt.title(f"{title_prefix} – Frequency {freq:.3f} Hz")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(motion_dir, f"{key.replace('%','')}_Freq_{freq:.3f}_Combined.png"), dpi=300)
            plt.close()

# Combined Excel (same structure as before, just different columns)
print("\nCreating combined Excel...")
for motion in ['Heave', 'Pitch']:
    motion_dir = os.path.join(base_path, motion)
    excel_path = os.path.join(motion_dir, f"{motion}_Combined_AmpPhase_Errors.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for key in ['Fy_Amp_%', 'Moment_Amp_%', 'Fy_Phase_deg', 'Moment_Phase_deg']:
            freq_data = {}
            for freq in sorted(combined_errors[motion][key].keys()):
                time_data = {}
                for tints, errs, setnum in combined_errors[motion][key][freq]:
                    for t, e in zip(tints, errs):
                        if t not in time_data:
                            time_data[t] = {}
                        time_data[t][f'Set_{setnum}'] = e
                if time_data:
                    df = pd.DataFrame.from_dict(time_data, orient='index')
                    df.index.name = 'Time_Interval'
                    df = df.sort_index()
                    df.columns = [f'Freq_{freq:.3f}_Hz_{col}' for col in df.columns]
                    freq_data[freq] = df
            if freq_data:
                df_all = pd.concat(freq_data.values(), axis=1)
                df_all.to_excel(writer, sheet_name=key.replace('%','_Err'), float_format='%.6f')

        # All data sheet
        all_rows = []
        for key in ['Fy_Amp_%', 'Moment_Amp_%', 'Fy_Phase_deg', 'Moment_Phase_deg']:
            for freq in sorted(combined_errors[motion][key].keys()):
                for tints, errs, setnum in combined_errors[motion][key][freq]:
                    for t, e in zip(tints, errs):
                        all_rows.append({
                            'Quantity': key,
                            'Frequency_Hz': freq,
                            'Time_Interval': t,
                            'Set': setnum,
                            'Value': e
                        })
        if all_rows:
            pd.DataFrame(all_rows).to_excel(writer, sheet_name='All_Data', index=False)

    print(f"  Saved: {excel_path}")

print("\nAll amplitude & phase convergence processing complete!")