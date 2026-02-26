import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

class SignalPhaseAnalyzer:
    """
    Analyzes two signals using FFT, matches their phases, and extracts 
    cosine and sine components.
    """
    
    def __init__(self, signal1, signal2, fs, freq):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        signal1 : array-like
            First signal (reference signal)
        signal2 : array-like
            Second signal (to be analyzed)
        fs : float
            Sampling frequency in Hz
        freq : float
            Frequency of interest in Hz
        """
        self.signal1 = np.array(signal1)
        self.signal2 = np.array(signal2)
        self.fs = fs
        self.freq = freq
        self.omega = 2 * np.pi * freq
        
        # Create time array
        self.n_samples = len(signal1)
        self.t = np.arange(self.n_samples) / fs
        
        # Results
        self.amplitude1 = None
        self.phase1 = None
        self.amplitude2 = None
        self.phase2 = None
        self.phase_difference = None
        self.signal2_aligned = None
        self.B = None  # Cosine component
        self.C = None  # Sine component
        self.D = None  # DC offset
        
    def extract_fft_component(self, sig):
        """
        Extract amplitude and phase at the frequency of interest using FFT.
        
        Returns:
        --------
        amplitude : float
            Amplitude at the frequency of interest
        phase : float
            Phase at the frequency of interest (in radians)
        """
        # Compute FFT
        fft_vals = np.fft.fft(sig)
        fft_freq = np.fft.fftfreq(len(sig), 1/self.fs)
        
        # Find the index closest to our frequency of interest
        # Look only at positive frequencies
        positive_freq_idx = np.where(fft_freq >= 0)
        freq_idx = positive_freq_idx[0][np.argmin(np.abs(fft_freq[positive_freq_idx] - self.freq))]
        
        # Extract complex coefficient at this frequency
        complex_coeff = fft_vals[freq_idx]
        
        # Amplitude (multiply by 2/N for single-sided spectrum, excluding DC)
        amplitude = 2.0 * np.abs(complex_coeff) / len(sig)
        
        # Phase (in radians)
        phase = np.angle(complex_coeff)
        
        return amplitude, phase, freq_idx, fft_freq[freq_idx]
    
    def compute_phase_difference(self):
        """
        Compute amplitude and phase for both signals using FFT,
        then calculate phase difference.
        """
        print("Extracting FFT components...")
        
        # Extract components for both signals
        self.amplitude1, self.phase1, idx1, actual_freq1 = self.extract_fft_component(self.signal1)
        self.amplitude2, self.phase2, idx2, actual_freq2 = self.extract_fft_component(self.signal2)
        
        print(f"Signal 1 - Amplitude: {self.amplitude1:.6f}, Phase: {np.degrees(self.phase1):.2f}°")
        print(f"Signal 2 - Amplitude: {self.amplitude2:.6f}, Phase: {np.degrees(self.phase2):.2f}°")
        print(f"Actual frequency from FFT: {actual_freq1:.3f} Hz")
        
        # Compute phase difference (phase of signal1 - phase of signal2)
        self.phase_difference = self.phase1 - self.phase2
        
        # Normalize to [-π, π]
        self.phase_difference = np.arctan2(np.sin(self.phase_difference), 
                                           np.cos(self.phase_difference))
        
        print(f"Phase difference: {np.degrees(self.phase_difference):.2f}°")
        
        return self.phase_difference
    
    def align_signal2_to_signal1(self):
        """
        Align signal2 to have the same phase as signal1 by applying phase shift.
        """
        # Create aligned version of signal2 by reconstructing with shifted phase
        # Original signal2 ≈ A2*cos(ωt + φ2) + other components
        # We want to shift it to match signal1's phase
        
        # Reconstruct the component at frequency of interest with adjusted phase
        component_aligned = self.amplitude2 * np.cos(self.omega * self.t + self.phase1)
        
        # Get the original component from signal2
        component_original = self.amplitude2 * np.cos(self.omega * self.t + self.phase2)
        
        # Shift the entire signal2 by the difference
        # signal2_aligned = signal2 - original_component + aligned_component
        self.signal2_aligned = self.signal2 - component_original + component_aligned
        
        return self.signal2_aligned
    
    def extract_cosine_sine_components(self):
        """
        Extract B*cos(ωt) + C*sin(ωt) + D components from aligned signal2.
        Uses least squares fitting.
        
        Since signal2 is now phase-aligned with signal1, we can extract:
        y(t) = B*cos(ωt) + C*sin(ωt) + D
        """
        # Create design matrix for least squares
        cos_term = np.cos(self.omega * self.t)
        sin_term = np.sin(self.omega * self.t)
        dc_term = np.ones_like(self.t)
        
        # Design matrix
        A = np.column_stack([cos_term, sin_term, dc_term])
        
        # Solve least squares: A * x = signal2_aligned
        # where x = [B, C, D]
        x, residuals, rank, s = np.linalg.lstsq(A, self.signal2_aligned, rcond=None)
        
        self.B = x[0]  # Cosine component
        self.C = x[1]  # Sine component
        self.D = x[2]  # DC offset
        
        # Reconstructed signal
        self.reconstructed = self.B * cos_term + self.C * sin_term + self.D
        
        # Calculate amplitude and phase from B and C
        self.fitted_amplitude = np.sqrt(self.B**2 + self.C**2)
        self.fitted_phase = np.arctan2(-self.C, self.B)
        
        # Calculate residual error
        self.residual_rms = np.sqrt(np.mean((self.signal2_aligned - self.reconstructed)**2))
        
        return self.B, self.C, self.D
    
    def analyze(self):
        """
        Perform complete analysis: FFT extraction, phase matching, 
        and component extraction.
        """
        print("="*60)
        print("SIGNAL PHASE ANALYSIS USING FFT")
        print("="*60)
        
        print("\nStep 1: Computing phase difference using FFT...")
        self.compute_phase_difference()
        
        print("\nStep 2: Aligning signal2 to match signal1's phase...")
        self.align_signal2_to_signal1()
        
        print("\nStep 3: Extracting cosine and sine components...")
        self.extract_cosine_sine_components()
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Frequency of interest: {self.freq} Hz")
        print(f"Sampling frequency: {self.fs} Hz")
        print(f"\nFFT Analysis:")
        print(f"  Signal 1 amplitude: {self.amplitude1:.6f}")
        print(f"  Signal 2 amplitude: {self.amplitude2:.6f}")
        print(f"  Phase difference:   {np.degrees(self.phase_difference):.2f}°")
        print(f"\nSignal2 (aligned) decomposition: y(t) = B*cos(ωt) + C*sin(ωt) + D")
        print(f"  B (Cosine component): {self.B:.6f}")
        print(f"  C (Sine component):   {self.C:.6f}")
        print(f"  D (DC offset):        {self.D:.6f}")
        print(f"\nAlternate form: y(t) = A*cos(ωt + φ) + D")
        print(f"  A (Amplitude): {self.fitted_amplitude:.6f}")
        print(f"  φ (Phase):     {np.degrees(self.fitted_phase):.2f}°")
        print(f"\nFitting quality:")
        print(f"  Residual RMS: {self.residual_rms:.6f}")
        print("="*60)
        
        return {
            'B': self.B,
            'C': self.C, 
            'D': self.D,
            'amplitude_fft': self.amplitude2,
            'amplitude_fitted': self.fitted_amplitude,
            'phase_fft': self.phase2,
            'phase_fitted': self.fitted_phase,
            'phase_difference': self.phase_difference,
            'residual_rms': self.residual_rms
        }
    
    def plot_results(self):
        """
        Plot the signals and results.
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        # Plot 1: Original signals
        axes[0].plot(self.t, self.signal1, 'b-', label='Signal 1 (Reference)', alpha=0.7)
        axes[0].plot(self.t, self.signal2, 'r-', label='Signal 2 (Original)', alpha=0.7)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Original Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Phase-matched signals
        axes[1].plot(self.t, self.signal1, 'b-', label='Signal 1 (Reference)', alpha=0.7)
        axes[1].plot(self.t, self.signal2_aligned, 'g-', 
                     label='Signal 2 (Phase-aligned)', alpha=0.7)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title(f'Phase-Aligned Signals (Δφ = {np.degrees(self.phase_difference):.2f}°)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Signal2 aligned and its reconstruction
        axes[2].plot(self.t, self.signal2_aligned, 'g-', 
                     label='Signal 2 (Phase-aligned)', alpha=0.7, linewidth=2)
        axes[2].plot(self.t, self.reconstructed, 'k--', 
                     label=f'Fit: {self.B:.3f}cos(ωt) + {self.C:.3f}sin(ωt) + {self.D:.3f}', 
                     alpha=0.8, linewidth=1.5)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title(f'Signal 2 Decomposition (RMS Error: {self.residual_rms:.4f})')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: FFT magnitude spectrum
        fft1 = np.fft.fft(self.signal1)
        fft2 = np.fft.fft(self.signal2)
        freq_axis = np.fft.fftfreq(len(self.signal1), 1/self.fs)
        
        # Only plot positive frequencies
        pos_mask = freq_axis >= 0
        axes[3].plot(freq_axis[pos_mask], 2*np.abs(fft1[pos_mask])/len(self.signal1), 
                     'b-', label='Signal 1', alpha=0.7)
        axes[3].plot(freq_axis[pos_mask], 2*np.abs(fft2[pos_mask])/len(self.signal2), 
                     'r-', label='Signal 2', alpha=0.7)
        axes[3].axvline(self.freq, color='k', linestyle='--', alpha=0.5, 
                        label=f'Target freq: {self.freq} Hz')
        axes[3].set_xlabel('Frequency (Hz)')
        axes[3].set_ylabel('Amplitude')
        axes[3].set_title('FFT Magnitude Spectrum')
        axes[3].set_xlim([0, min(50, self.fs/2)])  # Show up to 50 Hz or Nyquist
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate example signals

    """
    fs = 1000  # Sampling frequency (Hz)
    duration = 2  # Duration (seconds)
    t = np.arange(0, duration, 1/fs)
    
    # Frequency of interest
    freq = 10  # Hz
    omega = 2 * np.pi * freq
    
    # Signal 1: A1*cos(ωt + φ1) + noise
    A1 = 2.0
    phi1 = np.pi/6  # 30 degrees
    signal1 = A1 * np.cos(omega * t + phi1) + 0.5 * np.sin(2*omega*t) + 0.1 * np.random.randn(len(t))
    
    # Signal 2: A2*cos(ωt + φ2) + noise (different amplitude and phase)
    A2 = 3.0
    phi2 = np.pi/3  # 60 degrees (different phase)
    signal2 = A2 * np.cos(omega * t + phi2) + 0.3 * np.sin(3*omega*t) + 0.1 * np.random.randn(len(t))
    """
    path_base = "Fine_"

    n = 3660- 1136 # number of lines you want to remove from top
    frequency = 0.25
    sample_frequency = 250

    for i in range(5,6):

        current_path = path_base + str(i)
        path_CFD_results = os.path.join(current_path, "CFD_Results")
        path_results = os.path.join(current_path ,"Results")
        shutil.rmtree(path_results)
        os.makedirs(path_results, exist_ok=True)

        aerodynamic_force_path = os.path.join(path_CFD_results , "Aerodynamic_Forces.dat")
        aerodynamic_motion_path = os.path.join(path_CFD_results , "Input_Motion.dat")

        path_force_components = os.path.join(path_results, current_path + "_Fy_and_Components.png")
        path_moment_components = os.path.join(path_results, current_path + "_Mx_and_Components.png")
        path_motion_components = os.path.join(path_results, current_path + "_Motion_and_Components.png")
        path_DFT_Fy = os.path.join(path_results, current_path + "_DFT_Fy.png")
        path_DFT_moment = os.path.join(path_results, current_path + "_DFT_moment.png")
        path_fitted_signal = os.path.join(path_results, current_path + "_Fitted_Signal.png")
        path_flutter_derivatives = os.path.join(path_results, current_path + "_Flutter_Derivatives.txt")
        
        print(current_path)
        #Data Processing
        data_force = np.loadtxt(aerodynamic_force_path, skiprows=2)  # Read data
        data_force = data_force[n:len(data_force)]
        time, fx, fy, moment = data_force[:, 0], data_force[:, 1], data_force[:, 2], data_force[:, 6]  # Forces
        data_motion = np.loadtxt(aerodynamic_motion_path, skiprows = 2)
        data_motion = data_motion[n:len(data_motion)]
        u, v, theta = data_motion[:, 1], data_motion[:, 2], data_motion[:, 3]  # Displacement
        print("__________Length of list", len(fy), len(v))

    # Create analyzer and perform analysis
    analyzer = SignalPhaseAnalyzer(v, fy, sample_frequency, frequency)
    results = analyzer.analyze()
    
    # Plot results
    analyzer.plot_results()
    

    print(f"\nVerification:")
    print(f"Amplitude from FFT:    {results['amplitude_fft']:.6f}")
    print(f"Amplitude from fit:    {results['amplitude_fitted']:.6f}")
    print(f"√(B² + C²) =          {np.sqrt(results['B']**2 + results['C']**2):.6f}")
    print(f"Difference:            {abs(results['amplitude_fft'] - results['amplitude_fitted']):.6f}")