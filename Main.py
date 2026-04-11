import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
import wfdb
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class PanTompkinsQRS:
    """
    Implementation of Pan-Tompkins QRS Detection Algorithm
    with enhanced LMS adaptive thresholding
    """
    
    def __init__(self, fs=360):
        """
        Initialize the Pan-Tompkins detector
        
        Parameters:
        fs (int): Sampling frequency in Hz
        """
        self.fs = fs
        self.nyquist = fs / 2
        
        # Pan-Tompkins parameters
        self.integration_window = int(0.150 * fs)  # 150ms integration window
        
        # Thresholding parameters
        self.peak_threshold1 = 0
        self.peak_threshold2 = 0
        self.noise_threshold1 = 0
        self.noise_threshold2 = 0
        
        # Learning rates for thresholds
        self.alpha = 0.25
        self.gamma = 0.15
        
        # Refractory period (200ms)
        self.refractory_period = int(0.15 * fs)
        
        # LMS parameters
        self.lms_filter_length = 10
        self.lms_mu = 0.1
        
    def bandpass_filter(self, signal_data):
        """
        Stage 1: Bandpass filter (5-15 Hz)
        Removes baseline drift and high-frequency noise
        """
        # Design bandpass filter
        low_cutoff = 5.0 / self.nyquist
        high_cutoff = 15.0 / self.nyquist
        
        # Second-order Butterworth filter
        b, a = butter(2, [low_cutoff, high_cutoff], btype='band')
        
        # Apply zero-phase filtering
        filtered_signal = filtfilt(b, a, signal_data)
        
        self.bp_filter_coeff = (b, a)
        return filtered_signal
    
    def derivative_filter(self, signal_data):
        """
        Stage 2: Derivative filter
        Emphasizes QRS slope information
        H(z) = (1/8T)(-z^-2 - 2z^-1 + 2z^1 + z^2)
        """
        # Five-point derivative
        h = np.array([-1, -2, 0, 2, 1]) / (8 * (1/self.fs))
        
        # Apply derivative filter
        derivative_signal = np.convolve(signal_data, h, mode='same')
        
        self.deriv_filter_coeff = h
        return derivative_signal
    
    def squaring_function(self, signal_data):
        """
        Stage 3: Squaring function
        Emphasizes higher frequencies and makes all data positive
        """
        return signal_data ** 2
    
    def moving_window_integration(self, signal_data):
        """
        Stage 4: Moving window integration
        Smooths the signal and provides waveform feature information
        """
        # Integration window (typically 150ms)
        window = np.ones(self.integration_window) / self.integration_window
        
        # Apply moving window integration
        integrated_signal = np.convolve(signal_data, window, mode='same')
        
        # Store filter coefficients for analysis
        self.integration_filter_coeff = window
        
        return integrated_signal
    
    def analyze_filter_response(self, filter_coeff, filter_type=""):
        """
        Analyze filter frequency response, pole-zero plot, and group delay
        """
        if filter_type == "bandpass":
            b, a = filter_coeff
            w, h = signal.freqz(b, a, worN=8000, fs=self.fs)
            zeros, poles, _ = signal.tf2zpk(b, a)
            w_gd, gd = signal.group_delay((b, a), w=w, fs=self.fs)
            
        elif filter_type == "derivative":
            h_coeff = filter_coeff
            w, h = signal.freqz(h_coeff, 1, worN=8000, fs=self.fs)
            zeros, poles, _ = signal.tf2zpk(h_coeff, [1])
            w_gd, gd = signal.group_delay((h_coeff, [1]), w=w, fs=self.fs)
            
        elif filter_type == "integration":
            h_coeff = filter_coeff
            w, h = signal.freqz(h_coeff, 1, worN=8000, fs=self.fs)
            zeros, poles, _ = signal.tf2zpk(h_coeff, [1])
            w_gd, gd = signal.group_delay((h_coeff, [1]), w=w, fs=self.fs)
        
        return w, h, zeros, poles, w_gd, gd
    
    def plot_filter_analysis(self):
        """
        Plot magnitude, phase, pole-zero, and group delay for all three filters
        """
        fig, axes = plt.subplots(3, 4, figsize=(24, 12))

        # Analyze all filters
        filters = [
            ("bandpass", self.bp_filter_coeff),
            ("derivative", self.deriv_filter_coeff),
            ("integration", self.integration_filter_coeff)
        ]

        for row, (filter_type, coeff) in enumerate(filters):
            w, h, zeros, poles, w_gd, gd = self.analyze_filter_response(coeff, filter_type)

            # Magnitude
            axes[row, 0].semilogx(w, 20 * np.log10(abs(h) + 1e-10))
            axes[row, 0].set_title(f'{filter_type.capitalize()} Magnitude Response')
            axes[row, 0].set_xlabel('Frequency (Hz)')
            axes[row, 0].set_ylabel('Magnitude (dB)')
            axes[row, 0].grid(True)

            # Phase
            axes[row, 1].semilogx(w, np.angle(h))
            axes[row, 1].set_title(f'{filter_type.capitalize()} Phase Response')
            axes[row, 1].set_xlabel('Frequency (Hz)')
            axes[row, 1].set_ylabel('Phase (radians)')
            axes[row, 1].grid(True)

            # Pole-Zero
            axes[row, 2].plot(np.real(zeros), np.imag(zeros), 'o', label='Zeros', markersize=8)
            if len(poles) > 0:
                axes[row, 2].plot(np.real(poles), np.imag(poles), 'x', label='Poles', markersize=8)
            axes[row, 2].set_title(f'{filter_type.capitalize()} Pole-Zero Plot')
            axes[row, 2].set_xlabel('Real')
            axes[row, 2].set_ylabel('Imaginary')
            axes[row, 2].grid(True)
            axes[row, 2].legend()
            axes[row, 2].axis('equal')
            unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
            axes[row, 2].add_patch(unit_circle)

            # Group Delay
            axes[row, 3].plot(w_gd, gd)
            axes[row, 3].set_title(f'{filter_type.capitalize()} Group Delay')
            axes[row, 3].set_xlabel('Frequency (Hz)')
            axes[row, 3].set_ylabel('Group Delay (samples)')
            axes[row, 3].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_combined_response(self):
        """
        Plot the combined frequency response of all filters in cascade
        """
        # Get individual responses
        w_bp, h_bp, _, _, _, _ = self.analyze_filter_response(self.bp_filter_coeff, "bandpass")
        w_deriv, h_deriv, _, _, _, _ = self.analyze_filter_response(self.deriv_filter_coeff, "derivative")
        w_int, h_int, _, _, _, _ = self.analyze_filter_response(self.integration_filter_coeff, "integration")
        
        # Combined response (multiplication in frequency domain)
        h_combined = h_bp * h_deriv * h_int
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Individual responses
        axes[0, 0].semilogx(w_bp, 20 * np.log10(abs(h_bp)), 'b-', label='Bandpass', linewidth=2)
        axes[0, 0].semilogx(w_deriv, 20 * np.log10(abs(h_deriv) + 1e-10), 'r-', label='Derivative', linewidth=2)
        axes[0, 0].semilogx(w_int, 20 * np.log10(abs(h_int) + 1e-10), 'g-', label='Integration', linewidth=2)
        axes[0, 0].set_title('Individual Filter Responses')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Magnitude (dB)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Combined response
        axes[0, 1].semilogx(w_bp, 20 * np.log10(abs(h_combined) + 1e-10), 'k-', linewidth=2)
        axes[0, 1].set_title('Combined Filter Response (All Stages)')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Magnitude (dB)')
        axes[0, 1].grid(True)
        axes[0, 1].axvline(5, color='r', linestyle='--', alpha=0.7, label='QRS Band')
        axes[0, 1].axvline(15, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].legend()
        
        # Phase responses
        axes[1, 0].semilogx(w_bp, np.angle(h_bp), 'b-', label='Bandpass', linewidth=2)
        axes[1, 0].semilogx(w_deriv, np.angle(h_deriv), 'r-', label='Derivative', linewidth=2)
        axes[1, 0].semilogx(w_int, np.angle(h_int), 'g-', label='Integration', linewidth=2)
        axes[1, 0].set_title('Individual Phase Responses')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Phase (radians)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined phase response
        axes[1, 1].semilogx(w_bp, np.angle(h_combined), 'k-', linewidth=2)
        axes[1, 1].set_title('Combined Phase Response')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Phase (radians)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_individual_filter_analysis(self, filter_type, filter_coeff):
        """
        Plot magnitude, phase, pole-zero, and group delay for a single filter
        """
        w, h, zeros, poles, w_gd, gd = self.analyze_filter_response(filter_coeff, filter_type)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{filter_type.capitalize()} Filter Analysis', fontsize=16)

        # Magnitude response
        axes[0, 0].semilogx(w, 20 * np.log10(np.abs(h) + 1e-10))
        axes[0, 0].set_title('Magnitude Response')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Magnitude (dB)')
        axes[0, 0].grid(True)

        # Phase response
        axes[0, 1].semilogx(w, np.angle(h))
        axes[0, 1].set_title('Phase Response')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Phase (radians)')
        axes[0, 1].grid(True)

        # Pole-zero plot
        axes[1, 0].plot(np.real(zeros), np.imag(zeros), 'o', label='Zeros', markersize=8)
        if len(poles) > 0:
            axes[1, 0].plot(np.real(poles), np.imag(poles), 'x', label='Poles', markersize=8)
        axes[1, 0].set_title('Pole-Zero Plot')
        axes[1, 0].set_xlabel('Real')
        axes[1, 0].set_ylabel('Imaginary')
        axes[1, 0].grid(True)
        axes[1, 0].axis('equal')
        unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
        axes[1, 0].add_patch(unit_circle)
        axes[1, 0].legend()

        # Group delay
        axes[1, 1].plot(w_gd, gd)
        axes[1, 1].set_title('Group Delay')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Group Delay (samples)')
        axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

            

    
    def print_filter_characteristics(self):
        """
        Print detailed characteristics of each filter
        """
        print("\n" + "="*60)
        print("FILTER CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        print(f"\n1. BANDPASS FILTER (5-15 Hz):")
        print(f"   - Type: 2nd order Butterworth")
        print(f"   - Purpose: Remove baseline drift and high-frequency noise")
        print(f"   - Passband: 5-15 Hz (optimal for QRS detection)")
        print(f"   - Filter length: N/A (IIR filter)")
        
        print(f"\n2. DERIVATIVE FILTER:")
        print(f"   - Type: FIR, 5-point derivative approximation")
        print(f"   - Transfer function: H(z) = (1/8T)(-z^-2 - 2z^-1 + 2z^1 + z^2)")
        print(f"   - Purpose: Emphasize QRS slope information")
        print(f"   - Acts as high-pass filter, enhances sharp transitions")
        print(f"   - Filter length: 5 samples")
        
        print(f"\n3. INTEGRATION FILTER (Moving Average):")
        print(f"   - Type: FIR, rectangular window")
        print(f"   - Window length: {self.integration_window} samples ({self.integration_window/self.fs:.3f} seconds)")
        print(f"   - Purpose: Smooth signal and extract waveform features")
        print(f"   - Acts as low-pass filter, reduces noise")
        print(f"   - Provides information about QRS duration and energy")
        
        print(f"\n4. OVERALL CASCADE EFFECT:")
        print(f"   - Bandpass removes unwanted frequencies")
        print(f"   - Derivative emphasizes QRS slopes")
        print(f"   - Squaring makes all values positive and emphasizes large values")
        print(f"   - Integration smooths and provides feature extraction")
        print(f"   - Result: Enhanced QRS complexes with suppressed noise and artifacts")
        print("="*60)
        
       
    
    def original_thresholding(self, integrated_signal):
        """
        Original Pan-Tompkins thresholding logic
        """
        # Find peaks in integrated signal
        peaks, _ = find_peaks(integrated_signal, distance=self.refractory_period)
        
        if len(peaks) < 2:
            return []
        
        # Initialize thresholds
        peak_values = integrated_signal[peaks]
        sorted_peaks = np.sort(peak_values)
        
        # Initial thresholds (top 25% of peaks)
        threshold_idx = int(0.75 * len(sorted_peaks))
        initial_threshold = sorted_peaks[threshold_idx] if threshold_idx < len(sorted_peaks) else sorted_peaks[-1]
        
        self.peak_threshold1 = 0.25 * initial_threshold
        self.noise_threshold1 = 0.5 * self.peak_threshold1
        
        detected_peaks = []
        
        for i, peak_idx in enumerate(peaks):
            peak_value = integrated_signal[peak_idx]
            
            # Check if peak exceeds threshold
            if peak_value > self.peak_threshold1:
                detected_peaks.append(peak_idx)
                
                # Update thresholds
                self.peak_threshold1 = self.alpha * peak_value + (1 - self.alpha) * self.peak_threshold1
                self.noise_threshold1 = self.gamma * peak_value + (1 - self.gamma) * self.noise_threshold1
            else:
                # Update noise threshold
                self.noise_threshold1 = self.gamma * peak_value + (1 - self.gamma) * self.noise_threshold1
        
        return detected_peaks
    
    def lms_adaptive_threshold(self, integrated_signal):
        """
        Optimized LMS-based adaptive thresholding with improved peak validation
        """
        N = len(integrated_signal)
        detected_peaks = []
        adaptive_threshold = np.zeros(N)
        threshold = 0.4  # Starting threshold
        refractory = int(0.25 * self.fs)  # 250 ms
        mu = 0.03
        lms_window = 50

        for n in range(1, N):
            # Calculate local statistics
            local_start = max(0, n - lms_window)
            local_seg = integrated_signal[local_start:n+1]
            local_mean = np.mean(local_seg)
            local_std = np.std(local_seg)
            desired = local_mean + 0.2 * local_std

            # LMS update
            error = desired - threshold
            threshold += mu * error
            adaptive_threshold[n] = threshold

            # Peak detection with stricter validation
            if (
                integrated_signal[n] > threshold and
                integrated_signal[n] > integrated_signal[n - 1] and
                integrated_signal[n] > integrated_signal[n + 1 if n + 1 < N else n] and
                integrated_signal[n] > np.mean(integrated_signal[max(0, n - 10):n + 10]) and
                (integrated_signal[n] - integrated_signal[n - 1]) > 0.01  # min slope
            ):
                if len(detected_peaks) == 0 or n - detected_peaks[-1] > refractory:
                    detected_peaks.append(n)

        # Optional: Post-filter to suppress very close false peaks
        detected_peaks = self.suppress_close_peaks(detected_peaks, integrated_signal, refractory)

        return detected_peaks, adaptive_threshold



    def detect_qrs(self, ecg_signal, method='original'):
        """
        Complete QRS detection pipeline
        
        Parameters:
        ecg_signal: Input ECG signal
        method: 'original' or 'lms' for detection method
        
        Returns:
        Dictionary containing all processing stages and results
        """
        results = {}
        
        # Stage 1: Bandpass filtering
        filtered_signal = self.bandpass_filter(ecg_signal)
        results['filtered'] = filtered_signal
        
        # Stage 2: Derivative
        derivative_signal = self.derivative_filter(filtered_signal)
        results['derivative'] = derivative_signal
        
        # Stage 3: Squaring
        squared_signal = self.squaring_function(derivative_signal)
        results['squared'] = squared_signal
        
        # Stage 4: Moving window integration
        integrated_signal = self.moving_window_integration(squared_signal)
        results['integrated'] = integrated_signal
        
        # Stage 5: Peak detection and thresholding
        if method == 'original':
            qrs_peaks = self.original_thresholding(integrated_signal)
            results['qrs_peaks'] = qrs_peaks
            results['method'] = 'original'
        elif method == 'lms':
            qrs_peaks, adaptive_threshold = self.lms_adaptive_threshold(integrated_signal)
            results['qrs_peaks'] = qrs_peaks
            results['adaptive_threshold'] = adaptive_threshold
            results['method'] = 'lms'
        
        return results
    
    def plot_detection_results(self, ecg_signal, results):
        """
        Plot comprehensive detection results
        """
        fig, axes = plt.subplots(6, 1, figsize=(15, 12))
        
        time_axis = np.arange(len(ecg_signal)) / self.fs
        
        # Original ECG
        axes[0].plot(time_axis, ecg_signal)
        axes[0].set_title('Original ECG Signal')
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].grid(True)
        
        # Filtered signal
        axes[1].plot(time_axis, results['filtered'])
        axes[1].set_title('Bandpass Filtered (5-15 Hz)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True)
        
        # Derivative
        axes[2].plot(time_axis, results['derivative'])
        axes[2].set_title('Derivative Filter Output')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True)
        
        # Squared
        axes[3].plot(time_axis, results['squared'])
        axes[3].set_title('Squared Signal')
        axes[3].set_ylabel('Amplitude')
        axes[3].grid(True)
        
        # Integrated
        axes[4].plot(time_axis, results['integrated'])
        axes[4].set_title('Moving Window Integration')
        axes[4].set_ylabel('Amplitude')
        axes[4].grid(True)
        
        # QRS Detection
        axes[5].plot(time_axis, ecg_signal, 'b-', label='ECG Signal')
        qrs_times = np.array(results['qrs_peaks']) / self.fs
        qrs_amplitudes = ecg_signal[results['qrs_peaks']]
        axes[5].plot(qrs_times, qrs_amplitudes, 'ro', markersize=8, label='Detected QRS')
        
        if results['method'] == 'lms':
            axes[5].plot(time_axis, results['adaptive_threshold'] * 0.1, 'g--', 
                        label='Adaptive Threshold (scaled)')
        
        axes[5].set_title(f'QRS Detection Results ({results["method"].upper()} method)')
        axes[5].set_xlabel('Time (s)')
        axes[5].set_ylabel('Amplitude (mV)')
        axes[5].legend()
        axes[5].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_performance(self, detected_peaks, reference_peaks, tolerance=72):
        """
        Evaluate QRS detection performance
        
        Parameters:
        detected_peaks: Detected QRS peak locations
        reference_peaks: Reference (ground truth) peak locations
        tolerance: Tolerance window in samples (default: 200ms at 360Hz = 72 samples)
        
        Returns:
        Dictionary with performance metrics
        """
        if len(detected_peaks) == 0:
            return {'sensitivity': 0, 'ppv': 0, 'f1_score': 0, 'tp': 0, 'fp': len(detected_peaks), 'fn': len(reference_peaks)}
        
        # True positives: detected peaks within tolerance of reference peaks
        tp = 0
        matched_ref = set()
        
        for det_peak in detected_peaks:
            for ref_peak in reference_peaks:
                if abs(det_peak - ref_peak) <= tolerance and ref_peak not in matched_ref:
                    tp += 1
                    matched_ref.add(ref_peak)
                    break
        
        fp = len(detected_peaks) - tp  # False positives
        fn = len(reference_peaks) - tp  # False negatives
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0
        
        return {
            'sensitivity': sensitivity,
            'ppv': ppv,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    

    def suppress_close_peaks(self, peaks, signal, min_distance):
        """
        Removes peaks that are too close together by keeping the stronger one.
        """
        if not peaks:
            return []

        final_peaks = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] - final_peaks[-1] >= min_distance:
                final_peaks.append(peaks[i])
            else:
                # Keep the one with higher amplitude
                if signal[peaks[i]] > signal[final_peaks[-1]]:
                    final_peaks[-1] = peaks[i]

        return final_peaks


def load_mit_bih_record(record_name='100', duration=None):
    """
    Load MIT-BIH record
    
    Parameters:
    record_name: Record name (e.g., '100', '101')
    duration: Duration in seconds (None for full record)
    
    Returns:
    ECG signal and annotations
    """
    try:
        # Load recordAC
        record = wfdb.rdrecord(record_name, pn_dir='mitdb')
        
        # Get ECG signal (first channel)
        ecg_signal = record.p_signal[:, 0]
        
        if duration:
            samples = int(duration * record.fs)
            ecg_signal = ecg_signal[:samples]
        
        # Load annotations
        annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
        
        # Filter for normal beats and QRS annotations
        qrs_indices = []
        for i, symbol in enumerate(annotation.symbol):
            if symbol in ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/']:
                sample_idx = annotation.sample[i]
                if duration is None or sample_idx < len(ecg_signal):
                    qrs_indices.append(sample_idx)
        
        return ecg_signal, qrs_indices, record.fs
        
    except Exception as e:
        print(f"Error loading MIT-BIH record: {e}")
        print("Generating synthetic ECG data for demonstration...")
        
        # Generate synthetic ECG for demonstration
        fs = 360
        duration_samples = 3600 if duration is None else int(duration * fs)
        t = np.arange(duration_samples) / fs
        
        # Synthetic ECG with QRS complexes
        ecg = np.zeros_like(t)
        qrs_indices = []
        
        # Add QRS complexes every ~0.8 seconds (75 BPM)
        for i in range(0, len(t), int(0.8 * fs)):
            if i + 50 < len(t):
                # Simple QRS complex
                qrs_start = i
                qrs_end = min(i + 50, len(t))
                ecg[qrs_start:qrs_end] = signal.gausspulse(np.linspace(-1, 1, qrs_end - qrs_start), fc=5)
                qrs_indices.append(i + 25)  # Peak at center
        
        
        
        return ecg, qrs_indices, fs
    
def add_awgn_noise(signal, target_snr_db):
    """
    Adds Additive White Gaussian Noise (AWGN) to a signal to achieve a target SNR.
    
    Parameters:
        signal (np.array): Clean input ECG signal.
        target_snr_db (float): Desired Signal-to-Noise Ratio in dB.
    
    Returns:
        Noisy signal with the desired SNR.
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


# Example usage and comprehensive analysis
def main():
    """
    Main function demonstrating the complete Pan-Tompkins implementation
    """
    print("Pan-Tompkins QRS Detection Algorithm - DSP Course Project")
    print("=" * 60)
    
    # Load ECG data
    print("Loading MIT-BIH ECG data...")
    ecg_signal, reference_qrs, fs = load_mit_bih_record('100', duration=15)
    
    # Example: Apply SNR settings
    snr_level = 15  # change to 20, 15, 10, 5 depending on test
    ecg_signal = add_awgn_noise(ecg_signal, snr_level)

    # Initialize detector
    detector = PanTompkinsQRS(fs=fs)
    
    # Analyze filters
    print("Initializing filters by running detection on sample data...")
    temp_results = detector.detect_qrs(ecg_signal[:1000], method='original')

    # Plot individual filter analyses
    print("Plotting Bandpass Filter Analysis...")
    detector.plot_individual_filter_analysis("bandpass", detector.bp_filter_coeff)

    print("Plotting Derivative Filter Analysis...")
    detector.plot_individual_filter_analysis("derivative", detector.deriv_filter_coeff)

    print("Plotting Integration Filter Analysis...")
    detector.plot_individual_filter_analysis("integration", detector.integration_filter_coeff)
    
    # Detect QRS with original method
    print("Detecting QRS with original Pan-Tompkins method...")
    results_original = detector.detect_qrs(ecg_signal, method='original')
    
    # Detect QRS with LMS enhancement
    print("Detecting QRS with LMS adaptive thresholding...")
    results_lms = detector.detect_qrs(ecg_signal, method='lms')
    
    # Plot results
    print("Plotting detection results...")
    detector.plot_detection_results(ecg_signal, results_original)
    detector.plot_detection_results(ecg_signal, results_lms)
    
    # Evaluate performance
    print("\nPerformance Evaluation:")
    print("-" * 30)
    
    # Original method performance
    perf_original = detector.evaluate_performance(
        results_original['qrs_peaks'], reference_qrs)
    
    print(f"Original Pan-Tompkins Method:")
    print(f"  Sensitivity: {perf_original['sensitivity']:.3f}")
    print(f"  PPV: {perf_original['ppv']:.3f}")
    print(f"  F1-Score: {perf_original['f1_score']:.3f}")
    print(f"  TP: {perf_original['tp']}, FP: {perf_original['fp']}, FN: {perf_original['fn']}")
    
    # LMS method performance
    perf_lms = detector.evaluate_performance(
        results_lms['qrs_peaks'], reference_qrs)
    
    print(f"\nLMS Enhanced Method:")
    print(f"  Sensitivity: {perf_lms['sensitivity']:.3f}")
    print(f"  PPV: {perf_lms['ppv']:.3f}")
    print(f"  F1-Score: {perf_lms['f1_score']:.3f}")
    print(f"  TP: {perf_lms['tp']}, FP: {perf_lms['fp']}, FN: {perf_lms['fn']}")
    
    # Comparison
    print(f"\nImprovement with LMS:")
    print(f"  Sensitivity: {perf_lms['sensitivity'] - perf_original['sensitivity']:.3f}")
    print(f"  PPV: {perf_lms['ppv'] - perf_original['ppv']:.3f}")
    print(f"  F1-Score: {perf_lms['f1_score'] - perf_original['f1_score']:.3f}")


if __name__ == "__main__":
    main()