import numpy as np
import matplotlib.pyplot as plt


class Distortion(object):
    """Simulate nonlinear distortion by applying a nonlinear transformation to the input time-series data.

    Args:
        p                (float) : Probability of applying nonlinear distortion to the input signal.
        distortion_degree (float): Degree of nonlinearity applied to the signal (n > 1).
        harmonic_degree   (float): Degree of harmonic distortion added to the signal.
        phase_shift       (float): Degree of phase distortion (in radians).
        distortion_type   (str)  : Type of distortion to apply ('amplitude', 'harmonic', 'phase', 'all').
        seed             (int)   : A seed value for the random number generator.
    """

    def __init__(self, p=0.5, distortion_degree=2.0, harmonic_degree=0.1, phase_shift=0.1,
                 distortion_type='amplitude', seed=42):
        self.p = p
        self.distortion_degree = distortion_degree
        self.harmonic_degree = harmonic_degree
        self.phase_shift = phase_shift
        self.distortion_type = distortion_type
        self.seed = seed

    def apply_amplitude_distortion(self, signal):
        """Apply amplitude distortion to the signal."""
        return np.sign(signal) * np.abs(signal) ** self.distortion_degree

    def apply_harmonic_distortion(self, signal):
        """Apply harmonic distortion to the signal."""
        harmonic_signal = self.harmonic_degree * np.sin(2 * np.pi * np.linspace(0, 1, signal.shape[0]) * 5)
        return signal + harmonic_signal[:, np.newaxis]

    def apply_phase_distortion(self, signal):
        """Apply phase distortion by altering the phase of the signal's frequency components."""
        # Perform Fourier transform to move signal to the frequency domain
        signal_fft = np.fft.fft(signal, axis=0)
        # Create a phase shift array that applies the phase shift to each frequency component
        phase_shifts = np.exp(1j * self.phase_shift * np.linspace(0, signal.shape[0] - 1, signal.shape[0]))
        # Apply phase shift to each frequency component
        signal_fft = signal_fft * phase_shifts[:, np.newaxis]

        # Perform inverse Fourier transform to move back to time domain
        signal_phase_distorted = np.fft.ifft(signal_fft, axis=0).real

        return signal_phase_distorted

    def __call__(self, signal):
        """Apply nonlinear distortion to the signal.

        Args:
            signal: [sequence_length, input_dim] - The original physiological signal.

        Returns:
            Distorted signal with nonlinear transformations.
        """
        if np.random.uniform(0, 1) < self.p:
            np.random.seed(self.seed)
            signal_ = np.array(signal).copy()

            # Apply selected type of distortion
            if self.distortion_type == 'amplitude':
                signal_ = self.apply_amplitude_distortion(signal_)
            elif self.distortion_type == 'harmonic':
                signal_ = self.apply_harmonic_distortion(signal_)
            elif self.distortion_type == 'phase':
                signal_ = self.apply_phase_distortion(signal_)
            elif self.distortion_type == 'all':
                signal_ = self.apply_amplitude_distortion(signal_)
                signal_ = self.apply_harmonic_distortion(signal_)
                signal_ = self.apply_phase_distortion(signal_)
            else:
                raise ValueError("Invalid distortion type. Choose from 'amplitude', 'harmonic', 'phase', or 'all'.")

            return signal_
        return signal
