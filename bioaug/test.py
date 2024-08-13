import numpy as np
import matplotlib.pyplot as plt

class ImpedanceVariation(object):
    """Simulate impedance variation on a physiological signal using trigonometric, exponential, or linear functions.

    Args:
        p        (float) : Probability of applying impedance variation to the input signal.
        amplitude (float) : Amplitude of the modulation.
        frequency (float) : Frequency of the modulation (for trigonometric and exponential functions).
        func     (str)   : Type of function to use ('sin', 'cos', 'exp', 'linear').
        seed     (int)   : A seed value for the random number generator.
    """
    def __init__(self, p=0.5, amplitude=0.1, frequency=0.01, func='sin', seed=42):
        self.p = p
        self.amplitude = amplitude
        self.frequency = frequency
        self.func = func
        self.seed = seed

    def __call__(self, signal):
        """Apply impedance variation to the signal.
        
        Args:
            signal: [sequence_length, input_dim] - The original physiological signal.
        
        Returns:
            Modulated signal with impedance variation.
        """
        if np.random.uniform(0, 1) < self.p:
            np.random.seed(self.seed)
            signal_ = np.array(signal).copy()
            sequence_length, input_dim = signal_.shape[0], signal_.shape[1]
            
            # Generate a modulation signal based on the chosen function
            t = np.linspace(0, 1, sequence_length)
            
            if self.func == 'sin':
                modulation = 1 + self.amplitude * np.sin(2 * np.pi * self.frequency * t)
            elif self.func == 'exp':
                modulation = 1 + self.amplitude * np.exp(-t * self.frequency)
            elif self.func == 'linear':
                modulation = 1 + self.amplitude * t
            else:
                raise ValueError("Function type must be 'sin', 'cos', 'exp', or 'linear'")
            
            # Apply the modulation to all channels
            for i in range(input_dim):
                signal_[:, i] = signal_[:, i] * modulation
            
            return signal_
        return signal

# Example usage
if __name__ == '__main__':
    # Generate a simple sine wave signal
    t = np.linspace(0, 4 * np.pi, 500)
    original_signal = np.sin(t)
    original_signal = np.expand_dims(original_signal, axis=1)  # Make it 2D for multi-channel simulation
    
    # Apply Impedance Variation with different functions
    variation_sin = ImpedanceVariation(p=1.0, amplitude=0.1, frequency=0.1, func='sin')
    modulated_signal_sin = variation_sin(original_signal)
    
    variation_exp = ImpedanceVariation(p=1.0, amplitude=0.3, frequency=10, func='exp')
    modulated_signal_exp = variation_exp(original_signal)
    
    variation_linear = ImpedanceVariation(p=1.0, amplitude=0.3, func='linear')
    modulated_signal_linear = variation_linear(original_signal)
    
    # Plot the original and modulated signals
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(t, original_signal[:, 0], label='Original Signal')
    plt.title('Original Signal')

    plt.subplot(4, 1, 2)
    plt.plot(t, modulated_signal_sin[:, 0], label='Sinusoidal Modulation', color='orange')
    plt.title('Impedance Variation (Sine)')

    plt.subplot(4, 1, 3)
    plt.plot(t, modulated_signal_exp[:, 0], label='Exponential Modulation', color='green')
    plt.title('Impedance Variation (Exponential)')

    plt.subplot(4, 1, 4)
    plt.plot(t, modulated_signal_linear[:, 0], label='Linear Modulation', color='blue')
    plt.title('Impedance Variation (Linear)')
    plt.tight_layout()
    plt.show()
