import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_adc_data_complex(adc_data_complex, title="Frequency Domain Plot (FFT)"):
    """
    Flattens the 4D adc_data_complex array and plots the real and imaginary parts
    in one figure using different colors.

    Parameters:
    adc_data_complex (numpy.ndarray): 4D matrix containing complex ADC data 
                                      (numSamplePerChirp, numLoops, numRXPerDevice, numChirpPerLoop).
    title (str): The title for the plot.
    """
    # Flatten the 4D array into a 1D array
    adc_data_flat = adc_data_complex.flatten()
    
    # Create a figure and axis with a smaller size
    plt.figure(figsize=(8, 4))  # Smaller figure size
    
    # Plot the real part of the complex data
    plt.plot(np.real(adc_data_flat), color='blue', label='Real Part')
    
    # Plot the imaginary part of the complex data
    plt.plot(np.imag(adc_data_flat), color='red', label='Imaginary Part')
    
    # Add title and labels
    plt.title(f"Time domain - {title}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    
    # Add a legend
    plt.legend()
    
    # Optimize the layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_instantaneous_frequency(iq_data_chirp, title="Frequency Domain Magnitude Plot (FFT)",sample_rate=6e6):
    """
    Calculates and plots the instantaneous frequency of the IQ data for a single chirp.
    
    Parameters:
    iq_data_chirp (numpy.ndarray): IQ data for a single chirp (complex numbers).
    sample_rate (float): Sampling rate of the IQ data in Hz (default is 6 MHz).
    """
    # Check IQ data type
    print(f"IQ data type: {iq_data_chirp.dtype}")
    
    # Calculate the phase of the complex IQ data
    phase = np.angle(iq_data_chirp)
    
    # Unwrap the phase to prevent discontinuities
    unwrapped_phase = np.unwrap(phase)
    
    # Check unwrapped phase type and sample_rate
    print(f"Unwrapped phase type: {unwrapped_phase.dtype}")
    print(f"Sample rate type: {type(sample_rate)}")
    
    # Convert sample_rate to float if it's a string
    if isinstance(sample_rate, str):
        sample_rate = float(sample_rate)
    
    # Calculate the instantaneous frequency (derivative of phase)
    inst_freq = np.diff(unwrapped_phase) * sample_rate / (2.0 * np.pi)
    
    # Plot the instantaneous frequency
    time_axis = np.arange(len(inst_freq)) / sample_rate
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, inst_freq)
    plt.title('Instantaneous Frequency '+ title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True)
    plt.show()

def plot_fft(iq_data_chirp,  title="Frequency Domain Plot (FFT)",sample_rate=6e6):
    """
    Plots the FFT of the IQ data to visualize its frequency content.

    Parameters:
    iq_data_chirp (numpy.ndarray): IQ data for a single chirp (complex numbers).
    sample_rate (float): Sampling rate of the IQ data in Hz.
    title (str): The title for the plot.
    """
    # Perform FFT
    fft_data = np.fft.fft(iq_data_chirp)
    
    # Calculate the magnitude of the FFT
    magnitude = np.abs(fft_data)
    
    # Frequency axis
    freq_axis = np.fft.fftfreq(len(iq_data_chirp), d=1/sample_rate)
    
    # Plot the magnitude of the FFT
    plt.figure(figsize=(10, 6))
    plt.plot(freq_axis, magnitude)
    plt.title("FFT " +title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
def plot_fft_phase(iq_data_chirp,  title="Frequency Domain Phase Plot (FFT)",sample_rate=6e6):
    
    # Perform FFT
    fft_data = np.fft.fft(iq_data_chirp)
    
    # Calculate the magnitude of the FFT
    angle = np.angle(fft_data)
    
    # Frequency axis
    freq_axis = np.fft.fftfreq(len(iq_data_chirp), d=1/sample_rate)
    
    # Plot the magnitude of the FFT
    plt.figure(figsize=(10, 6))
    plt.plot(freq_axis, angle)
    plt.title("FFT Phase " +title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')
    plt.grid(True)
    plt.show()



    