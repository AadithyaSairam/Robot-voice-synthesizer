#imports
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt

# Load the original WAV file
sampling_rate, audio_data = wavfile.read("quote.wav")

# Down-sample if sampling rate > 16KHz
if sampling_rate > 16000:
    new_sampling_rate = 16000
    audio_data = signal.resample(
        audio_data, int(len(audio_data) * new_sampling_rate / sampling_rate))
    sampling_rate = new_sampling_rate

# Keep only one channel if stereo
if len(audio_data.shape) > 1:
    audio_data = audio_data[:, 0]

# Time-Segmentation with Overlap
chunk_duration = 15  # milliseconds
overlap_percentage = 50  # Overlap percentage

chunk_size = int(
    sampling_rate * chunk_duration / 1000
)  # Calculate the number of samples in each chunk based on the sampling rate and chunk duration
overlap_size = int(
    chunk_size * overlap_percentage /
    100)  # Calculate the number of samples in the overlap region
step_size = chunk_size - overlap_size  # Calculate the step size to move the segmentation window (chunk) by

# Create a list to store the segmented audio chunks with overlap
chunks = [
    audio_data[i:i + chunk_size]
    for i in range(0,
                   len(audio_data) - chunk_size + 1, step_size)
]

# Frequency-Domain Analysis
num_bands = 250
bandwidth = 100  # Hz
min_center_freq = (bandwidth / 2) + 1  # Minimum center frequency (in Hz)
max_center_freq = (sampling_rate / 2) - (min_center_freq)  # Nyquist frequency
center_frequencies = np.linspace(min_center_freq, max_center_freq, num_bands)

# Create a bank of band-pass filters
bpf_bank = [
    signal.butter(4, [f - bandwidth / 2, f + bandwidth / 2],
                  btype='band',
                  fs=sampling_rate,
                  output='sos') for f in center_frequencies
]

# Calculate the total length of the synthesized audio
total_samples = len(chunks) * step_size + chunk_size

# Initialize the robotic_voice array with zeros and the correct total length
robotic_voice = np.zeros(total_samples)

amplification_factor = 2  # Adjust the factor to increase or decrease loudness

# Synthesis
synthesis_index = 0  # Index to keep track of where to start adding synthesized audio
for chunk in chunks:
    synthesized_chunk = np.zeros(len(chunk))
    for i, (sos, center_freq) in enumerate(zip(bpf_bank, center_frequencies)):
        # Apply band-pass filter to the chunk
        filtered_chunk = signal.sosfilt(sos, chunk)

        # Calculate RMS (Root Mean Square) for the filtered chunk
        rms = np.sqrt(np.mean(filtered_chunk**2))

        # Synthesize a sine wave using the RMS and center frequency
        synthesized_wave = amplification_factor * rms * np.sin(
            2 * np.pi * center_freq * np.arange(len(chunk)) / sampling_rate)

        # Accumulate the synthesized wave
        synthesized_chunk += synthesized_wave

    # Add the synthesized chunk to the appropriate location in robotic_voice
    robotic_voice[synthesis_index:synthesis_index +
                  len(chunk)] += synthesized_chunk

    # Move the synthesis_index to the next chunk's starting point
    synthesis_index += step_size

# Trim the robotic_voice array to remove any trailing zeros (silence)
robotic_voice = robotic_voice[:synthesis_index]

# Save the synthesized robotic voice as a WAV file
wavfile.write("robotic_voice.wav", sampling_rate, np.int16(robotic_voice))

# Plot the original and synthesized waveforms
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(audio_data)) / sampling_rate, audio_data)
plt.title("Original Voice Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(robotic_voice)) / sampling_rate, robotic_voice)
plt.title("Synthesized Robotic Voice Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Perform FFT on the same length of data for original and synthesized signals
n = len(audio_data)
original_freq = np.fft.fft(audio_data, n)
synthesized_freq = np.fft.fft(robotic_voice, n)

# Frequencies for plotting
freqs = np.fft.fftfreq(n, d=1 / sampling_rate)

#must close previous plots to view this plot when running
# Plot the original and synthesized waveforms in frequency domain
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(freqs, np.abs(original_freq))
plt.title("Original Voice Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(freqs, np.abs(synthesized_freq))
plt.title("Synthesized Robotic Voice Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
