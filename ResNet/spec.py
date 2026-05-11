import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import numpy as np

def plot_mel_spectrogram(audio_path, n_mels=64):
    print(f"Loading audio file: {audio_path}")
    
    # 1. Load the audio file
    # sr=None preserves the native sample rate of the file
    y, sr = librosa.load(audio_path, sr=None)
    
    # 2. Compute the Mel-spectrogram
    # fmax=8000 is standard for capturing most environmental/human sounds
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    
    # 3. Convert power to decibels (log scale)
    # Neural networks (and human ears) process sound logarithmically!
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 4. Plot the results
    plt.figure(figsize=(10, 6))
    
    # Use librosa's built-in display tool for perfectly formatted axes
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='magma')
    
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram ({n_mels} bands)', fontsize=16, pad=15)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    
    plt.tight_layout()
    
    save_path = "sample_mel_spectrogram.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Success! Spectrogram saved to: {save_path}")
    

if __name__ == "__main__":
    # Replace this with the path to any .wav file on your computer!
    # Example: A file from your UrbanSound8K dataset
    SAMPLE_FILE = "D:/BTP/Dataset/UrbanSound8K/audio/fold1/46656-6-4-0.wav"
    
    plot_mel_spectrogram(SAMPLE_FILE, n_mels=64)