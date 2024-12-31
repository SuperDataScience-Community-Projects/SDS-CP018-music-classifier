import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Step 1: Load audio file
audio_path = "C:\\Salma Personal\\SDS-CP018-music-classifier\\notebooks\\salma-wahwah\\genres_original\\reggae\\reggae.00008.wav"
y, sr = librosa.load(audio_path, sr=None)

# Step 2: Generate Mel spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Step 3: Save the Mel spectrogram as a floating-point image
output_image_path = "mel_spectrogram.npy"
np.save(output_image_path, mel_spec_db)
print(f"Mel spectrogram saved as: {output_image_path}")

# Step 4: Load the saved Mel spectrogram
loaded_mel_spec_db = np.load(output_image_path)

# Step 5: Convert back from dB to power
loaded_mel_spec = librosa.db_to_power(loaded_mel_spec_db)

# Step 6: Reconstruct audio from the Mel spectrogram
reconstructed_audio = librosa.feature.inverse.mel_to_audio(
    loaded_mel_spec, sr=sr, n_fft=2048, hop_length=512, n_iter=100
)

# Step 7: Save the reconstructed audio
output_audio_path = "reconstructed_audio.wav"
sf.write(output_audio_path, reconstructed_audio, sr)
print(f"Reconstructed audio saved as: {output_audio_path}")
