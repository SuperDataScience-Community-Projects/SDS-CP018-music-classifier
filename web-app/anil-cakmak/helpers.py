import librosa
import numpy as np
import torch
import torchaudio.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torchvision.transforms import Resize
import matplotlib.pyplot as plt


# Load and pad or truncate audio to a target duration
def load_audio(file, target_duration=30, sample_rate=22050):
    audio, _ = librosa.load(file, sr=sample_rate)
    target_length = target_duration * sample_rate

    if len(audio) > target_length:
        audio = audio[:target_length]  # Truncate
    else:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')  # Pad

    return torch.tensor(audio)


# Convert audio to mel spectrogram
def audio_to_mel(audio, sample_rate=22050):
    mel_spectrogram = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
    to_db = transforms.AmplitudeToDB()
    mel = mel_spectrogram(audio)
    mel_db = to_db(mel)
    return mel_db


# Preprocess the spectrogram for model input
def preprocess_spectrogram(spectrogram):
    resize = Resize((224, 224))
    spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.repeat(3, 1, 1)
    spectrogram_resized = resize(spectrogram)
    return spectrogram_resized


# Plot the spectrogram
def plot_spectrogram(spectrogram):
    spectrogram = spectrogram.squeeze().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    return plt


def prediction(processed_spectrogram):
    genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(processed_spectrogram.unsqueeze(0))
        prediction = torch.argmax(output, dim=1).item()
        prediction = genre_labels[prediction]
    
    return prediction