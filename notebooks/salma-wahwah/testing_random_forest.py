import librosa
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('notebooks\\salma-wahwah\\best_rf_model.pkl')

# Define class labels
CLASS_LABELS = ['Classical', 'Jazz', 'Pop', 'Rock', 'Blues', 'Hip-hop', 'Reggae', 'Country', 'Electronic', 'Folk']

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        rms = librosa.feature.rms(y=audio)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        
        # Aggregate features
        features = np.hstack([ 
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
            np.mean(rms), np.std(rms),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to validate with an audio file
def validate_model(audio_file_path):
    features = extract_features(audio_file_path)
    if features is not None:
        # Reshape the feature vector for prediction
        features = features.reshape(1, -1)
        
        # Make a prediction
        predictions = model.predict(features)
        predicted_genre = CLASS_LABELS[predictions[0]]
        print(f"Predicted Genre: {predicted_genre}")
    else:
        print("Error in feature extraction.")

# Example: Validate the model with an audio file
audio_file_path = "C:\Salma Personal\Delete\Data\genres_original\jazz\jazz.00017.wav"  # Replace with the actual file path
validate_model(audio_file_path)


