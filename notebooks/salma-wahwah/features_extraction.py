import librosa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
base_path = os.path.abspath("genres_original")
output_path = os.path.abspath("notebooks/salma-wahwah/train")
features_csv_path = os.path.join(output_path, "features.csv")

# Function to extract features
def extract_features(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
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

# Process files and save features
all_features = []
labels = []

for genre in os.listdir(base_path):
    genre_path = os.path.join(base_path, genre)
    if not os.path.isdir(genre_path):
        continue

    print(f"Processing genre: {genre}")
    for file in os.listdir(genre_path):
        if file.endswith('.wav'):
            file_path = os.path.join(genre_path, file)
            features = extract_features(file_path)
            if features is not None:
                all_features.append(features)
                labels.append(genre)

# Save to CSV
columns = [f"feature_{i}" for i in range(len(all_features[0]))]
features_df = pd.DataFrame(all_features, columns=columns)
features_df['label'] = labels
features_df.to_csv(features_csv_path, index=False)
print(f"Features saved to {features_csv_path}")

# Visualizations
# Histogram of a single feature (e.g., feature_0)
plt.figure(figsize=(10, 6))
sns.histplot(features_df['feature_0'], kde=True, bins=30, color='blue')
plt.title('Distribution of Feature 0')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.show()

# Correlation Matrix
correlation_matrix = features_df.drop(columns=['label']).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", cbar=True)
plt.title('Feature Correlation Matrix')
plt.show()
