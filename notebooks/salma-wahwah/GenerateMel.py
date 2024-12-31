import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Path to the folder containing genres
base_path = os.path.abspath("notebooks/salma-wahwah/genres_original")
output_path = os.path.abspath("notebooks/salma-wahwah/train")

# Function to extract and preprocess Mel spectrogram
def extract_mel_spectrogram(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        y, sr = librosa.load(file_path, sr=None)  # Load audio with original sample rate
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db, sr
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process each genre folder
for genre in os.listdir(base_path):
    genre_path = os.path.join(base_path, genre)
    if not os.path.isdir(genre_path):
        continue

    print(f"Processing genre: {genre}")
    output_genre_path = os.path.join(output_path, genre)
    os.makedirs(output_genre_path, exist_ok=True)

    for file in os.listdir(genre_path):
        if file.endswith('.wav'):
            file_path = os.path.join(genre_path, file)
            try:
                file_name = os.path.splitext(file)[0]
                mel_spectrogram_db, sr=extract_mel_spectrogram(file_path)
                       
                # Save the Mel spectrogram as a .npy file
                npy_output_path = os.path.join(output_genre_path, f"{file_name}.npy")
                
                np.save(npy_output_path, mel_spectrogram_db)
                print(f"Mel spectrogram saved as: {npy_output_path}")
                
                
                # Step 3: Save the Mel spectrogram as a PNG image
                output_genre_path_img = os.path.join(output_path, genre)
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', fmax=8000, cmap='viridis',sr=sr)
                plt.colorbar(format='%+2.0f dB')
                png_output_path = os.path.join(output_genre_path_img, f"{file_name}.png")
                plt.title(f'Mel Spectrogram - {file_name}')
                plt.tight_layout()
                plt.savefig(png_output_path, dpi=300, format='png')
                plt.close()


                print(f"Mel spectrogram image saved as: {png_output_path}")
                

            except Exception as e:
                print(f"Error processing {file}: {e}")

