import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

import joblib
import tensorflow as tf



# Load the trained model
model_path = os.path.join(os.getcwd(), 'music_genre_cnn_model_regularized.h5')
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
label_encoder_path = os.path.join(os.getcwd(), 'label_encoder.pkl')

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load the saved scaler and label encoder
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

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
        # Normalize the features using the saved scaler
        features_scaled = scaler.transform(features.reshape(1, -1))  # Reshape and scale features

        # Reshape for Conv1D input
        features_scaled = features_scaled.reshape(1, features_scaled.shape[1], 1)

        # Make a prediction
        predictions = model.predict(features_scaled)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions, axis=1)

        # Decode the predicted label
        predicted_genre = label_encoder.inverse_transform(predicted_class_index)[0]
        confidence = np.max(predictions)
        return predicted_genre, confidence
    else:
        return None, None
# Streamlit UI
st.title("Music Genre Classification")
st.header("Record or Upload an Audio File to Classify the Genre")

# Define a processor for handling recorded audio
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv_audio(self, frame):
        self.audio_frames.append(frame)
        return frame

    def save_audio(self):
        if self.audio_frames:
             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                audio_data = np.concatenate([f.to_ndarray() for f in self.audio_frames])
                sf.write(temp_audio.name, audio_data, 16000)
                return temp_audio.name
        return None

# Add audio recording with WebRTC
# Function to initialize WebRTC when the user clicks "Start Recording"
def start_recording():
    audio_processor = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,  # Only sending audio
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    return audio_processor

# Button to start recording
if st.button("Start Recording"):
    audio_processor = start_recording()

    # Button to process the recorded audio
    if audio_processor and audio_processor.audio_processor:
        if st.button("Process Recorded Audio"):
            temp_audio_path = audio_processor.audio_processor.save_audio()
            if temp_audio_path:
                st.audio(temp_audio_path, format="audio/wav")
                predicted_genre, confidence = validate_model(temp_audio_path)
                if predicted_genre:
                    st.success(f"Predicted Genre: {predicted_genre}")
                    st.write(f"Confidence: {confidence:.2f}")
                else:
                    st.error("Could not extract features from the recorded audio.")
            else:
                st.error("No audio recorded. Please try again.")
    else:
        st.info("Please record audio first.")

# File upload option
audio_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3"])

if audio_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.getvalue())
        temp_audio_path = temp_audio.name

    st.audio(temp_audio_path, format="audio/wav")

    predicted_genre, confidence = validate_model(temp_audio_path)

    if predicted_genre:
        st.success(f"Predicted Genre: {predicted_genre}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.error("Error extracting features from the uploaded audio file.")