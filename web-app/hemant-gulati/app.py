import numpy as np
import joblib
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress Deprecation warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import pandas as pd

# Paths to the model, scaler, and label encoder
MODEL_PATH = "web-app/hemant-gulati/tuned_model.keras"
SCALER_PATH = "web-app/hemant-gulati/scaler.pkl"
LABEL_ENCODER_PATH = "web-app/hemant-gulati/label_encoder.pkl"

# Load the saved model, scaler, and label encoder
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    # st.write("Model, scaler, and label encoder loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or preprocessing objects: {e}")
    st.stop()

# Define expected feature keys for consistency
EXPECTED_FEATURE_KEYS = [
    "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
    "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var",
    "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var",
    "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo",
    *(f"mfcc{i}_mean" for i in range(1, 21)),
    *(f"mfcc{i}_var" for i in range(1, 21)),
    "spectral_centroid_to_bandwidth", "mfcc_mean_sum", "mfcc_var_sum"  # Interaction Features
]

def extract_audio_features(file_path):
    """Extract audio features, including interaction features."""
    try:
        y, sr = librosa.load(file_path, duration=30)
        features = {}

        # Basic features
        features["chroma_stft_mean"] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        features["chroma_stft_var"] = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
        features["rms_mean"] = np.mean(librosa.feature.rms(y=y))
        features["rms_var"] = np.var(librosa.feature.rms(y=y))
        features["spectral_centroid_mean"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features["spectral_centroid_var"] = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
        features["spectral_bandwidth_mean"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features["spectral_bandwidth_var"] = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features["rolloff_mean"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features["rolloff_var"] = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features["zero_crossing_rate_mean"] = np.mean(librosa.feature.zero_crossing_rate(y=y))
        features["zero_crossing_rate_var"] = np.var(librosa.feature.zero_crossing_rate(y=y))
        harmony, percussive = librosa.effects.hpss(y=y)
        features["harmony_mean"] = np.mean(harmony)
        features["harmony_var"] = np.var(harmony)
        features["perceptr_mean"] = np.mean(percussive)
        features["perceptr_var"] = np.var(percussive)

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
          # Ensure tempo is scalar

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f"mfcc{i}_mean"] = np.mean(mfccs[i - 1])
            features[f"mfcc{i}_var"] = np.var(mfccs[i - 1])

        # Interaction features
        features["spectral_centroid_to_bandwidth"] = (
            features["spectral_centroid_mean"] / (features["spectral_bandwidth_mean"] + 1e-6)
        )
        features["mfcc_mean_sum"] = sum(features[f"mfcc{i}_mean"] for i in range(1, 21))
        features["mfcc_var_sum"] = sum(features[f"mfcc{i}_var"] for i in range(1, 21))

        # Ensure all expected features are present
        feature_list = [features.get(key, 0.0) for key in EXPECTED_FEATURE_KEYS]
        return np.array(feature_list),y,sr
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None,None,None

def predict(input_features):
    """Predict the genre using the trained model."""
    try:
        # Ensure input features are aligned with expected feature names
        features_df = pd.DataFrame([input_features], columns=EXPECTED_FEATURE_KEYS)

        # Ensure the feature names match exactly with the scaler's feature names
        if list(features_df.columns) != list(scaler.feature_names_in_):
            raise ValueError("Feature names do not match the scaler's expected input. Please check the feature extraction process.")

        # Scale the features
        features_scaled = scaler.transform(features_df)

        # Predict class probabilities
        predictions = model.predict(features_scaled)

        # Get the predicted class index and decode it
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = label_encoder.inverse_transform([predicted_class_index])

        return predicted_class[0], predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Streamlit UI
st.title("Music Genre Classifier")
st.write("Upload an audio file to classify its genre.")

# Sidebar Information
st.sidebar.title("About This App")
st.sidebar.info(
    """
    ### **Music Genre Classifier**  
    This app uses a machine learning model to classify audio files into one of the predefined music genres.  

    **Purpose:**  
    To demonstrate the use of machine learning and audio feature extraction in classifying music genres.

    **Classification Classes:**  
    - Classical  
    - Jazz  
    - Rock  
    - Pop  
    - Hip-Hop  
    - Country  
    - Blues  

    **Key Features Extracted:**  
    - MFCCs (Mel-frequency cepstral coefficients)  
    - Spectral Centroid  
    - Zero Crossing Rate  
    - Chroma Features  

    **Developed by:**  
    **Hemant Gulati**  

    **Version:**  
    1.0  

    **Year:**  
    2025
    """
)

# Initialize session state for cleanup
if "results" not in st.session_state:
    st.session_state.results = None

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["wav"])

# Reset results if a new file is uploaded
if uploaded_file:
    if "last_uploaded_file" not in st.session_state or uploaded_file != st.session_state.last_uploaded_file:
        st.session_state.last_uploaded_file = uploaded_file
        st.session_state.results = None  # Clear previous results

# Process file if uploaded
if uploaded_file is not None:
    st.write("Processing audio file...")
    temp_file = "temp_audio.wav"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())

    # Extract features
    features,y,sr = extract_audio_features(temp_file)

    if features is not None:
        # Convert extracted features to a DataFrame
        features_df = pd.DataFrame([features], columns=EXPECTED_FEATURE_KEYS)

        # Debugging: Display extracted features and scaler's expected features
        # st.write("Extracted Feature Names:")
        # st.write(features_df.columns.tolist())
        # st.write("Scaler Expected Feature Names:")
        # st.write(scaler.feature_names_in_.tolist())

        # Align feature columns to scaler's expected feature names
        try:
            features_df = features_df[scaler.feature_names_in_]
            # st.write("Aligned Extracted Features to Scaler's Expected Order.")
        except KeyError as e:
            st.error(f"KeyError during alignment: {e}")
            st.stop()

        # Debugging: Display aligned feature names
        # st.write("Aligned Feature Names:")
        # st.write(features_df.columns.tolist())

        # Display spectrogram
        st.write("Spectrogram of the uploaded audio:")
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, fmax=8000)
        # Pass the spectrogram to the colorbar
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        st.pyplot(fig)


        # Scale the features
        try:
            features_scaled = scaler.transform(features_df)
            # st.write("Features scaled successfully.")
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            st.stop()

        # Predict genre
        try:
            predictions = model.predict(features_scaled)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class = label_encoder.inverse_transform([predicted_class_index])
            probabilities = predictions[0]

            # Store results in session state
            st.session_state.results = {
                "predicted_class": predicted_class[0],
                "probabilities": probabilities
            }
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Display results
if st.session_state.results:
    st.success(f"Predicted Class: {st.session_state.results['predicted_class']}")
    # st.write("Class Probabilities:")
    for i, prob in enumerate(st.session_state.results['probabilities']):
        genre = label_encoder.inverse_transform([i])[0]
        # st.write(f"{genre}: {prob:.2f}")

# Footer in the app
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2025 Hemant Gulati. All Rights Reserved."
    "</div>",
    unsafe_allow_html=True
)

