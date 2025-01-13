import streamlit as st
from streamlit_extras.let_it_rain import rain
from helpers import *

st.markdown("<h1 style='text-align: center;'>Super Music Genre Classifier</h1>", unsafe_allow_html=True)

st.text("""
         Want to try a super app that predicts the genre of the music you love better than you can?
         Upload a music file and watch the app generate its spectrogram and accurately predict its genre!
         Supported Genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock.
         """)

uploaded_file = st.file_uploader("Upload a music file", type=["wav", "aiff", "flac"])

if uploaded_file is not None:
    try:
        audio_tensor = load_audio(uploaded_file)
        mel_spectrogram = audio_to_mel(audio_tensor)
        processed_spectrogram = preprocess_spectrogram(mel_spectrogram)

        st.write("Mel Spectrogram:")
        plt = plot_spectrogram(mel_spectrogram)
        st.pyplot(plt)

        
        prediction = prediction(processed_spectrogram)
        st.markdown(f"<h1 style='text-align: center; font-size: 50px; font-weight: bold; color: red;'>{prediction.upper()}</h1>", unsafe_allow_html=True)

        rain(emoji="🎶", font_size=54, falling_speed=5, animation_length="infinite")

    except Exception as e:
        st.error(f"An error occurred: {e}")
