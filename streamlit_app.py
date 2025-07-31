import streamlit as st
import requests
import json
import base64
import pandas as pd
from io import BytesIO
import uuid
from PIL import Image

# Configure Streamlit page
st.set_page_config(
    page_title="Audio Spectograms Analyzer",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open("frontend/static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

# Backend configuration
BACKEND_URL = "http://localhost:5000"

def main():
    load_css()
    init_session_state()
    
    # Header
    st.title("🔊 Motor Audio Analysis System")
    st.markdown("**Analyze motor audio files using advanced spectrogram techniques**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # Theme toggle
        theme_toggle = st.toggle("Dark Theme", value=st.session_state.theme == 'dark')
        if theme_toggle != (st.session_state.theme == 'dark'):
            st.session_state.theme = 'dark' if theme_toggle else 'light'
            st.rerun()
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3'],
            help="Upload WAV or MP3 files (≤20 seconds)"
        )
        
        # Clear file button
        if st.button("🗑️ Clear File", type="secondary"):
            st.session_state.uploaded_file = None
            st.session_state.analysis_results = None
            cleanup_backend()
            st.rerun()
    
    # Main content area
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # Audio playback
        st.subheader("📻 Audio Playback")
        st.audio(uploaded_file, format='audio/wav')
        
        # Analyze button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔍 Analyze Audio", type="primary"):
                analyze_audio(uploaded_file)
        
        # Display results if available
        if st.session_state.analysis_results:
            display_analysis_results()
    
    else:
        st.info("👆 Please upload an audio file to begin analysis")

def analyze_audio(uploaded_file):
    """Send audio file to backend for analysis"""
    with st.spinner("🔄 Analyzing audio... This may take a few moments."):
        try:
            # Prepare file for upload
            files = {
                'audio': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            data = {'session_id': st.session_state.session_id}
            
            # Send to backend
            response = requests.post(f"{BACKEND_URL}/analyze", files=files, data=data)
            
            if response.status_code == 200:
                st.session_state.analysis_results = response.json()
                st.success("✅ Analysis completed successfully!")
            else:
                st.error(f"❌ Analysis failed: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Please ensure Flask server is running on port 5000.")
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

def display_analysis_results():
    """Display spectrogram visualizations and feature data"""
    results = st.session_state.analysis_results
    
    # Spectrograms section
    st.subheader("📊 Spectrogram Analysis")
    
    # Create tabs for different spectrograms
    spectrogram_tabs = st.tabs([
        "Mel-Spectrogram", "Constant-Q (CQT)", "Log-STFT", 
        "Wavelet Scalogram", "Spectral Kurtosis", "Modulation Spectrogram"
    ])
    
    spectrogram_types = [
        'mel_spectrogram', 'cqt', 'log_stft', 
        'wavelet_scalogram', 'spectral_kurtosis', 'modulation_spectrogram'
    ]
    
    spectrogram_descriptions = [
        "Energy imbalance, tonal shifts, soft degradation",
        "Harmonic noise, shifted frequency content",
        "Low-frequency rumble (imbalance or looseness)",
        "Short bursts, transient spikes",
        "Impulses and sudden power shifts",
        "Sideband-type modulation (winding faults)"
    ]
    
    for i, (tab, spec_type, description) in enumerate(zip(spectrogram_tabs, spectrogram_types, spectrogram_descriptions)):
        with tab:
            if spec_type in results['spectrograms']:
                # Decode base64 image
                img_data = base64.b64decode(results['spectrograms'][spec_type])
                img = Image.open(BytesIO(img_data))
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(img, caption=f"{spec_type.replace('_', ' ').title()}")
                with col2:
                    st.info(f"**What to look for:**\n{description}")
            else:
                st.warning(f"No data available for {spec_type}")
    
    # Features section
    st.subheader("📈 Extracted Features")
    
    if 'features' in results:
        features_df = pd.DataFrame([results['features']])
        st.dataframe(features_df, use_container_width=True)
        
        # Export buttons
        col1, col2 = st.columns(2)
        with col1:
            csv_data = features_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                file_name="audio_features.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = json.dumps(results['features'], indent=2)
            st.download_button(
                "📥 Download JSON",
                json_data,
                file_name="audio_features.json",
                mime="application/json"
            )

def cleanup_backend():
    """Clean up temporary files on backend"""
    try:
        requests.delete(
            f"{BACKEND_URL}/cleanup",
            data={'session_id': st.session_state.session_id}
        )
    except:
        pass  # Silent fail for cleanup

if __name__ == "__main__":
    main()
