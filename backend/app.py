"""
Flask Backend for Motor Audio Analysis
Handles audio processing, feature extraction, and spectrogram generation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
import logging

from audio_utils import AudioProcessor
from spectrograms import SpectrogramGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['TEMP_FOLDER'] = os.path.join(os.path.dirname(__file__), 'temp_files')

# Ensure temp directory exists
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

# Initialize processors
audio_processor = AudioProcessor()
spectrogram_generator = SpectrogramGenerator()

# Session storage for temporary files
session_files = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Motor Audio Analysis API is running'})

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """
    Main analysis endpoint
    Accepts audio file and returns spectrograms + features
    """
    try:
        # Validate request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'wav', 'mp3', 'flac', 'm4a'}
        file_ext = audio_file.filename.rsplit('.', 1)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(f"{session_id}_{audio_file.filename}")
        temp_path = os.path.join(app.config['TEMP_FOLDER'], filename)
        audio_file.save(temp_path)
        
        # Store file path for cleanup
        if session_id not in session_files:
            session_files[session_id] = []
        session_files[session_id].append(temp_path)
        
        logger.info(f"Processing audio file: {filename}")
        
        # Load and validate audio
        audio_data, sample_rate = audio_processor.load_audio(temp_path)
        
        # Check duration (≤20 seconds)
        duration = len(audio_data) / sample_rate
        if duration > 20:
            return jsonify({'error': f'Audio too long: {duration:.1f}s (max 20s)'}), 400
        
        # Extract features
        logger.info("Extracting audio features...")
        features = audio_processor.extract_features(audio_data, sample_rate)
        
        # Generate spectrograms
        logger.info("Generating spectrograms...")
        spectrograms = {}
        
        try:
            spectrograms['mel_spectrogram'] = spectrogram_generator.generate_mel_spectrogram(
                audio_data, sample_rate
            )
        except Exception as e:
            logger.warning(f"Failed to generate mel spectrogram: {e}")
        
        try:
            spectrograms['cqt'] = spectrogram_generator.generate_cqt(
                audio_data, sample_rate
            )
        except Exception as e:
            logger.warning(f"Failed to generate CQT: {e}")
        
        try:
            spectrograms['log_stft'] = spectrogram_generator.generate_log_stft(
                audio_data, sample_rate
            )
        except Exception as e:
            logger.warning(f"Failed to generate Log-STFT: {e}")
        
        try:
            spectrograms['wavelet_scalogram'] = spectrogram_generator.generate_wavelet_scalogram(
                audio_data, sample_rate
            )
        except Exception as e:
            logger.warning(f"Failed to generate wavelet scalogram: {e}")
        
        try:
            spectrograms['spectral_kurtosis'] = spectrogram_generator.generate_spectral_kurtosis(
                audio_data, sample_rate
            )
        except Exception as e:
            logger.warning(f"Failed to generate spectral kurtosis: {e}")
        
        try:
            spectrograms['modulation_spectrogram'] = spectrogram_generator.generate_modulation_spectrogram(
                audio_data, sample_rate
            )
        except Exception as e:
            logger.warning(f"Failed to generate modulation spectrogram: {e}")
        
        # Prepare response
        response = {
            'session_id': session_id,
            'filename': audio_file.filename,
            'duration': duration,
            'sample_rate': sample_rate,
            'features': features,
            'spectrograms': spectrograms,
            'message': 'Analysis completed successfully'
        }
        
        logger.info(f"Analysis completed for session: {session_id}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/cleanup', methods=['DELETE'])
def cleanup_session():
    """Clean up temporary files for a session"""
    try:
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session_id provided'}), 400
        
        # Remove files associated with session
        if session_id in session_files:
            for file_path in session_files[session_id]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove file {file_path}: {e}")
            
            del session_files[session_id]
        
        return jsonify({'message': 'Cleanup completed'})
        
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large (max 50MB)'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Motor Audio Analysis Backend...")
    app.run(debug=True, host='0.0.0.0', port=5000)
