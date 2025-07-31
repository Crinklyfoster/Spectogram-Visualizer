"""
Audio Processing Utilities
Handles audio loading and feature extraction
"""

import librosa
import numpy as np
from scipy import stats
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Main audio processing class"""
    
    def __init__(self):
        self.default_sr = 22050  # Default sample rate for consistency
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa
            audio_data, sample_rate = librosa.load(
                file_path, 
                sr=self.default_sr,  # Resample to consistent rate
                mono=True  # Convert to mono
            )
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            logger.info(f"Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract comprehensive audio features
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Time-domain features
            features.update(self._extract_time_domain_features(audio_data, sample_rate))
            
            # Frequency-domain features
            features.update(self._extract_frequency_domain_features(audio_data, sample_rate))
            
            # Spectral features
            features.update(self._extract_spectral_features(audio_data, sample_rate))
            
            # Rhythm and tempo features
            features.update(self._extract_rhythm_features(audio_data, sample_rate))
            
            logger.info(f"Extracted {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def _extract_time_domain_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract time-domain features"""
        features = {}
        
        # RMS Energy
        features['rms_energy'] = float(np.sqrt(np.mean(audio_data**2)))
        
        # Zero Crossing Rate
        features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)))
        
        # Peak amplitude
        features['peak_amplitude'] = float(np.max(np.abs(audio_data)))
        
        # Crest factor
        rms = features['rms_energy']
        features['crest_factor'] = features['peak_amplitude'] / rms if rms > 0 else 0
        
        # Statistical moments
        features['mean_amplitude'] = float(np.mean(np.abs(audio_data)))
        features['std_amplitude'] = float(np.std(audio_data))
        features['skewness'] = float(stats.skew(audio_data))
        features['kurtosis'] = float(stats.kurtosis(audio_data))
        
        return features
    
    def _extract_frequency_domain_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract frequency-domain features"""
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
        features['spectral_contrast_std'] = float(np.std(spectral_contrast))
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)
        features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        features['spectral_flatness_std'] = float(np.std(spectral_flatness))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        return features
    
    def _extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract advanced spectral features"""
        features = {}
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        for i in range(mfccs.shape[0]):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        
        # Tonnetz (Tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sample_rate)
        features['tonnetz_mean'] = float(np.mean(tonnetz))
        features['tonnetz_std'] = float(np.std(tonnetz))
        
        return features
    
    def _extract_rhythm_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract rhythm and tempo features"""
        features = {}
        
        try:
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
            
            # Rhythm patterns
            if len(beats) > 1:
                beat_intervals = np.diff(beats) / sample_rate
                features['beat_interval_mean'] = float(np.mean(beat_intervals))
                features['beat_interval_std'] = float(np.std(beat_intervals))
            else:
                features['beat_interval_mean'] = 0.0
                features['beat_interval_std'] = 0.0
                
        except Exception as e:
            logger.warning(f"Rhythm feature extraction failed: {e}")
            features['tempo'] = 0.0
            features['beat_count'] = 0
            features['beat_interval_mean'] = 0.0
            features['beat_interval_std'] = 0.0
        
        return features
