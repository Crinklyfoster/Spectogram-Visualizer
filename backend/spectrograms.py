"""
Spectrogram Generation Utilities
Generates various types of spectrograms for motor audio analysis
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis
import pywt
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class SpectrogramGenerator:
    """Generate various spectrograms for motor audio analysis"""
    
    def __init__(self, figsize=(10, 6), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
        # Set matplotlib backend for server environment
        plt.switch_backend('Agg')
    
    def _save_plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return img_base64
    
    def generate_mel_spectrogram(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Generate Mel-spectrogram
        Good for: Energy imbalance, tonal shifts, soft degradation
        """
        try:
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=sample_rate,
                n_mels=128,
                fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figsize)
            img = librosa.display.specshow(
                mel_spec_db, 
                x_axis='time', 
                y_axis='mel', 
                sr=sample_rate,
                fmax=8000,
                ax=ax
            )
            ax.set_title('Mel-Spectrogram')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Mel Frequency')
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            
            return self._save_plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Mel-spectrogram generation failed: {e}")
            raise
    
    def generate_cqt(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Generate Constant-Q Transform
        Good for: Harmonic noise, shifted frequency content
        """
        try:
            # Compute CQT
            cqt = librosa.cqt(audio_data, sr=sample_rate, n_bins=84)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figsize)
            img = librosa.display.specshow(
                cqt_db, 
                x_axis='time', 
                y_axis='cqt_note',
                sr=sample_rate,
                ax=ax
            )
            ax.set_title('Constant-Q Transform (CQT)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Note)')
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            
            return self._save_plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"CQT generation failed: {e}")
            raise
    
    def generate_log_stft(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Generate Log-STFT
        Good for: Low-frequency rumble (imbalance or looseness)
        """
        try:
            # Compute STFT
            stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
            stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            
            # Create plot with log frequency scale
            fig, ax = plt.subplots(figsize=self.figsize)
            img = librosa.display.specshow(
                stft_db, 
                x_axis='time', 
                y_axis='log',
                sr=sample_rate,
                ax=ax
            )
            ax.set_title('Log-STFT')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            
            return self._save_plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Log-STFT generation failed: {e}")
            raise
    
    def generate_wavelet_scalogram(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Generate Wavelet Scalogram using Continuous Wavelet Transform
        Good for: Short bursts, transient spikes
        """
        try:
            # Parameters for CWT
            scales = np.arange(1, 128)
            wavelet = 'morl'  # Morlet wavelet
            
            # Downsample for computation efficiency
            if len(audio_data) > 10000:
                audio_data = signal.decimate(audio_data, q=2)
                effective_sr = sample_rate // 2
            else:
                effective_sr = sample_rate
            
            # Compute CWT
            coefficients, frequencies = pywt.cwt(audio_data, scales, wavelet, 1/effective_sr)
            
            # Convert to dB scale
            power = np.abs(coefficients) ** 2
            power_db = 10 * np.log10(power + 1e-10)
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figsize)
            time = np.linspace(0, len(audio_data)/effective_sr, len(audio_data))
            
            img = ax.imshow(
                power_db, 
                extent=[0, time[-1], frequencies[-1], frequencies[0]], 
                cmap='viridis', 
                aspect='auto',
                origin='upper'
            )
            ax.set_title('Wavelet Scalogram')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            plt.colorbar(img, ax=ax, label='Power (dB)')
            
            return self._save_plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Wavelet scalogram generation failed: {e}")
            raise
    
    def generate_spectral_kurtosis(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Generate Spectral Kurtosis
        Good for: Impulses and sudden power shifts
        """
        try:
            # Compute STFT
            f, t, stft = signal.stft(audio_data, fs=sample_rate, nperseg=1024, noverlap=512)
            
            # Compute spectral kurtosis
            stft_magnitude = np.abs(stft)
            spectral_kurt = np.zeros_like(stft_magnitude)
            
            for i in range(stft_magnitude.shape[0]):
                if stft_magnitude.shape[1] > 3:  # Need minimum samples for kurtosis
                    spectral_kurt[i, :] = kurtosis(stft_magnitude[i, :])
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figsize)
            img = ax.pcolormesh(t, f, spectral_kurt, shading='gouraud', cmap='RdYlBu_r')
            ax.set_title('Spectral Kurtosis')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            plt.colorbar(img, ax=ax, label='Kurtosis')
            
            return self._save_plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Spectral kurtosis generation failed: {e}")
            raise
    
    def generate_modulation_spectrogram(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Generate Modulation Spectrogram
        Good for: Sideband-type modulation (winding faults)
        """
        try:
            # Compute STFT first
            stft = librosa.stft(audio_data, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            
            # Compute modulation spectrogram by taking FFT along time axis
            mod_spec = np.zeros((magnitude.shape[0], magnitude.shape[1]//2))
            
            for freq_bin in range(magnitude.shape[0]):
                if magnitude.shape[1] > 1:
                    # FFT along time axis for each frequency bin
                    time_series = magnitude[freq_bin, :]
                    mod_fft = np.fft.fft(time_series)
                    mod_spec[freq_bin, :] = np.abs(mod_fft[:mod_spec.shape[1]])
            
            # Convert to dB
            mod_spec_db = librosa.amplitude_to_db(mod_spec + 1e-10, ref=np.max)
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Time and frequency axes
            modulation_freqs = np.fft.fftfreq(magnitude.shape[1], d=256/sample_rate)[:mod_spec.shape[1]]
            acoustic_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
            
            img = ax.imshow(
                mod_spec_db, 
                extent=[0, modulation_freqs[-1], acoustic_freqs[-1], acoustic_freqs[0]], 
                cmap='viridis', 
                aspect='auto',
                origin='upper'
            )
            ax.set_title('Modulation Spectrogram')
            ax.set_xlabel('Modulation Frequency (Hz)')
            ax.set_ylabel('Acoustic Frequency (Hz)')
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            
            return self._save_plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Modulation spectrogram generation failed: {e}")
            raise
