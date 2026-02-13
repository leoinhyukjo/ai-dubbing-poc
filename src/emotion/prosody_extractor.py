"""
Prosody Extraction using librosa
Extracts pitch, energy, and speaking rate for emotion transfer
"""
from typing import Dict
import numpy as np
import librosa


class ProsodyExtractor:
    """Extracts prosodic features from audio"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def extract(self, audio_path: str) -> Dict[str, Dict]:
        """
        Extract prosodic features from audio

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with 'pitch', 'energy', and 'speaking_rate' keys
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Extract pitch using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )

        f0_clean = f0[~np.isnan(f0)]

        pitch_features = {
            'pitch_mean': float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0,
            'pitch_std': float(np.std(f0_clean)) if len(f0_clean) > 0 else 0,
            'pitch_range': float(np.ptp(f0_clean)) if len(f0_clean) > 0 else 0,
            'pitch_median': float(np.median(f0_clean)) if len(f0_clean) > 0 else 0
        }

        # Extract energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        energy_features = {
            'energy_mean': float(np.mean(rms)),
            'energy_std': float(np.std(rms)),
            'energy_max': float(np.max(rms)),
            'energy_min': float(np.min(rms))
        }

        # Extract speaking rate (approximate)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            backtrack=True
        )

        duration_seconds = len(y) / sr
        syllables_per_second = len(onsets) / duration_seconds if duration_seconds > 0 else 0

        speaking_rate_features = {
            'syllables_per_second': float(syllables_per_second),
            'total_syllables': len(onsets),
            'duration_seconds': float(duration_seconds)
        }

        return {
            'pitch': pitch_features,
            'energy': energy_features,
            'speaking_rate': speaking_rate_features
        }
