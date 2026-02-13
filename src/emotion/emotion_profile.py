"""
Emotion Profile data class
Stores extracted emotional and prosodic features for transfer to TTS
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class EmotionProfile:
    """Stores emotional and prosodic features from source audio."""
    emotion: str
    confidence: float
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    speaking_rate: float
    pitch_range: float = 0.0
    energy_std: float = 0.0

    def to_ssml_params(self) -> Dict[str, any]:
        """Convert emotion profile to Azure SSML parameters."""
        emotion_to_style = {
            'happy': 'cheerful',
            'sad': 'sad',
            'angry': 'angry',
            'excited': 'excited',
            'neutral': 'default'
        }
        style = emotion_to_style.get(self.emotion, 'default')

        # Pitch adjustment relative to baseline 200 Hz
        baseline_pitch = 200.0
        pitch_ratio = self.pitch_mean / baseline_pitch
        pitch_adjustment = f"{(pitch_ratio - 1.0) * 100:.1f}%"

        # Speaking rate relative to baseline 4 syllables/sec
        baseline_rate = 4.0
        rate_ratio = self.speaking_rate / baseline_rate
        rate_value = f"{rate_ratio:.2f}"

        # Energy to volume mapping
        if self.energy_mean > 0.7:
            volume = "loud"
        elif self.energy_mean > 0.4:
            volume = "medium"
        else:
            volume = "soft"

        return {
            'style': style,
            'pitch': pitch_adjustment,
            'rate': rate_value,
            'volume': volume
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'emotion': self.emotion,
            'confidence': self.confidence,
            'pitch_mean': self.pitch_mean,
            'pitch_std': self.pitch_std,
            'pitch_range': self.pitch_range,
            'energy_mean': self.energy_mean,
            'energy_std': self.energy_std,
            'speaking_rate': self.speaking_rate
        }
