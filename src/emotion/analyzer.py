"""
Emotion Analysis using SpeechBrain
Detects emotions from audio to preserve them in dubbing

Note: torchaudio 2.10+ removed list_audio_backends().
We monkeypatch it before importing SpeechBrain to avoid the AttributeError.
"""
import torchaudio

# Monkeypatch for torchaudio 2.10+ compatibility with SpeechBrain
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

from typing import Dict

import torch
from speechbrain.inference.interfaces import foreign_class


class EmotionAnalyzer:
    """Analyzes emotions in audio using SpeechBrain pretrained models.

    Uses the wav2vec2-IEMOCAP model which classifies audio into 4 emotions:
    angry, happy, sad, neutral.
    """

    # The IEMOCAP dataset has 5 labels: ang, hap, exc, sad, neu.
    # The wav2vec2-IEMOCAP model merges 'excited' into 'happy' during training,
    # so it only outputs 4 classes in practice. We still map 'exc' here so that
    # if any future model variant does output it, the mapping is correct.
    EMOTION_MAP = {
        'ang': 'angry',
        'hap': 'happy',
        'exc': 'excited',
        'sad': 'sad',
        'neu': 'neutral',
    }

    def __init__(self, model_source: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"):
        print(f"Loading emotion recognition model: {model_source}")
        try:
            self.classifier = foreign_class(
                source=model_source,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir="models/emotion_recognition"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load emotion model: {e}") from e
        # Cache label encoder mapping: {index: short_label}
        self._ind2lab = self.classifier.hparams.label_encoder.ind2lab
        print("Emotion analyzer initialized")

    def analyze(self, audio_path: str) -> Dict[str, any]:
        """Analyze emotions in an audio file.

        Parameters
        ----------
        audio_path : str
            Path to the audio file to analyze.

        Returns
        -------
        dict
            Dictionary with keys:
            - emotion: str - detected emotion (angry, happy, sad, neutral)
            - confidence: float - confidence score between 0 and 1
            - all_scores: dict - scores for all emotion categories
        """
        out_prob, score, index, text_lab = self.classifier.classify_file(audio_path)
        emotion_scores = out_prob.squeeze()
        emotion_label = text_lab[0]

        emotion = self.EMOTION_MAP.get(emotion_label, emotion_label)
        confidence = score.item()

        # Build all_scores using the label encoder's index mapping
        all_scores = {}
        for idx, short_label in self._ind2lab.items():
            full_label = self.EMOTION_MAP.get(short_label, short_label)
            all_scores[full_label] = float(emotion_scores[idx])

        return {
            'emotion': emotion,
            'confidence': float(confidence),
            'all_scores': all_scores
        }
