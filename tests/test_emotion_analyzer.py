import pytest
import numpy as np
import soundfile as sf

from src.emotion.analyzer import EmotionAnalyzer


@pytest.fixture
def sample_audio(tmp_path):
    """Create a synthetic audio sample for testing."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    audio_path = str(tmp_path / "test_sample.wav")
    sf.write(audio_path, audio, sr)
    return audio_path


def test_emotion_analyzer_detects_emotion(sample_audio):
    """Test that EmotionAnalyzer can detect emotions from audio"""
    analyzer = EmotionAnalyzer()
    result = analyzer.analyze(sample_audio)

    assert 'emotion' in result
    assert 'confidence' in result
    # The IEMOCAP model outputs 4 emotions (excited is merged into happy)
    assert result['emotion'] in ['happy', 'sad', 'angry', 'neutral']
    assert 0 <= result['confidence'] <= 1

    # Verify all_scores dict is present and well-formed
    assert 'all_scores' in result
    assert isinstance(result['all_scores'], dict)
    assert len(result['all_scores']) > 0
    for score in result['all_scores'].values():
        assert 0 <= score <= 1
