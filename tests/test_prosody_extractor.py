import pytest
import numpy as np
import soundfile as sf
from src.emotion.prosody_extractor import ProsodyExtractor

@pytest.fixture
def sample_audio(tmp_path):
    """Create a synthetic audio sample for testing"""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_path = str(tmp_path / "test_sample.wav")
    sf.write(audio_path, audio, sr)
    return audio_path

def test_prosody_extractor_extracts_pitch(sample_audio):
    """Test pitch extraction from audio"""
    extractor = ProsodyExtractor()
    result = extractor.extract(sample_audio)
    assert 'pitch' in result
    assert 'pitch_mean' in result['pitch']
    assert 'pitch_std' in result['pitch']
    assert 'pitch_range' in result['pitch']
    assert result['pitch']['pitch_mean'] > 0

def test_prosody_extractor_extracts_energy(sample_audio):
    """Test energy/volume extraction"""
    extractor = ProsodyExtractor()
    result = extractor.extract(sample_audio)
    assert 'energy' in result
    assert 'energy_mean' in result['energy']
    assert 'energy_std' in result['energy']

def test_prosody_extractor_extracts_speaking_rate(sample_audio):
    """Test speaking rate extraction"""
    extractor = ProsodyExtractor()
    result = extractor.extract(sample_audio)
    assert 'speaking_rate' in result
    assert result['speaking_rate']['syllables_per_second'] >= 0
