import pytest
from src.emotion.emotion_profile import EmotionProfile

def test_emotion_profile_creation():
    """Test creating an emotion profile"""
    profile = EmotionProfile(
        emotion='happy',
        confidence=0.85,
        pitch_mean=200.0,
        pitch_std=50.0,
        energy_mean=0.5,
        speaking_rate=4.5
    )
    assert profile.emotion == 'happy'
    assert profile.confidence == 0.85
    assert profile.pitch_mean == 200.0

def test_emotion_profile_to_ssml_params():
    """Test conversion to Azure SSML parameters"""
    profile = EmotionProfile(
        emotion='excited',
        confidence=0.90,
        pitch_mean=250.0,
        pitch_std=60.0,
        energy_mean=0.7,
        speaking_rate=5.0
    )
    ssml_params = profile.to_ssml_params()
    assert 'style' in ssml_params
    assert 'pitch' in ssml_params
    assert 'rate' in ssml_params
    assert 'volume' in ssml_params
    assert ssml_params['style'] == 'excited'

def test_emotion_profile_to_dict():
    """Test JSON serialization"""
    profile = EmotionProfile(
        emotion='sad',
        confidence=0.75,
        pitch_mean=150.0,
        pitch_std=30.0,
        energy_mean=0.3,
        speaking_rate=3.0
    )
    d = profile.to_dict()
    assert d['emotion'] == 'sad'
    assert d['confidence'] == 0.75
    assert d['pitch_mean'] == 150.0
