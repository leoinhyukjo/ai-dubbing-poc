import pytest
from src.tts.azure_neural import AzureNeuralTTS
from src.tts.ssml_builder import SSMLBuilder
from src.emotion.emotion_profile import EmotionProfile

def test_azure_tts_initialization():
    """Test Azure TTS initializes with API key"""
    tts = AzureNeuralTTS(
        api_key="test_key",
        region="eastus"
    )
    assert tts.api_key == "test_key"
    assert tts.region == "eastus"

def test_ssml_builder_simple():
    """Test simple SSML generation"""
    builder = SSMLBuilder(voice_name="en-US-JennyNeural")
    ssml = builder.build_simple("Hello world")
    assert 'en-US-JennyNeural' in ssml
    assert 'Hello world' in ssml
    assert '<speak' in ssml
    assert '</speak>' in ssml

def test_ssml_builder_with_emotion():
    """Test SSML generation with emotion parameters"""
    builder = SSMLBuilder(voice_name="en-US-JennyNeural")
    profile = EmotionProfile(
        emotion='happy',
        confidence=0.9,
        pitch_mean=220.0,
        pitch_std=50.0,
        energy_mean=0.6,
        speaking_rate=4.5
    )
    ssml = builder.build_with_emotion("Hello world", profile)
    assert 'express-as' in ssml
    assert 'cheerful' in ssml
    assert 'prosody' in ssml
    assert 'Hello world' in ssml

def test_azure_tts_synthesize_requires_api():
    """Test synthesis requires valid API key"""
    pytest.skip("Requires Azure API key and credits")
