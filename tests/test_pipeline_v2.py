"""
Tests for Emotion-Aware Dubbing Pipeline V2
Integration tests for the complete workflow
"""
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_pipeline_config_loads():
    """Test config_v2.yaml is valid and has required sections"""
    with open('config_v2.yaml', 'r') as f:
        config = yaml.safe_load(f)

    assert 'emotion' in config
    assert 'asr' in config
    assert 'translation' in config
    assert 'tts' in config
    assert 'voice_conversion' in config
    assert 'audio' in config
    assert 'pipeline' in config


def test_pipeline_v2_imports():
    """Test that EmotionAwareDubbingPipeline can be imported"""
    from src.pipeline_v2 import EmotionAwareDubbingPipeline
    assert EmotionAwareDubbingPipeline is not None


def test_pipeline_full_workflow():
    """Test complete dubbing workflow (requires all APIs)"""
    pytest.skip("Requires all APIs and trained RVC model")
