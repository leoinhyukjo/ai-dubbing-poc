import pytest
import json
from pathlib import Path
from src.voice_conversion.rvc_converter import RVCConverter
from src.voice_conversion.model_manager import RVCModelManager

def test_rvc_converter_initialization(tmp_path):
    """Test RVC converter initializes with model path"""
    # Create a fake model file
    model_file = tmp_path / "test_model.pth"
    model_file.touch()

    converter = RVCConverter(model_path=str(model_file))
    assert converter.model_path == str(model_file)

def test_rvc_converter_missing_model():
    """Test RVC converter raises error for missing model"""
    with pytest.raises(FileNotFoundError):
        RVCConverter(model_path="/nonexistent/model.pth")

def test_rvc_model_manager_register(tmp_path):
    """Test model registration"""
    manager = RVCModelManager(models_dir=str(tmp_path))

    # Create a fake model file
    model_file = tmp_path / "test_model.pth"
    model_file.touch()

    manager.register_model(
        name="test_voice",
        model_path=str(model_file),
        metadata={"creator": "Test"}
    )

    assert "test_voice" in manager.list_models()
    assert manager.get_model_path("test_voice") == str(model_file)

def test_rvc_model_manager_missing_model(tmp_path):
    """Test error on getting unregistered model"""
    manager = RVCModelManager(models_dir=str(tmp_path))

    with pytest.raises(ValueError):
        manager.get_model_path("nonexistent")

def test_rvc_converter_convert(tmp_path):
    """Test voice conversion (skipped without real model)"""
    pytest.skip("Requires trained RVC model")
