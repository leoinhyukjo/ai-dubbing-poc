"""
RVC Model Manager
Handles loading and managing RVC voice models
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional


class RVCModelManager:
    """Manages RVC voice models"""

    def __init__(self, models_dir: str = "models/rvc"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.models_index_path = self.models_dir / "models_index.json"
        self.models_index = self._load_models_index()

    def _load_models_index(self) -> Dict:
        if self.models_index_path.exists():
            with open(self.models_index_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_models_index(self):
        with open(self.models_index_path, 'w') as f:
            json.dump(self.models_index, f, indent=2)

    def register_model(
        self,
        name: str,
        model_path: str,
        metadata: Optional[Dict] = None
    ):
        """Register a new RVC model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.models_index[name] = {
            'path': model_path,
            'metadata': metadata or {}
        }
        self._save_models_index()
        print(f"Registered RVC model: {name}")

    def get_model_path(self, name: str) -> str:
        """Get path to registered model."""
        if name not in self.models_index:
            raise ValueError(f"Model not found: {name}. Available: {list(self.models_index.keys())}")
        return self.models_index[name]['path']

    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self.models_index.keys())
