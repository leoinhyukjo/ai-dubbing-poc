"""
RVC (Retrieval-based Voice Conversion) Wrapper
Converts TTS voice to creator's voice while preserving prosody
"""
import os
import subprocess
from typing import Optional


class RVCConverter:
    """Wrapper for RVC voice conversion.

    Requires RVC to be installed separately.
    See: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    """

    def __init__(
        self,
        model_path: str,
        rvc_python: str = "python",
        rvc_script_path: str = "external/rvc/infer_cli.py"
    ):
        self.model_path = model_path
        self.rvc_python = rvc_python
        self.rvc_script_path = rvc_script_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RVC model not found: {model_path}")

    def convert(
        self,
        input_audio: str,
        output_audio: str,
        pitch_shift: int = 0,
        filter_radius: int = 3,
        index_rate: float = 0.3
    ) -> str:
        """Convert voice using RVC."""
        cmd = [
            self.rvc_python,
            self.rvc_script_path,
            "--input", input_audio,
            "--output", output_audio,
            "--model", self.model_path,
            "--pitch", str(pitch_shift),
            "--filter_radius", str(filter_radius),
            "--index_rate", str(index_rate)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"RVC conversion: {input_audio} -> {output_audio}")
            return output_audio
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"RVC conversion failed: {e.stderr}")

    def convert_batch(
        self,
        input_audios: list[str],
        output_dir: str,
        **kwargs
    ) -> list[str]:
        """Convert multiple audio files in batch."""
        os.makedirs(output_dir, exist_ok=True)

        outputs = []
        for i, input_audio in enumerate(input_audios):
            output_name = f"converted_{i:04d}.wav"
            output_path = os.path.join(output_dir, output_name)
            self.convert(input_audio, output_path, **kwargs)
            outputs.append(output_path)

        return outputs
