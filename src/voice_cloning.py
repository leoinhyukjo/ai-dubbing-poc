"""
Voice Cloning Module
Supports ElevenLabs and Respeecher for high-quality voice synthesis
"""

import os
from typing import Dict, List, Optional
from pathlib import Path
import requests
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import time


class VoiceCloning:
    """Base class for voice cloning providers"""

    def __init__(self, config: Dict):
        self.config = config
        self.provider = config.get('provider', 'elevenlabs')

    def synthesize_segment(self, text: str, output_path: str) -> str:
        """Synthesize a single segment"""
        raise NotImplementedError

    def synthesize_segments(self, segments: List[Dict], output_dir: str) -> List[str]:
        """Synthesize multiple segments"""
        raise NotImplementedError


class ElevenLabsVoiceCloning(VoiceCloning):
    """ElevenLabs voice cloning implementation"""

    def __init__(self, config: Dict, voice_id: str):
        """
        Initialize ElevenLabs voice cloning

        Args:
            config: Voice cloning configuration
            voice_id: ElevenLabs voice ID (or voice name)
        """
        super().__init__(config)

        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment")

        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id

        # Get settings from config
        el_config = config.get('elevenlabs', {})
        self.model = el_config.get('model', 'eleven_multilingual_v2')
        self.voice_settings = VoiceSettings(
            stability=el_config.get('stability', 0.5),
            similarity_boost=el_config.get('similarity_boost', 0.75),
            style=el_config.get('style', 0.0),
            use_speaker_boost=el_config.get('use_speaker_boost', True)
        )

        print(f"✓ ElevenLabs initialized with voice: {voice_id}")
        print(f"  Model: {self.model}")
        print(f"  Settings: stability={self.voice_settings.stability}, "
              f"similarity={self.voice_settings.similarity_boost}")

    def synthesize_segment(self, text: str, output_path: str) -> str:
        """
        Synthesize a single text segment

        Args:
            text: Text to synthesize
            output_path: Path for output audio file

        Returns:
            Path to generated audio file
        """
        try:
            # Generate audio using ElevenLabs
            audio_generator = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id=self.model,
                voice_settings=self.voice_settings,
                output_format="mp3_44100_128"
            )

            # Save audio to file
            with open(output_path, 'wb') as f:
                for chunk in audio_generator:
                    f.write(chunk)

            return output_path

        except Exception as e:
            print(f"❌ ElevenLabs synthesis error: {e}")
            raise

    def synthesize_segments(
        self,
        segments: List[Dict],
        output_dir: str,
        show_progress: bool = True
    ) -> List[str]:
        """
        Synthesize multiple segments

        Args:
            segments: List of segments with 'text' field
            output_dir: Directory for output audio files
            show_progress: Show progress messages

        Returns:
            List of paths to generated audio files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"🎙️ Synthesizing {len(segments)} segments with ElevenLabs...")

        audio_files = []

        for i, segment in enumerate(segments):
            output_path = os.path.join(output_dir, f"segment_{segment['id']:04d}.mp3")

            try:
                self.synthesize_segment(segment['text'], output_path)
                audio_files.append(output_path)

                if show_progress:
                    print(f"  [{i+1}/{len(segments)}] ✓ {output_path}")

                # Rate limiting: ElevenLabs has request limits
                time.sleep(0.1)

            except Exception as e:
                print(f"  [{i+1}/{len(segments)}] ❌ Error: {e}")
                audio_files.append(None)

        print(f"✓ Voice synthesis complete: {len([f for f in audio_files if f])} / {len(segments)} succeeded")
        return audio_files

    def create_voice_from_samples(self, name: str, sample_files: List[str]) -> str:
        """
        Create a new cloned voice from sample audio files

        Args:
            name: Name for the new voice
            sample_files: List of paths to sample audio files (30-60 min recommended)

        Returns:
            Voice ID of the created voice
        """
        print(f"🎭 Creating voice clone: {name}")
        print(f"   Using {len(sample_files)} sample files")

        # Read sample files
        files = []
        for sample_path in sample_files:
            with open(sample_path, 'rb') as f:
                files.append(('files', (os.path.basename(sample_path), f.read(), 'audio/mpeg')))

        try:
            # Create voice using ElevenLabs API
            voice = self.client.voices.add(
                name=name,
                files=files,
                description=f"Custom voice clone created for dubbing"
            )

            print(f"✓ Voice created successfully: {voice.voice_id}")
            return voice.voice_id

        except Exception as e:
            print(f"❌ Voice creation error: {e}")
            raise


class RespeecherVoiceCloning(VoiceCloning):
    """Respeecher voice cloning implementation (Enterprise-grade STS)"""

    def __init__(self, config: Dict, voice_model_id: str):
        """
        Initialize Respeecher voice cloning

        Args:
            config: Voice cloning configuration
            voice_model_id: Respeecher voice model ID
        """
        super().__init__(config)

        self.api_key = os.getenv('RESPEECHER_API_KEY')
        self.project_id = os.getenv('RESPEECHER_PROJECT_ID')

        if not self.api_key or not self.project_id:
            raise ValueError("RESPEECHER_API_KEY or RESPEECHER_PROJECT_ID not found")

        self.voice_model_id = voice_model_id
        self.base_url = "https://api.respeecher.com/v1"

        print(f"✓ Respeecher initialized with model: {voice_model_id}")

    def synthesize_segment(self, text: str, output_path: str, guide_audio: Optional[str] = None) -> str:
        """
        Synthesize using Respeecher (STS mode if guide_audio provided)

        Args:
            text: Text to synthesize (for TTS mode)
            output_path: Path for output audio
            guide_audio: Optional guide audio for STS (Speech-to-Speech) mode

        Returns:
            Path to generated audio file
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Prepare request based on mode
        if guide_audio:
            # STS mode: convert guide audio to target voice
            with open(guide_audio, 'rb') as f:
                files = {'audio': f}
                data = {
                    'project_id': self.project_id,
                    'voice_model_id': self.voice_model_id,
                }
                response = requests.post(
                    f"{self.base_url}/speech-to-speech",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=data,
                    files=files
                )
        else:
            # TTS mode: synthesize from text
            data = {
                'project_id': self.project_id,
                'voice_model_id': self.voice_model_id,
                'text': text
            }
            response = requests.post(
                f"{self.base_url}/text-to-speech",
                headers=headers,
                json=data
            )

        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        else:
            raise Exception(f"Respeecher API error: {response.status_code} - {response.text}")

    def synthesize_segments(self, segments: List[Dict], output_dir: str) -> List[str]:
        """Synthesize multiple segments using Respeecher"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"🎙️ Synthesizing {len(segments)} segments with Respeecher...")

        audio_files = []

        for i, segment in enumerate(segments):
            output_path = os.path.join(output_dir, f"segment_{segment['id']:04d}.wav")

            try:
                # Check if guide audio exists for STS mode
                guide_audio = segment.get('guide_audio', None)
                self.synthesize_segment(segment['text'], output_path, guide_audio)
                audio_files.append(output_path)
                print(f"  [{i+1}/{len(segments)}] ✓")

            except Exception as e:
                print(f"  [{i+1}/{len(segments)}] ❌ Error: {e}")
                audio_files.append(None)

        print(f"✓ Voice synthesis complete")
        return audio_files


# Factory function
def create_voice_cloning(config: Dict, voice_id: str, provider: Optional[str] = None) -> VoiceCloning:
    """
    Factory function to create appropriate voice cloning instance

    Args:
        config: Voice cloning configuration
        voice_id: Voice/model ID for the provider
        provider: Optional provider override ('elevenlabs' or 'respeecher')

    Returns:
        VoiceCloning instance
    """
    if provider is None:
        provider = config.get('provider', 'elevenlabs')

    if provider == 'elevenlabs':
        return ElevenLabsVoiceCloning(config, voice_id)
    elif provider == 'respeecher':
        return RespeecherVoiceCloning(config, voice_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")
