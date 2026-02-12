"""
ASR (Automatic Speech Recognition) Module
Uses OpenAI Whisper for speech-to-text conversion
"""

import os
import whisper
from openai import OpenAI
from typing import Dict, List, Optional
from pathlib import Path
import json


class ASRModule:
    """Handles speech-to-text conversion using Whisper"""

    def __init__(self, config: Dict, use_api: bool = False):
        """
        Initialize ASR module

        Args:
            config: ASR configuration from config.yaml
            use_api: If True, use OpenAI Whisper API; if False, use local model
        """
        self.config = config
        self.use_api = use_api
        self.language = config.get('language', 'ko')
        self.model_name = config.get('model', 'large-v3')
        self.device = config.get('device', 'cpu')

        if use_api:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            print("✓ Using OpenAI Whisper API")
        else:
            print(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            print(f"✓ Whisper model loaded on {self.device}")

    def transcribe_local(self, audio_path: str) -> Dict:
        """
        Transcribe using local Whisper model

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with 'text', 'segments', and 'language'
        """
        print(f"🎤 Transcribing: {audio_path}")

        result = self.model.transcribe(
            audio_path,
            language=self.language,
            task='transcribe',
            verbose=False
        )

        # Format segments with timestamps
        formatted_segments = []
        for segment in result['segments']:
            formatted_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'id': segment['id']
            })

        return {
            'text': result['text'],
            'segments': formatted_segments,
            'language': result['language']
        }

    def transcribe_api(self, audio_path: str) -> Dict:
        """
        Transcribe using OpenAI Whisper API

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with 'text' and 'segments'
        """
        print(f"🎤 Transcribing via API: {audio_path}")

        with open(audio_path, 'rb') as audio_file:
            # Get transcript with timestamps
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=self.language,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        # Format segments
        formatted_segments = []
        if hasattr(transcript, 'segments'):
            for i, segment in enumerate(transcript.segments):
                formatted_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'id': i
                })

        return {
            'text': transcript.text,
            'segments': formatted_segments,
            'language': self.language
        }

    def transcribe(self, audio_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Main transcription method

        Args:
            audio_path: Path to audio file
            output_path: Optional path to save transcript JSON

        Returns:
            Transcription result dictionary
        """
        if self.use_api:
            result = self.transcribe_api(audio_path)
        else:
            result = self.transcribe_local(audio_path)

        print(f"✓ Transcription complete: {len(result['segments'])} segments")

        # Save to file if requested
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 Transcript saved: {output_path}")

        return result


# Utility function
def extract_audio_from_video(video_path: str, output_path: str) -> str:
    """
    Extract audio from video file using FFmpeg

    Args:
        video_path: Path to video file
        output_path: Path for output audio file

    Returns:
        Path to extracted audio file
    """
    import ffmpeg

    print(f"🎬 Extracting audio from: {video_path}")

    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"✓ Audio extracted: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        print(f"❌ FFmpeg error: {e.stderr.decode()}")
        raise
