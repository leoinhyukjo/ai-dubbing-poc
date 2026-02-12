"""
Video Synthesis Module
Combines original video with dubbed audio using FFmpeg
"""

import os
import ffmpeg
from typing import Dict, Optional
from pathlib import Path


class VideoSynthesizer:
    """Handles video synthesis and encoding"""

    def __init__(self, config: Dict):
        """
        Initialize video synthesizer

        Args:
            config: Video configuration from config.yaml
        """
        self.config = config
        self.output_format = config.get('output_format', 'mp4')
        self.video_codec = config.get('video_codec', 'libx264')
        self.audio_codec = config.get('audio_codec', 'aac')
        self.quality = config.get('quality', 'high')
        self.fps = config.get('fps', None)

        # Quality presets
        self.quality_presets = {
            'low': {'crf': 28, 'preset': 'veryfast'},
            'medium': {'crf': 23, 'preset': 'medium'},
            'high': {'crf': 18, 'preset': 'slow'}
        }

        print(f"✓ Video synthesizer initialized ({self.video_codec}/{self.audio_codec})")

    def replace_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        keep_original_audio: bool = False,
        original_audio_volume: float = 0.0
    ) -> str:
        """
        Replace video audio with dubbed audio

        Args:
            video_path: Path to original video
            audio_path: Path to dubbed audio
            output_path: Path for output video
            keep_original_audio: If True, mix with original audio
            original_audio_volume: Volume of original audio (0.0-1.0)

        Returns:
            Path to output video
        """
        print(f"🎬 Synthesizing final video...")

        try:
            # Get video stream
            video_stream = ffmpeg.input(video_path).video

            # Get dubbed audio
            dubbed_audio = ffmpeg.input(audio_path).audio

            # Prepare streams
            if keep_original_audio and original_audio_volume > 0:
                # Mix original and dubbed audio
                original_audio = ffmpeg.input(video_path).audio
                mixed_audio = ffmpeg.filter(
                    [original_audio, dubbed_audio],
                    'amix',
                    inputs=2,
                    weights=f"{original_audio_volume} {1.0}"
                )
                output_streams = [video_stream, mixed_audio]
            else:
                # Use only dubbed audio
                output_streams = [video_stream, dubbed_audio]

            # Get quality settings
            quality_settings = self.quality_presets.get(self.quality, self.quality_presets['high'])

            # Build output
            output = ffmpeg.output(
                *output_streams,
                output_path,
                vcodec=self.video_codec,
                acodec=self.audio_codec,
                **quality_settings,
                **({'r': self.fps} if self.fps else {})
            )

            # Run FFmpeg
            output.overwrite_output().run(quiet=True)

            print(f"✓ Video synthesis complete: {output_path}")
            return output_path

        except ffmpeg.Error as e:
            print(f"❌ FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    def add_subtitles(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str,
        subtitle_style: Optional[Dict] = None
    ) -> str:
        """
        Burn subtitles into video

        Args:
            video_path: Path to input video
            subtitle_path: Path to SRT subtitle file
            output_path: Path for output video
            subtitle_style: Optional dict with style parameters

        Returns:
            Path to output video with subtitles
        """
        print(f"📝 Adding subtitles to video...")

        # Default subtitle style
        if subtitle_style is None:
            subtitle_style = {
                'FontName': 'Arial',
                'FontSize': 24,
                'PrimaryColour': '&HFFFFFF',  # White
                'OutlineColour': '&H000000',  # Black outline
                'Outline': 2,
                'Shadow': 1,
                'Alignment': 2  # Bottom center
            }

        # Create subtitle filter
        style_str = ','.join([f"{k}={v}" for k, v in subtitle_style.items()])

        try:
            # Apply subtitle filter
            output = (
                ffmpeg
                .input(video_path)
                .output(
                    output_path,
                    vf=f"subtitles={subtitle_path}:force_style='{style_str}'",
                    vcodec=self.video_codec,
                    acodec='copy'  # Copy audio stream
                )
                .overwrite_output()
            )

            output.run(quiet=True)

            print(f"✓ Subtitles added: {output_path}")
            return output_path

        except ffmpeg.Error as e:
            print(f"❌ FFmpeg subtitle error: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    def create_preview(
        self,
        video_path: str,
        output_path: str,
        duration: int = 30,
        start_time: int = 0
    ) -> str:
        """
        Create a preview clip from video

        Args:
            video_path: Path to input video
            output_path: Path for output preview
            duration: Duration of preview in seconds
            start_time: Start time in seconds

        Returns:
            Path to preview video
        """
        print(f"🎞️ Creating {duration}s preview from {start_time}s...")

        try:
            output = (
                ffmpeg
                .input(video_path, ss=start_time, t=duration)
                .output(output_path, vcodec=self.video_codec, acodec=self.audio_codec)
                .overwrite_output()
            )

            output.run(quiet=True)

            print(f"✓ Preview created: {output_path}")
            return output_path

        except ffmpeg.Error as e:
            print(f"❌ FFmpeg preview error: {e.stderr.decode() if e.stderr else str(e)}")
            raise


# Utility functions
def get_video_info(video_path: str) -> Dict:
    """
    Get video metadata using FFprobe

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata
    """
    try:
        probe = ffmpeg.probe(video_path)

        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

        return {
            'duration': float(probe['format']['duration']),
            'size': int(probe['format']['size']),
            'video': {
                'codec': video_info['codec_name'],
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'fps': eval(video_info['r_frame_rate']),
                'bitrate': int(video_info.get('bit_rate', 0))
            },
            'audio': {
                'codec': audio_info['codec_name'] if audio_info else None,
                'sample_rate': int(audio_info['sample_rate']) if audio_info else None,
                'channels': int(audio_info['channels']) if audio_info else None
            } if audio_info else None
        }

    except Exception as e:
        print(f"❌ Error getting video info: {e}")
        return {}


def extract_thumbnail(video_path: str, output_path: str, timestamp: float = 0) -> str:
    """
    Extract a thumbnail from video

    Args:
        video_path: Path to video file
        output_path: Path for thumbnail image
        timestamp: Timestamp in seconds

    Returns:
        Path to thumbnail image
    """
    try:
        (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output(output_path, vframes=1)
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except ffmpeg.Error as e:
        print(f"❌ Thumbnail extraction error: {e.stderr.decode() if e.stderr else str(e)}")
        raise
