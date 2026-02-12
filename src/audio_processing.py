"""
Audio Processing Module
Handles audio manipulation, BGM separation, and timeline alignment
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
from pydub import AudioSegment
import librosa
import soundfile as sf


class AudioProcessor:
    """Audio processing utilities"""

    def __init__(self, config: Dict):
        """
        Initialize audio processor

        Args:
            config: Audio configuration from config.yaml
        """
        self.config = config
        self.sample_rate = config.get('sample_rate', 44100)
        self.channels = config.get('channels', 1)
        self.separate_bgm = config.get('separate_bgm', True)
        self.demucs_model = config.get('demucs_model', 'htdemucs')

        print(f"✓ Audio processor initialized (SR: {self.sample_rate}Hz)")

    def separate_vocals_bgm(self, audio_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Separate vocals and background music using Demucs

        Args:
            audio_path: Path to input audio file
            output_dir: Directory for separated outputs

        Returns:
            Tuple of (vocals_path, bgm_path)
        """
        if not self.separate_bgm:
            print("⚠️ BGM separation disabled in config")
            return audio_path, None

        print(f"🎵 Separating vocals and BGM using Demucs...")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            # Run Demucs
            cmd = [
                'demucs',
                '--two-stems', 'vocals',
                '--out', output_dir,
                '--name', self.demucs_model,
                audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"⚠️ Demucs failed: {result.stderr}")
                return audio_path, None

            # Find output files
            audio_name = Path(audio_path).stem
            demucs_output = Path(output_dir) / self.demucs_model / audio_name

            vocals_path = str(demucs_output / "vocals.wav")
            bgm_path = str(demucs_output / "no_vocals.wav")

            if os.path.exists(vocals_path) and os.path.exists(bgm_path):
                print(f"✓ Vocals: {vocals_path}")
                print(f"✓ BGM: {bgm_path}")
                return vocals_path, bgm_path
            else:
                print("⚠️ Output files not found, using original audio")
                return audio_path, None

        except Exception as e:
            print(f"⚠️ BGM separation error: {e}")
            return audio_path, None

    def combine_segments(
        self,
        segment_files: List[str],
        timestamps: List[Tuple[float, float]],
        output_path: str,
        sample_rate: Optional[int] = None,
        enable_time_stretch: bool = True
    ) -> str:
        """
        Combine multiple audio segments into a single file with proper timing

        Args:
            segment_files: List of audio file paths
            timestamps: List of (start, end) timestamps in seconds
            output_path: Path for output combined audio
            sample_rate: Optional sample rate (defaults to config)
            enable_time_stretch: Apply time-stretching to match expected duration

        Returns:
            Path to combined audio file
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        print(f"🔗 Combining {len(segment_files)} audio segments...")
        if enable_time_stretch:
            print(f"   Time-stretching enabled for duration matching")

        # Calculate total duration
        total_duration = max([end for start, end in timestamps])

        # Create silent base track
        silence = AudioSegment.silent(
            duration=int(total_duration * 1000),  # milliseconds
            frame_rate=sample_rate
        )

        # Overlay each segment at its timestamp
        combined = silence
        stretched_count = 0

        for i, (audio_file, (start, end)) in enumerate(zip(segment_files, timestamps)):
            if audio_file is None or not os.path.exists(audio_file):
                print(f"  [{i+1}/{len(segment_files)}] ⚠️ Missing file, skipping")
                continue

            try:
                # Expected duration
                expected_duration = end - start

                # Load segment to check actual duration
                segment = AudioSegment.from_file(audio_file)
                actual_duration = len(segment) / 1000.0  # Convert ms to seconds

                # Time-stretch if needed (tolerance: 5% difference)
                duration_ratio = actual_duration / expected_duration
                if enable_time_stretch and abs(duration_ratio - 1.0) > 0.05:
                    # Create temp file for stretched audio
                    temp_stretched = audio_file.replace('.mp3', '_stretched.wav')

                    # Apply time-stretching using librosa
                    self.match_audio_duration(audio_file, expected_duration, temp_stretched)

                    # Load the stretched audio
                    segment = AudioSegment.from_file(temp_stretched)
                    stretched_count += 1

                    # Clean up temp file
                    try:
                        os.remove(temp_stretched)
                    except:
                        pass

                # Convert to target sample rate if needed
                if segment.frame_rate != sample_rate:
                    segment = segment.set_frame_rate(sample_rate)

                # Set to mono if needed
                if self.channels == 1 and segment.channels > 1:
                    segment = segment.set_channels(1)

                # Calculate position in milliseconds
                position_ms = int(start * 1000)

                # Overlay segment
                combined = combined.overlay(segment, position=position_ms)

                print(f"  [{i+1}/{len(segment_files)}] ✓ @{start:.2f}s")

            except Exception as e:
                print(f"  [{i+1}/{len(segment_files)}] ❌ Error: {e}")

        # Export combined audio
        combined.export(output_path, format="wav")
        print(f"✓ Combined audio saved: {output_path}")
        if enable_time_stretch:
            print(f"   Time-stretched segments: {stretched_count}/{len(segment_files)}")

        return output_path

    def mix_audio_tracks(
        self,
        voice_path: str,
        bgm_path: str,
        output_path: str,
        voice_volume: float = 1.0,
        bgm_volume: float = 0.3
    ) -> str:
        """
        Mix voice and background music tracks

        Args:
            voice_path: Path to voice audio
            bgm_path: Path to background music
            output_path: Path for output mixed audio
            voice_volume: Volume multiplier for voice (default: 1.0)
            bgm_volume: Volume multiplier for BGM (default: 0.3)

        Returns:
            Path to mixed audio file
        """
        print(f"🎚️ Mixing voice and BGM...")

        # Load audio files
        voice = AudioSegment.from_file(voice_path)
        bgm = AudioSegment.from_file(bgm_path)

        # Ensure same frame rate
        if voice.frame_rate != bgm.frame_rate:
            bgm = bgm.set_frame_rate(voice.frame_rate)

        # Ensure same length (pad or trim BGM)
        if len(bgm) < len(voice):
            # Loop BGM if too short
            loops = (len(voice) // len(bgm)) + 1
            bgm = bgm * loops

        bgm = bgm[:len(voice)]

        # Adjust volumes (convert to dB)
        voice_db = 20 * (voice_volume if voice_volume > 0 else 0.01)
        bgm_db = 20 * (bgm_volume if bgm_volume > 0 else 0.01)

        voice = voice + voice_db
        bgm = bgm + bgm_db

        # Mix tracks
        mixed = voice.overlay(bgm)

        # Export
        mixed.export(output_path, format="wav")
        print(f"✓ Mixed audio saved: {output_path}")

        return output_path

    def normalize_audio(self, audio_path: str, output_path: str, target_db: float = -20.0) -> str:
        """
        Normalize audio to target loudness

        Args:
            audio_path: Input audio file
            output_path: Output audio file
            target_db: Target loudness in dB

        Returns:
            Path to normalized audio
        """
        audio = AudioSegment.from_file(audio_path)

        # Calculate current loudness
        current_db = audio.dBFS

        # Calculate adjustment needed
        adjustment = target_db - current_db

        # Apply normalization
        normalized = audio + adjustment

        # Export
        normalized.export(output_path, format="wav")
        print(f"✓ Audio normalized: {current_db:.1f}dB → {target_db:.1f}dB")

        return output_path

    def match_audio_duration(
        self,
        audio_path: str,
        target_duration: float,
        output_path: str
    ) -> str:
        """
        Adjust audio speed to match target duration (time-stretching)

        Args:
            audio_path: Input audio file
            target_duration: Target duration in seconds
            output_path: Output audio file

        Returns:
            Path to adjusted audio
        """
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Calculate current duration
        current_duration = len(y) / sr

        # Calculate stretch rate
        rate = current_duration / target_duration

        print(f"⏱️ Time-stretching: {current_duration:.2f}s → {target_duration:.2f}s (rate: {rate:.3f})")

        # Apply time-stretching
        y_stretched = librosa.effects.time_stretch(y, rate=rate)

        # Save
        sf.write(output_path, y_stretched, sr)
        print(f"✓ Duration matched: {output_path}")

        return output_path


# Utility function
def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0  # Convert ms to seconds
