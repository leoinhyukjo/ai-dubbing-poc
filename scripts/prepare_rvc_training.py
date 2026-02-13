#!/usr/bin/env python3
"""
Prepare audio samples for RVC training
Resamples, denoises, and splits audio files
"""
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np


def prepare_sample(input_path: str, output_dir: Path, target_sr: int = 40000):
    """Prepare a single audio sample for RVC training."""
    print(f"Processing: {input_path}")
    y, sr = librosa.load(input_path, sr=target_sr)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    chunk_length = target_sr * 8  # 8 seconds
    chunks = [
        y_trimmed[i:i + chunk_length]
        for i in range(0, len(y_trimmed), chunk_length)
    ]

    input_name = Path(input_path).stem
    for i, chunk in enumerate(chunks):
        if len(chunk) > target_sr * 2:  # At least 2 seconds
            output_path = output_dir / f"{input_name}_chunk_{i:03d}.wav"
            sf.write(output_path, chunk, target_sr)
            print(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Prepare audio samples for RVC training")
    parser.add_argument('--input_dir', required=True, help='Input directory with voice samples')
    parser.add_argument('--output_dir', required=True, help='Output directory for prepared samples')
    parser.add_argument('--sample_rate', type=int, default=40000, help='Target sample rate')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.glob('**/*.wav')) + list(input_dir.glob('**/*.mp3'))
    print(f"Found {len(audio_files)} audio files")

    for audio_file in audio_files:
        prepare_sample(str(audio_file), output_dir, args.sample_rate)

    print(f"\nPrepared {len(audio_files)} files -> {output_dir}")


if __name__ == '__main__':
    main()
