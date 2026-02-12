"""
Voice Clone Creation Example
How to create a cloned voice from sample audio files
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.voice_cloning import ElevenLabsVoiceCloning
import yaml


def main():
    """Create a voice clone from sample audio files"""

    # Load config
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize ElevenLabs
    voice_cloner = ElevenLabsVoiceCloning(
        config['voice_cloning'],
        voice_id="placeholder"  # Will be replaced
    )

    # STEP 1: Prepare sample audio files
    # Requirements:
    # - 30-60 minutes of clean audio recommended
    # - Single speaker only
    # - Good audio quality
    # - Various emotions and speaking styles
    sample_files = [
        "samples/creator_voice_sample1.mp3",
        "samples/creator_voice_sample2.mp3",
        "samples/creator_voice_sample3.mp3",
        # Add more samples...
    ]

    # Verify files exist
    existing_files = [f for f in sample_files if os.path.exists(f)]
    if not existing_files:
        print("❌ No sample files found! Please add audio samples to the 'samples' directory.")
        print("\nTo extract audio from videos, you can use:")
        print("  ffmpeg -i your_video.mp4 -vn -acodec mp3 sample.mp3")
        return

    print(f"Found {len(existing_files)} sample files")

    # STEP 2: Create voice clone
    try:
        voice_name = "MyCreator_EnglishVoice"  # Give it a descriptive name

        print(f"\n🎭 Creating voice clone: {voice_name}")
        print("This may take several minutes...")

        voice_id = voice_cloner.create_voice_from_samples(
            name=voice_name,
            sample_files=existing_files
        )

        print(f"\n✅ Voice clone created successfully!")
        print(f"Voice ID: {voice_id}")
        print(f"\nSave this Voice ID - you'll use it in your dubbing pipeline:")
        print(f"  python pipeline.py video.mp4 {voice_id}")

        # Save to file for reference
        with open('voice_ids.txt', 'a') as f:
            f.write(f"{voice_name}: {voice_id}\n")
        print(f"\n💾 Voice ID saved to voice_ids.txt")

    except Exception as e:
        print(f"\n❌ Error creating voice: {e}")
        print("\nTroubleshooting:")
        print("1. Check that your ELEVENLABS_API_KEY is valid")
        print("2. Verify you have sufficient API quota")
        print("3. Ensure audio files are in supported format (mp3, wav)")


if __name__ == '__main__':
    main()
