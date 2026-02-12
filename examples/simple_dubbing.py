"""
Simple Dubbing Example
Quick start guide for using the AI dubbing pipeline
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline import DubbingPipeline


def main():
    """Simple example: Dub a video"""

    # STEP 1: Set up your API keys in .env file
    # See .env.example for required keys

    # STEP 2: Initialize pipeline
    pipeline = DubbingPipeline(config_path='../config.yaml')

    # STEP 3: Prepare your video and voice ID
    video_path = "path/to/your/korean_video.mp4"
    voice_id = "your_elevenlabs_voice_id"  # Or create one using create_voice_example.py
    output_path = "output/dubbed_video.mp4"

    # STEP 4: Run the pipeline!
    print("🚀 Starting dubbing process...")
    results = pipeline.process_video(
        video_path=video_path,
        voice_id=voice_id,
        output_path=output_path,
        use_whisper_api=False  # Set to True to use OpenAI Whisper API
    )

    # STEP 5: Check results
    if results['status'] == 'success':
        print(f"\n✅ Success! Your dubbed video is ready:")
        print(f"   {results['output_path']}")
        print(f"\nSubtitles are also available:")
        print(f"   {results['steps']['translation']['subtitles']}")
    else:
        print(f"\n❌ Dubbing failed: {results.get('error', 'Unknown error')}")


if __name__ == '__main__':
    main()
