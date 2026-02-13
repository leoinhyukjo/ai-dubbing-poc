#!/usr/bin/env python3
"""
ElevenLabs Dubbing API Test
One API call handles: speaker detection, translation, timing, emotion, voice cloning
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
_args = [a for a in sys.argv[1:] if not a.startswith('--')]
INPUT_AUDIO = _args[0] if _args else "/Users/leo/Downloads/2024 BEST .mp3"
OUTPUT_DIR = Path("temp/dubbing_api_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_LANG = "ko"
TARGET_LANG = "en"


def main():
    print("=" * 70)
    print("ElevenLabs Dubbing API Test")
    print("=" * 70)

    if not os.path.exists(INPUT_AUDIO):
        print(f"\nERROR: Input file not found: {INPUT_AUDIO}")
        sys.exit(1)

    print(f"\nInput: {INPUT_AUDIO}")
    print(f"Direction: {SOURCE_LANG} -> {TARGET_LANG}")
    print(f"Output dir: {OUTPUT_DIR}")

    from elevenlabs.client import ElevenLabs as ElevenLabsClient

    client = ElevenLabsClient(api_key=os.getenv('ELEVENLABS_API_KEY'))

    # ========== STEP 1: Create Dubbing Job ==========
    print("\n" + "=" * 70)
    print("STEP 1: Creating Dubbing Job")
    print("=" * 70)

    with open(INPUT_AUDIO, 'rb') as f:
        result = client.dubbing.create(
            file=f,
            source_lang=SOURCE_LANG,
            target_lang=TARGET_LANG,
            num_speakers=1,
            drop_background_audio=False,
        )

    dubbing_id = result.dubbing_id
    print(f"  Dubbing ID: {dubbing_id}")
    print(f"  Expected duration: {result.expected_duration_sec:.0f}s")

    # ========== STEP 2: Poll for Completion ==========
    print("\n" + "=" * 70)
    print("STEP 2: Waiting for Dubbing to Complete")
    print("=" * 70)

    max_wait = 600  # 10 minutes
    poll_interval = 10
    elapsed = 0

    while elapsed < max_wait:
        status = client.dubbing.get(dubbing_id=dubbing_id)

        if status.status == "dubbed":
            print(f"\n  Dubbing complete! (took ~{elapsed}s)")
            break
        elif status.status == "failed":
            print(f"\n  ERROR: Dubbing failed!")
            if hasattr(status, 'error'):
                print(f"  Error: {status.error}")
            sys.exit(1)
        else:
            print(f"  [{elapsed:>3}s] Status: {status.status}...")
            time.sleep(poll_interval)
            elapsed += poll_interval
    else:
        print(f"\n  TIMEOUT: Dubbing took longer than {max_wait}s")
        sys.exit(1)

    # ========== STEP 3: Download Dubbed Audio ==========
    print("\n" + "=" * 70)
    print("STEP 3: Downloading Dubbed Audio")
    print("=" * 70)

    dubbed_audio = client.dubbing.audio.get(
        dubbing_id=dubbing_id,
        language_code=TARGET_LANG,
    )

    output_file = OUTPUT_DIR / f"dubbed_{TARGET_LANG}.mp3"
    with open(output_file, 'wb') as f:
        for chunk in dubbed_audio:
            f.write(chunk)

    print(f"  Saved: {output_file}")

    # ========== STEP 4: Get Transcript ==========
    print("\n" + "=" * 70)
    print("STEP 4: Getting Transcript")
    print("=" * 70)

    try:
        transcript = client.dubbing.transcript.get(
            dubbing_id=dubbing_id,
            language_code=TARGET_LANG,
            format_type="json",
        )
        import json
        transcript_path = OUTPUT_DIR / f"transcript_{TARGET_LANG}.json"
        with open(transcript_path, 'w') as f:
            json.dump(transcript.dict() if hasattr(transcript, 'dict') else str(transcript), f, indent=2, ensure_ascii=False)
        print(f"  Saved: {transcript_path}")
    except Exception as e:
        print(f"  Transcript fetch failed (non-critical): {e}")

    # ========== DONE ==========
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nResults in: {OUTPUT_DIR}/")
    print(f"  - dubbed_{TARGET_LANG}.mp3  <-- final output")
    print(f"  - transcript_{TARGET_LANG}.json")
    print(f"\nDubbing ID: {dubbing_id} (for future reference)")


if __name__ == '__main__':
    main()
