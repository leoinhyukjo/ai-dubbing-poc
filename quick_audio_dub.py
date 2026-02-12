"""
Quick Audio Dubbing Script
Process audio file through the dubbing pipeline
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from asr_module import ASRModule
from translation import TranslationModule, create_srt
from voice_cloning import create_voice_cloning
from audio_processing import AudioProcessor

# Load environment
load_dotenv()

# Load config
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Paths
audio_path = 'temp/test_2min_sample.wav'
voice_id = 'EPNTq1UhYta9iHCfyaKd'  # 나니까 목소리
output_path = 'temp/dubbed_2min_output.wav'

print("=" * 70)
print("🎙️  QUICK AUDIO DUBBING")
print("=" * 70)
print(f"Input: {audio_path}")
print(f"Voice: 나니까 목소리 ({voice_id})")
print(f"Output: {output_path}")
print()

# Create work directory
work_dir = Path('temp/work')
work_dir.mkdir(parents=True, exist_ok=True)

try:
    # STEP 1: ASR (Speech-to-Text)
    print("─" * 70)
    print("STEP 1: Speech Recognition (Whisper)")
    print("─" * 70)
    asr = ASRModule(config['asr'], use_api=False)
    transcript_path = str(work_dir / "transcript.json")
    transcript = asr.transcribe(audio_path, transcript_path)
    print(f"✅ Transcribed {len(transcript['segments'])} segments")
    print(f"📝 Original text: {transcript['text'][:200]}...")
    print()

    # STEP 2: Translation
    print("─" * 70)
    print("STEP 2: Translation (Claude)")
    print("─" * 70)
    translator = TranslationModule(config['translation'])
    translation = translator.translate_full_script(transcript)
    translation_path = str(work_dir / "translation.json")
    translator.save_translation(translation, translation_path)
    print(f"✅ Translated to {config['translation']['target_language']}")
    print(f"📝 Translated text: {translation['text'][:200]}...")
    print()

    # STEP 3: Voice Cloning
    print("─" * 70)
    print("STEP 3: Voice Synthesis (ElevenLabs)")
    print("─" * 70)
    voice_cloner = create_voice_cloning(config['voice_cloning'], voice_id)
    segments_dir = work_dir / "voice_segments"
    audio_segments = voice_cloner.synthesize_segments(
        translation['segments'],
        str(segments_dir)
    )
    print(f"✅ Generated {len(audio_segments)} voice segments")
    print()

    # STEP 4: Audio Assembly
    print("─" * 70)
    print("STEP 4: Audio Assembly")
    print("─" * 70)
    audio_processor = AudioProcessor(config['audio'])
    timestamps = [(s['start'], s['end']) for s in translation['segments']]
    audio_processor.combine_segments(
        audio_segments,
        timestamps,
        output_path
    )
    print(f"✅ Final audio saved: {output_path}")
    print()

    # Success!
    print("=" * 70)
    print("✅ DUBBING COMPLETE!")
    print("=" * 70)
    print(f"📁 Output: {output_path}")
    print(f"📝 Transcript: {transcript_path}")
    print(f"📝 Translation: {translation_path}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
