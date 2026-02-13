#!/usr/bin/env python3
"""
V2 Pipeline Test - ElevenLabs Edition
Emotion analysis + Whisper ASR + Claude translation + ElevenLabs TTS (emotion-adjusted)
Skips RVC (no trained model yet)
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
VOICE_ID = "EPNTq1UhYta9iHCfyaKd"  # nanika voice
# Parse args: filter out flags like --resume
_args = [a for a in sys.argv[1:] if not a.startswith('--')]
INPUT_AUDIO = _args[0] if _args else "temp/sample_1min.wav"
OUTPUT_DIR = Path("temp/v2_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def map_emotion_to_elevenlabs(emotion_profile):
    """Map EmotionProfile to ElevenLabs VoiceSettings parameters."""
    emotion = emotion_profile.emotion

    # Emotion-based parameter mapping
    emotion_params = {
        'happy': {'stability': 0.35, 'similarity_boost': 0.75, 'style': 0.6},
        'angry': {'stability': 0.25, 'similarity_boost': 0.70, 'style': 0.7},
        'sad':   {'stability': 0.55, 'similarity_boost': 0.80, 'style': 0.4},
        'neutral': {'stability': 0.65, 'similarity_boost': 0.80, 'style': 0.1},
        'excited': {'stability': 0.30, 'similarity_boost': 0.75, 'style': 0.7},
    }

    params = emotion_params.get(emotion, emotion_params['neutral'])
    print(f"  Emotion: {emotion} -> stability={params['stability']}, style={params['style']}")
    return params


def main():
    print("=" * 70)
    print("V2 Pipeline Test (ElevenLabs Edition)")
    print("=" * 70)

    # Check for --resume flag to skip stages 1-3
    resume_mode = '--resume' in sys.argv

    if not resume_mode and not os.path.exists(INPUT_AUDIO):
        print(f"\nERROR: Input file not found: {INPUT_AUDIO}")
        print(f"Usage: ./venv/bin/python3 test_v2_elevenlabs.py <audio_file>")
        print(f"       ./venv/bin/python3 test_v2_elevenlabs.py --resume  (skip stages 1-3)")
        sys.exit(1)

    print(f"\nInput: {INPUT_AUDIO}")
    print(f"Output dir: {OUTPUT_DIR}")
    if resume_mode:
        print("** RESUME MODE: Skipping stages 1-3, loading cached results **")

    # ========== STAGE 1: Emotion Analysis ==========
    emotion_cache = OUTPUT_DIR / "emotion_profile.json"
    if resume_mode and emotion_cache.exists():
        print("\n" + "=" * 70)
        print("STAGE 1: Emotion Analysis (CACHED)")
        print("=" * 70)
        from src.emotion.emotion_profile import EmotionProfile
        with open(emotion_cache) as f:
            ep_data = json.load(f)
        emotion_profile = EmotionProfile(**ep_data)
        print(f"  Loaded: {emotion_profile.emotion} (confidence: {emotion_profile.confidence:.2f})")
    else:
        print("\n" + "=" * 70)
        print("STAGE 1: Emotion Analysis")
        print("=" * 70)

        from src.emotion.analyzer import EmotionAnalyzer
        from src.emotion.prosody_extractor import ProsodyExtractor
        from src.emotion.emotion_profile import EmotionProfile

        analyzer = EmotionAnalyzer()
        emotion_result = analyzer.analyze(INPUT_AUDIO)

        extractor = ProsodyExtractor()
        prosody_result = extractor.extract(INPUT_AUDIO)

        emotion_profile = EmotionProfile(
            emotion=emotion_result['emotion'],
            confidence=emotion_result['confidence'],
            pitch_mean=prosody_result['pitch']['pitch_mean'],
            pitch_std=prosody_result['pitch']['pitch_std'],
            pitch_range=prosody_result['pitch']['pitch_range'],
            energy_mean=prosody_result['energy']['energy_mean'],
            energy_std=prosody_result['energy']['energy_std'],
            speaking_rate=prosody_result['speaking_rate']['syllables_per_second']
        )

        print(f"  Detected: {emotion_profile.emotion} (confidence: {emotion_profile.confidence:.2f})")
        print(f"  Pitch: {emotion_profile.pitch_mean:.1f} Hz (std: {emotion_profile.pitch_std:.1f})")
        print(f"  Energy: {emotion_profile.energy_mean:.4f}")
        print(f"  Speaking rate: {emotion_profile.speaking_rate:.1f} syl/sec")

        with open(emotion_cache, 'w') as f:
            json.dump(emotion_profile.to_dict(), f, indent=2, ensure_ascii=False)

    # ========== STAGE 2: ASR ==========
    transcript_path = OUTPUT_DIR / "transcript.json"
    if resume_mode and transcript_path.exists():
        print("\n" + "=" * 70)
        print("STAGE 2: Speech Recognition (CACHED)")
        print("=" * 70)
        with open(transcript_path) as f:
            transcript = json.load(f)
        print(f"  Loaded {len(transcript['segments'])} segments")
    else:
        print("\n" + "=" * 70)
        print("STAGE 2: Speech Recognition (Whisper)")
        print("=" * 70)

        from src.asr_module import ASRModule

        asr_config = {'model': 'large-v3', 'language': 'ko', 'device': 'cpu'}
        asr = ASRModule(asr_config, use_api=False)

        transcript = asr.transcribe(INPUT_AUDIO, str(transcript_path))

        print(f"  Segments: {len(transcript['segments'])}")
        for seg in transcript['segments'][:3]:
            print(f"    [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
        if len(transcript['segments']) > 3:
            print(f"    ... +{len(transcript['segments']) - 3} more")

    # ========== STAGE 3: Translation ==========
    translation_path = OUTPUT_DIR / "translation.json"
    if resume_mode and translation_path.exists():
        print("\n" + "=" * 70)
        print("STAGE 3: Translation (CACHED)")
        print("=" * 70)
        with open(translation_path) as f:
            translated_segments = json.load(f)
        print(f"  Loaded {len(translated_segments)} translated segments")
    else:
        print("\n" + "=" * 70)
        print("STAGE 3: Translation (Claude)")
        print("=" * 70)

        from src.translation import TranslationModule

        translation_config = {
            'provider': 'claude',
            'model': 'claude-sonnet-4-5-20250929',
            'target_language': 'en'
        }
        translator = TranslationModule(translation_config)

        translated_segments = translator.translate_segments(transcript['segments'])

        with open(translation_path, 'w') as f:
            json.dump(translated_segments, f, indent=2, ensure_ascii=False)

        print(f"  Translated {len(translated_segments)} segments")

    for seg in translated_segments[:3]:
        print(f"    [{seg['start']:.1f}s] {seg['text']}")

    # ========== STAGE 4: ElevenLabs TTS (emotion-adjusted) ==========
    print("\n" + "=" * 70)
    print("STAGE 4: Voice Synthesis (ElevenLabs + Emotion)")
    print("=" * 70)

    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs as ElevenLabsClient

    client = ElevenLabsClient(api_key=os.getenv('ELEVENLABS_API_KEY'))
    el_params = map_emotion_to_elevenlabs(emotion_profile)

    voice_settings = VoiceSettings(
        stability=el_params['stability'],
        similarity_boost=el_params['similarity_boost'],
        style=el_params['style'],
        use_speaker_boost=True
    )

    tts_dir = OUTPUT_DIR / "tts_segments"
    tts_dir.mkdir(exist_ok=True)

    tts_files = []
    for i, seg in enumerate(translated_segments):
        output_path = tts_dir / f"seg_{i:04d}.mp3"
        text = seg['text']

        audio_gen = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=voice_settings,
            output_format="mp3_44100_128"
        )

        with open(output_path, 'wb') as f:
            for chunk in audio_gen:
                f.write(chunk)

        tts_files.append({
            'path': str(output_path),
            'start': seg['start'],
            'end': seg['end'],
            'text': text
        })
        print(f"  [{i+1}/{len(translated_segments)}] {output_path.name}")

    # ========== STAGE 5: Combine Audio ==========
    print("\n" + "=" * 70)
    print("STAGE 5: Combining Audio")
    print("=" * 70)

    from pydub import AudioSegment

    # Get total duration from original audio
    original = AudioSegment.from_file(INPUT_AUDIO)
    total_duration_ms = len(original)

    # Create silent base
    combined = AudioSegment.silent(duration=total_duration_ms)

    for tts_info in tts_files:
        seg_audio = AudioSegment.from_mp3(tts_info['path'])
        start_ms = int(tts_info['start'] * 1000)
        combined = combined.overlay(seg_audio, position=start_ms)

    output_file = OUTPUT_DIR / "dubbed_output.mp3"
    combined.export(str(output_file), format="mp3")

    print(f"  Output: {output_file}")
    print(f"  Duration: {len(combined) / 1000:.1f}s")

    # ========== DONE ==========
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nResults in: {OUTPUT_DIR}/")
    print(f"  - emotion_profile.json")
    print(f"  - transcript.json")
    print(f"  - translation.json")
    print(f"  - tts_segments/")
    print(f"  - dubbed_output.mp3  <-- final output")


if __name__ == '__main__':
    main()
