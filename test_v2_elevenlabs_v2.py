#!/usr/bin/env python3
"""
V2 Pipeline Test - ElevenLabs Edition (Improved)
Fixes: accent consistency via segment merging, English model, previous_text continuity
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
VOICE_ID = "EPNTq1UhYta9iHCfyaKd"  # nanika voice
_args = [a for a in sys.argv[1:] if not a.startswith('--')]
INPUT_AUDIO = _args[0] if _args else "/Users/leo/Downloads/nanika sample.mp3"
OUTPUT_DIR = Path("temp/v2_test_improved")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TTS settings
MODEL_ID = "eleven_turbo_v2_5"  # English-focused, consistent accent
MAX_CHUNK_CHARS = 400  # Max characters per TTS call
MERGE_GAP_THRESHOLD = 0.5  # Merge segments with gap < this (seconds)


def merge_segments(segments, gap_threshold=MERGE_GAP_THRESHOLD, max_chars=MAX_CHUNK_CHARS):
    """Merge adjacent short segments into larger chunks for consistent TTS."""
    if not segments:
        return []

    chunks = []
    current = {
        'start': segments[0]['start'],
        'end': segments[0]['end'],
        'text': segments[0]['text'],
    }

    for i in range(1, len(segments)):
        seg = segments[i]
        gap = seg['start'] - current['end']
        combined_text = current['text'] + ' ' + seg['text']

        if gap < gap_threshold and len(combined_text) <= max_chars:
            current['end'] = seg['end']
            current['text'] = combined_text
        else:
            chunks.append(current)
            current = {
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
            }

    chunks.append(current)
    return chunks


def map_emotion_to_elevenlabs(emotion_profile):
    """Map EmotionProfile to ElevenLabs VoiceSettings - tuned for accent consistency."""
    emotion = emotion_profile.emotion

    # Higher stability = more consistent accent
    emotion_params = {
        'happy':   {'stability': 0.55, 'similarity_boost': 0.80, 'style': 0.4},
        'angry':   {'stability': 0.45, 'similarity_boost': 0.75, 'style': 0.5},
        'sad':     {'stability': 0.65, 'similarity_boost': 0.85, 'style': 0.3},
        'neutral': {'stability': 0.70, 'similarity_boost': 0.85, 'style': 0.1},
        'excited': {'stability': 0.50, 'similarity_boost': 0.80, 'style': 0.5},
    }

    params = emotion_params.get(emotion, emotion_params['neutral'])
    print(f"  Emotion: {emotion} -> stability={params['stability']}, style={params['style']}")
    return params


def main():
    print("=" * 70)
    print("V2 Pipeline Test - Improved (Consistent Accent)")
    print("=" * 70)

    # Load cached Stage 1-3 results from previous run
    prev_dir = Path("temp/v2_test")
    emotion_cache = prev_dir / "emotion_profile.json"
    translation_cache = prev_dir / "translation.json"

    if not emotion_cache.exists() or not translation_cache.exists():
        print("\nERROR: Run test_v2_elevenlabs.py first to generate cached results")
        print(f"  Missing: {emotion_cache if not emotion_cache.exists() else translation_cache}")
        sys.exit(1)

    # --- Load emotion profile ---
    print("\n" + "=" * 70)
    print("STAGE 1-3: Loading cached results")
    print("=" * 70)

    from src.emotion.emotion_profile import EmotionProfile
    with open(emotion_cache) as f:
        ep_data = json.load(f)
    emotion_profile = EmotionProfile(**ep_data)
    print(f"  Emotion: {emotion_profile.emotion} (confidence: {emotion_profile.confidence:.2f})")

    with open(translation_cache) as f:
        translated_segments = json.load(f)
    print(f"  Translated segments: {len(translated_segments)}")

    # --- Merge segments ---
    print("\n" + "=" * 70)
    print("STAGE 3.5: Merging Segments for Consistency")
    print("=" * 70)

    chunks = merge_segments(translated_segments)
    print(f"  {len(translated_segments)} segments -> {len(chunks)} chunks")
    print(f"  Avg chunk length: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars")
    for c in chunks[:5]:
        print(f"    [{c['start']:6.1f}s - {c['end']:6.1f}s] {c['text'][:70]}...")

    # Save merged chunks
    with open(OUTPUT_DIR / "merged_chunks.json", 'w') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # --- ElevenLabs TTS with consistency improvements ---
    print("\n" + "=" * 70)
    print(f"STAGE 4: Voice Synthesis (ElevenLabs {MODEL_ID})")
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
    prev_text = ""

    for i, chunk in enumerate(chunks):
        output_path = tts_dir / f"chunk_{i:04d}.mp3"
        text = chunk['text']

        # Build next_text for context
        next_text = chunks[i + 1]['text'][:100] if i + 1 < len(chunks) else ""

        # TTS with previous/next context for accent continuity
        tts_kwargs = dict(
            voice_id=VOICE_ID,
            text=text,
            model_id=MODEL_ID,
            voice_settings=voice_settings,
            output_format="mp3_44100_128",
        )

        # Add context for continuity (if supported by SDK)
        if prev_text:
            tts_kwargs['previous_text'] = prev_text[-200:]
        if next_text:
            tts_kwargs['next_text'] = next_text[:200]

        try:
            audio_gen = client.text_to_speech.convert(**tts_kwargs)
        except TypeError:
            # Fallback if previous_text/next_text not supported
            tts_kwargs.pop('previous_text', None)
            tts_kwargs.pop('next_text', None)
            audio_gen = client.text_to_speech.convert(**tts_kwargs)

        with open(output_path, 'wb') as f:
            for audio_chunk in audio_gen:
                f.write(audio_chunk)

        tts_files.append({
            'path': str(output_path),
            'start': chunk['start'],
            'end': chunk['end'],
            'text': text
        })
        prev_text = text
        print(f"  [{i+1}/{len(chunks)}] chunk_{i:04d}.mp3 ({len(text)} chars)")

    # --- Combine Audio ---
    print("\n" + "=" * 70)
    print("STAGE 5: Combining Audio")
    print("=" * 70)

    from pydub import AudioSegment

    original = AudioSegment.from_file(INPUT_AUDIO)
    total_duration_ms = len(original)

    combined = AudioSegment.silent(duration=total_duration_ms)

    for tts_info in tts_files:
        seg_audio = AudioSegment.from_mp3(tts_info['path'])
        start_ms = int(tts_info['start'] * 1000)
        combined = combined.overlay(seg_audio, position=start_ms)

    output_file = OUTPUT_DIR / "dubbed_output.mp3"
    combined.export(str(output_file), format="mp3")

    print(f"  Output: {output_file}")
    print(f"  Duration: {len(combined) / 1000:.1f}s")

    # --- Done ---
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nImprovements applied:")
    print(f"  - Model: eleven_multilingual_v2 -> {MODEL_ID}")
    print(f"  - Segments: {len(translated_segments)} -> {len(chunks)} merged chunks")
    print(f"  - Stability: 0.35 -> {el_params['stability']} (more consistent accent)")
    print(f"  - Context: previous_text/next_text for continuity")
    print(f"\nOutput: {output_file}")


if __name__ == '__main__':
    main()
