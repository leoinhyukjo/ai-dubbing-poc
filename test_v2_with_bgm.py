#!/usr/bin/env python3
"""
V2 Pipeline Test - BGM Separation + ElevenLabs
Full pipeline: BGM separation → Emotion → ASR → Translation → TTS → Mix with BGM
"""
import os
import sys
import json
import torch
import torchaudio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
VOICE_ID = "EPNTq1UhYta9iHCfyaKd"  # nanika voice
_args = [a for a in sys.argv[1:] if not a.startswith('--')]
INPUT_AUDIO = _args[0] if _args else "/Users/leo/Downloads/2024 BEST .mp3"
OUTPUT_DIR = Path("temp/v2_bgm_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TTS settings
MODEL_ID = "eleven_multilingual_v2"  # must match nanika voice clone model
MAX_CHUNK_CHARS = 400
MERGE_GAP_THRESHOLD = 0.5
BGM_VOLUME_DB = -10  # BGM volume reduction in dB


def separate_bgm(audio_path, output_dir):
    """Separate vocals and BGM using torchaudio's HDEMUCS model."""
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

    print("  Loading HDEMUCS model...")
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    sample_rate = bundle.sample_rate  # 44100

    print(f"  Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != sample_rate:
        print(f"  Resampling {sr} -> {sample_rate}")
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # Ensure stereo (model expects 2 channels)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    # Process in chunks to manage memory
    print("  Separating sources (this may take a few minutes)...")
    waveform = waveform.unsqueeze(0).to(device)  # (1, channels, samples)

    # Split into manageable segments (~30 sec chunks)
    segment_length = sample_rate * 30
    total_length = waveform.shape[-1]
    vocals_parts = []
    bgm_parts = []

    with torch.no_grad():
        for start in range(0, total_length, segment_length):
            end = min(start + segment_length, total_length)
            chunk = waveform[..., start:end]
            sources = model(chunk)
            # HDEMUCS outputs: drums, bass, other, vocals (index 3)
            vocals_parts.append(sources[0, 3])  # vocals
            bgm_parts.append(sources[0, :3].sum(dim=0))  # drums + bass + other
            progress = min(100, int(end / total_length * 100))
            print(f"    Progress: {progress}%")

    vocals = torch.cat(vocals_parts, dim=-1)
    bgm = torch.cat(bgm_parts, dim=-1)

    # Save separated tracks
    vocals_path = output_dir / "vocals.wav"
    bgm_path = output_dir / "bgm.wav"

    torchaudio.save(str(vocals_path), vocals.cpu(), sample_rate)
    torchaudio.save(str(bgm_path), bgm.cpu(), sample_rate)

    print(f"  Vocals: {vocals_path}")
    print(f"  BGM: {bgm_path}")
    return str(vocals_path), str(bgm_path)


def merge_segments(segments, gap_threshold=MERGE_GAP_THRESHOLD, max_chars=MAX_CHUNK_CHARS):
    """Merge adjacent short segments into larger chunks."""
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
    print("V2 Pipeline - BGM Separation + Emotion-Aware Dubbing")
    print("=" * 70)

    resume_mode = '--resume' in sys.argv

    if not resume_mode and not os.path.exists(INPUT_AUDIO):
        print(f"\nERROR: Input file not found: {INPUT_AUDIO}")
        sys.exit(1)

    print(f"\nInput: {INPUT_AUDIO}")
    print(f"Output dir: {OUTPUT_DIR}")

    # ========== STAGE 0: BGM Separation ==========
    vocals_path = OUTPUT_DIR / "vocals.wav"
    bgm_path = OUTPUT_DIR / "bgm.wav"

    if resume_mode and vocals_path.exists():
        print("\n" + "=" * 70)
        print("STAGE 0: BGM Separation (CACHED)")
        print("=" * 70)
        vocals_path = str(vocals_path)
        bgm_path = str(bgm_path)
        print(f"  Loaded cached vocals and BGM")
    else:
        print("\n" + "=" * 70)
        print("STAGE 0: BGM Separation (HDEMUCS)")
        print("=" * 70)
        vocals_path, bgm_path = separate_bgm(INPUT_AUDIO, OUTPUT_DIR)

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
        print("STAGE 1: Emotion Analysis (on vocals only)")
        print("=" * 70)

        from src.emotion.analyzer import EmotionAnalyzer
        from src.emotion.prosody_extractor import ProsodyExtractor
        from src.emotion.emotion_profile import EmotionProfile

        analyzer = EmotionAnalyzer()
        emotion_result = analyzer.analyze(vocals_path)

        extractor = ProsodyExtractor()
        prosody_result = extractor.extract(vocals_path)

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
        with open(emotion_cache, 'w') as f:
            json.dump(emotion_profile.to_dict(), f, indent=2, ensure_ascii=False)

    # ========== STAGE 2: ASR (on vocals only) ==========
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
        print("STAGE 2: Speech Recognition (Whisper on vocals)")
        print("=" * 70)

        from src.asr_module import ASRModule
        asr_config = {'model': 'large-v3', 'language': 'ko', 'device': 'cpu'}
        asr = ASRModule(asr_config, use_api=False)
        transcript = asr.transcribe(vocals_path, str(transcript_path))

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

    # ========== STAGE 3.5: Merge Segments ==========
    print("\n" + "=" * 70)
    print("STAGE 3.5: Merging Segments")
    print("=" * 70)

    chunks = merge_segments(translated_segments)
    print(f"  {len(translated_segments)} segments -> {len(chunks)} chunks")

    # ========== STAGE 4: ElevenLabs TTS ==========
    tts_cache = OUTPUT_DIR / "tts_segments"
    tts_done_flag = OUTPUT_DIR / "tts_complete.flag"

    if resume_mode and tts_done_flag.exists():
        print("\n" + "=" * 70)
        print("STAGE 4: Voice Synthesis (CACHED)")
        print("=" * 70)
        tts_files = []
        for i, chunk in enumerate(chunks):
            tts_files.append({
                'path': str(tts_cache / f"chunk_{i:04d}.mp3"),
                'start': chunk['start'],
                'end': chunk['end'],
                'text': chunk['text']
            })
        print(f"  Loaded {len(tts_files)} cached TTS files")
    else:
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

        tts_cache.mkdir(exist_ok=True)
        tts_files = []
        prev_text = ""

        for i, chunk in enumerate(chunks):
            output_path = tts_cache / f"chunk_{i:04d}.mp3"
            text = chunk['text']
            next_text = chunks[i + 1]['text'][:100] if i + 1 < len(chunks) else ""

            tts_kwargs = dict(
                voice_id=VOICE_ID,
                text=text,
                model_id=MODEL_ID,
                voice_settings=voice_settings,
                output_format="mp3_44100_128",
            )
            if prev_text:
                tts_kwargs['previous_text'] = prev_text[-200:]
            if next_text:
                tts_kwargs['next_text'] = next_text[:200]

            try:
                audio_gen = client.text_to_speech.convert(**tts_kwargs)
            except TypeError:
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

        # Mark TTS as complete
        tts_done_flag.touch()

    # ========== STAGE 5: Combine Voice + Mix with BGM ==========
    print("\n" + "=" * 70)
    print("STAGE 5: Combining Voice + Mixing with BGM")
    print("=" * 70)

    from pydub import AudioSegment

    # Get original duration
    original = AudioSegment.from_file(INPUT_AUDIO)
    total_duration_ms = len(original)
    print(f"  Original duration: {total_duration_ms / 1000:.1f}s")

    # Create dubbed voice track (with time-stretch to prevent overlap)
    import numpy as np
    import librosa
    import soundfile as sf
    import tempfile

    voice_track = AudioSegment.silent(duration=total_duration_ms)

    MAX_SPEED = 1.3  # Max speed-up before trimming instead

    for idx, tts_info in enumerate(tts_files):
        seg_audio = AudioSegment.from_mp3(tts_info['path'])
        tts_dur_ms = len(seg_audio)
        start_ms = int(tts_info['start'] * 1000)

        # Calculate available window (until next chunk starts or end of audio)
        if idx + 1 < len(tts_files):
            next_start_ms = int(tts_files[idx + 1]['start'] * 1000)
            available_ms = next_start_ms - start_ms
        else:
            available_ms = total_duration_ms - start_ms

        if tts_dur_ms > available_ms and available_ms > 0:
            speed_ratio = tts_dur_ms / available_ms

            if speed_ratio <= MAX_SPEED:
                # Time-stretch: speed up audio to fit window
                samples = np.array(seg_audio.get_array_of_samples(), dtype=np.float32)
                if seg_audio.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                samples = samples / (2**15)

                stretched = librosa.effects.time_stretch(samples, rate=speed_ratio)

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, stretched, seg_audio.frame_rate)
                    seg_audio = AudioSegment.from_wav(tmp.name)
                    os.unlink(tmp.name)

                print(f"    chunk {idx}: stretched {speed_ratio:.2f}x ({tts_dur_ms}ms -> {available_ms}ms)")
            else:
                # Trim: cut audio to fit window
                seg_audio = seg_audio[:available_ms]
                print(f"    chunk {idx}: trimmed ({tts_dur_ms}ms -> {available_ms}ms, would need {speed_ratio:.1f}x)")

        voice_track = voice_track.overlay(seg_audio, position=start_ms)

    # Save voice-only track
    voice_only_path = OUTPUT_DIR / "dubbed_voice_only.mp3"
    voice_track.export(str(voice_only_path), format="mp3")
    print(f"  Voice-only: {voice_only_path}")

    # Mix with original BGM
    if os.path.exists(bgm_path):
        bgm_audio = AudioSegment.from_file(bgm_path)

        # Match duration
        if len(bgm_audio) > total_duration_ms:
            bgm_audio = bgm_audio[:total_duration_ms]
        elif len(bgm_audio) < total_duration_ms:
            bgm_audio = bgm_audio + AudioSegment.silent(duration=total_duration_ms - len(bgm_audio))

        # Reduce BGM volume
        bgm_audio = bgm_audio + BGM_VOLUME_DB

        # Mix
        final = voice_track.overlay(bgm_audio)
        output_file = OUTPUT_DIR / "dubbed_with_bgm.mp3"
        final.export(str(output_file), format="mp3")
        print(f"  BGM volume: {BGM_VOLUME_DB} dB")
        print(f"  Final (voice + BGM): {output_file}")
    else:
        output_file = voice_only_path
        print(f"  No BGM track found, using voice-only output")

    print(f"  Duration: {total_duration_ms / 1000:.1f}s")

    # ========== DONE ==========
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nResults in: {OUTPUT_DIR}/")
    print(f"  - vocals.wav           (separated vocals)")
    print(f"  - bgm.wav              (separated BGM)")
    print(f"  - emotion_profile.json")
    print(f"  - transcript.json")
    print(f"  - translation.json")
    print(f"  - tts_segments/")
    print(f"  - dubbed_voice_only.mp3")
    print(f"  - dubbed_with_bgm.mp3  <-- final output")


if __name__ == '__main__':
    main()
