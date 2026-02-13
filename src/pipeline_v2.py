"""
Emotion-Aware Dubbing Pipeline V2
Complete workflow with automated emotional transfer
"""
import os
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from src.emotion.analyzer import EmotionAnalyzer
from src.emotion.prosody_extractor import ProsodyExtractor
from src.emotion.emotion_profile import EmotionProfile
from src.asr_module import ASRModule
from src.translation import TranslationModule
from src.tts.azure_neural import AzureNeuralTTS
from src.voice_conversion.rvc_converter import RVCConverter
from src.voice_conversion.model_manager import RVCModelManager


class EmotionAwareDubbingPipeline:
    """Complete dubbing pipeline with emotion preservation"""

    def __init__(self, config_path: str = "config_v2.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.temp_dir = Path(self.config['pipeline']['temp_dir'])
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = self.config['pipeline'].get('verbose', True)

        self._modules_initialized = False

    def _init_modules(self):
        """Lazy-initialize all pipeline modules."""
        if self._modules_initialized:
            return

        print("Initializing Emotion-Aware Dubbing Pipeline V2...")

        self.emotion_analyzer = EmotionAnalyzer(
            model_source=self.config['emotion']['model']
        )
        self.prosody_extractor = ProsodyExtractor()

        self.asr = ASRModule(
            self.config['asr'],
            use_api=False
        )

        self.translator = TranslationModule(self.config['translation'])

        self.tts = AzureNeuralTTS(
            voice_name=self.config['tts']['voice_name']
        )

        # Load RVC model
        model_manager = RVCModelManager()
        rvc_model_path = model_manager.get_model_path(
            self.config['voice_conversion']['model_name']
        )
        self.voice_converter = RVCConverter(model_path=rvc_model_path)

        self._modules_initialized = True
        print("All modules initialized")

    def process_audio(
        self,
        audio_path: str,
        output_path: str,
        save_intermediate: bool = True
    ) -> Dict:
        """
        Process audio through complete emotion-aware pipeline.

        Args:
            audio_path: Input audio (Korean)
            output_path: Output audio (English, creator's voice)
            save_intermediate: Save intermediate files for debugging

        Returns:
            Dict with results and intermediate file paths
        """
        self._init_modules()

        results = {
            'status': 'processing',
            'intermediate_files': {}
        }

        work_dir = self.temp_dir / "work"
        work_dir.mkdir(exist_ok=True)

        # STAGE 1: Emotion Analysis
        print("\n" + "=" * 70)
        print("STAGE 1: Emotional Analysis")
        print("=" * 70)

        emotion_result = self.emotion_analyzer.analyze(audio_path)
        prosody_result = self.prosody_extractor.extract(audio_path)

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

        print(f"Detected emotion: {emotion_profile.emotion} ({emotion_profile.confidence:.2f})")
        print(f"Pitch: {emotion_profile.pitch_mean:.1f} Hz")
        print(f"Speaking rate: {emotion_profile.speaking_rate:.1f} syl/sec")

        if save_intermediate:
            emotion_path = work_dir / "emotion_profile.json"
            with open(emotion_path, 'w') as f:
                json.dump(emotion_profile.to_dict(), f, indent=2)
            results['intermediate_files']['emotion'] = str(emotion_path)

        # STAGE 2: ASR + Translation
        print("\n" + "=" * 70)
        print("STAGE 2: Speech Recognition & Translation")
        print("=" * 70)

        transcript_path = work_dir / "transcript.json"
        transcript = self.asr.transcribe(audio_path, str(transcript_path))

        translation = self.translator.translate_full_script(transcript)
        translation_path = work_dir / "translation.json"
        self.translator.save_translation(translation, str(translation_path))

        print(f"Transcribed {len(transcript['segments'])} segments")
        print(f"Translated to {self.config['translation']['target_language']}")

        results['intermediate_files']['transcript'] = str(transcript_path)
        results['intermediate_files']['translation'] = str(translation_path)

        # STAGE 3: Emotion-Aware TTS
        print("\n" + "=" * 70)
        print("STAGE 3: Emotion-Controlled TTS")
        print("=" * 70)

        full_text = translation['text']
        tts_output = work_dir / "tts_output.wav"

        self.tts.synthesize(
            text=full_text,
            emotion_profile=emotion_profile,
            output_path=str(tts_output)
        )

        print(f"Synthesized with emotion: {emotion_profile.emotion}")
        results['intermediate_files']['tts'] = str(tts_output)

        # STAGE 4: Voice Conversion
        print("\n" + "=" * 70)
        print("STAGE 4: Voice Conversion (RVC)")
        print("=" * 70)

        rvc_output = work_dir / "rvc_output.wav"

        self.voice_converter.convert(
            input_audio=str(tts_output),
            output_audio=str(rvc_output),
            pitch_shift=self.config['voice_conversion']['pitch_shift']
        )

        print("Applied creator's voice")
        results['intermediate_files']['rvc'] = str(rvc_output)

        # STAGE 5: Final Output
        shutil.copy(rvc_output, output_path)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Output: {output_path}")

        results['status'] = 'success'
        results['output_path'] = output_path

        return results
