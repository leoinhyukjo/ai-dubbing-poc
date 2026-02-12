"""
Main Dubbing Pipeline
End-to-end video dubbing automation
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml
from dotenv import load_dotenv
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from asr_module import ASRModule, extract_audio_from_video
from translation import TranslationModule, create_srt
from voice_cloning import create_voice_cloning
from audio_processing import AudioProcessor
from video_synthesis import VideoSynthesizer, get_video_info


class DubbingPipeline:
    """Complete dubbing pipeline orchestrator"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize dubbing pipeline

        Args:
            config_path: Path to configuration file
        """
        # Load environment variables
        load_dotenv()

        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Initialize temp directory
        self.temp_dir = Path(self.config['pipeline']['temp_dir'])
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.verbose = self.config['pipeline'].get('verbose', True)
        self.keep_temp_files = self.config['pipeline'].get('keep_temp_files', False)

        print("=" * 70)
        print("🎬 AI DUBBING PIPELINE INITIALIZED")
        print("=" * 70)

    def process_video(
        self,
        video_path: str,
        voice_id: str,
        output_path: str,
        use_whisper_api: bool = False
    ) -> Dict:
        """
        Process a single video through the complete dubbing pipeline

        Args:
            video_path: Path to input video
            voice_id: ElevenLabs voice ID or Respeecher model ID
            output_path: Path for output dubbed video
            use_whisper_api: If True, use OpenAI Whisper API; else use local

        Returns:
            Dictionary with processing results and metadata
        """
        print(f"\n📹 Processing video: {video_path}")

        # Create working directory for this video
        video_name = Path(video_path).stem
        work_dir = self.temp_dir / video_name
        work_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'video_path': video_path,
            'output_path': output_path,
            'work_dir': str(work_dir),
            'steps': {}
        }

        try:
            # Get video info
            print("\n" + "─" * 70)
            print("STEP 0: Video Analysis")
            print("─" * 70)
            video_info = get_video_info(video_path)
            results['video_info'] = video_info
            print(f"Duration: {video_info['duration']:.2f}s")
            print(f"Resolution: {video_info['video']['width']}x{video_info['video']['height']}")

            # STEP 1: Extract audio from video
            print("\n" + "─" * 70)
            print("STEP 1: Audio Extraction")
            print("─" * 70)
            audio_path = str(work_dir / "original_audio.wav")
            extract_audio_from_video(video_path, audio_path)
            results['steps']['audio_extraction'] = {'output': audio_path}

            # STEP 1.5 (Optional): Separate vocals and BGM
            vocals_path = audio_path
            bgm_path = None

            if self.config['audio'].get('separate_bgm', False):
                print("\n" + "─" * 70)
                print("STEP 1.5: BGM Separation")
                print("─" * 70)
                audio_processor = AudioProcessor(self.config['audio'])
                separation_dir = work_dir / "separated"
                vocals_path, bgm_path = audio_processor.separate_vocals_bgm(
                    audio_path,
                    str(separation_dir)
                )
                results['steps']['bgm_separation'] = {
                    'vocals': vocals_path,
                    'bgm': bgm_path
                }

            # STEP 2: ASR (Speech-to-Text)
            print("\n" + "─" * 70)
            print("STEP 2: Speech Recognition")
            print("─" * 70)
            asr = ASRModule(self.config['asr'], use_api=use_whisper_api)
            transcript_path = str(work_dir / "transcript.json")
            transcript = asr.transcribe(vocals_path, transcript_path)
            results['steps']['asr'] = {
                'transcript': transcript_path,
                'segments': len(transcript['segments']),
                'text_preview': transcript['text'][:200] + '...'
            }

            # STEP 3: Translation
            print("\n" + "─" * 70)
            print("STEP 3: Translation")
            print("─" * 70)
            translator = TranslationModule(self.config['translation'])
            translation = translator.translate_full_script(transcript)
            translation_path = str(work_dir / "translation.json")
            translator.save_translation(translation, translation_path)

            # Create SRT subtitle file
            srt_path = str(work_dir / "subtitles.srt")
            create_srt(translation['segments'], srt_path)

            results['steps']['translation'] = {
                'translation': translation_path,
                'subtitles': srt_path,
                'text_preview': translation['text'][:200] + '...'
            }

            # STEP 4: Voice Cloning
            print("\n" + "─" * 70)
            print("STEP 4: Voice Synthesis")
            print("─" * 70)
            voice_cloner = create_voice_cloning(
                self.config['voice_cloning'],
                voice_id
            )
            segments_dir = work_dir / "voice_segments"
            audio_segments = voice_cloner.synthesize_segments(
                translation['segments'],
                str(segments_dir)
            )
            results['steps']['voice_cloning'] = {
                'segments_dir': str(segments_dir),
                'segments_count': len(audio_segments)
            }

            # STEP 5: Audio Assembly
            print("\n" + "─" * 70)
            print("STEP 5: Audio Assembly")
            print("─" * 70)
            audio_processor = AudioProcessor(self.config['audio'])

            # Combine voice segments
            timestamps = [(s['start'], s['end']) for s in translation['segments']]
            combined_voice_path = str(work_dir / "dubbed_voice.wav")
            audio_processor.combine_segments(
                audio_segments,
                timestamps,
                combined_voice_path
            )

            # Mix with BGM if available
            if bgm_path and os.path.exists(bgm_path):
                final_audio_path = str(work_dir / "final_audio.wav")
                audio_processor.mix_audio_tracks(
                    combined_voice_path,
                    bgm_path,
                    final_audio_path,
                    voice_volume=self.config['audio'].get('voice_volume', 1.0),
                    bgm_volume=self.config['audio'].get('bgm_volume', 0.3)
                )
            else:
                final_audio_path = combined_voice_path

            results['steps']['audio_assembly'] = {
                'final_audio': final_audio_path
            }

            # STEP 6: Video Synthesis
            print("\n" + "─" * 70)
            print("STEP 6: Video Synthesis")
            print("─" * 70)
            video_synthesizer = VideoSynthesizer(self.config['video'])
            video_synthesizer.replace_audio(
                video_path,
                final_audio_path,
                output_path
            )

            results['steps']['video_synthesis'] = {
                'output': output_path
            }

            # Success!
            print("\n" + "=" * 70)
            print("✅ DUBBING COMPLETE!")
            print("=" * 70)
            print(f"📁 Output: {output_path}")
            print(f"📝 Subtitles: {srt_path}")

            results['status'] = 'success'

        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            results['status'] = 'failed'
            results['error'] = str(e)

        finally:
            # Clean up temp files if configured
            if not self.keep_temp_files and work_dir.exists():
                print(f"\n🧹 Cleaning up temporary files...")
                shutil.rmtree(work_dir)

        return results


def main():
    """Command-line interface for dubbing pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='AI Video Dubbing Pipeline')
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('voice_id', help='ElevenLabs voice ID or Respeecher model ID')
    parser.add_argument('-o', '--output', help='Output video path')
    parser.add_argument('-c', '--config', default='config.yaml', help='Config file path')
    parser.add_argument('--api', action='store_true', help='Use Whisper API instead of local')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files')

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        video_path = Path(args.video)
        output_path = str(video_path.parent / f"{video_path.stem}_dubbed{video_path.suffix}")

    # Create pipeline
    pipeline = DubbingPipeline(args.config)

    if args.keep_temp:
        pipeline.keep_temp_files = True

    # Process video
    results = pipeline.process_video(
        args.video,
        args.voice_id,
        output_path,
        use_whisper_api=args.api
    )

    # Exit with appropriate code
    sys.exit(0 if results['status'] == 'success' else 1)


if __name__ == '__main__':
    main()
