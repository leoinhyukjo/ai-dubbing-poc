"""
Azure Neural TTS with emotion control
Uses SSML for precise prosody and emotion control
"""
import os
import tempfile
from typing import Optional

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SDK_AVAILABLE = True
except ImportError:
    speechsdk = None
    AZURE_SDK_AVAILABLE = False

from src.emotion.emotion_profile import EmotionProfile


class AzureNeuralTTS:
    """Azure Neural Text-to-Speech with emotion control"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        voice_name: str = "en-US-JennyNeural"
    ):
        self.api_key = api_key or os.getenv('AZURE_SPEECH_KEY')
        self.region = region or os.getenv('AZURE_SPEECH_REGION')
        self.voice_name = voice_name

    def synthesize(
        self,
        text: str,
        emotion_profile: Optional[EmotionProfile] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Synthesize speech with emotion control

        Args:
            text: Text to synthesize
            emotion_profile: Emotion parameters for synthesis
            output_path: Where to save audio (temp file if None)

        Returns:
            Path to generated audio file
        """
        if not AZURE_SDK_AVAILABLE:
            raise RuntimeError(
                "azure-cognitiveservices-speech is not installed. "
                "Install it with: pip install azure-cognitiveservices-speech"
            )

        if not self.api_key or not self.region:
            raise ValueError("Azure API key and region required")

        from src.tts.ssml_builder import SSMLBuilder
        ssml_builder = SSMLBuilder(self.voice_name)

        if emotion_profile:
            ssml = ssml_builder.build_with_emotion(text, emotion_profile)
        else:
            ssml = ssml_builder.build_simple(text)

        if output_path is None:
            output_path = tempfile.mktemp(suffix='.wav')

        speech_config = speechsdk.SpeechConfig(
            subscription=self.api_key,
            region=self.region
        )
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )

        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Synthesized to {output_path}")
            return output_path
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise RuntimeError(f"Speech synthesis canceled: {cancellation.reason}")

        return output_path
