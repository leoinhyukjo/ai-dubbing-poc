"""
SSML Builder for Azure Neural TTS
Constructs SSML with emotion and prosody tags
"""
from src.emotion.emotion_profile import EmotionProfile


class SSMLBuilder:
    """Builds SSML markup for Azure TTS with emotion control"""

    def __init__(self, voice_name: str = "en-US-JennyNeural"):
        self.voice_name = voice_name

    def build_simple(self, text: str) -> str:
        """Build simple SSML without emotion control"""
        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            f'<voice name="{self.voice_name}">'
            f'{text}'
            '</voice>'
            '</speak>'
        )

    def build_with_emotion(self, text: str, profile: EmotionProfile) -> str:
        """Build SSML with emotion and prosody control"""
        params = profile.to_ssml_params()
        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
            ' xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">'
            f'<voice name="{self.voice_name}">'
            f'<mstts:express-as style="{params["style"]}">'
            f'<prosody pitch="{params["pitch"]}" rate="{params["rate"]}" volume="{params["volume"]}">'
            f'{text}'
            '</prosody>'
            '</mstts:express-as>'
            '</voice>'
            '</speak>'
        )
