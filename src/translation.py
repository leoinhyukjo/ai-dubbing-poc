"""
Translation Module
Supports both Claude (Anthropic) and GPT-4o (OpenAI) for high-quality translation
"""

import os
from typing import Dict, List, Optional
import json


class TranslationModule:
    """Handles translation using Claude or GPT-4o"""

    def __init__(self, config: Dict):
        """
        Initialize translation module

        Args:
            config: Translation configuration from config.yaml
        """
        self.config = config
        self.provider = config.get('provider', 'claude')  # 'claude' or 'openai'
        self.model = config.get('model', 'claude-sonnet-4-5-20250929')
        self.target_language = config.get('target_language', 'en')
        self.system_prompt = config.get('system_prompt', self._default_system_prompt())

        # Initialize appropriate client
        if self.provider == 'claude':
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            print(f"✓ Translation module initialized (Claude → {self.target_language})")
        elif self.provider == 'openai':
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            print(f"✓ Translation module initialized (GPT-4o → {self.target_language})")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _default_system_prompt(self) -> str:
        """Default system prompt for translation"""
        return """You are a professional video subtitle translator specializing in YouTube content.

CRITICAL RULES:
- Provide ONLY ONE final translation - NO alternatives, NO options, NO explanations
- Output ONLY the translated text - NO markdown formatting, NO notes, NO commentary
- Do NOT include phrases like "Alternative:", "Note:", "Primary:", or any meta-text
- Do NOT add dashes (---), asterisks (**), or any formatting markers

Translate the Korean transcript to natural, conversational English that:
- Maintains the creator's personality and tone
- Uses casual, engaging language appropriate for social media
- Preserves slang, memes, and cultural references (explain if needed)
- Matches the approximate length of the original for lip-sync compatibility
- Keeps punctuation and emphasis markers

Output format: Just the translated text, nothing else."""

    def translate_text(self, text: str, context: Optional[str] = None) -> str:
        """
        Translate a single text segment

        Args:
            text: Text to translate
            context: Optional context for better translation

        Returns:
            Translated text
        """
        user_message = text
        if context:
            user_message = f"Context: {context}\n\nTranslate: {text}"

        if self.provider == 'claude':
            return self._translate_with_claude(user_message)
        elif self.provider == 'openai':
            return self._translate_with_openai(user_message)

    def _translate_with_claude(self, user_message: str) -> str:
        """Translate using Claude API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.3,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response.content[0].text.strip()

    def _translate_with_openai(self, user_message: str) -> str:
        """Translate using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def translate_segments(
        self,
        segments: List[Dict],
        preserve_timing: bool = True
    ) -> List[Dict]:
        """
        Translate transcript segments while preserving timing

        Args:
            segments: List of segment dictionaries with 'text', 'start', 'end'
            preserve_timing: If True, add length constraints to translation

        Returns:
            List of translated segments with same structure
        """
        print(f"🌐 Translating {len(segments)} segments to {self.target_language} using {self.provider.upper()}...")

        translated_segments = []

        for i, segment in enumerate(segments):
            original_text = segment['text']
            duration = segment['end'] - segment['start']

            # Add context from previous segment if available
            context = None
            if i > 0:
                context = f"Previous: {translated_segments[-1]['text']}"

            # Add timing constraint if needed
            if preserve_timing:
                timing_hint = f"\n\n[Note: Keep translation concise - original duration: {duration:.1f}s]"
                original_text = original_text + timing_hint

            try:
                translated_text = self.translate_text(original_text, context)

                # Clean up timing hint if it leaked into translation
                if '[Note:' in translated_text:
                    translated_text = translated_text.split('[Note:')[0].strip()

                translated_segments.append({
                    'id': segment['id'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': translated_text,
                    'original_text': segment['text']
                })

                print(f"  [{i+1}/{len(segments)}] ✓")

            except Exception as e:
                print(f"  [{i+1}/{len(segments)}] ❌ Error: {e}")
                # Keep original text on error
                translated_segments.append({
                    'id': segment['id'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'original_text': segment['text'],
                    'error': str(e)
                })

        print(f"✓ Translation complete")
        return translated_segments

    def translate_full_script(self, transcript: Dict) -> Dict:
        """
        Translate entire transcript including metadata

        Args:
            transcript: Full transcript dictionary from ASR module

        Returns:
            Translated transcript with same structure
        """
        translated_segments = self.translate_segments(
            transcript['segments'],
            preserve_timing=True
        )

        # Combine all translated text
        full_translated_text = ' '.join([s['text'] for s in translated_segments])

        return {
            'text': full_translated_text,
            'segments': translated_segments,
            'source_language': transcript.get('language', 'ko'),
            'target_language': self.target_language,
            'original_text': transcript['text']
        }

    def save_translation(self, translation: Dict, output_path: str):
        """Save translation to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translation, f, ensure_ascii=False, indent=2)
        print(f"💾 Translation saved: {output_path}")


# Utility: Create SRT subtitle file
def create_srt(segments: List[Dict], output_path: str):
    """
    Create SRT subtitle file from segments

    Args:
        segments: List of segments with 'start', 'end', 'text'
        output_path: Path for output SRT file
    """
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text']}\n\n")

    print(f"📝 SRT file created: {output_path}")
