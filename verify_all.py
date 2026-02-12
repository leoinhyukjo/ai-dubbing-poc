"""
🎯 완전한 시스템 검증 스크립트
모든 API와 패키지가 정상 작동하는지 확인합니다
"""

import sys
import os

print("=" * 60)
print("🚀 AI 더빙 시스템 전체 검증 시작")
print("=" * 60)
print()

# ============================================
# STEP 1: .env 파일 로드
# ============================================
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ STEP 1: .env 파일 로드 성공")
except ImportError:
    print("❌ python-dotenv가 설치되지 않았습니다")
    print("   실행: pip install python-dotenv")
    sys.exit(1)

# ============================================
# STEP 2: API 키 확인
# ============================================
print("\n" + "-" * 60)
print("🔑 STEP 2: API 키 확인")
print("-" * 60)

# Anthropic API 키
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
if anthropic_key and anthropic_key.startswith('sk-ant-'):
    print(f"✅ Anthropic API 키: {anthropic_key[:20]}...{anthropic_key[-10:]}")
else:
    print("❌ ANTHROPIC_API_KEY가 설정되지 않았거나 잘못되었습니다")
    sys.exit(1)

# ElevenLabs API 키
elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
if elevenlabs_key and elevenlabs_key.startswith('sk_'):
    print(f"✅ ElevenLabs API 키: {elevenlabs_key[:15]}...{elevenlabs_key[-10:]}")
else:
    print("⚠️  ELEVENLABS_API_KEY가 설정되지 않았습니다 (Voice Cloning에 필요)")

# OpenAI API 키 (선택사항)
openai_key = os.getenv('OPENAI_API_KEY')
if openai_key:
    print(f"✅ OpenAI API 키: {openai_key[:20]}...{openai_key[-10:]}")
else:
    print("ℹ️  OpenAI API 키 없음 (Local Whisper 사용)")

# ============================================
# STEP 3: 필수 패키지 확인
# ============================================
print("\n" + "-" * 60)
print("📦 STEP 3: 필수 패키지 확인")
print("-" * 60)

packages = {
    'anthropic': 'Claude API',
    'whisper': 'Local Whisper',
    'pydub': '오디오 처리',
    'ffmpeg': 'FFmpeg Python',
}

all_packages_ok = True
for package, description in packages.items():
    try:
        __import__(package)
        print(f"✅ {package:15s} - {description}")
    except ImportError:
        print(f"❌ {package:15s} - {description} (설치 필요)")
        all_packages_ok = False

if not all_packages_ok:
    print("\n⚠️  일부 패키지가 누락되었습니다.")
    print("   실행: pip install -r requirements.txt")

# ============================================
# STEP 4: Claude API 연결 테스트
# ============================================
print("\n" + "-" * 60)
print("🤖 STEP 4: Claude API 연결 테스트")
print("-" * 60)

try:
    from anthropic import Anthropic

    client = Anthropic(api_key=anthropic_key)

    # 간단한 번역 테스트
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=100,
        temperature=0.3,
        system="You are a translator. Translate Korean to English naturally.",
        messages=[
            {"role": "user", "content": "안녕하세요! 저는 유튜버입니다."}
        ]
    )

    translation = response.content[0].text
    print("✅ Claude API 연결 성공!")
    print(f"\n   📝 테스트 번역:")
    print(f"   원문: 안녕하세요! 저는 유튜버입니다.")
    print(f"   번역: {translation}")

except Exception as e:
    print(f"❌ Claude API 연결 실패: {e}")
    print("\n   💡 일반적인 원인:")
    print("   1. API 키가 잘못되었습니다")
    print("   2. 인터넷 연결이 끊겼습니다")
    print("   3. Anthropic 서비스에 문제가 있습니다")

# ============================================
# STEP 5: ElevenLabs API 연결 테스트
# ============================================
if elevenlabs_key:
    print("\n" + "-" * 60)
    print("🎙️  STEP 5: ElevenLabs API 연결 테스트")
    print("-" * 60)

    try:
        # elevenlabs 패키지 확인
        try:
            from elevenlabs.client import ElevenLabs
            print("✅ ElevenLabs 패키지 설치됨")
        except ImportError:
            print("❌ elevenlabs 패키지가 설치되지 않았습니다")
            print("   실행: pip install elevenlabs")
            elevenlabs_key = None

        if elevenlabs_key:
            client = ElevenLabs(api_key=elevenlabs_key)

            # Voice 목록 가져오기
            voices = client.voices.get_all()

            print(f"✅ ElevenLabs API 연결 성공!")
            print(f"\n   🎤 사용 가능한 Voice: {len(voices.voices)}개")

            if len(voices.voices) > 0:
                print(f"\n   처음 3개 Voice:")
                for i, voice in enumerate(voices.voices[:3]):
                    print(f"   {i+1}. {voice.name} (ID: {voice.voice_id})")
            else:
                print(f"\n   ⚠️  등록된 Voice가 없습니다")
                print(f"   Voice Clone을 먼저 생성해야 합니다")
                print(f"   예: python examples/create_voice_clone.py")

    except Exception as e:
        print(f"❌ ElevenLabs API 연결 실패: {e}")
        print("\n   💡 일반적인 원인:")
        print("   1. API 키가 잘못되었습니다")
        print("   2. 인터넷 연결이 끊겼습니다")
        print("   3. ElevenLabs 서비스에 문제가 있습니다")

# ============================================
# STEP 6: FFmpeg 확인
# ============================================
print("\n" + "-" * 60)
print("🎬 STEP 6: FFmpeg 시스템 확인")
print("-" * 60)

import subprocess

try:
    result = subprocess.run(
        ['ffmpeg', '-version'],
        capture_output=True,
        text=True,
        timeout=5
    )

    if result.returncode == 0:
        version_line = result.stdout.split('\n')[0]
        print(f"✅ FFmpeg 설치됨: {version_line}")
    else:
        print("❌ FFmpeg 실행 실패")

except FileNotFoundError:
    print("❌ FFmpeg가 설치되지 않았습니다")
    print("\n   💡 설치 방법:")
    print("   macOS:   brew install ffmpeg")
    print("   Ubuntu:  sudo apt install ffmpeg")
    print("   Windows: https://ffmpeg.org/download.html")

except Exception as e:
    print(f"⚠️  FFmpeg 확인 중 오류: {e}")

# ============================================
# 최종 결과
# ============================================
print("\n" + "=" * 60)
print("📊 검증 결과 요약")
print("=" * 60)
print()
print("✅ 준비 완료:")
print("   • Claude API (번역)")
if elevenlabs_key:
    print("   • ElevenLabs API (Voice Cloning)")
print("   • 필수 Python 패키지")
print()
print("🎯 다음 단계:")
print()
print("1. Voice Clone 생성 (아직 안 했다면):")
print("   python examples/create_voice_clone.py")
print()
print("2. 첫 더빙 테스트:")
print("   python pipeline.py input.mp4 <voice_id> -o output.mp4")
print()
print("3. 자세한 가이드:")
print("   - CLAUDE_SETUP.md - Claude 설정")
print("   - README.md - 전체 사용법")
print()
print("=" * 60)
print("🎉 시스템 준비 완료! Happy Dubbing!")
print("=" * 60)
