"""
설정 테스트 스크립트
각 단계를 순차적으로 확인합니다
"""

import sys
import os

# .env 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ STEP 1: .env 파일 로드 성공")
except ImportError:
    print("❌ python-dotenv가 설치되지 않았습니다")
    print("   실행: pip install python-dotenv")
    sys.exit(1)

# API 키 확인
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
if anthropic_key and anthropic_key.startswith('sk-ant-'):
    print("✅ STEP 2: Anthropic API 키 확인 완료")
    print(f"   키: {anthropic_key[:20]}...{anthropic_key[-10:]}")
else:
    print("❌ ANTHROPIC_API_KEY가 설정되지 않았거나 잘못되었습니다")
    sys.exit(1)

# Anthropic 패키지 확인
try:
    from anthropic import Anthropic
    print("✅ STEP 3: anthropic 패키지 설치 확인")
except ImportError:
    print("❌ anthropic 패키지가 설치되지 않았습니다")
    print("   실행: pip install anthropic")
    sys.exit(1)

# Claude API 연결 테스트
print("\n🔍 STEP 4: Claude API 연결 테스트 중...")
try:
    client = Anthropic(api_key=anthropic_key)

    # 간단한 번역 테스트
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=100,
        temperature=0.3,
        system="You are a translator. Translate Korean to English.",
        messages=[
            {"role": "user", "content": "안녕하세요! 저는 유튜버입니다."}
        ]
    )

    translation = response.content[0].text
    print(f"✅ STEP 4: API 연결 성공!")
    print(f"\n   📝 테스트 번역:")
    print(f"   원문: 안녕하세요! 저는 유튜버입니다.")
    print(f"   번역: {translation}")

except Exception as e:
    print(f"❌ API 연결 실패: {e}")
    sys.exit(1)

# 모든 테스트 통과
print("\n" + "="*50)
print("🎉 모든 설정이 완료되었습니다!")
print("="*50)
print("\n다음 단계:")
print("1. Voice Clone 생성 (ElevenLabs API 키 필요)")
print("2. 실제 영상 더빙 테스트")
