# 🤖 Claude Code 실행 프롬프트

**이 프롬프트를 Claude Code CLI에 복사해서 붙여넣으면 자동으로 설치부터 테스트까지 진행됩니다!**

---

## 📋 전체 프롬프트 (복사해서 사용)

```
AI 더빙 시스템을 설치하고 테스트해줘.

다음 순서로 진행:

1. 프로젝트 다운로드 및 설정
   - ~/ai-dubbing-poc 폴더 생성
   - 필요한 파일들 복사

2. Python 가상환경 설정
   - venv 생성
   - 패키지 설치 (requirements.txt)

3. API 키 설정
   - .env 파일에 다음 키들 추가:
     * ANTHROPIC_API_KEY=your-anthropic-api-key-here
     * ELEVENLABS_API_KEY=your-elevenlabs-api-key-here

4. 시스템 검증
   - python verify_all.py 실행
   - 모든 API 연결 테스트

5. Voice Clone 생성 준비
   - ~/Downloads/나니까 샘플.mp3 파일로 Voice Clone 생성
   - Voice ID 저장

6. 테스트 더빙 실행
   - 샘플 오디오로 간단한 더빙 테스트
   - 결과 확인 및 리포트

각 단계마다 결과를 보고하고, 에러가 있으면 해결책 제시해줘.
```

---

## 🎯 단계별 프롬프트 (나눠서 실행하고 싶다면)

### 1단계: 설치

```
AI 더빙 프로젝트를 ~/ai-dubbing-poc에 설치해줘.

1. /sessions/gifted-admiring-turing/mnt/outputs/ai-dubbing-poc 전체를 ~/ai-dubbing-poc로 복사
2. Python 가상환경 생성 (python -m venv venv)
3. requirements.txt로 패키지 설치
4. FFmpeg 설치 확인 (없으면 안내)

완료되면 설치 결과를 요약해줘.
```

### 2단계: API 설정

```
AI 더빙 프로젝트의 API 키를 설정해줘.

1. ~/ai-dubbing-poc/.env 파일 생성
2. 다음 내용 추가:
   ANTHROPIC_API_KEY=your-anthropic-api-key-here
   ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
   TARGET_LANGUAGE=en
   SOURCE_LANGUAGE=ko

3. verify_all.py 실행해서 API 연결 테스트
4. 결과 리포트
```

### 3단계: Voice Clone 생성

```
ElevenLabs Voice Clone을 생성해줘.

1. ~/Downloads/나니까 샘플.mp3 파일 확인
2. ElevenLabs API로 Voice Clone 생성:
   - 이름: "나니까 목소리"
   - 설명: "유튜브 더빙용"
3. 생성된 Voice ID 저장 및 출력
4. Voice 정보 확인 (이름, ID, 설정)
```

### 4단계: 테스트 더빙

```
샘플 오디오로 더빙 테스트를 실행해줘.

1. 짧은 테스트 영상 또는 오디오 준비
   (없으면 간단한 TTS로 테스트 오디오 생성)

2. 전체 파이프라인 실행:
   - ASR (음성→텍스트)
   - Translation (한국어→영어, Claude 사용)
   - Voice Synthesis (ElevenLabs)

3. 결과 확인:
   - 원본 텍스트
   - 번역 텍스트
   - 생성된 오디오

4. 품질 평가 및 리포트
```

---

## 🚀 빠른 실행 (올인원)

```
AI 더빙 시스템 전체 설치 + 테스트를 실행해줘.

프로젝트 경로: /sessions/gifted-admiring-turing/mnt/outputs/ai-dubbing-poc
설치 경로: ~/ai-dubbing-poc
샘플 파일: ~/Downloads/나니까 샘플.mp3

API 키:
- Anthropic: your-anthropic-api-key-here
- ElevenLabs: your-elevenlabs-api-key-here

작업 순서:
1. 프로젝트 복사
2. venv 생성 및 패키지 설치
3. .env 설정
4. 시스템 검증 (verify_all.py)
5. Voice Clone 생성
6. 테스트 더빙 실행
7. 최종 리포트

각 단계 진행 상황과 결과를 실시간으로 보고해줘.
에러가 있으면 자동으로 해결하거나 해결 방법 제시해줘.
```

---

## 💡 문제 해결 프롬프트

### FFmpeg 설치가 필요한 경우

```
FFmpeg를 설치해줘.

OS 감지해서:
- macOS: brew install ffmpeg
- Ubuntu/Debian: sudo apt install ffmpeg
- 기타: 설치 가이드 출력

설치 후 버전 확인해서 정상 작동 확인해줘.
```

### API 연결 테스트

```
AI 더빙 시스템의 API 연결을 테스트해줘.

1. Claude API:
   - 간단한 번역 테스트 ("안녕하세요" → "Hello")
   - 응답 시간 측정

2. ElevenLabs API:
   - Voice 목록 가져오기
   - 사용 가능한 Voice 출력

3. 결과 요약 및 상태 리포트
```

### Voice 품질 테스트

```
생성된 Voice Clone의 품질을 테스트해줘.

1. 짧은 테스트 문장 3개로 음성 생성:
   - "안녕하세요, 반갑습니다"
   - "오늘 날씨가 정말 좋네요"
   - "이 영상을 시청해주셔서 감사합니다"

2. 각 음성 파일 생성 및 저장
3. 파일 정보 출력 (크기, 길이, 경로)
4. 품질 평가 가이드 제공
```

---

## 📊 상태 확인 프롬프트

```
AI 더빙 시스템의 현재 상태를 확인해줘.

체크리스트:
1. ✅/❌ 프로젝트 설치 (~/ai-dubbing-poc)
2. ✅/❌ Python 패키지 설치
3. ✅/❌ API 키 설정
4. ✅/❌ Claude API 연결
5. ✅/❌ ElevenLabs API 연결
6. ✅/❌ FFmpeg 설치
7. ✅/❌ Voice Clone 생성
8. ✅/❌ 테스트 실행 완료

각 항목의 상태와 다음 필요한 작업을 리포트해줘.
```

---

## 🔧 커스터마이징 프롬프트

### 일본어로 변경

```
AI 더빙 시스템을 일본어로 설정해줘.

1. config.yaml에서 target_language를 "ja"로 변경
2. system_prompt를 일본어 번역에 최적화
3. 설정 저장 및 확인
4. 간단한 테스트 (한국어 → 일본어)
```

### 번역 품질 개선

```
번역 품질을 개선하기 위해 system_prompt를 최적화해줘.

현재 용도: 유튜브 게임 콘텐츠 더빙
타겟 시청자: 10-30대 게이머
톤: 친근하고 에너지 넘치는

1. 현재 prompt 분석
2. 개선된 prompt 작성
3. config.yaml에 적용
4. 전후 비교 테스트
```

---

## 🎬 실전 사용 프롬프트

```
실제 유튜브 영상을 더빙해줘.

영상 경로: ~/Videos/my_video.mp4
Voice ID: [앞서 생성한 Voice ID]
출력 경로: ~/Desktop/dubbed_video.mp4

전체 파이프라인 실행:
1. 오디오 추출
2. BGM 분리
3. ASR (한국어 → 텍스트)
4. 번역 (한국어 → 영어, Claude)
5. TTS (영어 음성, ElevenLabs)
6. 오디오 믹싱 (음성 + BGM)
7. 영상 합성

각 단계 진행률과 예상 시간 출력해줘.
최종 결과 파일 정보와 품질 리포트 제공해줘.
```

---

## 📝 사용 예시

### 터미널에서 Claude Code 실행

```bash
# Claude Code CLI 실행
claude

# 프롬프트 입력 (위의 프롬프트 중 하나 복사)
> AI 더빙 시스템을 설치하고 테스트해줘...
```

### 또는 직접 명령

```bash
# 한 줄로 실행
claude "AI 더빙 시스템 전체 설치 + 테스트를 실행해줘. 프로젝트 경로: /sessions/..."
```

---

## 💡 팁

1. **처음 실행할 때**: "전체 프롬프트" 사용
2. **단계별로 확인하고 싶을 때**: "단계별 프롬프트" 사용
3. **에러 발생 시**: "문제 해결 프롬프트" 사용
4. **상태 확인**: "상태 확인 프롬프트" 사용

---

**이 프롬프트들을 복사해서 Claude Code에 바로 붙여넣으면 됩니다!** 🚀

**최종 업데이트**: 2026-02-11
