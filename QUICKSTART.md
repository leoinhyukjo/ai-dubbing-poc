# 🚀 빠른 시작 가이드

**5분 안에 첫 더빙 완성하기!**

---

## ✅ 준비 사항

### 이미 완료된 것들:
- ✅ Claude API 키 설정 완료
- ✅ ElevenLabs API 키 설정 완료
- ✅ 코드 작성 완료
- ✅ `.env` 파일 설정 완료

---

## 📥 STEP 1: 프로젝트 다운로드

프로젝트 폴더를 로컬 컴퓨터에 복사하세요:

```bash
# 이 프로젝트 폴더 전체를 복사
cp -r /sessions/gifted-admiring-turing/mnt/outputs/ai-dubbing-poc ~/ai-dubbing-poc

# 폴더로 이동
cd ~/ai-dubbing-poc
```

---

## 🔧 STEP 2: 패키지 설치

```bash
# 전체 설치 (권장)
pip install -r requirements.txt

# 또는 자동 스크립트 실행
./setup.sh        # macOS/Linux
setup.bat         # Windows
```

**설치 예상 시간**: 3-5분

---

## ✅ STEP 3: 설정 확인

```bash
python verify_all.py
```

**기대 결과:**
```
🚀 AI 더빙 시스템 전체 검증 시작
✅ STEP 1: .env 파일 로드 성공
✅ STEP 2: API 키 확인
✅ STEP 3: 필수 패키지 확인
✅ STEP 4: Claude API 연결 성공!
✅ STEP 5: ElevenLabs API 연결 성공!
✅ STEP 6: FFmpeg 설치됨
🎉 시스템 준비 완료!
```

**에러가 나면?** → [문제 해결](#-문제-해결) 섹션 참고

---

## 🎤 STEP 4: Voice Clone 생성

ElevenLabs에 목소리를 등록하세요:

### 방법 1: Web UI (추천)

1. [ElevenLabs Voice Lab](https://elevenlabs.io/voice-lab) 접속
2. "Add Instant Voice Clone" 클릭
3. 목소리 샘플 업로드 (1-3분 길이)
4. Voice ID 복사

### 방법 2: Python 스크립트

```bash
python examples/create_voice_clone.py \
  --name "내 목소리" \
  --samples voice_sample.mp3 \
  --description "유튜브 더빙용"
```

**Voice ID를 저장하세요!** 예: `21m00Tcm4TlvDq8ikWAM`

---

## 🎬 STEP 5: 첫 더빙 실행!

```bash
python pipeline.py \
  input_video.mp4 \
  YOUR_VOICE_ID \
  -o output.mp4
```

**예시:**
```bash
python pipeline.py \
  ~/Downloads/my_video.mp4 \
  21m00Tcm4TlvDq8ikWAM \
  -o ~/Desktop/dubbed_video.mp4
```

### 파라미터 설명:
- `input_video.mp4`: 원본 영상 (한국어)
- `YOUR_VOICE_ID`: ElevenLabs Voice ID
- `-o output.mp4`: 출력 파일명

### 진행 과정:
```
🎥 [1/6] 영상에서 오디오 추출 중...
🎵 [2/6] BGM과 음성 분리 중...
🎙️  [3/6] 음성을 텍스트로 변환 중...
🌐 [4/6] 한국어 → 영어 번역 중... (Claude)
🗣️  [5/6] 영어 음성 생성 중... (ElevenLabs)
🎬 [6/6] 최종 영상 합성 중...

✅ 완성! output.mp4
```

**예상 소요 시간:**
- 5분 영상: ~3분
- 10분 영상: ~5분
- 30분 영상: ~15분

---

## 🎯 고급 옵션

### 번역 언어 변경

`config.yaml` 파일 수정:
```yaml
translation:
  target_language: "ja"  # 일본어로 변경
```

**지원 언어:**
- `en`: English
- `ja`: Japanese
- `es`: Spanish
- `fr`: French
- `de`: German

### 음성 설정 커스터마이징

`config.yaml`에서 조정:
```yaml
voice_cloning:
  stability: 0.75      # 안정성 (0-1)
  similarity_boost: 0.8  # 유사도 (0-1)
  style: 0.5           # 스타일 강도 (0-1)
```

### 배치 처리 (여러 영상 한 번에)

```bash
# 폴더 내 모든 영상 처리
for video in videos/*.mp4; do
  python pipeline.py "$video" YOUR_VOICE_ID -o "dubbed_$(basename $video)"
done
```

---

## 💡 유용한 팁

### Tip 1: 빠른 테스트
짧은 영상(1-2분)으로 먼저 테스트해보세요!

### Tip 2: 품질 체크포인트
- 자막 파일(`.srt`) 확인 → ASR 정확도
- 번역 파일 확인 → 번역 품질
- 음성 샘플 확인 → Voice Clone 품질

### Tip 3: 비용 모니터링
- [Anthropic Console](https://console.anthropic.com/settings/usage) - Claude 사용량
- [ElevenLabs Dashboard](https://elevenlabs.io/usage) - Voice 사용량

### Tip 4: 프롬프트 최적화
유튜버 특성에 맞게 번역 프롬프트 수정:
```yaml
system_prompt: |
  당신은 게임 유튜버 전문 번역가입니다.
  게임 용어와 밈을 자연스럽게 번역하세요.
```

---

## 🐛 문제 해결

### "ModuleNotFoundError: No module named 'anthropic'"
```bash
pip install anthropic
```

### "FFmpeg not found"
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# https://ffmpeg.org/download.html 에서 다운로드
```

### "Invalid API key"
`.env` 파일 확인:
```bash
cat .env
```
API 키가 올바르게 입력되었는지 확인하세요.

### "Rate limit exceeded"
잠시 기다렸다가 재시도하거나, 세그먼트당 딜레이 추가:
```yaml
translation:
  delay_between_segments: 0.5  # 0.5초 딜레이
```

### 번역 품질이 이상해요
`config.yaml`에서 system_prompt 수정:
```yaml
system_prompt: |
  더 자세한 컨텍스트 추가...
  톤앤매너 가이드 추가...
```

---

## 📚 더 알아보기

### 상세 가이드
- **[README.md](README.md)** - 전체 문서
- **[CLAUDE_SETUP.md](CLAUDE_SETUP.md)** - Claude 설정 가이드
- **[COST_COMPARISON.md](COST_COMPARISON.md)** - 비용 분석

### 예제 코드
- **`examples/simple_dubbing.py`** - 간단한 예제
- **`examples/create_voice_clone.py`** - Voice Clone 생성

### 설정 파일
- **`config.yaml`** - 시스템 설정
- **`.env`** - API 키

---

## 🤝 도움이 필요하신가요?

질문이나 문제가 있으면 언제든 물어보세요!

---

**Happy Dubbing!** 🎬✨

**최종 업데이트**: 2026-02-11
