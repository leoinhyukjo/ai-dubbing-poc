# 🎬 AI Video Dubbing - PoC

**자동 음성 더빙 솔루션**: 한국 크리에이터의 영상을 영어/일본어/중국어로 자동 더빙하여 글로벌 시장 진출을 지원합니다.

---

## 📋 목차

- [특징](#-특징)
- [시스템 아키텍처](#-시스템-아키텍처)
- [설치](#-설치)
- [설정](#-설정)
- [사용법](#-사용법)
- [비용 분석](#-비용-분석)
- [로드맵](#-로드맵)
- [문제 해결](#-문제-해결)

---

## ✨ 특징

### 🎯 핵심 기능
- **자동 음성 인식 (ASR)**: OpenAI Whisper를 사용한 고정확도 한국어 음성 인식
- **AI 번역**: Claude (권장) 또는 GPT-4o를 통한 맥락 보존 자연스러운 번역 ⭐
- **음성 클로닝**: ElevenLabs/Respeecher를 활용한 크리에이터 목소리 복제
- **BGM 분리**: Demucs를 사용한 음성/배경음 자동 분리
- **자동 자막 생성**: SRT 형식 자막 자동 생성
- **비디오 합성**: FFmpeg를 사용한 고품질 최종 영상 생성

### 🆕 Pipeline V2: 감정 보존 더빙 (Beta)
- **감정 분석**: SpeechBrain으로 원본 음성의 감정 자동 감지 (기쁨, 슬픔, 분노, 중립)
- **운율 추출**: librosa로 피치, 에너지, 말하기 속도 분석
- **감정 제어 TTS**: Azure Neural TTS + SSML로 감정을 반영한 음성 합성
- **보이스 변환**: RVC로 크리에이터 목소리 적용 (감정/운율 유지)
- **자연스러운 타이밍**: time-stretching 없이 자연스러운 발화

### 📊 성능 지표
- **비용**: 전통적 성우 더빙 대비 **100배 절감** ($50-300/분 → $0.5-10/분)
- **시간**: 제작 기간 **90% 단축** (수 주 → 72시간 이내)
- **품질**: 상업적 수준의 음성 품질 (ElevenLabs Professional)
- **효과**: 자막 대비 **24% 높은 시청 완료율**, 최대 **400배 조회수 증가**

---

## 🏗️ 시스템 아키텍처

### Pipeline V1 (ElevenLabs 기반)

```
┌─────────────────────────────────────────────────────┐
│                   한국어 영상 입력                    │
│                   (MP4, MOV 등)                      │
└──────────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  STEP 1: 전처리 (Optional)  │
    │  • Demucs: 음성/BGM 분리   │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  STEP 2: ASR                │
    │  • Whisper Large-v3         │
    │  • 타임스탬프 추출           │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  STEP 3: 번역               │
    │  • Claude (권장) ⭐         │
    │  • 톤앤매너 보존             │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  STEP 4: Voice Cloning      │
    │  • ElevenLabs Professional  │
    │  • 또는 Respeecher          │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  STEP 5: 오디오 합성        │
    │  • 타임라인 정렬            │
    │  • BGM 믹싱                 │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  STEP 6: 비디오 생성        │
    │  • FFmpeg 인코딩            │
    └──────────────┬──────────────┘
                   │
            ┌──────▼───────┐
            │  더빙된 영상  │
            │  + 자막 파일  │
            └──────────────┘
```

### Pipeline V2 (감정 보존 기반 - Beta)

```
┌──────────────────────────────────────────────┐
│              한국어 오디오 입력                │
└──────────────────┬───────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  STAGE 1: 감정 분석         │
    │  • SpeechBrain: 감정 감지  │
    │  • librosa: 운율 추출      │
    │  → EmotionProfile 생성     │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  STAGE 2: ASR + 번역        │
    │  • Whisper Large-v3        │
    │  • Claude 번역 (길이 제어) │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  STAGE 3: 감정 제어 TTS     │
    │  • Azure Neural TTS        │
    │  • SSML 감정/운율 태그     │
    │  • 자연스러운 영어 발화    │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  STAGE 4: 보이스 변환       │
    │  • RVC v2                  │
    │  • 크리에이터 목소리 적용  │
    │  • 감정/운율 보존          │
    └──────────────┬──────────────┘
                   │
            ┌──────▼───────┐
            │  더빙된 오디오 │
            └──────────────┘
```

### V1 vs V2 비교

| 기능 | V1 | V2 |
|------|----|----|
| 감정 보존 | 없음 | 자동 감지 및 전달 |
| 음성 품질 | ~90% | 95-97% (목표) |
| 타이밍 | Time-stretching (아티팩트) | 자연스러운 타이밍 |
| 보이스 | ElevenLabs 클로닝 | RVC 변환 (감정 유지) |
| 월 비용 | ~$400 | ~$500 |

---

## 🚀 설치

### 1️⃣ 시스템 요구사항

- **Python**: 3.10 이상
- **FFmpeg**: 오디오/비디오 처리에 필수
- **GPU**: 선택사항 (로컬 Whisper 사용 시 권장)

### 2️⃣ FFmpeg 설치

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
1. [FFmpeg 공식 사이트](https://ffmpeg.org/download.html)에서 다운로드
2. PATH 환경 변수에 추가

### 3️⃣ Python 패키지 설치

```bash
# 저장소 클론 (또는 압축 해제)
cd ai-dubbing-poc

# 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 4️⃣ Demucs 설치 (선택사항, BGM 분리용)

```bash
pip install demucs
```

---

## ⚙️ 설정

### 1️⃣ API 키 설정

`.env.example`을 복사하여 `.env` 파일 생성:

```bash
cp .env.example .env
```

`.env` 파일을 열어 API 키 입력:

```env
# Anthropic API Key (번역용 - 권장) ⭐ NEW!
ANTHROPIC_API_KEY=sk-ant-your-actual-api-key-here

# OpenAI API Key (Whisper API용, 번역은 선택사항)
OPENAI_API_KEY=sk-proj-your-actual-api-key-here

# ElevenLabs API Key (Voice Cloning)
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here

# Respeecher (선택사항)
RESPEECHER_API_KEY=your-respeecher-api-key-here
RESPEECHER_PROJECT_ID=your-project-id
```

> **💡 번역 엔진 권장**: Claude (Anthropic)를 번역에 사용하는 것을 권장합니다!
> - 한국어 이해도 우수, 품질 향상
> - 상세 가이드: [CLAUDE_SETUP.md](CLAUDE_SETUP.md)

### 2️⃣ API 키 발급 방법

#### Anthropic (Claude - 번역용, 권장) ⭐
1. [Anthropic Console](https://console.anthropic.com/)에 가입
2. API Keys 메뉴에서 새 키 생성
3. $5 무료 크레딧으로 시작 (약 10개 영상 테스트 가능)
4. 상세 가이드: [CLAUDE_SETUP.md](CLAUDE_SETUP.md)

#### OpenAI (Whisper API용, 번역은 선택사항)
1. [OpenAI Platform](https://platform.openai.com/)에 가입
2. API Keys 메뉴에서 새 키 생성
3. 결제 정보 등록 (사용량 기반 과금)

#### ElevenLabs (Voice Cloning)
1. [ElevenLabs](https://elevenlabs.io/)에 가입
2. Professional 플랜 구독 ($330/월, 권장)
3. Profile → API Keys에서 키 생성

#### Respeecher (선택사항, 엔터프라이즈급)
1. [Respeecher](https://www.respeecher.com/)에 문의
2. 맞춤 견적 및 계약 진행

### 3️⃣ 설정 파일 커스터마이징

`config.yaml` 파일을 열어 필요에 따라 수정:

```yaml
# 예: 일본어로 더빙하려면
translation:
  target_language: "ja"

# 예: 로컬 Whisper 모델 크기 변경
asr:
  model: "large-v3"  # tiny, base, small, medium, large, large-v3
```

---

## 📖 사용법

### Quick Start: 명령줄 사용

```bash
# 기본 사용법
python pipeline.py input_video.mp4 <VOICE_ID>

# 출력 경로 지정
python pipeline.py input_video.mp4 <VOICE_ID> -o output/dubbed.mp4

# Whisper API 사용 (로컬 모델 대신)
python pipeline.py input_video.mp4 <VOICE_ID> --api

# 임시 파일 보존 (디버깅용)
python pipeline.py input_video.mp4 <VOICE_ID> --keep-temp
```

### 1️⃣ Voice Clone 생성 (최초 1회)

크리에이터의 목소리를 학습시켜 Voice ID를 생성합니다:

```bash
python examples/create_voice_clone.py
```

**준비물:**
- 크리에이터의 음성 샘플 (30-60분 권장)
- 깨끗한 오디오 (잡음 최소화)
- 다양한 감정/톤 포함

**출력:**
- Voice ID (예: `vXYZ123abc...`)
- 이 ID를 더빙 시 사용

### 2️⃣ 영상 더빙 실행

```python
from pipeline import DubbingPipeline

# 파이프라인 초기화
pipeline = DubbingPipeline('config.yaml')

# 더빙 실행
results = pipeline.process_video(
    video_path='videos/korean_content.mp4',
    voice_id='vXYZ123abc',  # 위에서 생성한 Voice ID
    output_path='output/dubbed_english.mp4',
    use_whisper_api=False  # True = API, False = 로컬
)

# 결과 확인
if results['status'] == 'success':
    print(f"완료! {results['output_path']}")
```

### 3️⃣ 출력 파일

더빙 완료 후 생성되는 파일:
- `dubbed_video.mp4`: 더빙된 최종 영상
- `subtitles.srt`: 타임스탬프 포함 자막 파일
- `transcript.json`: 원본 한국어 전사 (선택사항)
- `translation.json`: 번역 결과 (선택사항)

---

## 💰 비용 분석

### PoC 단계 월 예상 비용 (10분 영상 기준)

| 항목 | 서비스 | 비용 | 비고 |
|------|--------|------|------|
| **음성 인식** | Whisper (로컬) | 무료 | GPU 권장 |
| | Whisper API | ~$0.60 | 10분 × $0.006/분 |
| **번역** | GPT-4o | ~$2-5 | 약 3,000 토큰 예상 |
| **Voice Cloning** | ElevenLabs Pro | $330/월 | 무제한 사용 |
| | Respeecher | 맞춤 견적 | 엔터프라이즈 |
| **총계** | | **~$333-335/월** | 무제한 더빙 가능 |

### 전통적 성우 더빙과 비교

| 구분 | 전통 방식 | AI 더빙 (본 시스템) | 절감율 |
|------|----------|-------------------|--------|
| **10분 영상 비용** | $500 - $3,000 | $3 - $10 | **99% 절감** |
| **제작 기간** | 2-4주 | 1-2시간 | **95% 단축** |
| **다국어 확장** | 언어당 재작업 | 동시 처리 가능 | **무제한** |

**ROI 계산 예시:**
- 월 10개 영상 더빙 시: 전통 방식 $15,000 → AI 방식 $400 (97% 절감)
- 6개월 후 손익분기점 도달

---

## 🗺️ 로드맵

### ✅ Phase 1: PoC (완료)
- [x] 기본 파이프라인 구축 (V1)
- [x] ElevenLabs/Respeecher 통합
- [x] 단일 언어 (한→영) 지원
- [x] 명령줄 인터페이스

### ✅ Phase 1.5: 감정 보존 파이프라인 (완료)
- [x] SpeechBrain 감정 분석 모듈
- [x] librosa 운율 추출 모듈
- [x] Azure Neural TTS + SSML 감정 제어
- [x] RVC 보이스 변환 래퍼
- [x] V2 통합 파이프라인
- [x] 단위 테스트 (16 passed)

### 🚧 Phase 2: MVP (다음 단계)
- [ ] DTW 정렬 (세그먼트별 타이밍 맞춤)
- [ ] BGM 분리/믹싱 통합
- [ ] 비디오 합성 연동
- [ ] 웹 UI 구축 (Streamlit/Gradio)
- [ ] 다국어 동시 지원 (영어, 일본어, 중국어)
- [ ] 배치 처리 기능

### 🔮 Phase 3: 프로덕션
- [ ] 클라우드 배포 (AWS/GCP)
- [ ] 사용자 대시보드
- [ ] API 서비스 제공
- [ ] 유튜브 직접 업로드 연동
- [ ] A/B 테스트 도구
- [ ] 자동 립싱크 (Lip-sync) 기능

---

## 🔧 문제 해결

### 일반적인 문제

#### 1. FFmpeg 오류
```
Error: FFmpeg not found
```
**해결:** FFmpeg가 PATH에 있는지 확인
```bash
ffmpeg -version  # 버전 확인
which ffmpeg     # 설치 위치 확인
```

#### 2. OpenAI API 오류
```
Error: Invalid API key
```
**해결:**
- `.env` 파일에 정확한 API 키 입력했는지 확인
- OpenAI 대시보드에서 결제 정보 확인
- API 키 권한 확인 (GPT-4o 접근 권한 필요)

#### 3. ElevenLabs 할당량 초과
```
Error: Quota exceeded
```
**해결:**
- Professional 플랜 구독 확인
- 월 사용량 확인 (Pro는 무제한이지만 rate limit 존재)
- 세그먼트 간 0.1초 딜레이 적용됨 (코드에 포함)

#### 4. 메모리 부족 (로컬 Whisper)
```
Error: CUDA out of memory
```
**해결:**
- 더 작은 모델 사용 (`config.yaml`에서 `large-v3` → `medium`)
- Whisper API 사용 (`--api` 플래그)
- GPU 메모리 확인

#### 5. Demucs 분리 실패
```
Warning: BGM separation failed
```
**해결:**
- Demucs가 설치되어 있는지 확인: `pip install demucs`
- 오디오 파일 형식 확인 (WAV 권장)
- `config.yaml`에서 `separate_bgm: false`로 비활성화 가능

---

## 📚 추가 리소스

### 참고 문서
- [OpenAI Whisper 공식 문서](https://github.com/openai/whisper)
- [ElevenLabs API 문서](https://elevenlabs.io/docs/api-reference)
- [Respeecher 가이드](https://www.respeecher.com/documentation)
- [FFmpeg 가이드](https://ffmpeg.org/documentation.html)

### 관련 연구
- K-FAST 프로젝트 (한국 정부 AI 더빙 이니셔티브)
- 글로벌 AI 비디오 더빙 시장 리포트 (CAGR 44.4%)

---

## 🤝 기여 & 라이선스

이 프로젝트는 PoC(Proof of Concept) 단계로, 내부 평가 및 개발 목적으로 제작되었습니다.

**주의사항:**
- 음성 클론 사용 시 크리에이터의 명시적 동의 필요
- 저작권 및 초상권 보호 준수
- 상업적 사용 시 각 API의 라이선스 확인 필요

---

## 📞 문의

프로젝트 관련 문의:
- Notion 프로젝트 보드 참조
- 이메일: [담당자 이메일]

---

**Made with ❤️ for Korean Creators Going Global** 🌍

---

## 🆕 Pipeline V2: 감정 보존 더빙 (Beta)

원본 한국어 음성의 **감정과 운율을 자동으로 분석**하여 영어 더빙에 반영하는 차세대 파이프라인입니다.

### 추가 요구사항

V1의 기본 설정에 더해 다음이 필요합니다:

```bash
# Azure Speech Services API 키 (.env에 추가)
AZURE_SPEECH_KEY=your-azure-speech-key
AZURE_SPEECH_REGION=eastus
```

RVC 모델 훈련이 필요합니다: [RVC 훈련 가이드](docs/RVC_TRAINING.md)

### 사용법

```python
from src.pipeline_v2 import EmotionAwareDubbingPipeline

pipeline = EmotionAwareDubbingPipeline('config_v2.yaml')
results = pipeline.process_audio('input.wav', 'output.wav')
```

### 기술 스택

| 모듈 | 기술 | 역할 |
|------|------|------|
| 감정 분석 | SpeechBrain wav2vec2 | 기쁨/슬픔/분노/중립 감지 |
| 운율 추출 | librosa | 피치, 에너지, 말하기 속도 |
| TTS | Azure Neural TTS | SSML 감정 제어 음성 합성 |
| 보이스 변환 | RVC v2 | 크리에이터 목소리 적용 |

**상세 문서**: [Pipeline V2 가이드](docs/PIPELINE_V2_GUIDE.md)
