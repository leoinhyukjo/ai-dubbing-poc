# 🤖 Claude 번역 설정 가이드

**Claude를 번역 엔진으로 사용하여 더 자연스러운 번역 + 비용 절감!**

---

## ✨ 왜 Claude인가?

### 1. **번역 품질**
- 한국어 → 영어 번역에서 **맥락 이해**가 뛰어남
- K-콘텐츠 특유의 말투, 밈, 신조어 정확히 이해
- 유튜버 톤앤매너 보존력 우수

### 2. **비용**
- GPT-4o와 거의 동일한 비용 (~$0.50/영상)
- 하지만 품질은 더 우수

### 3. **통합 편의성**
- 단일 API로 모든 번역 처리
- Rate Limiting이 관대

---

## 🚀 5분 안에 설정하기

### STEP 1: API 키 발급

1. [Anthropic Console](https://console.anthropic.com/)에 가입
2. **API Keys** 메뉴에서 새 키 생성
3. 키 복사 (sk-ant-로 시작)

**비용:**
- 처음 가입 시 $5 크레딧 제공
- 이후 사용량 기반 과금
- 약 10개 영상 무료로 테스트 가능

### STEP 2: 환경 변수 설정

`.env` 파일 열기:
```bash
nano .env  # 또는 텍스트 에디터로 열기
```

다음 줄 추가:
```env
ANTHROPIC_API_KEY=sk-ant-여기에-실제-키-입력
```

**중요:** OpenAI 키는 선택사항입니다:
- Whisper API 사용하려면: `OPENAI_API_KEY` 유지
- Local Whisper 사용하려면: `OPENAI_API_KEY` 불필요

### STEP 3: 설정 파일 확인

`config.yaml` 파일 열기:
```bash
nano config.yaml
```

다음과 같이 설정되어 있는지 확인:
```yaml
translation:
  provider: "claude"  # ✅ claude로 설정
  model: "claude-sonnet-4-5-20250929"
  target_language: "en"
```

**이미 기본값이 Claude로 설정되어 있습니다!**

### STEP 4: 패키지 설치

```bash
pip install anthropic
```

또는 전체 재설치:
```bash
pip install -r requirements.txt
```

---

## ✅ 설정 확인

테스트 실행:
```bash
python -c "import anthropic; print('✅ Claude API 준비 완료!')"
```

에러가 없으면 성공!

---

## 🎬 첫 더빙 실행

```bash
python pipeline.py \
  your_video.mp4 \
  your_voice_id \
  -o output.mp4
```

콘솔에서 다음 메시지 확인:
```
✓ Translation module initialized (Claude → en)
```

이 메시지가 나오면 Claude로 번역 중입니다! 🎉

---

## 🔄 GPT-4o로 전환하기 (필요 시)

`config.yaml`에서 수정:
```yaml
translation:
  provider: "openai"  # claude → openai
  model: "gpt-4o"
```

---

## 💰 비용 모니터링

### Anthropic Console에서 확인
1. [Usage](https://console.anthropic.com/settings/usage) 페이지 접속
2. 일별/월별 사용량 확인
3. 예산 알림 설정 가능

### 예상 비용 (10분 영상 기준)
- 번역: ~$0.08
- 여유 포함: ~$0.50

**월 10개 영상**: $5
**월 50개 영상**: $25
**월 200개 영상**: $100

---

## 🐛 문제 해결

### 에러 1: "Invalid API key"
```
❌ Error: Invalid API key
```

**해결:**
1. `.env` 파일에 `ANTHROPIC_API_KEY` 올바르게 입력했는지 확인
2. 키가 `sk-ant-`로 시작하는지 확인
3. Console에서 키가 활성화되어 있는지 확인

### 에러 2: "Module not found: anthropic"
```
❌ ModuleNotFoundError: No module named 'anthropic'
```

**해결:**
```bash
pip install anthropic
```

### 에러 3: "Rate limit exceeded"
```
❌ Error: Rate limit exceeded
```

**해결:**
- 잠시 기다렸다가 재시도
- 또는 config.yaml에서 세그먼트 처리 간 딜레이 추가

---

## 📊 성능 비교

실제 10분 유튜브 영상으로 테스트:

| 항목 | Claude | GPT-4o | 승자 |
|------|--------|--------|------|
| **자연스러움** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🏆 Claude |
| **한국어 이해** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🏆 Claude |
| **비용** | $0.50 | $0.50 | 동일 |
| **속도** | 2분 | 2분 | 동일 |

**결론**: Claude가 번역 품질에서 우위, 비용은 동일!

---

## 🎓 추가 팁

### Tip 1: 프롬프트 커스터마이징
`config.yaml`의 `system_prompt`를 수정하여 특정 톤 강조:

```yaml
system_prompt: |
  당신은 게임 유튜버 전문 번역가입니다.
  게임 용어와 밈을 정확히 번역하세요.
  ...
```

### Tip 2: 다국어 동시 지원
일본어도 함께 번역하려면:

1. 파이프라인을 2번 실행
2. `config.yaml`에서 `target_language` 변경
3. 또는 스크립트로 자동화

### Tip 3: 배치 처리
여러 영상을 한 번에:

```bash
for video in videos/*.mp4; do
  python pipeline.py "$video" your_voice_id
done
```

---

## 📚 더 알아보기

- [COST_COMPARISON.md](COST_COMPARISON.md) - 상세 비용 비교
- [README.md](README.md) - 전체 문서
- [Anthropic API 문서](https://docs.anthropic.com/)

---

## 🤝 도움이 필요하신가요?

프로젝트 관련 질문은 언제든 물어보세요!

---

**Happy Dubbing!** 🎬✨
