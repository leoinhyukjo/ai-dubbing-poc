# RVC Voice Model Training Guide

## Overview

RVC (Retrieval-based Voice Conversion) enables high-quality voice cloning by training
a model on a target speaker's voice samples. In the AI Dubbing pipeline, RVC converts
TTS-generated speech into the creator's natural voice, preserving emotion and prosody
while matching the original speaker's timbre.

Training a dedicated RVC model for each creator ensures:
- **Natural-sounding output** instead of generic TTS voices
- **Consistent identity** across all dubbed content
- **Emotional fidelity** - the model learns the creator's vocal characteristics

---

## Prerequisites

### Voice Samples
- **Duration**: 30-60 minutes of clean speech (more is better, diminishing returns after 2 hours)
- **Format**: WAV or MP3 (WAV preferred for lossless quality)
- **Content**: Natural speech covering a range of emotions and intonations
- **Quality**: Minimal background noise, no music, no overlapping speakers

### Hardware
- **GPU**: NVIDIA with 8GB+ VRAM (RTX 3060 or better recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free disk space for training data and checkpoints

### Software
- Python 3.9+
- RVC WebUI (cloned into `external/rvc`)
- `librosa`, `soundfile` (installed via `requirements.txt`)

---

## Training Steps

### Step 1: Prepare Voice Samples

Use the provided preparation script to process raw audio into training-ready chunks:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the preparation script
python scripts/prepare_rvc_training.py \
  --input_dir /path/to/raw_voice_samples \
  --output_dir /path/to/prepared_samples \
  --sample_rate 40000
```

The script will:
- Resample audio to the target sample rate (40000 Hz default for RVC)
- Trim silence from the beginning and end
- Split long recordings into 8-second chunks
- Discard chunks shorter than 2 seconds

**Recommended structure:**
```
raw_voice_samples/
  interview_01.wav
  podcast_episode_03.wav
  voiceover_session.mp3
```

### Step 2: Install RVC

Clone the RVC repository into the project:

```bash
cd external/
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git rvc
cd rvc
pip install -r requirements.txt
```

### Step 3: Train the Model

#### Option A: WebUI (Recommended for first-time users)

```bash
cd external/rvc
python infer-web.py
```

1. Open the WebUI in your browser (typically `http://localhost:7865`)
2. Go to the **Train** tab
3. Set the following parameters:
   - **Experiment name**: e.g., `creator_name_v1`
   - **Training data path**: Point to your prepared samples directory
   - **Sample rate**: 40000
   - **Training epochs**: 200-400 (start with 200, increase if quality is insufficient)
   - **Batch size**: Adjust based on your VRAM (8 for 8GB, 16 for 12GB+)
4. Click **Train Model**
5. Monitor the training loss - it should decrease steadily

#### Option B: CLI

```bash
cd external/rvc
python trainset_preprocess_pipeline_print.py \
  /path/to/prepared_samples 40000 12 logs/creator_name_v1 True

python train_nsf_sim_cache_sid_load_pretrain.py \
  -e creator_name_v1 \
  -sr 40000 \
  -f0 1 \
  -bs 8 \
  -te 200 \
  -pg pretrained/f0G40k.pth \
  -pd pretrained/f0D40k.pth \
  -l 1 \
  -c 0
```

### Step 4: Export the Model

After training completes, copy the trained model file to the project:

```bash
# Create model directory
mkdir -p models/rvc/

# Copy the trained .pth file
cp external/rvc/logs/creator_name_v1/G_200.pth models/rvc/creator_name_v1.pth
```

### Step 5: Register the Model with RVCModelManager

```python
from src.voice_conversion.model_manager import RVCModelManager

manager = RVCModelManager(models_dir="models/rvc")

manager.register_model(
    name="creator_name",
    model_path="models/rvc/creator_name_v1.pth",
    metadata={
        "language": "ko",
        "training_epochs": 200,
        "sample_rate": 40000,
        "description": "Creator Name - trained on interview + podcast audio"
    }
)

# Verify registration
print(manager.list_models())  # ['creator_name']
```

### Step 6: Test the Model

Run a quick conversion test to verify quality:

```python
from src.voice_conversion.rvc_converter import RVCConverter

converter = RVCConverter(model_name="creator_name")
result = converter.convert(
    input_audio="test_input.wav",
    output_path="test_output.wav"
)

# Listen to test_output.wav and compare with the creator's real voice
```

---

## Tips for Best Results

### Audio Quality
- **Use a good microphone**: Condenser mics produce cleaner training data than phone recordings
- **Record in a quiet room**: Background noise degrades model quality significantly
- **Avoid reverb**: Dry recordings train better than echoey ones

### Sample Diversity
- Include various **emotions**: happy, neutral, serious, excited
- Include various **speaking styles**: conversational, reading, presenting
- Include various **pitch ranges**: questions (rising), statements (falling), emphasis

### Training Parameters
- **Start with 200 epochs** and evaluate; add more if the voice sounds thin
- **Do not overtrain**: Too many epochs (500+) can cause artifacts and overfitting
- **Use f0 (pitch) guidance**: Always enable for singing or expressive speech models

### Consistency
- All samples should be from the **same speaker**
- Avoid samples where the speaker is sick, whispering, or using an unusual voice
- Remove any segments with laughter, coughing, or non-speech sounds

---

## Troubleshooting

### Robotic or Metallic Sound
- **Cause**: Insufficient training data or too few epochs
- **Fix**: Add more voice samples (aim for 45+ minutes) and train for more epochs (300+)
- **Also check**: Sample rate mismatch between training and inference

### Pitch Issues (Too High / Too Low)
- **Cause**: Pitch extraction errors or f0 method mismatch
- **Fix**: Try a different f0 method (`harvest` vs `crepe` vs `rmvpe`)
- **For cross-gender conversion**: Adjust the transpose parameter (+12 for male-to-female, -12 for female-to-male)

### Audio Artifacts (Clicks, Pops, Distortion)
- **Cause**: Noisy training data or chunks with silence/noise at boundaries
- **Fix**: Re-run the preparation script with stricter trimming, manually remove bad chunks
- **Also try**: Lower the `index_rate` during inference (0.5-0.7 instead of 1.0)

### Model Fails to Load
- **Cause**: Corrupted model file or version mismatch
- **Fix**: Verify the .pth file is complete, re-export from training logs
- **Check**: Ensure the model was registered with the correct path via `RVCModelManager`

### Training Runs Out of VRAM
- **Cause**: Batch size too large for available GPU memory
- **Fix**: Reduce batch size (try 4 or even 2)
- **Alternative**: Use Google Colab with a free T4 GPU for training

---

## Quick Reference

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| Sample Rate | 40000 Hz | Standard for RVC v2 |
| Training Epochs | 200-400 | Start low, increase if needed |
| Batch Size | 8-16 | Depends on VRAM |
| Chunk Length | 8 seconds | Set in preparation script |
| Min Chunk Length | 2 seconds | Shorter chunks are discarded |
| f0 Method | rmvpe | Most accurate for speech |
| Index Rate | 0.75 | Balance between similarity and clarity |
