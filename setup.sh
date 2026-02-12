#!/bin/bash
# AI Dubbing PoC Setup Script

echo "🎬 AI Dubbing PoC - Setup Script"
echo "================================"
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found: Python $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null; then
    echo "❌ Error: Python 3.10 or higher is required"
    exit 1
fi

# Check FFmpeg
echo ""
echo "📌 Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version 2>&1 | head -n1)
    echo "   ✓ $ffmpeg_version"
else
    echo "   ❌ FFmpeg not found"
    echo ""
    echo "   Please install FFmpeg:"
    echo "   - macOS: brew install ffmpeg"
    echo "   - Ubuntu: sudo apt install ffmpeg"
    echo "   - Windows: Download from https://ffmpeg.org"
    exit 1
fi

# Create virtual environment
echo ""
echo "📌 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   ⚠️  Virtual environment already exists"
    read -p "   Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "   ✓ Virtual environment recreated"
    fi
else
    python3 -m venv venv
    echo "   ✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "📌 Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"

# Install dependencies
echo ""
echo "📌 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "   ✓ Dependencies installed"
else
    echo "   ❌ Error installing dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "📌 Creating directories..."
mkdir -p temp output samples
echo "   ✓ Directories created"

# Setup environment file
echo ""
echo "📌 Setting up environment file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   ✓ .env file created from template"
    echo ""
    echo "   ⚠️  IMPORTANT: Edit .env file and add your API keys:"
    echo "      - OPENAI_API_KEY"
    echo "      - ELEVENLABS_API_KEY"
else
    echo "   ℹ️  .env file already exists"
fi

# Final message
echo ""
echo "================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys"
echo "2. Prepare audio samples in 'samples/' directory"
echo "3. Run: python examples/create_voice_clone.py"
echo "4. Run: python pipeline.py your_video.mp4 <VOICE_ID>"
echo ""
echo "For more information, see README.md"
echo ""
