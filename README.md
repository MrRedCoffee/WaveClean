# WaveClean an Advanced Audio Processing Tool 🎧

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Professional audio processing pipeline with multi-stage noise reduction and intelligent silence handling. Perfect for podcast editing, voice recording cleanup, and audio post-production.

## Features ✨

- **Smart Noise Reduction** - Two-stage spectral noise reduction
- **Breath Control** - High-pass filtering for breath sounds
- **Silence Trimming** - Context-aware silence removal
- **Dynamic Processing** - Compression and expansion
- **Format Conversion** - Supports 20+ audio formats via FFmpeg

## Installation 📦

### Prerequisites
- FFmpeg ([Installation Guide](https://ffmpeg.org/download.html))
- Python 3.8+

### Setup
```bash
# Clone repository
git clone https://github.com/MrRedCoffee/WaveClean.git
cd WaveClean

# Install dependencies
pip install -r requirements.txt

```

### Usage
```bash
python audio_processor.py input.mp3 output.mp3
```

### Advanced
```bash
python audio_processor.py raw_audio.m4a cleaned_audio.wav \
  --silence_threshold -45 \
  --min_silence_len 0.2 \
  --breath_cutoff 150 \
  --target_dBFS -16.0```
```

## Troubleshooting 🛠️
### Common Issues
Click artifacts during silence removal:
```bash
# Increase fade time in code (line 84)
fade_time = min(100, duration // 4)  # Increased from 50ms to 100ms
```

FFmpeg not found:
```bash
# Verify FFmpeg installation
ffmpeg -version
# Windows users: Add FFmpeg to PATH
```

High CPU Usage:
- Use smaller FFT sizes (512 instead of 1024)
- Reduce number of processing stages

