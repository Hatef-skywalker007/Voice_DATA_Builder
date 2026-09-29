# Persian Whisper Dataset Builder

## امکانات

- Whisper large-v3-turbo
- CPU با int8
- GPU با float16
- آپلود WAV, MP3, M4A, FLAC, OGG و AAC
- ضبط مستقیم با Streamlit
- FFmpeg برای تبدیل به Mono / 16 kHz
- Silero VAD
- حذف بخش های بدون گفتار
- Peak Normalization بعد از VAD
- اصلاح دستی Transcript
- ثبت نام لهجه
- استخراج RMS، ZCR، Spectral Centroid، Spectral Bandwidth و میانگین 13 MFCC
- جدول موقت در زمان اجرای برنامه
- Save as CSV
- Save Full Dataset به صورت ZIP
- بدون PyAudio
- بدون pyaudioop
- بدون pydub

## ترتیب پردازش

```text
Audio
↓
FFmpeg
↓
Mono / 16 kHz
↓
Silero VAD
↓
Speech-only Audio
↓
Normalization
↓
Whisper large-v3-turbo
↓
Manual Transcript Correction
↓
Dialect
↓
Audio Features
↓
Temporary Dataset
↓
Save
```

## ساختار خروجی

```text
whisper_dataset_YYYYMMDD_HHMMSS/
├── dataset.csv
├── dataset_audio/
│   ├── 00001_sample.wav
│   ├── 00002_sample.wav
│   └── ...
└── raw_audio/
    ├── 00001_sample.m4a
    ├── 00002_sample.wav
    └── ...
```

CSV مسیر WAV پردازش شده و فایل خام را نگه می دارد و خود فایل صوتی را داخل سلول CSV قرار نمی دهد.

## نصب

ابتدا FFmpeg را نصب کنید.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Google Colab

```bash
!apt-get -qq update
!apt-get -qq install -y ffmpeg
%cd /content/whisper_dataset_builder_v2
!pip install -q -r requirements.txt
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > /content/streamlit.log 2>&1 &
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared
!chmod +x /content/cloudflared
!/content/cloudflared tunnel --url http://localhost:8501
```

آدرس موقت چاپ شده توسط tunnel را باز کنید.

## Python 3.13+

این پروژه PyAudio، pyaudioop و pydub ندارد. تبدیل فایل با FFmpeg انجام می شود تا حذف audioop در Python 3.13 باعث خطا نشود.
