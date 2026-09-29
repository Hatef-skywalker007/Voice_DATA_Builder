import hashlib
import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
from faster_whisper import WhisperModel

APP_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
MODEL_NAME = "large-v3-turbo"

if "records" not in st.session_state:
    st.session_state.records = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None


def check_ffmpeg():
    return shutil.which("ffmpeg") is not None


@st.cache_resource(show_spinner="در حال بارگذاری Whisper large-v3-turbo ...")
def load_whisper():
    if torch.cuda.is_available():
        return WhisperModel(MODEL_NAME, device="cuda", compute_type="float16"), "CUDA / float16"
    return WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(1, (os.cpu_count() or 4) // 2),
    ), "CPU / int8"


@st.cache_resource(show_spinner="در حال بارگذاری Silero VAD ...")
def load_vad():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    return model, utils[0]


def convert_to_wav(audio_bytes, extension):
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg پیدا نشد. ابتدا FFmpeg را نصب کنید.")
    extension = extension.lower().lstrip(".") or "wav"
    with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as src:
        src.write(audio_bytes)
        src_path = src.name
    wav_path = f"{src_path}_16k.wav"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", src_path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-c:a", "pcm_s16le", wav_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "FFmpeg خطای ناشناخته داد.")
        y, sr = sf.read(wav_path, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return np.asarray(y, dtype=np.float32), int(sr)
    finally:
        for path in (src_path, wav_path):
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass


def run_vad(y, sr):
    vad_model, get_speech_timestamps = load_vad()
    timestamps = get_speech_timestamps(
        torch.from_numpy(y),
        vad_model,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        speech_pad_ms=150,
        return_seconds=False,
    )
    if not timestamps:
        return y.copy(), []
    pieces = [y[int(x["start"]):int(x["end"])] for x in timestamps if int(x["end"]) > int(x["start"])]
    if not pieces:
        return y.copy(), []
    return np.concatenate(pieces).astype(np.float32), timestamps


def normalize_audio(y):
    if len(y) == 0:
        return y.astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak <= 0:
        return y.astype(np.float32)
    return ((y / peak) * 0.95).astype(np.float32)


def extract_features(y, sr, original_duration, speech_duration):
    base = {
        "sample_rate": sr,
        "channels": 1,
        "original_duration_sec": round(original_duration, 4),
        "speech_duration_sec": round(speech_duration, 4),
        "speech_ratio": round(speech_duration / original_duration, 4) if original_duration else 0.0,
    }
    if len(y) == 0:
        return {
            **base,
            "rms_db": -120.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            **{f"mfcc_{i}_mean": 0.0 for i in range(1, 14)},
        }
    rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    features = {
        **base,
        "rms_db": round(20 * np.log10(max(rms, 1e-9)), 4),
        "zero_crossing_rate": round(float(np.mean(librosa.feature.zero_crossing_rate(y=y))), 6),
        "spectral_centroid_hz": round(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))), 4),
        "spectral_bandwidth_hz": round(float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))), 4),
    }
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, value in enumerate(np.mean(mfcc, axis=1), start=1):
        features[f"mfcc_{i}_mean"] = round(float(value), 4)
    return features


def transcribe_wav(wav_bytes):
    model, runtime = load_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
        file.write(wav_bytes)
        wav_path = file.name
    try:
        segments, info = model.transcribe(
            wav_path,
            language="fa",
            task="transcribe",
            beam_size=3,
            best_of=1,
            temperature=0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        return text, info.language, float(info.language_probability), runtime
    finally:
        try:
            Path(wav_path).unlink()
        except FileNotFoundError:
            pass


def process_audio(audio_bytes, original_name):
    extension = Path(original_name).suffix.lower().lstrip(".") or "wav"
    original_audio, sr = convert_to_wav(audio_bytes, extension)
    original_duration = len(original_audio) / sr
    speech_audio, vad_timestamps = run_vad(original_audio, sr)
    speech_duration = len(speech_audio) / sr
    normalized_audio = normalize_audio(speech_audio)
    with io.BytesIO() as buffer:
        sf.write(buffer, normalized_audio, sr, format="WAV", subtype="PCM_16")
        processed_wav = buffer.getvalue()
    transcript, detected_language, probability, runtime = transcribe_wav(processed_wav)
    return {
        "raw_bytes": audio_bytes,
        "raw_name": original_name,
        "processed_wav": processed_wav,
        "transcript": transcript,
        "dialect": "",
        "detected_language": detected_language,
        "language_probability": round(probability, 4),
        "runtime": runtime,
        "features": extract_features(normalized_audio, sr, original_duration, speech_duration),
        "vad_segments": len(vad_timestamps),
    }


def make_hash(data):
    return hashlib.sha256(data).hexdigest()


def make_row(pending, row_id):
    return {
        "id": row_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_filename": pending["raw_name"],
        "audio_path": "",
        "raw_audio_path": "",
        "transcript": pending["transcript"].strip(),
        "dialect": pending["dialect"].strip(),
        "whisper_model": MODEL_NAME,
        "detected_language": pending["detected_language"],
        "language_probability": pending["language_probability"],
        **pending["features"],
    }


def build_export(records):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    processed_audio = []
    raw_audio = []
    for index, record in enumerate(records, start=1):
        pending = record["_pending"]
        original_stem = Path(pending["raw_name"]).stem.replace(" ", "_")
        stem = f"{index:05d}_{original_stem}"
        wav_name = f"{stem}.wav"
        raw_extension = Path(pending["raw_name"]).suffix.lower() or ".bin"
        raw_name = f"{stem}{raw_extension}"
        row = dict(record)
        row.pop("_pending", None)
        row["audio_path"] = f"dataset_audio/{wav_name}"
        row["raw_audio_path"] = f"raw_audio/{raw_name}"
        rows.append(row)
        processed_audio.append((f"dataset_audio/{wav_name}", pending["processed_wav"]))
        raw_audio.append((f"raw_audio/{raw_name}", pending["raw_bytes"]))
    dataframe = pd.DataFrame(rows)
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")
    zip_buffer = io.BytesIO()
    root_name = f"whisper_dataset_{stamp}"
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{root_name}/dataset.csv", csv_bytes)
        for path, data in processed_audio:
            archive.writestr(f"{root_name}/{path}", data)
        for path, data in raw_audio:
            archive.writestr(f"{root_name}/{path}", data)
    return csv_bytes, zip_buffer.getvalue()


st.set_page_config(page_title="Persian Whisper Dataset Builder", page_icon="🎙️", layout="wide")
st.title("🎙️ Persian Whisper Dataset Builder")
st.write("آپلود یا ضبط صدا، VAD، نرمال‌سازی، Whisper large-v3-turbo، اصلاح Transcript، ثبت لهجه و ساخت دیتاست")

with st.sidebar:
    st.header("تنظیمات")
    st.write(f"Whisper: `{MODEL_NAME}`")
    st.write("Language: Persian")
    st.write("Sample rate: 16 kHz")
    st.write("VAD: Silero VAD")
    if torch.cuda.is_available():
        st.success("GPU: فعال")
    else:
        st.info("GPU پیدا نشد؛ CPU / int8 فعال است")
    if check_ffmpeg():
        st.success("FFmpeg: فعال")
    else:
        st.error("FFmpeg: پیدا نشد")
    st.write(f"تعداد رکوردهای موقت: **{len(st.session_state.records)}**")

source = st.radio("روش ورود صدا", ["آپلود فایل", "ضبط با میکروفون"], horizontal=True)
audio_input = None

if source == "آپلود فایل":
    audio_input = st.file_uploader("فایل صوتی را انتخاب کنید", type=["wav", "mp3", "m4a", "flac", "ogg", "aac"])
else:
    audio_input = st.audio_input("صدای جلسه یا نمونه را ضبط کنید", sample_rate=SAMPLE_RATE)

if audio_input is not None:
    audio_bytes = audio_input.getvalue()
    original_name = getattr(audio_input, "name", "recorded_audio.wav")
    current_hash = make_hash(audio_bytes)
    st.audio(audio_bytes)
    if current_hash != st.session_state.last_audio_hash and st.session_state.pending is None:
        with st.spinner("در حال پردازش VAD و Whisper large-v3-turbo ..."):
            try:
                st.session_state.pending = process_audio(audio_bytes, original_name)
                st.session_state.last_audio_hash = current_hash
            except Exception as error:
                st.error("پردازش فایل انجام نشد.")
                st.exception(error)
                st.session_state.pending = None

if st.session_state.pending is not None:
    pending = st.session_state.pending
    st.subheader("نتیجه پردازش")
    first, second, third, fourth = st.columns(4)
    first.metric("Detected language", pending["detected_language"])
    second.metric("Language probability", pending["language_probability"])
    third.metric("VAD segments", pending["vad_segments"])
    fourth.metric("Runtime", pending["runtime"])
    st.audio(pending["processed_wav"], format="audio/wav")
    transcript = st.text_area("Transcript را در صورت نیاز اصلاح کنید", value=pending["transcript"], height=180, key=f"transcript_{st.session_state.last_audio_hash}")
    dialect = st.text_input("نام لهجه", value=pending["dialect"], placeholder="مثلاً Northern، Tehrani، Isfahani", key=f"dialect_{st.session_state.last_audio_hash}")
    st.subheader("ویژگی‌های صوتی")
    features_dataframe = pd.DataFrame([{"feature": key, "value": value} for key, value in pending["features"].items()])
    st.dataframe(features_dataframe, use_container_width=True, hide_index=True)
    if st.button("➕ افزودن به دیتاست", type="primary", use_container_width=True):
        pending["transcript"] = transcript
        pending["dialect"] = dialect
        row_id = len(st.session_state.records) + 1
        row = make_row(pending, row_id)
        row["_pending"] = pending
        st.session_state.records.append(row)
        st.session_state.pending = None
        st.session_state.last_audio_hash = None
        st.success("رکورد به جدول موقت اضافه شد.")
        st.rerun()

st.divider()
st.subheader(f"جدول دیتاست موقت — {len(st.session_state.records)} رکورد")

if st.session_state.records:
    display_rows = [{key: value for key, value in record.items() if key != "_pending"} for record in st.session_state.records]
    dataframe = pd.DataFrame(display_rows)
    st.dataframe(dataframe, use_container_width=True, hide_index=True)
    st.info("رکوردها تا زمان گرفتن خروجی در حافظه برنامه هستند.")
    csv_bytes, zip_bytes = build_export(st.session_state.records)
    left, right = st.columns(2)
    with left:
        st.download_button("💾 Save as CSV", data=csv_bytes, file_name="dataset.csv", mime="text/csv", use_container_width=True)
    with right:
        st.download_button("📦 Save Full Dataset", data=zip_bytes, file_name="whisper_dataset.zip", mime="application/zip", use_container_width=True)
    if st.button("🗑️ پاک کردن کل دیتاست موقت", use_container_width=True):
        st.session_state.records = []
        st.session_state.pending = None
        st.session_state.last_audio_hash = None
        st.rerun()
else:
    st.info("هنوز رکوردی به دیتاست اضافه نشده است.")
