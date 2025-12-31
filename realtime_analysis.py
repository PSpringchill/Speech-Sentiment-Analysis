import argparse
import csv
import json
import math
import os
import queue
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Source Separation imports
try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model as get_demucs_model
except ImportError:
    apply_model = None
    get_demucs_model = None

try:
    from df.enhance import enhance, init_df, load_audio, save_audio
except ImportError:
    enhance = None
    init_df = None

import librosa
import numpy as np
import scipy.stats
import tensorflow as tf
from tensorflow import keras
import pickle
from sklearn.preprocessing import MinMaxScaler

# === CRITICAL FIX: Allow TensorFlow and PyTorch to share GPU ===
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    # Check for Apple Silicon (MPS)
    gpus = tf.config.list_physical_devices('Metal')
    
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow memory growth enabled for {len(gpus)} device(s).")
    except RuntimeError as e:
        print(f"TensorFlow memory growth config error: {e}")
# ===============================================================

# Global Demucs model placeholder
_demucs_model = None
_df_model = None
_df_state = None

def get_df():
    global _df_model, _df_state
    if _df_model is None and init_df is not None:
        print("Loading DeepFilterNet for real-time enhancement...")
        _df_model, _df_state, _ = init_df()
    return _df_model, _df_state

def enhance_audio_mic(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Enhances mic audio by removing noise/music using DeepFilterNet."""
    model, state = get_df()
    if model is None or enhance is None:
        return audio_data
    
    # DeepFilterNet expects 48kHz usually, handles internal resampling
    # but let's convert numpy to torch if needed
    audio_tensor = torch.from_numpy(audio_data).float()
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    enhanced = enhance(model, state, audio_tensor)
    return enhanced.squeeze().cpu().numpy()

def get_demucs():
    global _demucs_model
    if _demucs_model is None and get_demucs_model is not None:
        print("Loading Demucs (htdemucs) for source separation...")
        _demucs_model = get_demucs_model("htdemucs")
        device = "cuda" if torch and torch.cuda.is_available() else ("mps" if torch and torch.backends.mps.is_available() else "cpu")
        _demucs_model.to(device)
        _demucs_model.eval()
    return _demucs_model

def isolate_vocals(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Separates vocals from background music using Demucs."""
    model = get_demucs()
    if model is None or apply_model is None:
        return audio_data

    device = next(model.parameters()).device
    
    # Prepare tensor [1, Channels, Time]
    wav = torch.from_numpy(audio_data).float().to(device)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0) # [1, Time]
    elif wav.ndim == 2 and wav.shape[0] > wav.shape[1]:
        wav = wav.t() # [Channels, Time]
    
    wav = wav.unsqueeze(0) # [1, Channels, Time]

    # Normalize
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    with torch.no_grad():
        # sources: [Batch, Sources, Channels, Time]
        # htdemucs sources: 0=drums, 1=bass, 2=other, 3=vocals
        sources = apply_model(model, wav, shifts=0, split=True, overlap=0.25)
    
    # Extract vocals and mix to mono
    vocals = sources[0, 3, :, :] 
    vocals_mono = vocals.mean(dim=0).cpu().numpy()
    
    return vocals_mono

# Diarization imports
try:
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    import torch
except ImportError:
    EncDecSpeakerLabelModel = None
    torch = None

try:
    import sounddevice as sd
except Exception:
    sd = None

# Transcription imports
try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
except ImportError:
    torch = None

try:
    from pythainlp.util import normalize as thai_normalize
except ImportError:
    thai_normalize = None

try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import sounddevice as sd
except Exception:
    sd = None

# Suppress warnings
warnings.filterwarnings("ignore")


def extract_features_from_audio(data: np.ndarray, sample_rate: int, trim: bool = True) -> np.ndarray:
    y = np.asarray(data, dtype=np.float32).reshape(-1)

    # 1. Normalize amplitude (Dataset files are usually normalized)
    if np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)

    # 2. Trim silence (Notebook skips 0.3s or trims)
    if trim:
        y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < 1024: # Avoid too short audio after trim
        y = np.asarray(data, dtype=np.float32).reshape(-1)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate).T, axis=0)
    energy = np.mean(librosa.feature.rms(y=y).T, axis=0)

    st_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
    entropy_of_energy = np.mean([scipy.stats.entropy(e) for e in st_energy])

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sample_rate).T, axis=0)

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sample_rate).T, axis=0
    )
    spectral_spread = np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sample_rate).T, axis=0
    )
    spectral_rolloff = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sample_rate).T, axis=0
    )

    features = np.hstack(
        (
            mfcc,
            energy,
            entropy_of_energy,
            zcr,
            mel,
            spectral_centroid,
            spectral_spread,
            spectral_rolloff,
        )
    )
    return features


def extract_features(data_path: str | None = None, audio_data: np.ndarray | None = None, sample_rate: int | None = None) -> np.ndarray:
    if audio_data is not None:
        if sample_rate is None:
            raise ValueError("sample_rate must be provided if audio_data is used")
        return extract_features_from_audio(audio_data, int(sample_rate))
    
    if data_path is None:
        raise ValueError("Either data_path or audio_data must be provided")
        
    data, sr = librosa.load(data_path, mono=True)
    return extract_features_from_audio(data, int(sr))


def softmax_entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


class SpeakerManager:
    """Tracks and identifies speakers in real-time using TitaNet embeddings."""
    def __init__(self, threshold=0.65, model_name="titanet_large"):
        if EncDecSpeakerLabelModel is None:
            self.model = None
            print("Warning: NeMo not installed. Diarization disabled.")
            return

        print(f"Loading speaker embedding model: {model_name}...")
        self.model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)
        if torch and torch.cuda.is_available():
            self.model = self.model.cuda()
        elif torch and torch.backends.mps.is_available():
            self.model = self.model.to("mps")
        self.model.eval()
        
        self.gallery = [] # List of dicts: {'embedding': np.array, 'id': int, 'count': int}
        self.threshold = threshold
        self.next_id = 0
        self.history_embeddings = []
        self.history_labels = []

    def identify(self, audio_signal, sample_rate):
        if self.model is None or torch is None:
            return "Unknown"

        # === FIX: Resample to 16000Hz for TitaNet ===
        target_sr = 16000
        if sample_rate != target_sr:
            audio_signal = librosa.resample(audio_signal, orig_sr=sample_rate, target_sr=target_sr)
        # ============================================

        # Prepare audio for NeMo
        audio_tensor = torch.from_numpy(audio_signal).float()
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        device = next(self.model.parameters()).device
        audio_tensor = audio_tensor.to(device)
        audio_len = torch.tensor([audio_tensor.shape[1]]).to(device)

        with torch.no_grad():
            _, embedding = self.model.forward(input_signal=audio_tensor, input_signal_length=audio_len)
            embedding = embedding.cpu().numpy()[0]
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Compare with gallery
        best_sim = -1
        best_id = -1

        for g_item in self.gallery:
            g_emb = g_item['embedding']
            sim = np.dot(embedding, g_emb) # Both are normalized, so dot is cosine sim

            if sim > best_sim:
                best_sim = sim
                best_id = g_item['id']

        if best_sim > self.threshold:
            # Update gallery embedding with moving average to "track" the speaker's voice drift
            for g_item in self.gallery:
                if g_item['id'] == best_id:
                    # Update centroid: new_avg = (old_avg * count + current) / (count + 1)
                    # But using a weighted moving average is often better for real-time drift
                    g_item['embedding'] = 0.8 * g_item['embedding'] + 0.2 * embedding
                    # Re-normalize
                    g_item['embedding'] /= np.linalg.norm(g_item['embedding'])
                    g_item['count'] += 1
                    break
            
            # LOGIC FOR CLONES:
            # If similarity is High (>0.85), it's very likely the same person.
            # If similarity is Medium (threshold to 0.80), it might be a Clone or tone change.
            hint = ""
            if best_sim > 0.85:
                hint = " [Match: High]"
            elif best_sim > 0.70:
                hint = " [Match: Medium - Check for Clone]"
            else:
                hint = " [Match: Low - Tone Change?]"
                
            return f"Speaker_{best_id}{hint}"
        else:
            # New speaker
            new_id = self.next_id
            self.next_id += 1
            self.gallery.append({
                'embedding': embedding,
                'id': new_id,
                'count': 1
            })
            print(f"Detected New Speaker: Speaker_{new_id} (Best Sim: {best_sim:.4f})")
            result_id = f"Speaker_{new_id}"
        
        # SAVE FOR PAPER/RESEARCH
        self.history_embeddings.append(embedding)
        self.history_labels.append(result_id)
        
        return result_id

    def save_for_paper(self, filename="experiment_data.npz"):
        if not self.history_embeddings:
            print("No embedding history to save.")
            return
        np.savez(filename, 
                 embeddings=np.array(self.history_embeddings), 
                 labels=np.array(self.history_labels))
        print(f"Data saved for research analysis: {filename}")

@dataclass
class Metrics:
    classes: list[str]

    def __post_init__(self) -> None:
        self.total = 0
        self.counts = {c: 0 for c in self.classes}
        self.prob_sum = np.zeros((len(self.classes),), dtype=np.float64)
        self.conf_sum = 0.0
        self.entropy_sum = 0.0

    def update(self, probs: np.ndarray) -> dict:
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        conf = float(probs[pred_idx])
        ent = softmax_entropy(probs)

        self.total += 1
        self.counts[pred_class] += 1
        self.prob_sum += probs
        self.conf_sum += conf
        self.entropy_sum += ent

        return {
            "pred_class": pred_class,
            "confidence": conf,
            "entropy": ent,
            "probs": probs,
        }

    def summary(self) -> dict:
        if self.total == 0:
            avg_probs = np.zeros((len(self.classes),), dtype=np.float64)
            avg_conf = 0.0
            avg_entropy = 0.0
        else:
            avg_probs = self.prob_sum / float(self.total)
            avg_conf = self.conf_sum / float(self.total)
            avg_entropy = self.entropy_sum / float(self.total)

        percentages = {
            c: (0.0 if self.total == 0 else (self.counts[c] / self.total) * 100.0)
            for c in self.classes
        }

        top_class = None
        if self.total > 0:
            top_class = max(self.classes, key=lambda c: self.counts[c])

        return {
            "total": self.total,
            "counts": dict(self.counts),
            "percentages": percentages,
            "avg_probs": {c: float(avg_probs[i]) for i, c in enumerate(self.classes)},
            "avg_confidence": avg_conf,
            "avg_entropy": avg_entropy,
            "top_class": top_class,
        }


def prepare_input(features: np.ndarray) -> np.ndarray:
    raise RuntimeError("prepare_input is initialized at runtime")


def load_scaler(path: Path) -> MinMaxScaler:
    with path.open("rb") as f:
        scaler = pickle.load(f)
    if not isinstance(scaler, MinMaxScaler):
        raise TypeError(f"Expected MinMaxScaler in {path}, got {type(scaler)}")
    return scaler


def save_scaler(path: Path, scaler: MinMaxScaler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(scaler, f)


def fit_minmax_scaler(
    *, input_dir: Path, extensions: set[str], max_files: int
) -> MinMaxScaler:
    files = iter_media_files(input_dir, extensions)
    if not files:
        raise ValueError(f"No media files found under: {input_dir}")

    if max_files > 0 and len(files) > max_files:
        step = max(1, len(files) // max_files)
        files = files[::step][:max_files]

    print(f"Calibrating MinMaxScaler using up to {len(files)} files...")
    
    feats_list: list[np.ndarray] = []
    iterator = tqdm(files, desc="Extracting features") if tqdm else files
    
    for p in iterator:
        try:
            feats_list.append(extract_features(str(p)))
        except Exception:
            continue

    if not feats_list:
        raise ValueError("Failed to extract features for scaler calibration")

    X = np.stack(feats_list, axis=0)
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler


def make_prepare_input(scaler: MinMaxScaler | None, *, clip_scaled: bool, baseline: np.ndarray | None = None):
    def _prepare(features: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=np.float32).copy()
        
        # 1. Baseline subtraction (relative features)
        if baseline is not None:
            feats = feats - baseline
            
        feats = feats.reshape(1, -1)
        
        # 2. Scaling
        if scaler is not None:
            feats = scaler.transform(feats)
            if clip_scaled:
                feats = np.clip(feats, 0.0, 1.0)
        
        # 3. Reshape for CNN (samples, features, 1)
        x = feats.reshape(-1, 1)
        x = np.expand_dims(x, axis=0)
        return x

    return _prepare


def load_neutral_baseline(path: Path, sample_rate: int) -> np.ndarray:
    print(f"Loading neutral baseline from: {path}")
    audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return extract_features_from_audio(audio, sample_rate)


def top_k_str(classes: list[str], probs: np.ndarray, k: int) -> str:
    k = max(1, int(k))
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    idxs = np.argsort(p)[::-1][:k]
    return ",".join(f"{classes[int(i)]}:{float(p[int(i)]):.4f}" for i in idxs)


class ASRManager:
    def __init__(self, mode="thai", device=None):
        self.mode = mode
        if device is None:
            if torch and torch.cuda.is_available():
                self.device = "cuda"
            elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if torch is None:
            raise ImportError("torch and transformers are required for ASR. Please install them.")

        self.processor = None
        self.model = None

        if mode == "thai":
            # Using AIResearch model which is the standard for Thai Wav2Vec2
            model_id = "airesearch/wav2vec2-large-xlsr-53-th"
        else:
            # Standard English model
            model_id = "facebook/wav2vec2-base-960h"

        print(f"Loading ASR model '{model_id}' on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            # Wav2Vec2 models are almost always 16k
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        if self.mode == "thai" and thai_normalize:
            transcription = thai_normalize(transcription)
            
        return transcription.lower() if self.mode == "eng" else transcription


def ensure_csv_header(path: Path, classes: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ts_utc",
                "file",
                "speaker",
                "pred_class",
                "confidence",
                "entropy",
                "rms",
                "text",
                *[f"prob_{c}" for c in classes],
            ]
        )


def append_csv_row(path: Path, classes: list[str], row: dict) -> None:
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["ts_utc"],
                row["file"],
                row.get("speaker", "Unknown"),
                row["pred_class"],
                f"{row['confidence']:.8f}",
                f"{row['entropy']:.8f}",
                f"{row.get('rms', 0.0):.8f}",
                row.get("text", ""),
                *[f"{row['probs_by_class'][c]:.8f}" for c in classes],
            ]
        )


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


MEDIA_EXTENSIONS_DEFAULT = ".wav,.mp3,.flac,.m4a,.aac,.ogg,.mp4,.mov,.mkv,.webm"


def parse_extensions(extensions: str) -> set[str]:
    exts: set[str] = set()
    for raw in extensions.split(","):
        ext = raw.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.add(ext)
    return exts


def emit_summary(metrics: Metrics, summary_json_path: Path | None) -> dict:
    summary = metrics.summary()
    summary["ts_utc"] = now_utc_iso()
    print(json.dumps(summary, indent=2))

    if summary_json_path:
        with summary_json_path.open("w") as f:
            json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_model = repo_root / "models" / "model_2" / "CNN_model.keras"

    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["watch", "mic"],
        default="watch",
        help="watch = monitor a folder for media files, mic = stream from microphone",
    )
    p.add_argument(
        "--model",
        default=str(default_model),
        help="Path to a .keras model file",
    )
    p.add_argument(
        "--input-dir",
        default=str(repo_root / "data" / "input"),
        help="Directory to watch for new media files (recursively)",
    )
    p.add_argument(
        "--extensions",
        default=MEDIA_EXTENSIONS_DEFAULT,
        help="Comma-separated file extensions to process in watch mode",
    )
    p.add_argument(
        "--scaler-path",
        default="artifacts/minmax_scaler.pkl",
        help="Path to a pickled sklearn MinMaxScaler (relative paths are relative to repo root)",
    )
    p.add_argument(
        "--calibrate-dir",
        default="",
        help="Directory of media files to fit MinMaxScaler if scaler-path does not exist",
    )
    p.add_argument(
        "--calibrate-max-files",
        type=int,
        default=1000,
        help="Maximum number of files to use when fitting MinMaxScaler",
    )
    p.add_argument(
        "--no-scaler",
        action="store_true",
        help="Disable MinMax scaling (not recommended; for debugging only)",
    )
    p.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable clipping scaled features to [0,1] (not recommended)",
    )
    p.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval for scanning input-dir",
    )
    p.add_argument(
        "--summary-seconds",
        type=float,
        default=10.0,
        help="How often to print the metrics summary",
    )
    p.add_argument(
        "--log-csv",
        default="",
        help="If set, append per-file predictions to this CSV path",
    )
    p.add_argument(
        "--summary-json",
        default="",
        help="If set, periodically write the latest summary to this JSON path",
    )
    p.add_argument(
        "--move-to",
        default="",
        help="If set, move processed wav files into this directory (keeps relative structure)",
    )
    p.add_argument(
        "--classes",
        default="angry,calm,fear,happy,sad,surprise",
        help="Comma-separated class labels to SHOW. 'disgust' is excluded by default as it often causes bias.",
    )

    p.add_argument(
        "--mic-device",
        default="",
        help="Microphone device name or index (empty = default device)",
    )
    p.add_argument(
        "--mic-sample-rate",
        type=int,
        default=22050,
        help="Microphone sample rate",
    )
    p.add_argument(
        "--mic-channels",
        type=int,
        default=1,
        help="Number of mic channels to capture (will be mixed to mono)",
    )
    p.add_argument(
        "--window-seconds",
        type=float,
        default=3.0,
        help="Window size (seconds) for each mic prediction",
    )
    p.add_argument(
        "--hop-seconds",
        type=float,
        default=1.0,
        help="Time between mic predictions (seconds)",
    )
    p.add_argument(
        "--mic-duration-seconds",
        type=float,
        default=0.0,
        help="If > 0, stop mic mode after this many seconds",
    )
    p.add_argument(
        "--min-rms",
        type=float,
        default=0.01,
        help="Mic mode: skip windows with RMS below this threshold",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help="Mic mode: EMA smoothing factor for probabilities (0 disables)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Print top-k probabilities (1 disables extra output)",
    )
    p.add_argument(
        "--asr",
        choices=["none", "thai", "eng"],
        default="none",
        help="Enable real-time transcription (none, thai, eng)",
    )
    p.add_argument(
        "--asr-device",
        default="",
        help="Device for ASR (cpu, cuda, mps). Empty = auto-detect.",
    )
    p.add_argument(
        "--neutral-baseline-file",
        default="",
        help="Path to a wav file containing neutral voice for initial calibration",
    )
    p.add_argument(
        "--dynamic-calibration",
        action="store_true",
        help="Enable dynamic baseline adjustment based on high-confidence calm/neutral detections",
    )
    p.add_argument(
        "--calibration-frames",
        type=int,
        default=5,
        help="Number of initial frames to use for calibration if no baseline file is provided",
    )
    p.add_argument(
        "--auto-bias",
        action="store_true",
        help="Capture average prediction bias during calibration and subtract it from future results",
    )
    p.add_argument(
        "--diarization",
        action="store_true",
        help="Enable real-time speaker diarization using TitaNet-L",
    )
    p.add_argument(
        "--speaker-threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for identifying the same speaker (0.0 to 1.0). Default 0.6 is more lenient.",
    )
    p.add_argument(
        "--source-separation",
        action="store_true",
        help="Enable background music removal using Demucs (watch mode only)",
    )
    p.add_argument(
        "--mic-source-separation",
        action="store_true",
        help="Enable real-time noise/music removal using DeepFilterNet (mic mode)",
    )
    return p.parse_args()


def iter_media_files(input_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def run_watch(
    *,
    model: keras.Model,
    show_classes: list[str],
    keep_indices: list[int],
    metrics: Metrics,
    input_dir: Path,
    extensions: set[str],
    poll_seconds: float,
    summary_seconds: float,
    log_csv_path: Path | None,
    summary_json_path: Path | None,
    move_to: Path | None,
    asr_manager: ASRManager | None = None,
    source_separation: bool = False,
) -> None:
    processed: set[Path] = set()
    last_summary_ts = 0.0

    while True:
        files = iter_media_files(input_dir, extensions)

        for media_path in files:
            if media_path in processed:
                continue

            try:
                # Transcribe/Enhance if enabled
                transcription = ""
                audio = None
                sr = 16000 # Default target for model consistency
                
                # If we need the audio data for ASR or Source Separation
                if asr_manager or source_separation:
                    audio, sr = librosa.load(str(media_path), sr=16000, mono=True)
                    
                    if source_separation and len(audio) > sr * 1.0:
                        print(f"Removing background music for {media_path.name}...")
                        audio = isolate_vocals(audio, sr)
                    
                    if asr_manager:
                        transcription = asr_manager.transcribe(audio, sr)

                if audio is not None:
                    feats = extract_features(audio_data=audio, sample_rate=sr)
                else:
                    feats = extract_features(str(media_path))

                x = prepare_input(feats)
                raw_probs = model.predict(x, verbose=0)[0]
                
                # Filter and re-normalize probabilities
                probs = raw_probs[keep_indices]
                if np.sum(probs) > 0:
                    probs = probs / np.sum(probs)
                else:
                    probs = np.ones(len(show_classes)) / len(show_classes)
            except Exception as e:
                ts_utc = now_utc_iso()
                print(f"[{ts_utc}] ERROR {media_path}: {e}")
                processed.add(media_path)
                continue

            upd = metrics.update(probs)
            ts_utc = now_utc_iso()
            probs_by_class = {show_classes[i]: float(probs[i]) for i in range(len(show_classes))}

            text_part = f" | text: \"{transcription}\"" if transcription else ""
            print(
                f"[{ts_utc}] {media_path.name} -> {upd['pred_class']} (conf={upd['confidence']:.4f}, ent={upd['entropy']:.4f}){text_part}"
            )

            if log_csv_path:
                # Calculate RMS if not already calculated (e.g. if asr_manager was not used)
                rms_val = 0.0
                if 'audio' in locals():
                    rms_val = float(np.sqrt(np.mean(np.square(audio))))
                
                append_csv_row(
                    log_csv_path,
                    show_classes,
                    {
                        "ts_utc": ts_utc,
                        "file": str(media_path.relative_to(input_dir)),
                        "speaker": "N/A",
                        "pred_class": upd["pred_class"],
                        "confidence": upd["confidence"],
                        "entropy": upd["entropy"],
                        "rms": rms_val,
                        "text": transcription,
                        "probs_by_class": probs_by_class,
                    },
                )

            processed.add(media_path)

            if move_to:
                dest = move_to / media_path.relative_to(input_dir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    media_path.rename(dest)
                except OSError:
                    pass

        now_ts = time.time()
        if now_ts - last_summary_ts >= float(summary_seconds):
            emit_summary(metrics, summary_json_path)
            last_summary_ts = now_ts

        time.sleep(float(poll_seconds))


def run_mic(
    *,
    model: keras.Model,
    show_classes: list[str],
    keep_indices: list[int],
    metrics: Metrics,
    sample_rate: int,
    channels: int,
    device: str,
    window_seconds: float,
    hop_seconds: float,
    duration_seconds: float,
    summary_seconds: float,
    log_csv_path: Path | None,
    summary_json_path: Path | None,
    min_rms: float,
    ema_alpha: float,
    top_k: int,
    asr_manager: ASRManager | None = None,
    initial_baseline: np.ndarray | None = None,
    dynamic_calibration: bool = False,
    calibration_frames: int = 5,
    scaler: MinMaxScaler | None = None,
    clip_scaled: bool = True,
    auto_bias: bool = False,
    diarization: bool = False,
    speaker_threshold: float = 0.75,
    mic_source_separation: bool = False,
) -> None:
    if sd is None:
        raise RuntimeError(
            "Microphone mode requires the 'sounddevice' package. Install it in your venv: pip install sounddevice"
        )

    if window_seconds <= 0:
        raise ValueError("window-seconds must be > 0")
    if hop_seconds <= 0:
        raise ValueError("hop-seconds must be > 0")

    window_samples = int(math.ceil(window_seconds * float(sample_rate)))
    buf: deque[float] = deque(maxlen=window_samples)
    lock = threading.Lock()
    seg_idx = 0
    last_summary_ts = 0.0
    start_ts = time.time()
    ema_probs: np.ndarray | None = None
    
    current_baseline = initial_baseline
    calibration_buffer = []
    bias_buffer = []
    prob_bias = np.zeros(len(show_classes))
    is_calibrating = (initial_baseline is None or auto_bias)

    speaker_manager = None
    if diarization:
        speaker_manager = SpeakerManager(threshold=speaker_threshold)
    
    # State-based prepare_input that uses current_baseline
    def _dynamic_prepare(features: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=np.float32).copy()
        if current_baseline is not None:
            feats = feats - current_baseline
        
        feats = feats.reshape(1, -1)
        if scaler is not None:
            feats = scaler.transform(feats)
            if clip_scaled:
                feats = np.clip(feats, 0.0, 1.0)
        
        x = feats.reshape(-1, 1)
        x = np.expand_dims(x, axis=0)
        return x

    def callback(indata: np.ndarray, frames: int, time_info: dict, status) -> None:
        if status:
            print(status, file=sys.stderr)
        if indata.ndim == 1:
            mono = indata
        else:
            mono = indata.mean(axis=1)
        with lock:
            buf.extend(mono.astype(np.float32).tolist())

    sd_device = None
    if device != "":
        try:
            sd_device = int(device)
        except ValueError:
            sd_device = device

    print("--- Starting Microphone Stream ---")
    if is_calibrating:
        print(f"Calibration Phase: Please speak neutrally for {calibration_frames} segments...")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        device=sd_device,
        dtype="float32",
        callback=callback,
    ):
        try:
            while True:
                time.sleep(float(hop_seconds))

                if duration_seconds > 0 and (time.time() - start_ts) >= float(duration_seconds):
                    break

                with lock:
                    if len(buf) < window_samples:
                        continue
                    y = np.asarray(buf, dtype=np.float32).copy()

                rms = float(np.sqrt(np.mean(np.square(y))))
                if rms < float(min_rms):
                    continue

                try:
                    if mic_source_separation:
                        y = enhance_audio_mic(y, sample_rate)
                        rms = float(np.sqrt(np.mean(np.square(y))))

                    feats = extract_features_from_audio(y, sample_rate)
                    
                    if is_calibrating:
                        if initial_baseline is None:
                            calibration_buffer.append(feats)
                        
                        if len(calibration_buffer) >= calibration_frames:
                            if initial_baseline is None:
                                current_baseline = np.mean(calibration_buffer, axis=0)
                                print("Feature baseline established.")
                            
                            if auto_bias and len(bias_buffer) < calibration_frames:
                                x = _dynamic_prepare(feats)
                                raw_p = model.predict(x, verbose=0)[0]
                                p = raw_p[keep_indices]
                                if np.sum(p) > 0:
                                    p = p / np.sum(p)
                                else:
                                    p = np.ones(len(show_classes)) / len(show_classes)
                                
                                bias_buffer.append(p)
                                print(f"Bias Calibration: {len(bias_buffer)}/{calibration_frames}")
                                if len(bias_buffer) >= calibration_frames:
                                    prob_bias = np.mean(bias_buffer, axis=0)
                                    is_calibrating = False
                                    print(f"Auto-bias established: {top_k_str(show_classes, prob_bias, 3)}")
                            else:
                                is_calibrating = False
                                print("Calibration Complete!")
                        continue

                    transcription = ""
                    if asr_manager:
                        transcription = asr_manager.transcribe(y, sample_rate)

                    speaker_tag = ""
                    if speaker_manager:
                        speaker_tag = speaker_manager.identify(y, sample_rate)
                        speaker_tag = f"[{speaker_tag}] "

                    x = _dynamic_prepare(feats)
                    raw_probs = model.predict(x, verbose=0)[0]
                    
                    probs = raw_probs[keep_indices]
                    if np.sum(probs) > 0:
                        probs = probs / np.sum(probs)
                    else:
                        probs = np.ones(len(show_classes)) / len(show_classes)
                    
                    if auto_bias:
                        probs = probs - prob_bias
                        probs = np.maximum(probs, 0.0)
                        if np.sum(probs) > 0:
                            probs = probs / np.sum(probs)
                        else:
                            probs = np.ones(len(show_classes)) / len(show_classes)
                except Exception as e:
                    ts_utc = now_utc_iso()
                    print(f"[{ts_utc}] ERROR mic segment {seg_idx}: {e}")
                    seg_idx += 1
                    continue

                probs = np.asarray(probs, dtype=np.float64)
                if ema_alpha > 0:
                    if ema_probs is None:
                        ema_probs = probs
                    else:
                        ema_probs = (float(ema_alpha) * probs) + (1.0 - float(ema_alpha)) * ema_probs
                    probs_to_use = ema_probs
                else:
                    probs_to_use = probs

                upd = metrics.update(probs_to_use)
                ts_utc = now_utc_iso()
                
                if dynamic_calibration and upd["pred_class"] == "calm" and upd["confidence"] > 0.8:
                    current_baseline = 0.95 * current_baseline + 0.05 * feats

                probs_by_class = {
                    show_classes[i]: float(probs_to_use[i]) for i in range(len(show_classes))
                }

                extra = ""
                if int(top_k) > 1:
                    extra = f" top{int(top_k)}=[{top_k_str(show_classes, probs_to_use, int(top_k))}]"
                
                text_part = f" | text: \"{transcription}\"" if transcription else ""

                print(
                    f"[{ts_utc}] {speaker_tag}mic[{seg_idx}] -> {upd['pred_class']} (conf={upd['confidence']:.4f}, ent={upd['entropy']:.4f}, rms={rms:.4f}){extra}{text_part}"
                )

                if log_csv_path:
                    append_csv_row(
                        log_csv_path,
                        show_classes,
                        {
                            "ts_utc": ts_utc,
                            "file": f"mic:{seg_idx}",
                            "speaker": speaker_tag.strip("[] "),
                            "pred_class": upd["pred_class"],
                            "confidence": upd["confidence"],
                            "entropy": upd["entropy"],
                            "rms": rms,
                            "text": transcription,
                            "probs_by_class": probs_by_class,
                        },
                    )

                seg_idx += 1

                now_ts = time.time()
                if now_ts - last_summary_ts >= float(summary_seconds):
                    emit_summary(metrics, summary_json_path)
                    last_summary_ts = now_ts
        except KeyboardInterrupt:
            print("\nMicrophone stream stopped by user.")
        finally:
            if speaker_manager:
                speaker_manager.save_for_paper()
            emit_summary(metrics, summary_json_path)


def main() -> None:
    args = parse_args()

    # The model was trained with 7 specific classes in this EXACT order
    TRAINED_CLASSES = ["angry", "calm", "disgust", "fear", "happy", "sad", "surprise"]
    
    # User-requested classes (to show)
    show_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not show_classes:
        raise ValueError("classes must be non-empty")

    # Indices of classes to keep from the model output
    keep_indices = [i for i, c in enumerate(TRAINED_CLASSES) if c in show_classes]
    
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = keras.models.load_model(str(model_path))
    
    # Metrics should only track the classes we are showing
    metrics = Metrics(classes=show_classes)

    log_csv_path = Path(args.log_csv).expanduser().resolve() if args.log_csv else None
    if log_csv_path:
        ensure_csv_header(log_csv_path, show_classes)

    summary_json_path = (
        Path(args.summary_json).expanduser().resolve() if args.summary_json else None
    )
    if summary_json_path:
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent

    scaler_path = Path(args.scaler_path)
    if not scaler_path.is_absolute():
        scaler_path = repo_root / scaler_path
    scaler_path = scaler_path.expanduser().resolve()

    scaler: MinMaxScaler | None
    if args.no_scaler:
        scaler = None
    else:
        if scaler_path.exists():
            scaler = load_scaler(scaler_path)
        else:
            if args.calibrate_dir:
                cal_dir = Path(args.calibrate_dir).expanduser().resolve()
            else:
                cal_dir = repo_root / "data" / "input"
            extensions_for_cal = parse_extensions(args.extensions)
            scaler = fit_minmax_scaler(
                input_dir=cal_dir,
                extensions=extensions_for_cal,
                max_files=int(args.calibrate_max_files),
            )
            save_scaler(scaler_path, scaler)

    global prepare_input
    prepare_input = make_prepare_input(scaler, clip_scaled=(not bool(args.no_clip)))

    initial_baseline = None
    if args.neutral_baseline_file:
        baseline_path = Path(args.neutral_baseline_file).expanduser().resolve()
        if baseline_path.exists():
            initial_baseline = load_neutral_baseline(baseline_path, int(args.mic_sample_rate))
        else:
            print(f"Warning: Neutral baseline file not found: {baseline_path}")

    asr_manager = None
    if args.asr != "none":
        asr_device = args.asr_device if args.asr_device else None
        asr_manager = ASRManager(mode=args.asr, device=asr_device)

    if args.mode == "watch":
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        extensions = parse_extensions(args.extensions)
        if not extensions:
            raise ValueError("extensions must be non-empty")

        move_to = Path(args.move_to).expanduser().resolve() if args.move_to else None
        if move_to:
            move_to.mkdir(parents=True, exist_ok=True)

        run_watch(
            model=model,
            show_classes=show_classes,
            keep_indices=keep_indices,
            metrics=metrics,
            input_dir=input_dir,
            extensions=extensions,
            poll_seconds=float(args.poll_seconds),
            summary_seconds=float(args.summary_seconds),
            log_csv_path=log_csv_path,
            summary_json_path=summary_json_path,
            move_to=move_to,
            asr_manager=asr_manager,
            source_separation=bool(args.source_separation),
        )
        return

    run_mic(
        model=model,
        show_classes=show_classes,
        keep_indices=keep_indices,
        metrics=metrics,
        sample_rate=int(args.mic_sample_rate),
        channels=int(args.mic_channels),
        device=str(args.mic_device),
        window_seconds=float(args.window_seconds),
        hop_seconds=float(args.hop_seconds),
        duration_seconds=float(args.mic_duration_seconds),
        summary_seconds=float(args.summary_seconds),
        log_csv_path=log_csv_path,
        summary_json_path=summary_json_path,
        min_rms=float(args.min_rms),
        ema_alpha=float(args.ema_alpha),
        top_k=int(args.top_k),
        asr_manager=asr_manager,
        initial_baseline=initial_baseline,
        dynamic_calibration=bool(args.dynamic_calibration),
        calibration_frames=int(args.calibration_frames),
        scaler=scaler,
        clip_scaled=(not bool(args.no_clip)),
        auto_bias=bool(args.auto_bias),
        diarization=bool(args.diarization),
        speaker_threshold=float(args.speaker_threshold),
        mic_source_separation=bool(args.mic_source_separation),
    )


if __name__ == "__main__":
    main()
