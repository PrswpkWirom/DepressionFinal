# ========= CONFIG =========
ROOT = "/media/popsatorn/timeshift_backup/DAIC-WOZ"  # parent of 300, 301, ...
OUT_DIR = "/media/popsatorn/timeshift_backup/DAIC-WOZ/DAIC_embeddings"
DEBUG = True
SR_TARGET = 16_000
TEXT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"  # 768-d
AUDIO_MODEL_ID = "microsoft/wavlm-base-plus"               # 768-d
TEXT_BATCH = 64
AUDIO_BATCH = 16  # (kept for signature compat; streaming ignores it)
STREAM_WINDOW_SECONDS = 20.0  # <= reduce if you still see OOM
USE_FP16_FOR_WAVLM = True     # set False if your GPU dislikes fp16
MIN_SAMPLES_WAVLM = 400  # ~25 ms at 16k; safe minimum for WavLM conv stack
# NEW: resume behavior — skip sessions that already have <sid>.pt in OUT_DIR
RESUME_SKIP_EXISTING = True

# =========================

# ---- (Optional) help CUDA fragmentation; must be set before importing torch
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import os, math, pathlib
from glob import glob
from typing import List, Dict, Tuple, Optional

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"In 2\.9, this function's implementation will be changed to use torchaudio\.load_with_torchcodec",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Support for mismatched key_padding_mask and attn_mask is deprecated",
    category=UserWarning,
)

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transformers import AutoFeatureExtractor, WavLMModel

# Treat weird Windows/legacy NaN/Inf tokens as missing
BAD_NA_TOKENS = {
    "-1.#IND", "1.#IND", "1.#QNAN", "-1.#QNAN",
    "1.#INF", "-1.#INF", "#IND", "#QNAN",
    "IND", "INF", "-INF", "NaN", "NAN", "nan", "Null", "NULL", "null", ""
}

# --------- Setup ---------
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEBUG:
    print("Selected device:", device)
torch.backends.cudnn.benchmark = True

# Load models once
# Put text model on CPU to save VRAM
txt_model = SentenceTransformer(TEXT_MODEL_ID, device="cpu")

wavlm_processor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
wavlm_model = WavLMModel.from_pretrained(AUDIO_MODEL_ID)
wavlm_model.eval()
# Keep WavLM off GPU until needed
wavlm_model.to("cpu")
if DEBUG:
    print("WavLM initial device:", next(wavlm_model.parameters()).device,
          "dtype:", next(wavlm_model.parameters()).dtype)

def to_path(x): return pathlib.Path(x)

# ---- tiny context to borrow GPU then free it
from contextlib import contextmanager
@contextmanager
def wavlm_on_gpu(dev="cuda:0", dtype=None):
    """Temporarily move WavLM to GPU, then back to CPU and flush VRAM."""
    if torch.cuda.is_available():
        wavlm_model.to(dev)
        if dtype is not None:
            wavlm_model.to(dtype=dtype)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            wavlm_model.to("cpu")
            torch.cuda.empty_cache()

# --------- Helpers ---------
def clean_string(s: str) -> str:
    return (s or "").strip()

def _read_dataframe_smart(path: str) -> pd.DataFrame:
    """
    Robust CSV/TSV reader:
    - tries comma; if single column, retries with '\t'
    - finally uses sep=None + engine='python' (csv.Sniffer)
    - treats Windows-style tokens like '-1.#IND' as NaN
    - strips BOM from header cells and trims stray spaces in string cells
    """
    def _strip_bom_cols(df):
        df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
        return df

    def _post(df):
        df = _strip_bom_cols(df)
        # trim leading/trailing spaces in object columns so ' -1.#IND' -> '-1.#IND'
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()
        return df

    common = dict(
        low_memory=False,           # avoid DtypeWarning + better type inference
        na_values=BAD_NA_TOKENS,    # normalize weird tokens to NaN
        keep_default_na=True,
        skipinitialspace=True,      # trim spaces right after delimiters
    )

    # try comma
    try:
        df = pd.read_csv(path, **common)
        df = _post(df)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # try tab
    try:
        df = pd.read_csv(path, sep="\t", **common)
        df = _post(df)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # try sniffer
    df = pd.read_csv(path, sep=None, engine="python", **common)
    df = _post(df)
    return df


def is_participant(name: str) -> bool:
    if not isinstance(name, str): return False
    name = name.strip().lower()
    return name == "participant" or name.startswith("participant")

def discover_sessions(root: str) -> List[str]:
    items = sorted([p for p in glob(os.path.join(root, "*")) if os.path.isdir(p)])
    nums = [p for p in items if to_path(p).name.isdigit()]
    return nums if nums else items

def find_session_files(sess_dir: str, sid: str) -> Dict[str, Optional[str]]:
    d = to_path(sess_dir)
    g = lambda suf: next((str(p) for p in d.glob(f"*{suf}")), None)
    return {
        "audio": g("_AUDIO.wav"),
        "transcript": g("_TRANSCRIPT.csv"),
        "clnf_aus": g("_CLNF_AUs.csv"),
        "clnf_f3d": g("_CLNF_features3D.csv"),
        "clnf_gaze": g("_CLNF_gaze.csv"),
        "clnf_pose": g("_CLNF_pose.csv"),
    }

def load_wav_resampled_mono(path: str, sr_target: int = SR_TARGET) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # (C, S)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sr_target)
    return wav.squeeze(0).contiguous().float()

def slice_audio(wav: torch.Tensor, start_t: float, stop_t: float, sr: int = SR_TARGET) -> torch.Tensor:
    if math.isnan(start_t) or math.isnan(stop_t):
        return torch.empty(0)
    s = max(0, int(round(start_t * sr)))
    e = max(0, int(round(stop_t * sr)))
    e = min(e, wav.numel())
    if e <= s:
        return torch.empty(0)
    return wav[s:e].clone()

# ---- STREAMING, OOM-SAFE AUDIO EMBEDDING ----
@torch.no_grad()
def embed_audio_fragments(frags: List[torch.Tensor], batch_size: int = AUDIO_BATCH) -> torch.Tensor:
    """
    Mean-pool WavLM frame features per fragment → (N, 768).
    Robust to ultra-short fragments by zero-padding to MIN_SAMPLES_WAVLM
    and supplying a matching sample-level attention mask.
    Empty fragment → zero vector.
    """
    outs = []
    i = 0
    model_dev   = next(wavlm_model.parameters()).device
    model_dtype = next(wavlm_model.parameters()).dtype

    while i < len(frags):
        batch = frags[i:i+batch_size]

        batch_arr: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        lens: List[int] = []

        for x in batch:
            L = int(x.numel()) if isinstance(x, torch.Tensor) else int(len(x))
            lens.append(L)

            if L == 0:
                # pad to minimum; mask is all zeros so it contributes nothing
                arr  = np.zeros((MIN_SAMPLES_WAVLM,), dtype=np.float32)
                mask = np.zeros((MIN_SAMPLES_WAVLM,), dtype=np.int64)
            else:
                arr = x.detach().cpu().numpy().astype(np.float32)
                if L < MIN_SAMPLES_WAVLM:
                    pad = MIN_SAMPLES_WAVLM - L
                    arr  = np.pad(arr, (0, pad), mode="constant")
                    mask = np.concatenate(
                        [np.ones((L,), dtype=np.int64), np.zeros((pad,), dtype=np.int64)]
                    )
                else:
                    mask = np.ones((L,), dtype=np.int64)
            batch_arr.append(arr)
            mask_list.append(mask)

        # Use our masks so padding is respected even for manually padded items
        inputs = wavlm_processor(
            batch_arr,
            sampling_rate=SR_TARGET,
            return_tensors="pt",
            padding=True,                 # pad to longest in this batch
            attention_mask=mask_list,     # <-- our per-sample masks
        )

        input_values   = inputs["input_values"].to(device=model_dev, dtype=model_dtype)
        attention_mask = inputs["attention_mask"].to(device=model_dev, dtype=torch.bool)

        out   = wavlm_model(input_values=input_values, attention_mask=attention_mask)
        feats = out.last_hidden_state  # (B, T_frame, 768)

        # Convert sample-mask → frame-mask using WavLM’s helper
        valid_samples = attention_mask.sum(-1)  # (B,)
        frame_lengths = wavlm_model._get_feat_extract_output_lengths(valid_samples).to(feats.device)  # (B,)
        T = feats.size(1)
        frame_mask = (torch.arange(T, device=feats.device).unsqueeze(0) < frame_lengths.unsqueeze(1))
        frame_mask = frame_mask.unsqueeze(-1).to(feats.dtype)  # (B, T, 1)

        summed  = (feats * frame_mask).sum(dim=1)       # (B, 768)
        lengths = frame_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        embs    = summed / lengths                      # (B, 768)

        # Keep "empty" fragments as zeros
        for j, L in enumerate(lens):
            if L == 0:
                embs[j].zero_()

        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        outs.append(embs.detach().cpu().float())
        i += batch_size

        # Batch-level cleanup (helps long runs)
        del inputs, input_values, attention_mask, out, feats, frame_mask, summed, lengths, embs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not outs:
        return torch.zeros((0, 768), dtype=torch.float32)
    return torch.vstack(outs)


def embed_text(sentences: List[str]) -> torch.Tensor:
    if not sentences:
        return torch.zeros((0, 768), dtype=torch.float32)
    vecs = txt_model.encode(
        sentences, batch_size=TEXT_BATCH, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False
    )
    return torch.from_numpy(vecs.astype(np.float32))

def read_transcript_keep_participant(path_csv: str) -> pd.DataFrame:
    df = _read_dataframe_smart(path_csv)

    # normalize header names
    norm = lambda s: s.lower().strip().replace("\ufeff", "")
    raw_cols = {norm(c): c for c in df.columns}

    # accept common aliases just in case
    aliases = {
        "start_time": ["start_time", "start", "start time", "onset"],
        "stop_time":  ["stop_time", "stop", "stop time", "offset", "end_time", "end"],
        "speaker":    ["speaker", "participant_type", "role"],
        "value":      ["value", "transcript", "utterance", "text"],
    }

    def pick(name):
        for cand in aliases[name]:
            if cand in raw_cols:
                return raw_cols[cand]
        raise ValueError(
            f"Transcript missing column '{name}' in {path_csv}. "
            f"Found columns: {list(df.columns)}"
        )

    c_start = pick("start_time")
    c_stop  = pick("stop_time")
    c_spk   = pick("speaker")
    c_val   = pick("value")

    # filter participant
    df = df.rename(columns={c_start:"start_time", c_stop:"stop_time",
                            c_spk:"speaker", c_val:"value"})
    m = df["speaker"].astype(str).map(is_participant)
    df = df[m].copy()

    # clean + sort
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["stop_time"]  = pd.to_numeric(df["stop_time"],  errors="coerce")
    df["value"]      = df["value"].astype(str).fillna("")
    df = df[(df["stop_time"] > df["start_time"]) & df["start_time"].notna() & df["stop_time"].notna()]
    df = df.sort_values("start_time").reset_index(drop=True)
    return df[["start_time", "stop_time", "value"]]

def read_clnf(path_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if not path_csv or not os.path.exists(path_csv):
        return None
    df = _read_dataframe_smart(path_csv)

    # find/construct timestamp
    if "timestamp" not in df.columns:
        # try common alternates
        for alt in ["time", "ts", "t", " frame_time"]:
            if alt in df.columns:
                df["timestamp"] = pd.to_numeric(df[alt], errors="coerce")
                break
        else:
            if "frame" in df.columns and "fps" in df.columns:
                df["timestamp"] = pd.to_numeric(df["frame"], errors="coerce") / float(pd.to_numeric(df["fps"].iloc[0], errors="coerce"))
            else:
                raise ValueError(f"{path_csv} has no 'timestamp' column and no recoverable alternative.")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].reset_index(drop=True)
    return df

def time_crop_df(df: pd.DataFrame, start: float, stop: float, drop_meta=True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    sub = df[(df["timestamp"] >= start) & (df["timestamp"] < stop)]
    ts = sub["timestamp"].to_numpy(dtype=np.float32)
    if drop_meta:
        drop = [c for c in ["frame","success","confidence","timestamp"," face_id","face_id","fps"] if c in sub.columns]
        X = sub.drop(columns=drop, errors="ignore")
    else:
        X = sub.drop(columns=["timestamp"], errors="ignore")

    # Ensure every column is numeric; non-numeric parses become NaN (safe for float32)
    X = X.apply(pd.to_numeric, errors="coerce")

    cols = list(X.columns)
    X = X.to_numpy(dtype=np.float32, copy=True)
    return X, ts, cols


def _stat1d(arr):
    if len(arr) == 0:
        return (0, 0, 0)
    a = np.asarray(arr)
    return (int(np.min(a)), int(np.median(a)), int(np.max(a)))


def _clnf_stats(clnf_list):
    lens = [int(x["X"].shape[0]) for x in clnf_list]
    D = int(clnf_list[0]["X"].shape[1]) if (len(clnf_list) and clnf_list[0]["X"].ndim == 2 and clnf_list[0]["X"].numel()>0) else (int(clnf_list[0]["X"].shape[1]) if len(clnf_list) else 0)
    total = int(np.sum(lens)) if lens else 0
    mn, md, mx = _stat1d(lens)
    return {"D": D, "frames_total": total, "len_min": mn, "len_med": md, "len_max": mx, "lens": lens}

def _print_session_debug(sid, N, text_emb, audio_emb, empty_audio_count, clnf_lists):
    if not DEBUG: return
    print(f"\n=== [{sid}] DEBUG ===")
    print(f"Utterances (Participant only): N = {N}")
    print(f"text_emb  shape: {tuple(text_emb.shape)}")
    print(f"audio_emb shape: {tuple(audio_emb.shape)}   (empty audio slices: {empty_audio_count}/{N})")
    for name in ["aus", "features3d", "gaze", "pose"]:
        st = _clnf_stats(clnf_lists[name])
        print(f"CLNF[{name}] -> D={st['D']}, frames_total={st['frames_total']}, "
              f"per-utt len (min/med/max) = ({st['len_min']}/{st['len_med']}/{st['len_max']})")

def save_pt(obj, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(obj, out_path)

# --------- Core per-session ---------
from typing import Optional, Tuple

def process_session(sess_dir: str) -> Optional[Tuple[str, dict]]:
    sid = to_path(sess_dir).name
    out_path = os.path.join(OUT_DIR, f"{sid}.pt")

    # NEW: safety skip inside the worker too
    if RESUME_SKIP_EXISTING and os.path.exists(out_path):
        if DEBUG:
            print(f"[{sid}] SKIP - already exists: {out_path}")
        return None

    files = find_session_files(sess_dir, sid)
    if files["transcript"] is None or files["audio"] is None:
        if DEBUG:
            miss = [k for k, v in files.items() if v is None]
            print(f"[{sid}] SKIP - missing: {miss}")
        return None

    # 1) Transcript → participant-only utterances
    tr = read_transcript_keep_participant(files["transcript"])
    sentences = [clean_string(s) for s in tr["value"].tolist()]
    times = tr[["start_time", "stop_time"]].to_numpy(dtype=np.float32)
    N = len(sentences)

    # 2) Text embeddings (CPU)
    text_emb = embed_text(sentences)

    # 3) Audio embeddings (GPU only inside context, with streaming)
    wav = load_wav_resampled_mono(files["audio"], sr_target=SR_TARGET)
    frags = [slice_audio(wav, float(s), float(e), sr=SR_TARGET) for s, e in times]
    dtype = torch.float16 if (USE_FP16_FOR_WAVLM and torch.cuda.is_available()) else None
    with wavlm_on_gpu(dev=device, dtype=dtype):
        audio_emb = embed_audio_fragments(frags).to(torch.float32)
    empty_audio_count = sum(1 for x in frags if x.numel() == 0)

    if DEBUG:
        if text_emb.shape[0] != N:
            print(f"[{sid}] WARN text_emb rows ({text_emb.shape[0]}) != N ({N})")
        if audio_emb.shape[0] != N:
            print(f"[{sid}] WARN audio_emb rows ({audio_emb.shape[0]}) != N ({N})")

    # 4) CLNF slices (unpooled)
    df_aus  = read_clnf(files["clnf_aus"])
    df_f3d  = read_clnf(files["clnf_f3d"])
    df_gaze = read_clnf(files["clnf_gaze"])
    df_pose = read_clnf(files["clnf_pose"])

    clnf_lists = {"aus": [], "features3d": [], "gaze": [], "pose": []}
    cols_map = {}

    for (start, stop) in times:
        def crop(df):
            if df is None:
                return np.zeros((0, 0), np.float32), np.zeros((0,), np.float32), []
            return time_crop_df(df, float(start), float(stop), drop_meta=True)

        X_aus,  ts_aus,  c_aus  = crop(df_aus)
        X_f3d,  ts_f3d,  c_f3d  = crop(df_f3d)
        X_gaze, ts_gaze, c_gaze = crop(df_gaze)
        X_pose, ts_pose, c_pose = crop(df_pose)

        clnf_lists["aus"].append(  {"t": torch.from_numpy(ts_aus),  "X": torch.from_numpy(X_aus)} )
        clnf_lists["features3d"].append({"t": torch.from_numpy(ts_f3d), "X": torch.from_numpy(X_f3d)} )
        clnf_lists["gaze"].append( {"t": torch.from_numpy(ts_gaze), "X": torch.from_numpy(X_gaze)} )
        clnf_lists["pose"].append( {"t": torch.from_numpy(ts_pose), "X": torch.from_numpy(X_pose)} )

        if not cols_map:
            cols_map = {"aus": c_aus, "features3d": c_f3d, "gaze": c_gaze, "pose": c_pose}

    # 5) Debug print
    _print_session_debug(sid, N, text_emb, audio_emb, empty_audio_count, clnf_lists)

    # 6) Pack debug + output object
    dbg = {
        "N": N,
        "text_emb_shape": tuple(text_emb.shape),
        "audio_emb_shape": tuple(audio_emb.shape),
        "empty_audio_fragments": int(empty_audio_count),
        "clnf_stats": {k: _clnf_stats(v) for k, v in clnf_lists.items()},
    }

    out_obj = {
        "session_id": int(sid) if sid.isdigit() else sid,
        "sentences": sentences,
        "times": torch.from_numpy(times),          # (N, 2)
        "text_emb": text_emb.to(torch.float32),    # (N, 768)
        "audio_emb": audio_emb.to(torch.float32),  # (N, 768)
        "clnf": clnf_lists,                        # dict of lists of dicts
        "meta": {
            "audio_sr": SR_TARGET,
            "text_model": TEXT_MODEL_ID,
            "audio_model": AUDIO_MODEL_ID,
            "paths": files,
            "device_used": device,
            "notes": "WavLM mean-pooled per utterance via streaming; CLNF kept as raw per-frame features.",
        },
        "clnf_cols": cols_map,
        "debug": dbg,
    }

    save_pt(out_obj, out_path)

    # Build the row for CSV (no .pt reload)
    row = {
        "session": str(sid),
        "N": dbg["N"],
        "text_shape": str(dbg["text_emb_shape"]),
        "audio_shape": str(dbg["audio_emb_shape"]),
        "empty_audio": dbg["empty_audio_fragments"],
        "aus_D": dbg["clnf_stats"]["aus"]["D"],
        "aus_total": dbg["clnf_stats"]["aus"]["frames_total"],
        "f3d_D": dbg["clnf_stats"]["features3d"]["D"],
        "f3d_total": dbg["clnf_stats"]["features3d"]["frames_total"],
        "gaze_D": dbg["clnf_stats"]["gaze"]["D"],
        "gaze_total": dbg["clnf_stats"]["gaze"]["frames_total"],
        "pose_D": dbg["clnf_stats"]["pose"]["D"],
        "pose_total": dbg["clnf_stats"]["pose"]["frames_total"],
    }

    # 7) FREE per-session memory (CPU & GPU)
    try:
        del wav, frags, text_emb, audio_emb, df_aus, df_f3d, df_gaze, df_pose
        del clnf_lists, cols_map, tr, sentences, times, out_obj, dbg
    except NameError:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path, row

def main():
    sessions = discover_sessions(ROOT)
    print(f"Found {len(sessions)} session folders under {ROOT}")

    # NEW: make a fast skip list from existing .pt files
    existing = {to_path(p).stem for p in glob(os.path.join(OUT_DIR, "*.pt"))}
    if DEBUG and existing:
        print(f"Resume mode: {len(existing)} sessions already have .pt files and will be skipped.")

    made = []
    debug_rows = []
    for sdir in tqdm(sessions, desc="sessions"):
        sid = to_path(sdir).name
        if RESUME_SKIP_EXISTING and (sid in existing):
            if DEBUG:
                print(f"[{sid}] SKIP - already exists in OUT_DIR")
            continue

        result = process_session(sdir)
        if result:
            outp, row = result
            made.append(outp)
            debug_rows.append(row)

    print(f"Done. Wrote {len(made)} new .pt files to {OUT_DIR}")

    # Merge/append debug summary without duplicates on 'session'
    csv_path = os.path.join(OUT_DIR, "preprocess_debug_summary.csv")
    if debug_rows:
        df_new = pd.DataFrame(debug_rows)
        if os.path.exists(csv_path):
            try:
                df_old = pd.read_csv(csv_path)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["session"], keep="first")
            except Exception:
                df_all = df_new
        else:
            df_all = df_new
        df_all = df_all.sort_values("session")
        df_all.to_csv(csv_path, index=False)
        if DEBUG:
            print(f"Debug summary updated at: {csv_path}")

if __name__ == "__main__":
    main()
