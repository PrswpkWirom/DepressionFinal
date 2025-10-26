# PCA for sequential embeddings (fit on train only; apply to all splits).
# Assumptions:
#   - You already moved files into:
#       /media/popsatorn/timeshift_backup/DAIC-WOZ/DAIC_embeddings/{train,validate,test}
#   - Each .pt contains a dict with keys like 'text_emb' and 'audio_emb', each [T, D]
#     (fallback keys handled below).
#
# Output:
#   - Saves PCA models: pca_text_200.joblib, pca_audio_200.joblib
#   - Writes updated .pt files with added keys 'text_pca' and 'audio_pca' (shape [T,200])
#
# Tip: If you want whitening, set PCA_WHITEN=True (usually leave False for embeddings).

from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
import joblib

# ---------- Config ----------
ROOT = Path("/media/popsatorn/external/depressionFinalProject/DAIC-WOZ/DAIC_embeddings")
SPLITS = ["train", "validate", "test"]
N_COMPONENTS = 64
PCA_BATCH_ROWS = 8192      # rows = time steps; tune if you have more/less RAM
PCA_RANDOM_STATE = 42
PCA_WHITEN = False         # True -> decorrelate & unit-variance PCs (changes scale)
SAVE_MODELS_TO = Path(__file__).resolve().parent / "pca64_model"    # folder to store joblib PCA models
WRITE_IN_PLACE = True      # True: overwrite same .pt file (add new keys); False: write to new files
SUFFIX_WHEN_COPY = "_with_pca"  # used only if WRITE_IN_PLACE=False
# ---------------------------

def find_arrays(d):
    """
    Extract text and audio arrays from a loaded torch dict.
    Returns (text_np, audio_np) possibly None if missing.
    Accepts multiple common key names.
    """
    # possible keys
    text_keys = ["text_emb", "text_embed", "text", "sentence_emb", "sentence_embed"]
    audio_keys = ["audio_emb", "audio_embed", "audio"]

    def pick(keys):
        for k in keys:
            if k in d and d[k] is not None:
                x = d[k]
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                else:
                    x = np.asarray(x)
                # ensure 2D [T, D]
                if x.ndim == 1:
                    x = x.reshape(-1, x.shape[-1])
                return x.astype(np.float32, copy=False)
        return None

    return pick(text_keys), pick(audio_keys)

def iter_sessions(split_dir: Path):
    """Yield (path, data_dict) for all .pt files in a split directory."""
    for p in sorted(split_dir.glob("*.pt")):
        try:
            d = torch.load(p, map_location="cpu")
            if not isinstance(d, dict):
                # wrap non-dict
                d = {"data": d}
            yield p, d
        except Exception as e:
            print(f"⚠️  Skipping {p.name}: cannot load ({e})")

def fit_incremental_pca_on_train(train_dir: Path):
    """Fit IPCA for text and audio using rows aggregated across all train sessions."""
    ipca_text = IncrementalPCA(n_components=N_COMPONENTS, whiten=PCA_WHITEN)
    ipca_audio = IncrementalPCA(n_components=N_COMPONENTS, whiten=PCA_WHITEN)

    # For reproducibility of randomized SVD path
    # (IncrementalPCA uses LAPACK by default; still fix np RNG for any internal sampling)
    rng = np.random.RandomState(PCA_RANDOM_STATE)

    # We’ll stream batches of rows across all sessions
    buf_text, rows_text = [], 0
    buf_audio, rows_audio = [], 0

    n_text_dim, n_audio_dim = None, None
    n_text_rows, n_audio_rows = 0, 0

    def flush(which):
        nonlocal buf_text, rows_text, buf_audio, rows_audio, n_text_rows, n_audio_rows
        if which in ("text", "both") and rows_text > 0:
            X = np.vstack(buf_text)
            ipca_text.partial_fit(X)
            n_text_rows += X.shape[0]
            buf_text, rows_text = [], 0
        if which in ("audio", "both") and rows_audio > 0:
            Y = np.vstack(buf_audio)
            ipca_audio.partial_fit(Y)
            n_audio_rows += Y.shape[0]
            buf_audio, rows_audio = [], 0

    for path, d in iter_sessions(train_dir):
        x_text, x_audio = find_arrays(d)
        if x_text is not None:
            if n_text_dim is None:
                n_text_dim = x_text.shape[1]
            elif x_text.shape[1] != n_text_dim:
                print(f"⚠️  {path.name} text dim {x_text.shape[1]} != {n_text_dim}; skipping text.")
                x_text = None
        if x_audio is not None:
            if n_audio_dim is None:
                n_audio_dim = x_audio.shape[1]
            elif x_audio.shape[1] != n_audio_dim:
                print(f"⚠️  {path.name} audio dim {x_audio.shape[1]} != {n_audio_dim}; skipping audio.")
                x_audio = None

        if x_text is not None:
            buf_text.append(x_text)
            rows_text += x_text.shape[0]
            if rows_text >= PCA_BATCH_ROWS:
                flush("text")
        if x_audio is not None:
            buf_audio.append(x_audio)
            rows_audio += x_audio.shape[0]
            if rows_audio >= PCA_BATCH_ROWS:
                flush("audio")

    # Flush leftovers
    flush("both")

    if n_text_rows == 0:
        print("❌ No text rows seen in train.")
    else:
        print(f"✅ Text PCA fitted on {n_text_rows} time steps, dim={n_text_dim} → {N_COMPONENTS}")

    if n_audio_rows == 0:
        print("❌ No audio rows seen in train.")
    else:
        print(f"✅ Audio PCA fitted on {n_audio_rows} time steps, dim={n_audio_dim} → {N_COMPONENTS}")

    return ipca_text if n_text_rows > 0 else None, ipca_audio if n_audio_rows > 0 else None

def transform_and_save(ipca_text, ipca_audio, split_dir: Path):
    out_stats = {"written": 0, "skipped": 0}
    for path, d in iter_sessions(split_dir):
        x_text, x_audio = find_arrays(d)
        changed = False

        if (ipca_text is not None) and (x_text is not None) and (x_text.shape[1] == ipca_text.n_features_in_):
            Xt = ipca_text.transform(x_text).astype(np.float32, copy=False)
            d["text_pca64"] = torch.from_numpy(Xt)
            changed = True
        elif x_text is not None:
            print(f"⚠️  {path.name}: text dim {x_text.shape[1]} != PCA.n_features_in_={ipca_text.n_features_in_ if ipca_text else 'N/A'}; skipping text.")

        if (ipca_audio is not None) and (x_audio is not None) and (x_audio.shape[1] == ipca_audio.n_features_in_):
            Ya = ipca_audio.transform(x_audio).astype(np.float32, copy=False)
            d["audio_pca64"] = torch.from_numpy(Ya)
            changed = True
        elif x_audio is not None:
            print(f"⚠️  {path.name}: audio dim {x_audio.shape[1]} != PCA.n_features_in_={ipca_audio.n_features_in_ if ipca_audio else 'N/A'}; skipping audio.")

        if changed:
            if WRITE_IN_PLACE:
                torch.save(d, path)
            else:
                new_path = path.with_name(path.stem + SUFFIX_WHEN_COPY + path.suffix)
                torch.save(d, new_path)
            out_stats["written"] += 1
        else:
            out_stats["skipped"] += 1

    print(f"→ {split_dir.name}: wrote {out_stats['written']} files, skipped {out_stats['skipped']}.")
    return out_stats

def main():
    train_dir = ROOT / "train"
    assert train_dir.exists(), f"Train dir not found: {train_dir}"

    # 1) Fit PCA on train only (streaming rows to avoid RAM blow-up)
    ipca_text, ipca_audio = fit_incremental_pca_on_train(train_dir)

    # 2) Save PCA models for reuse/repro
    if ipca_text is not None:
        joblib.dump(ipca_text, SAVE_MODELS_TO / f"pca_text_{N_COMPONENTS}.joblib")
    if ipca_audio is not None:
        joblib.dump(ipca_audio, SAVE_MODELS_TO / f"pca_audio_{N_COMPONENTS}.joblib")

    # 3) Transform each split with the frozen PCAs
    for split in SPLITS:
        split_dir = ROOT / split
        if split_dir.exists():
            transform_and_save(ipca_text, ipca_audio, split_dir)
        else:
            print(f"ℹ️  Skip missing split dir: {split_dir}")

if __name__ == "__main__":
    main()
