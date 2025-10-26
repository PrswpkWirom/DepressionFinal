# dataloader_mffnet.py
import os
import csv
from glob import glob
from typing import List, Tuple, Dict, Optional
import math
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
# ---------- Helpers ----------
def _read_labels(csv_path: str) -> Dict[int, int]:
    """
    Read labels from a CSV with columns:
    Participant_ID,PHQ_Binary,PHQ_Score,Gender
    Returns {participant_id: phq_binary}
    """
    labels = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            pid = int(row["Participant_ID"])
            y = int(row["PHQ_Binary"])
            labels[pid] = y
    return labels


# ---------- Dataset ----------
class DAICEmbeddingsDataset(Dataset):
    """
    Each .pt contains at least:
      - 'text_pca'  : FloatTensor [T, d_text]
      - 'audio_pca' : FloatTensor [T, d_audio]
    """
    def __init__(
        self,
        root_dir: str,
        split: str,                        # 'train' | 'validate' | 'test'
        labels_dir: Optional[str] = None,  # defaults to <root_dir>/label
        text_key: str = "text_pca",
        audio_key: str = "audio_pca",
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        assert os.path.isdir(self.split_dir), f"Missing split dir: {self.split_dir}"

        # labels
        if labels_dir is None:
            labels_dir = os.path.join(root_dir, "label")
        csv_path = os.path.join(labels_dir, f"{'validate' if split=='validate' else split}.csv")
        assert os.path.isfile(csv_path), f"Missing label CSV: {csv_path}"
        self.labels = _read_labels(csv_path)

        # collect files in split
        self.files = sorted(glob(os.path.join(self.split_dir, "*.pt")))
        assert len(self.files) > 0, f"No .pt files in {self.split_dir}"

        # only keep files that have labels
        files_keep = []
        for fp in self.files:
            pid = int(os.path.splitext(os.path.basename(fp))[0])
            if pid in self.labels:
                files_keep.append(fp)
        self.files = files_keep
        assert len(self.files) > 0, "No files matched with labels. Check IDs."

        self.text_key = text_key
        self.audio_key = audio_key

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        pid = int(os.path.splitext(os.path.basename(fp))[0])

        obj = torch.load(fp, map_location="cpu")
        text = obj[self.text_key].to(torch.float32)   # [T, d_text]
        audio = obj[self.audio_key].to(torch.float32) # [T, d_audio]
        assert text.dim() == 2 and audio.dim() == 2, f"Bad shapes in {fp}"
        assert text.size(0) == audio.size(0), f"Length mismatch in {fp}"

        length = text.size(0)
        y = int(self.labels[pid])  # 0/1

        return {
            "pid": pid,
            "audio": audio,   # [T, d_a]
            "text": text,     # [T, d_t]
            "length": length,
            "label": y,
        }


# ---------- Collate (pad to max-T in batch) ----------
def collate_mffnet(batch: List[dict]):
    """
    Returns:
      audio_padded: FloatTensor (B, T_max, d_audio)
      text_padded : FloatTensor (B, T_max, d_text)
      mask        : BoolTensor  (B, T_max)  True=valid
      labels      : LongTensor  (B,)
      pids        : List[int]
    """
    B = len(batch)
    lengths = [b["length"] for b in batch]
    T_max = max(lengths)

    d_audio = batch[0]["audio"].size(1)
    d_text  = batch[0]["text"].size(1)

    audio_pad = torch.zeros(B, T_max, d_audio, dtype=torch.float32)
    text_pad  = torch.zeros(B, T_max, d_text,  dtype=torch.float32)
    mask      = torch.zeros(B, T_max, dtype=torch.bool)
    labels    = torch.zeros(B, dtype=torch.long)
    pids      = []

    for i, b in enumerate(batch):
        L = b["length"]
        audio_pad[i, :L] = b["audio"]
        text_pad[i,  :L] = b["text"]
        mask[i, :L] = True
        labels[i] = b["label"]
        pids.append(b["pid"])

    return audio_pad, text_pad, mask, labels, pids


def collate_mffnet_fragmented(
    batch: List[dict],
    seg_len: int = 16,
    pad_threshold: float = 0.20,  # allow padding if missing < 20% of seg_len
    debug: bool = False,
):
    """
    Splits each sequence into non-overlapping segments of length `seg_len`.
    - Full segments are always kept.
    - The trailing remainder is kept and padded *only if* its missing part is < pad_threshold * seg_len.
      (i.e., remainder_len >= ceil((1 - pad_threshold) * seg_len)).
    - Otherwise the trailing remainder is discarded.

    Returns:
      audio: FloatTensor (N_total, seg_len, d_audio)
      text : FloatTensor (N_total, seg_len, d_text)
      mask : BoolTensor  (N_total, seg_len)  True = valid
      labels: LongTensor (N_total,)
      pids: List[int]    (duplicated per produced fragment)
    """
    assert 0 < pad_threshold < 1.0, "pad_threshold should be in (0, 1)."
    assert seg_len > 0, "seg_len must be positive."

    # Peek dims from the first item
    d_audio = batch[0]["audio"].size(1)
    d_text  = batch[0]["text"].size(1)

    audio_segs = []
    text_segs  = []
    masks      = []
    labels     = []
    pids       = []

    min_keep_remainder = math.ceil((1.0 - pad_threshold) * seg_len)
    if debug:
        print(f"[DEBUG] seg_len={seg_len}, pad_threshold={pad_threshold}")
        print(f"[DEBUG] min_keep_remainder={min_keep_remainder}")

    for i, b in enumerate(batch):
        pid     = b["pid"]
        y       = b["label"]
        audio   = b["audio"]  # [T, d_a]
        text    = b["text"]   # [T, d_t]
        T       = b["length"]

        if debug:
            print(f"\n[DEBUG] Sample {i}: pid={pid}, label={y}, length={T}")
            print(f"[DEBUG] Audio shape: {tuple(audio.shape)}, Text shape: {tuple(text.shape)}")

        # Full non-overlapping chunks
        n_full = T // seg_len
        if debug:
            print(f"[DEBUG] n_full={n_full}, remainder={T - n_full * seg_len}")

        for k in range(n_full):
            s = k * seg_len
            e = s + seg_len
            if debug:
                print(f"  [DEBUG] Segment {k}: indices [{s}:{e}]")
            audio_segs.append(audio[s:e])
            text_segs.append(text[s:e])
            masks.append(torch.ones(seg_len, dtype=torch.bool))
            labels.append(y)
            pids.append(pid)

        # Tail handling
        rem = T - n_full * seg_len
        if rem > 0:
            if rem >= min_keep_remainder:
                if debug:
                    print(f"  [DEBUG] Keeping tail (len={rem}), padding to {seg_len}")
                a_tail = torch.zeros(seg_len, d_audio, dtype=audio.dtype)
                t_tail = torch.zeros(seg_len, d_text,  dtype=text.dtype)
                a_tail[:rem] = audio[-rem:]
                t_tail[:rem] = text[-rem:]
                m_tail = torch.zeros(seg_len, dtype=torch.bool)
                m_tail[:rem] = True

                audio_segs.append(a_tail)
                text_segs.append(t_tail)
                masks.append(m_tail)
                labels.append(y)
                pids.append(pid)
            else:
                if debug:
                    print(f"  [DEBUG] Discarding tail (len={rem} < {min_keep_remainder})")

    if len(audio_segs) == 0:
        raise RuntimeError(
            f"No segments produced. Consider lowering seg_len={seg_len} "
            f"or increasing pad_threshold={pad_threshold}."
        )

    audio_out = torch.stack(audio_segs, dim=0)  # (N_total, seg_len, d_a)
    text_out  = torch.stack(text_segs,  dim=0)  # (N_total, seg_len, d_t)
    mask_out  = torch.stack(masks,      dim=0)  # (N_total, seg_len)
    labels_out = torch.tensor(labels, dtype=torch.long)

    if debug:
        print(f"\n[DEBUG] Final output shapes:")
        print(f"  audio_out: {tuple(audio_out.shape)}")
        print(f"  text_out : {tuple(text_out.shape)}")
        print(f"  mask_out : {tuple(mask_out.shape)}")
        print(f"  labels_out: {tuple(labels_out.shape)}")
        print(f"  Total segments: {len(pids)}")

    return audio_out, text_out, mask_out, labels_out, pids

# ---------- Factory ----------
def make_loader(
    root_dir: str,
    split: str,
    labels_dir: Optional[str] = None,
    batch_size: int = 8,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    shuffle defaults to True for train, False otherwise.
    """
    if shuffle is None:
        shuffle = split == "train"
    ds = DAICEmbeddingsDataset(root_dir=root_dir, split=split, labels_dir=labels_dir)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_mffnet,
        #collate_fn=partial(collate_mffnet_fragmented, seg_len=64, pad_threshold=0.2),
        drop_last=False,
    )
    return ds, dl
