"""
RSNA 2025 Intracranial Aneurysm Detection — Kaggle Notebook (No Evaluation API)

Overview
- Single-notebook baseline designed for clarity, reproducibility, and extensibility.
- Trains a 2.5D CNN on train.csv and evaluates via stratified patient-series cross-validation.
- Implements the weighted column-wise AUROC consistent with competition description.

Constraints
- Internet disabled. No external downloads. Runs within Kaggle time limits.
- No evaluation API here (cannot score hidden test). Produces OOF metrics and checkpoints.

Usage
- Paste into a Kaggle GPU Notebook. Attach the dataset: rsna-intracranial-aneurysm-detection.
- Start with CFG.limit_series small and CFG.epochs low, then scale.

Author: Your Name
"""

# =====================
# Imports & Utilities
# =====================
import os
import gc
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.models as tvm
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import pydicom
except Exception:
    pydicom = None


def report_environment():
    print("Environment report:")
    print(f"  torch {torch.__version__}")
    try:
        import torchvision as _tv
        print(f"  torchvision {_tv.__version__}")
    except Exception:
        pass
    print(f"  device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"  cuda name: {torch.cuda.get_device_name(0)}")
    print(f"  pydicom: {'available' if pydicom is not None else 'not available'}")


# ===============
# Configuration
# ===============
@dataclass
class CFG:
    data_root: str = "/kaggle/input/rsna-intracranial-aneurysm-detection"
    train_csv: str = os.path.join(data_root, "train.csv")
    series_dir: str = os.path.join(data_root, "series")

    seed: int = 3407
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    dry_run: bool = False
    limit_series: int | None = None

    img_size: Tuple[int, int] = (384, 384)
    in_channels: int = 3
    backbone: str = "resnet50"
    pretrained: bool = False

    label_order: List[str] = field(
        default_factory=lambda: [
            "Left Infraclinoid Internal Carotid Artery",
            "Right Infraclinoid Internal Carotid Artery",
            "Left Supraclinoid Internal Carotid Artery",
            "Right Supraclinoid Internal Carotid Artery",
            "Left Middle Cerebral Artery",
            "Right Middle Cerebral Artery",
            "Anterior Communicating Artery",
            "Left Anterior Cerebral Artery",
            "Right Anterior Cerebral Artery",
            "Left Posterior Communicating Artery",
            "Right Posterior Communicating Artery",
            "Basilar Tip",
            "Other Posterior Circulation",
            "Aneurysm Present",
        ]
    )
    present_index: int = 13

    folds: int = 5
    epochs: int = 30
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 5e-5
    amp: bool = True
    grad_accum: int = 4
    grad_clip: float | None = 1.0

    warmup_epochs: int = 3
    ema_decay: float = 0.9995
    early_stop_patience: int = 5
    train_augment: bool = True


CFG = CFG()


def seed_everything(seed: int = 42) -> None:
    import os as _os, random as _random
    import numpy as _np
    _os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.seed)
report_environment()


def _worker_init_fn(worker_id: int):
    s = CFG.seed + worker_id
    import random as _random
    import numpy as _np
    _random.seed(s)
    _np.random.seed(s)


def _standardize_channels(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def _train_augment(x: torch.Tensor) -> torch.Tensor:
    if not CFG.train_augment:
        return x
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(2,))
    contrast = 1.0 + (torch.rand(()) - 0.5) * 0.1
    brightness = (torch.rand(()) - 0.5) * 0.05
    x = (x * contrast + brightness).clamp(0.0, 1.0)
    gamma = 1.0 + (torch.rand(()) - 0.5) * 0.1
    x = x.clamp(0.0, 1.0) ** gamma
    return x


# ======================
# Metric (weighted AUC)
# ======================
class WeightedColumnwiseAUC:
    def __init__(self, present_index: int, num_classes: int, present_weight: float = 13.0, other_weight: float = 1.0):
        self.present_index = present_index
        self.num_classes = num_classes
        self.present_weight = present_weight
        self.other_weight = other_weight

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        assert y_true.shape == y_pred.shape
        assert y_true.shape[1] == self.num_classes
        aucs, weights = [], []
        per_label = {}
        for i in range(self.num_classes):
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            except ValueError:
                auc = 0.5
            per_label[f"auc_{i}"] = float(auc)
            w = self.present_weight if i == self.present_index else self.other_weight
            aucs.append(auc * w)
            weights.append(w)
        per_label["weighted_auc"] = float(np.sum(aucs) / np.sum(weights))
        return per_label


# =====================
# Data: Loader/Reader
# =====================
class RSNAAneurysmSeries(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        series_dir: str,
        label_cols: List[str],
        img_size: Tuple[int, int] = (224, 224),
        in_channels: int = 3,
        dry_run: bool = False,
        is_train: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.series_dir = series_dir
        self.label_cols = label_cols
        self.img_size = img_size
        self.in_channels = in_channels
        self.dry_run = dry_run
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _load_series_tensor(self, series_uid: str) -> torch.Tensor:
        H, W = self.img_size
        if self.dry_run or pydicom is None:
            x = torch.rand(self.in_channels, H, W, dtype=torch.float32)  # [0,1]
            x = _standardize_channels(x)
            return x
        folder = os.path.join(self.series_dir, series_uid)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.dcm')]
        if not files:
            return torch.zeros(self.in_channels, H, W, dtype=torch.float32)

        def _sort_key(path):
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                ipp = getattr(ds, 'ImagePositionPatient', None)
                if ipp is not None and len(ipp) == 3:
                    return float(ipp[2])
                return int(getattr(ds, 'InstanceNumber', 0))
            except Exception:
                return 0

        files.sort(key=_sort_key)
        idxs = [len(files)//2 - 1, len(files)//2, len(files)//2 + 1]
        idxs = [max(0, min(len(files)-1, i)) for i in idxs]
        imgs: List[torch.Tensor] = []
        for i in idxs[: self.in_channels]:
            try:
                ds = pydicom.dcmread(files[i], force=True)
                arr = ds.pixel_array.astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[arr.shape[0] // 2]
                elif arr.ndim != 2:
                    raise ValueError("Unexpected DICOM pixel_array dims")
                slope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
                inter = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
                arr = arr * slope + inter
                if getattr(ds, 'PhotometricInterpretation', '').upper() == 'MONOCHROME1':
                    arr = arr.max() - arr
                mod = str(getattr(ds, 'Modality', ''))
                if mod in ('CT', 'CTA'):
                    wc, ww = 300.0, 700.0
                    lo, hi = wc - ww/2.0, wc + ww/2.0
                    arr = (arr - lo) / (hi - lo + 1e-6)
                else:
                    p1, p99 = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
                    arr = (arr - p1) / (p99 - p1 + 1e-6)
                arr = np.clip(arr, 0.0, 1.0)
            except Exception:
                arr = np.zeros((H, W), dtype=np.float32)
            t = torch.from_numpy(arr)[None, ...]
            t = F.interpolate(t[None, ...], size=(H, W), mode="bilinear", align_corners=False)[0]
            imgs.append(t)
        while len(imgs) < self.in_channels:
            imgs.append(imgs[-1].clone())
        x = torch.cat(imgs[: self.in_channels], dim=0)
        if self.is_train and not self.dry_run:
            x = _train_augment(x)
        x = _standardize_channels(x)
        return x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        x = self._load_series_tensor(row["SeriesInstanceUID"])
        y = torch.from_numpy(row[self.label_cols].values.astype(np.float32))
        return {"image": x, "target": y, "series_uid": row["SeriesInstanceUID"], "modality": row.get("Modality", "")}


class UnlabeledSeries(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        series_dir: str,
        img_size: Tuple[int, int] = (224, 224),
        in_channels: int = 3,
        dry_run: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.series_dir = series_dir
        self.img_size = img_size
        self.in_channels = in_channels
        self.dry_run = dry_run

    def __len__(self) -> int:
        return len(self.df)

    def _load_series_tensor(self, series_uid: str) -> torch.Tensor:
        H, W = self.img_size
        if self.dry_run or pydicom is None:
            x = torch.rand(self.in_channels, H, W, dtype=torch.float32)
            x = _standardize_channels(x)
            return x
        folder = os.path.join(self.series_dir, series_uid)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.dcm')]
        if not files:
            return torch.zeros(self.in_channels, H, W, dtype=torch.float32)
        def _sort_key(path):
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                ipp = getattr(ds, 'ImagePositionPatient', None)
                if ipp is not None and len(ipp) == 3:
                    return float(ipp[2])
                return int(getattr(ds, 'InstanceNumber', 0))
            except Exception:
                return 0
        files.sort(key=_sort_key)
        idxs = [len(files)//2 - 1, len(files)//2, len(files)//2 + 1]
        idxs = [max(0, min(len(files)-1, i)) for i in idxs]
        imgs: List[torch.Tensor] = []
        for i in idxs[: self.in_channels]:
            try:
                ds = pydicom.dcmread(files[i], force=True)
                arr = ds.pixel_array.astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[arr.shape[0] // 2]
                elif arr.ndim != 2:
                    raise ValueError("Unexpected DICOM pixel_array dims")
                slope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
                inter = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
                arr = arr * slope + inter
                if getattr(ds, 'PhotometricInterpretation', '').upper() == 'MONOCHROME1':
                    arr = arr.max() - arr
                mod = str(getattr(ds, 'Modality', ''))
                if mod in ('CT', 'CTA'):
                    wc, ww = 300.0, 700.0
                    lo, hi = wc - ww/2.0, wc + ww/2.0
                    arr = (arr - lo) / (hi - lo + 1e-6)
                else:
                    p1, p99 = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
                    arr = (arr - p1) / (p99 - p1 + 1e-6)
                arr = np.clip(arr, 0.0, 1.0)
            except Exception:
                arr = np.zeros((H, W), dtype=np.float32)
            t = torch.from_numpy(arr)[None, ...]
            t = F.interpolate(t[None, ...], size=(H, W), mode="bilinear", align_corners=False)[0]
            imgs.append(t)
        while len(imgs) < self.in_channels:
            imgs.append(imgs[-1].clone())
        x = torch.cat(imgs[: self.in_channels], dim=0)
        x = _standardize_channels(x)
        return x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        x = self._load_series_tensor(row["SeriesInstanceUID"]) 
        return {"image": x, "series_uid": row["SeriesInstanceUID"]}


# =============
# Model: 2.5D
# =============
class ResNet2_5D(nn.Module):
    def __init__(self, backbone: str = "resnet18", in_channels: int = 3, num_classes: int = 14, pretrained: bool = False):
        super().__init__()
        if backbone == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet34":
            net = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        if in_channels != 3:
            w = net.conv1.weight
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if w.shape[1] == 3 and in_channels > 0:
                with torch.no_grad():
                    net.conv1.weight[:, : min(3, in_channels)] = w[:, : min(3, in_channels)]
                    if in_channels > 3:
                        mean_w = w.mean(dim=1, keepdim=True)
                        net.conv1.weight[:, 3:in_channels] = mean_w.repeat(1, in_channels - 3, 1, 1)
        self.backbone = nn.Sequential(*(list(net.children())[:-1]))
        self.head = nn.Linear(feat_dim, len(CFG.label_order))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = feat.flatten(1)
        return self.head(feat)


# =====================
# Training/Evaluation
# =====================
class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy as _copy
        # Deepcopy the model to ensure matching state structure
        self.ema = _copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for k in ema_state.keys():
            ema_v = ema_state[k]
            v = model_state[k]
            if isinstance(ema_v, torch.Tensor):
                if ema_v.dtype.is_floating_point:
                    ema_v.mul_(self.decay).add_(v.data, alpha=1.0 - self.decay)
                else:
                    ema_v.copy_(v.data)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.ema.state_dict(), strict=False)


class EarlyStopping:
    def __init__(self, patience: int = 3):
        self.best = -float('inf')
        self.patience = patience
        self.count = 0
    def step(self, score: float) -> bool:
        if score > self.best + 1e-8:
            self.best = score
            self.count = 0
            return False
        self.count += 1
        return self.count > self.patience


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, amp: bool) -> float:
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), 1):
        x = batch["image"].to(device)
        y = batch["target"].to(device)
        try:
            with torch.amp.autocast(device_type=('cuda' if torch.cuda.is_available() else 'cpu'), enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
        except Exception:
            from torch.cuda.amp import autocast as cuda_autocast
            with cuda_autocast(enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
        # average loss across accumulation steps
        loss_to_backprop = loss / max(1, CFG.grad_accum)
        scaler.scale(loss_to_backprop).backward()
        if step % CFG.grad_accum == 0:
            if CFG.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running += loss.item()
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    preds, targs = [], []
    for batch in tqdm(loader, desc="valid", leave=False):
        x = batch["image"].to(device)
        y = batch["target"].to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds.append(probs.cpu())
        targs.append(y.cpu())
    pred = torch.cat(preds).numpy()
    targ = torch.cat(targs).numpy()
    metric = WeightedColumnwiseAUC(present_index=CFG.present_index, num_classes=len(CFG.label_order))
    return metric(targ, pred)


# ============================
# DataFrame and CV utilities
# ============================
ALL_LABELS = CFG.label_order


def load_train_df() -> pd.DataFrame:
    if not os.path.exists(CFG.train_csv):
        n = 100
        data = {"SeriesInstanceUID": [f"dry_{i:05d}" for i in range(n)], "Modality": ["CTA"] * n}
        rng = np.random.default_rng(CFG.seed)
        for col in ALL_LABELS:
            data[col] = rng.binomial(1, 0.2, size=n)
        return pd.DataFrame(data)
    df = pd.read_csv(CFG.train_csv)
    missing = [c for c in ALL_LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in train.csv: {missing}")
    return df


def make_folds(df: pd.DataFrame, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = df["Aneurysm Present"].astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.seed)
    return list(skf.split(df, y))


# =================================
# Optional submission file builder
# =================================
SUBMISSION_COLUMNS = ALL_LABELS


def build_submission_stub(save_path: str) -> None:
    pd.DataFrame(columns=SUBMISSION_COLUMNS).to_csv(save_path, index=False)


def get_submission_series_ids(train_df: pd.DataFrame) -> List[str]:
    """Resolve test SeriesInstanceUIDs from dataset artifacts with sensible fallbacks."""
    # Prefer sample_submission.csv
    ss_path = os.path.join(CFG.data_root, "sample_submission.csv")
    if os.path.exists(ss_path):
        try:
            ss = pd.read_csv(ss_path)
            col = "SeriesInstanceUID" if "SeriesInstanceUID" in ss.columns else ss.columns[0]
            ids = ss[col].astype(str).dropna().tolist()
            if len(ids) > 0:
                return ids
        except Exception:
            pass
    # Try explicit test.csv
    test_csv = os.path.join(CFG.data_root, "test.csv")
    if os.path.exists(test_csv):
        try:
            tdf = pd.read_csv(test_csv)
            col = "SeriesInstanceUID" if "SeriesInstanceUID" in tdf.columns else tdf.columns[0]
            ids = tdf[col].astype(str).dropna().tolist()
            if len(ids) > 0:
                return ids
        except Exception:
            pass
    # Try listing series_dir and excluding known train IDs
    if os.path.isdir(CFG.series_dir):
        try:
            all_ids = [d for d in os.listdir(CFG.series_dir) if os.path.isdir(os.path.join(CFG.series_dir, d))]
            train_ids = set(train_df["SeriesInstanceUID"].astype(str).tolist())
            cand = [i for i in all_ids if i not in train_ids]
            if len(cand) > 0:
                return cand
        except Exception:
            pass
    # Fallback: use train IDs as mock (not valid for Kaggle scoring, but useful to ensure pipeline works)
    return train_df["SeriesInstanceUID"].astype(str).tolist()


@torch.no_grad()
def predict_submission(ids: List[str]) -> pd.DataFrame:
    """Run inference over provided IDs, averaging predictions across available fold checkpoints."""
    df_ids = pd.DataFrame({"SeriesInstanceUID": ids})
    ds = UnlabeledSeries(df_ids, CFG.series_dir, CFG.img_size, CFG.in_channels, dry_run=CFG.dry_run)
    loader = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    fold_preds: List[np.ndarray] = []
    for fold in range(CFG.folds):
        model = ResNet2_5D(backbone=CFG.backbone, in_channels=CFG.in_channels, num_classes=len(ALL_LABELS), pretrained=CFG.pretrained).to(CFG.device)
        ckpt_best = f"/kaggle/working/model_fold{fold+1}_best.pt"
        loaded = False
        if os.path.exists(ckpt_best):
            try:
                state = torch.load(ckpt_best, map_location=CFG.device)
                model.load_state_dict(state.get("model_state", state), strict=False)
                loaded = True
            except Exception:
                pass
        if not loaded:
            # Try last checkpoint as a fallback
            ckpt_last = f"/kaggle/working/model_fold{fold+1}_last.pt"
            if os.path.exists(ckpt_last):
                try:
                    state = torch.load(ckpt_last, map_location=CFG.device)
                    model.load_state_dict(state.get("model_state", state), strict=False)
                    loaded = True
                except Exception:
                    pass
        model.eval()
        preds: List[np.ndarray] = []
        for batch in loader:
            x = batch["image"].to(CFG.device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            preds.append(p)
        fold_pred = np.concatenate(preds, axis=0)
        fold_preds.append(fold_pred)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if len(fold_preds) == 0:
        pred = np.zeros((len(ids), len(ALL_LABELS)), dtype=np.float32)
    else:
        pred = np.mean(np.stack(fold_preds, axis=0), axis=0)
    sub_df = pd.DataFrame(pred, columns=ALL_LABELS)
    sub_df.insert(0, "SeriesInstanceUID", ids)
    return sub_df


# ======
# Main
# ======
if __name__ == "__main__":
    start_time = time.time()
    df = load_train_df()
    if CFG.limit_series is not None and len(df) > CFG.limit_series:
        df = df.sample(CFG.limit_series, random_state=CFG.seed).reset_index(drop=True)

    folds = make_folds(df, CFG.folds)
    oof_preds = np.zeros((len(df), len(ALL_LABELS)), dtype=np.float32)
    oof_targs = df[ALL_LABELS].values.astype(np.float32)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        print(f"Fold {fold+1}/{CFG.folds}: train {len(trn_idx)} | val {len(val_idx)}")
        trn_df = df.iloc[trn_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = RSNAAneurysmSeries(trn_df, CFG.series_dir, ALL_LABELS, CFG.img_size, CFG.in_channels, dry_run=CFG.dry_run, is_train=True)
        val_ds = RSNAAneurysmSeries(val_df, CFG.series_dir, ALL_LABELS, CFG.img_size, CFG.in_channels, dry_run=CFG.dry_run, is_train=False)

        generator = torch.Generator().manual_seed(CFG.seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            generator=generator,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
        )

        model = ResNet2_5D(backbone=CFG.backbone, in_channels=CFG.in_channels, num_classes=len(ALL_LABELS), pretrained=CFG.pretrained).to(CFG.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        try:
            scaler = torch.amp.GradScaler(enabled=CFG.amp)
        except Exception:
            from torch.cuda.amp import GradScaler as CudaGradScaler
            scaler = CudaGradScaler(enabled=CFG.amp)

        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
        warmup_epochs = max(0, int(CFG.warmup_epochs))
        total_cosine_epochs = max(1, CFG.epochs - warmup_epochs)
        warmup = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / max(1, warmup_epochs)) if warmup_epochs > 0 else 1.0)
        cosine = CosineAnnealingLR(optimizer, T_max=total_cosine_epochs)

        ema = ModelEma(model, decay=CFG.ema_decay)
        stopper = EarlyStopping(patience=CFG.early_stop_patience)
        best_auc = -1.0

        for epoch in range(1, CFG.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, CFG.device, scaler, CFG.amp)
            ema.update(model)
            ema.copy_to(model)
            val_metrics = evaluate(model, val_loader, CFG.device)
            cur_auc = val_metrics['weighted_auc']
            print(f"fold {fold+1} epoch {epoch}: train_loss {train_loss:.4f} | val_weighted_auc {cur_auc:.4f}")
            # Save best
            os.makedirs("/kaggle/working", exist_ok=True)
            ckpt_best = f"/kaggle/working/model_fold{fold+1}_best.pt"
            if cur_auc > best_auc:
                best_auc = cur_auc
                torch.save({
                    "model_state": model.state_dict(),
                    "cfg": CFG.__dict__,
                    "fold": fold+1,
                    "metric": cur_auc,
                }, ckpt_best)
            # Scheduler step
            if epoch <= warmup_epochs:
                warmup.step()
            else:
                cosine.step()
            # Early stopping
            if stopper.step(cur_auc):
                print(f"Early stopping at epoch {epoch} (no improvement for {CFG.early_stop_patience} epochs)")
                break

        # OOF preds
        model.eval()
        preds = []
        for batch in val_loader:
            x = batch["image"].to(CFG.device)
            with torch.no_grad():
                p = torch.sigmoid(model(x)).cpu().numpy()
            preds.append(p)
        preds = np.concatenate(preds, axis=0)
        oof_preds[val_idx] = preds

        ckpt_last = f"/kaggle/working/model_fold{fold+1}_last.pt"
        torch.save({"model_state": model.state_dict(), "cfg": CFG.__dict__, "fold": fold+1}, ckpt_last)
        print(f"Saved checkpoint: {ckpt_last}")

        del model, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        torch.cuda.empty_cache()

    metric = WeightedColumnwiseAUC(present_index=CFG.present_index, num_classes=len(ALL_LABELS))
    oof_res = metric(oof_targs, oof_preds)
    print("OOF metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in oof_res.items() if k == 'weighted_auc'})

    oof_df = pd.DataFrame(oof_preds, columns=ALL_LABELS)
    oof_df.insert(0, "SeriesInstanceUID", df["SeriesInstanceUID"].values)
    oof_path = "/kaggle/working/oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions to {oof_path}")

    # Build a REAL submission by running inference on test IDs
    ids = get_submission_series_ids(df)
    sub_df = predict_submission(ids)
    save_path = os.path.join(".", "submission.csv")
    sub_df.to_csv(save_path, index=False)
    elapsed = time.time() - start_time
    print(f"Wrote submission with {len(sub_df)} rows to {save_path}.")
    print(f"Total elapsed: {elapsed/60.0:.1f} min")
