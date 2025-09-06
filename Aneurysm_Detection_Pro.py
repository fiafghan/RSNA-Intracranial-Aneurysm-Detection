"""
RSNA 2025 Intracranial Aneurysm Detection — Kaggle Notebook (No Evaluation API)

Overview
- Single-notebook baseline designed for clarity, reproducibility, and extensibility.
- Trains a 2.5D CNN on train.csv and evaluates via patient-series cross-validation.
- Implements the weighted column-wise AUROC consistent with competition description.

Compliance and Constraints
- Internet disabled. No external downloads. Uses CPU/GPU within Kaggle limits.
- No evaluation API used. Therefore, no hidden test predictions are generated here.
- A stub submission.csv header is produced for completeness; it does not contain test rows.

Usage
- Paste into a Kaggle Notebook (GPU recommended). Ensure the input dataset is attached.
- Start with a small limit_series and epochs, then scale.

Author: Your Name
"""

# =====================
# Imports & Utilities
# =====================
import os
import math
import random
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
from torch.cuda.amp import autocast, GradScaler

import torchvision.models as tvm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Optional imports for real DICOM loading (commented if not needed)
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
    # Paths (for Kaggle, dataset usually mounted under /kaggle/input/rsna-intracranial-aneurysm-detection)
    data_root: str = "/kaggle/input/rsna-intracranial-aneurysm-detection"
    train_csv: str = os.path.join(data_root, "train.csv")
    series_dir: str = os.path.join(data_root, "series")

    # Runtime
    seed: int = 3407
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    dry_run: bool = False  # set True to use synthetic data
    limit_series: int | None = 200  # None for all; keep small for quick iterations

    # Data
    img_size: Tuple[int, int] = (224, 224)
    in_channels: int = 3  # 2.5D (3 slices)

    # Labels order (13 locations + Aneurysm Present)
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

    # Training
    folds: int = 5
    epochs: int = 1  # increase for real training
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    grad_accum: int = 1
    grad_clip: float | None = 1.0


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
        aucs = []
        weights = []
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
        weighted = float(np.sum(aucs) / np.sum(weights))
        per_label["weighted_auc"] = weighted
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
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.series_dir = series_dir
        self.label_cols = label_cols
        self.img_size = img_size
        self.in_channels = in_channels
        self.dry_run = dry_run

    def __len__(self) -> int:
        return len(self.df)

    def _load_series_tensor(self, series_uid: str) -> torch.Tensor:
        if self.dry_run or pydicom is None:
            # Synthetic 2.5D tensor
            H, W = self.img_size
            return torch.randn(self.in_channels, H, W, dtype=torch.float32)
        # Real: read DICOM slices from folder series/<SeriesInstanceUID>/*.dcm
        folder = os.path.join(self.series_dir, series_uid)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.dcm')]
        if not files:
            H, W = self.img_size
            return torch.zeros(self.in_channels, H, W, dtype=torch.float32)
        # Read and sort by ImagePositionPatient (z) if available, else InstanceNumber
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
        # Pick central 3 slices (2.5D)
        idxs = [len(files)//2 - 1, len(files)//2, len(files)//2 + 1]
        idxs = [max(0, min(len(files)-1, i)) for i in idxs]
        imgs = []
        for i in idxs[: self.in_channels]:
            try:
                ds = pydicom.dcmread(files[i], force=True)
                arr = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
                inter = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
                arr = arr * slope + inter
                # Invert MONOCHROME1 if present
                if getattr(ds, 'PhotometricInterpretation', '').upper() == 'MONOCHROME1':
                    arr = arr.max() - arr
                # modality-aware simple normalization
                # CTA tends to benefit from HU windowing; MR from percentile scaling
                mod = str(getattr(ds, 'Modality', ''))
                if mod == 'CT' or mod == 'CTA':
                    # Vessel-like window (rough): center ~ 300, width ~ 700
                    wc, ww = 300.0, 700.0
                    lo, hi = wc - ww/2.0, wc + ww/2.0
                    arr = (arr - lo) / (hi - lo + 1e-6)
                else:
                    p1, p99 = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
                    arr = (arr - p1) / (p99 - p1 + 1e-6)
                arr = np.clip(arr, 0.0, 1.0)
            except Exception:
                arr = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.float32)
            t = torch.from_numpy(arr)[None, ...]  # (1,H,W)
            t = F.interpolate(t[None, ...], size=self.img_size, mode="bilinear", align_corners=False)[0]  # (1,H,W)
            imgs.append(t)
        if len(imgs) < self.in_channels:
            # pad channels with zeros
            H, W = self.img_size
            for _ in range(self.in_channels - len(imgs)):
                imgs.append(torch.zeros(1, H, W, dtype=torch.float32))
        x = torch.cat(imgs, dim=0)  # (C,H,W)
        return x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        series_uid = row["SeriesInstanceUID"]
        x = self._load_series_tensor(series_uid)
        y = torch.from_numpy(row[self.label_cols].values.astype(np.float32))
        return {"image": x, "target": y, "series_uid": series_uid, "modality": row.get("Modality", "")}


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
        # adapt first conv
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
@dataclass
class TrainParams:
    epochs: int
    amp: bool
    grad_accum: int


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler: GradScaler, amp: bool) -> float:
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        x = batch["image"].to(device)
        y = batch["target"].to(device)
        with autocast(enabled=amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        if CFG.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        if step % CFG.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running += loss.item()
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    preds, targs = [], []
    for batch in loader:
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
        # synthetic small df for dry_run
        n = 100
        data = {"SeriesInstanceUID": [f"dry_{i:05d}" for i in range(n)], "Modality": ["CTA"] * n}
        rng = np.random.default_rng(CFG.seed)
        for col in ALL_LABELS:
            data[col] = rng.binomial(1, 0.2, size=n)
        return pd.DataFrame(data)
    df = pd.read_csv(CFG.train_csv)
    # Ensure columns exist
    missing = [c for c in ALL_LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in train.csv: {missing}")
    return df


def make_folds(df: pd.DataFrame, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = df["Aneurysm Present"].astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.seed)
    splits = list(skf.split(df, y))
    return splits


# =================================
# Optional submission file builder
# =================================
SUBMISSION_COLUMNS = ALL_LABELS


def build_submission_stub(save_path: str) -> None:
    # Creates a submission.csv header without rows (no test set available without API)
    pd.DataFrame(columns=SUBMISSION_COLUMNS).to_csv(save_path, index=False)


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

        train_ds = RSNAAneurysmSeries(trn_df, CFG.series_dir, ALL_LABELS, CFG.img_size, CFG.in_channels, dry_run=CFG.dry_run)
        val_ds = RSNAAneurysmSeries(val_df, CFG.series_dir, ALL_LABELS, CFG.img_size, CFG.in_channels, dry_run=CFG.dry_run)

        train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

        model = ResNet2_5D(backbone="resnet18", in_channels=CFG.in_channels, num_classes=len(ALL_LABELS), pretrained=False).to(CFG.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        scaler = GradScaler(enabled=CFG.amp)

        for epoch in range(1, CFG.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, CFG.device, scaler, CFG.amp)
            val_metrics = evaluate(model, val_loader, CFG.device)
            print(f"fold {fold+1} epoch {epoch}: train_loss {train_loss:.4f} | val_weighted_auc {val_metrics['weighted_auc']:.4f}")

        # collect OOF predictions for validation summary
        model.eval()
        preds = []
        for batch in val_loader:
            x = batch["image"].to(CFG.device)
            with torch.no_grad():
                p = torch.sigmoid(model(x)).cpu().numpy()
            preds.append(p)
        preds = np.concatenate(preds, axis=0)
        oof_preds[val_idx] = preds

        # Save checkpoint per fold
        os.makedirs("/kaggle/working", exist_ok=True)
        ckpt_path = f"/kaggle/working/model_fold{fold+1}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "cfg": CFG.__dict__,
            "fold": fold+1,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # memory cleanup per fold
        del model, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        torch.cuda.empty_cache()

    # OOF metrics
    metric = WeightedColumnwiseAUC(present_index=CFG.present_index, num_classes=len(ALL_LABELS))
    oof_res = metric(oof_targs, oof_preds)
    print("OOF metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in oof_res.items() if k == 'weighted_auc'})
    # Save OOF predictions for audit
    oof_df = pd.DataFrame(oof_preds, columns=ALL_LABELS)
    oof_df.insert(0, "SeriesInstanceUID", df["SeriesInstanceUID"].values)
    oof_path = "/kaggle/working/oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions to {oof_path}")

    # Create a stub submission file (no rows, only header) to satisfy local checks if desired
    save_path = os.path.join(".", "submission.csv")
    build_submission_stub(save_path)
    elapsed = time.time() - start_time
    print(f"Wrote stub submission at {save_path} (no rows; test served only via evaluation API).")
    print(f"Total elapsed: {elapsed/60.0:.1f} min")
