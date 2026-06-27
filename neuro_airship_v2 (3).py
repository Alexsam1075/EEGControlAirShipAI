"""
neuro_airship_v2.py
===================
Единый скрипт управления нейродирижаблем-акулой:

  1. Симуляция (без гарнитуры) — 2D поле, отработка клавиш, тест ИИ
  2. Подключение к нейрогарнитуре — кнопка «Найти / Подключить»
  3. Улучшенный ИИ — MLP с Adam, Dropout, class-balancing, ансамблевое
     голосование нескольких скользящих окон
  4. Подключение акулы BLE — кнопка «Подключить акулу»
  5. Плавное отключение акулы — heartbeat + защита от резкого обрыва
  6. Гарнитура управляет акулой: двойное моргание = пауза/возобновление
  7. Профили сохраняются/загружаются автоматически

Зависимости: numpy, bleak, CapsuleSDK (уже в проекте)
"""
from __future__ import annotations

import json
import os
import platform
import queue
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Optional
import tkinter as tk

import numpy as np

# ─── SDK imports (graceful fallback to simulation) ───────────────────────────
try:
    from CapsuleSDK.Capsule import Capsule
    from CapsuleSDK.Device import Device
    from CapsuleSDK.DeviceLocator import DeviceLocator
    from CapsuleSDK.DeviceType import DeviceType
    from CapsuleSDK.EEGTimedData import EEGTimedData
    from CapsuleSDK.MEMS import MEMS, MEMSTimedData
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

import pc_manual_control  # BLE shark bridge (всегда рядом)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
APP_DIR           = Path(__file__).resolve().parent
PROFILE_ROOT      = APP_DIR / "mini_ai_data"
PROFILE_DEFAULT   = "shark_profile"

SAMPLE_RATE       = 250.0
EEG_WIN_SEC       = 2.5       # размер окна для извлечения признаков
CTRL_WIN_SEC      = 0.8       # размер скользящего окна управления
CTRL_TICK_MS      = 120       # мс между тиками управления акулой

SEARCH_TIMEOUT    = 16
CONNECT_TIMEOUT   = 20
DEVICE_CONNECTED  = 1
ANY_SEARCH_EXTRA_SEC = 8.0
SEARCH_SETTLE_SEC = 2.0

LABELS = ("idle", "forward", "back", "left", "right")
DIRECTION_LABELS = ("forward", "back", "left", "right")
MEMORY_PREDICT_MIN_SAMPLES = 8
MEMORY_PREDICT_MAX_SAMPLES = 240
MEMORY_PREDICT_TOP_K = 9
TRAIN_RECENT_MAX_SAMPLES = 400

BUTTON_TAGS = {
    "forward": pc_manual_control.BUTTON_TAGS["forward"],
    "back":    pc_manual_control.BUTTON_TAGS["back"],
    "left":    pc_manual_control.BUTTON_TAGS["left"],
    "right":   pc_manual_control.BUTTON_TAGS["right"],
}
COMMAND_TO_TAG = {
    "idle":    None,
    "forward": BUTTON_TAGS["forward"],
    "back":    BUTTON_TAGS["back"],
    "left":    BUTTON_TAGS["left"],
    "right":   BUTTON_TAGS["right"],
}

KEY_TO_LABEL = {
    "w": "forward", "up": "forward",
    "s": "back",    "down": "back",
    "a": "left",    "left": "left",
    "d": "right",   "right": "right",
    "space": "idle",
}
ACTION_TO_ACCEL = {
    "idle":    np.array([0.0,  0.0], dtype=np.float32),
    "forward": np.array([0.0, -1.0], dtype=np.float32),
    "back":    np.array([0.0,  1.0], dtype=np.float32),
    "left":    np.array([-1.0, 0.0], dtype=np.float32),
    "right":   np.array([1.0,  0.0], dtype=np.float32),
}

# Параметры обнаружения двойного моргания
BLINK_WIN_SEC        = 1.2
BLINK_THRESH_Z       = 8.0
BLINK_PEAK_Z         = 10.0
BLINK_MIN_GAP_SEC    = 0.08
BLINK_MAX_GAP_SEC    = 0.55
BLINK_MIN_WIDTH_SEC  = 0.02
BLINK_MAX_WIDTH_SEC  = 0.18
BLINK_MIN_PEAK_RATIO = 2.5
BLINK_COOLDOWN_SEC   = 1.2

# Параметры BLE keep-alive
BLE_RESEND_SEC  = 0.35        # переотправка команды если акула не меняет движение
BLE_MIN_HOLD    = 0.18        # минимальное время удержания кнопки
SHARK_HEARTBEAT = 0.5         # интервал проверки соединения акулы

DEVICE_SEARCH_TYPES = ()
if _SDK_AVAILABLE:
    DEVICE_SEARCH_TYPES = (
        DeviceType.Any, DeviceType.Band, DeviceType.Buds,
        DeviceType.BrainBit, DeviceType.Headphones, DeviceType.Impulse,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_int(v: str, default: int) -> int:
    try: return int(v)
    except Exception: return default


def _safe_float(v: str, default: float) -> float:
    try: return float(v)
    except Exception: return default


def normalize_serial(value) -> str:
    return re.sub(r"\D+", "", str(value or "")).strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  EEG / MEMS BUFFERS
# ═══════════════════════════════════════════════════════════════════════════════
class EEGBuffer:
    def __init__(self, n_channels: int, max_samples: int):
        self.n_channels  = int(n_channels)
        self.max_samples = int(max_samples)
        self.samples: deque[np.ndarray] = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()

    def append_block(self, block: np.ndarray):
        if block.size == 0:
            return
        with self.lock:
            for i in range(block.shape[1]):
                self.samples.append(block[:, i].astype(np.float32, copy=False))

    def get_last(self, n: int) -> np.ndarray:
        with self.lock:
            if not self.samples:
                return np.zeros((self.n_channels, 0), dtype=np.float32)
            n = min(int(n), len(self.samples))
            arr = np.asarray(list(self.samples)[-n:], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr.T.copy()


class MEMSBuffer:
    def __init__(self, max_samples: int = 2500):
        self.max_samples = int(max_samples)
        self.gyro:  deque[np.ndarray] = deque(maxlen=self.max_samples)
        self.accel: deque[np.ndarray] = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()

    def append(self, g: np.ndarray, a: np.ndarray):
        with self.lock:
            self.gyro.append(g.astype(np.float32, copy=False))
            self.accel.append(a.astype(np.float32, copy=False))

    def get_last(self, n: int):
        with self.lock:
            if not self.gyro:
                empty = np.zeros((0, 3), dtype=np.float32)
                return empty, empty
            n = min(int(n), len(self.gyro))
            return (np.asarray(list(self.gyro)[-n:], dtype=np.float32),
                    np.asarray(list(self.accel)[-n:], dtype=np.float32))


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (улучшенная версия)
# ═══════════════════════════════════════════════════════════════════════════════
def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=np.float32)
    return psd[:, mask].mean(axis=1).astype(np.float32)


def compute_eeg_features(block: np.ndarray, sfreq: float) -> np.ndarray:
    """48-мерный вектор EEG признаков на одном окне."""
    if block.size == 0 or block.shape[1] < 8:
        return np.zeros(48, dtype=np.float32)
    x = block.astype(np.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)

    std      = x.std(axis=1)
    rms      = np.sqrt(np.mean(x * x, axis=1))
    mean_abs = np.mean(np.abs(x), axis=1)
    skew     = np.mean(x ** 3, axis=1) / (std ** 3 + 1e-6)

    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / sfreq)
    psd   = np.abs(np.fft.rfft(x, axis=1)) ** 2

    delta = _band_power(freqs, psd, 0.5, 4.0)
    theta = _band_power(freqs, psd, 4.0, 8.0)
    alpha = _band_power(freqs, psd, 8.0, 13.0)
    beta  = _band_power(freqs, psd, 13.0, 30.0)
    gamma = _band_power(freqs, psd, 30.0, 45.0)
    total = delta + theta + alpha + beta + gamma + 1e-6

    return np.concatenate([
        mean_abs, std, rms, skew,
        alpha / total, beta / total, theta / total, delta / total, gamma / total,
        beta / (alpha + 1e-6),   # beta/alpha ratio — маркер концентрации
    ]).astype(np.float32)


def _axis_stats(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return np.zeros(21, dtype=np.float32)
    mean     = data.mean(axis=0)
    std      = data.std(axis=0)
    mean_abs = np.mean(np.abs(data), axis=0)
    max_abs  = np.max(np.abs(data), axis=0)
    ptp      = np.ptp(data, axis=0)
    energy   = np.mean(data ** 2, axis=0)
    diff     = np.diff(data, axis=0)
    mean_d   = diff.mean(axis=0) if diff.size else np.zeros(3, dtype=np.float32)
    return np.concatenate([mean, std, mean_abs, max_abs, ptp, energy, mean_d]).astype(np.float32)


def compute_mems_features(gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
    return np.concatenate([_axis_stats(gyro), _axis_stats(accel)]).astype(np.float32)


# Многоокновая «стековая» векторизация
STACK_WINDOWS = 3
STACK_STEP_SEC = 0.18

def extract_feature_vector(eeg: np.ndarray, gyro: np.ndarray, accel: np.ndarray, sfreq: float) -> np.ndarray:
    """Строим вектор из нескольких скользящих окон для устойчивости."""
    if eeg.size == 0:
        dim = (48 + 42) * STACK_WINDOWS
        return np.zeros(dim, dtype=np.float32)

    base = max(8, int(CTRL_WIN_SEC * sfreq))
    step = max(2, int(STACK_STEP_SEC * sfreq))
    total = eeg.shape[1]

    parts: list[np.ndarray] = []
    end = total
    for _ in range(STACK_WINDOWS):
        start = max(0, end - base)
        eeg_w   = eeg[:, start:end]
        n_mems  = end - start
        g_start = max(0, gyro.shape[0] - n_mems) if gyro.ndim == 2 else 0
        gyro_w  = gyro[g_start:] if gyro.ndim == 2 else np.zeros((0, 3), dtype=np.float32)
        acc_w   = accel[g_start:] if accel.ndim == 2 else np.zeros((0, 3), dtype=np.float32)
        parts.append(compute_eeg_features(eeg_w, sfreq))
        parts.append(compute_mems_features(gyro_w, acc_w))
        if start == 0:
            parts *= STACK_WINDOWS  # дублируем если данных мало
            break
        end = max(0, end - step)

    while len(parts) < STACK_WINDOWS * 2:
        parts.extend(parts[-2:])
    return np.concatenate(parts[:STACK_WINDOWS * 2]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  УЛУЧШЕННАЯ ML-МОДЕЛЬ  (MLP с Adam + class weighting + ансамбль)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Prediction:
    label: str
    confidence: float
    probs: dict[str, float]


class NeuroMLP:
    """
    MLP обученный с:
      - Adam-оптимайзером (сходится в 3–4× быстрее SGD)
      - Label smoothing (уменьшает overfit на маленьких датасетах)
      - Class-balanced loss (не игнорировать редкие классы)
      - Dropout (при обучении зануляем 30% нейронов)
      - Ансамбль предсказаний из 3 скользящих окон при инференсе
    """

    def __init__(
        self,
        labels: list[str],
        hidden: tuple[int, ...] = (256, 128, 64),
        lr: float = 3e-4,
        epochs: int = 200,
        batch_size: int = 64,
        dropout: float = 0.30,
        label_smooth: float = 0.10,
    ):
        self.labels       = list(labels)
        self.hidden       = tuple(int(h) for h in hidden)
        self.lr           = float(lr)
        self.epochs       = int(epochs)
        self.batch_size   = int(batch_size)
        self.dropout      = float(dropout)
        self.label_smooth = float(label_smooth)

        self.mean_:    Optional[np.ndarray] = None
        self.scale_:   Optional[np.ndarray] = None
        self.weights_: list[np.ndarray]     = []
        self.biases_:  list[np.ndarray]     = []

    # ── architecture ──────────────────────────────────────────────────────────
    def _init_weights(self, in_dim: int):
        dims = [in_dim, *self.hidden, len(self.labels)]
        rng  = np.random.default_rng(7)
        self.weights_ = []
        self.biases_  = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / max(1, d_in))          # He init
            self.weights_.append(rng.standard_normal((d_in, d_out)).astype(np.float32) * scale)
            self.biases_.append(np.zeros(d_out, dtype=np.float32))

    def _forward(self, X: np.ndarray, training: bool = False):
        acts, pres = [X], []
        cur = X
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = cur @ W + b
            pres.append(z)
            if i < len(self.weights_) - 1:
                cur = np.maximum(z, 0.0)            # ReLU
                if training and self.dropout > 0:
                    mask = (np.random.rand(*cur.shape) > self.dropout).astype(np.float32)
                    cur  = cur * mask / (1.0 - self.dropout + 1e-8)
            else:
                cur = z                             # logits
            acts.append(cur)
        return acts, pres

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z).astype(np.float32)
        return e / (e.sum(axis=1, keepdims=True) + 1e-8)

    def _smooth_one_hot(self, y: list[str], weights: np.ndarray) -> np.ndarray:
        n_cls = len(self.labels)
        idx   = {lb: i for i, lb in enumerate(self.labels)}
        Y     = np.full((len(y), n_cls), self.label_smooth / n_cls, dtype=np.float32)
        for row, lb in enumerate(y):
            Y[row, idx[lb]] += 1.0 - self.label_smooth
        # масштабирование строк весами классов
        for row, lb in enumerate(y):
            Y[row] *= weights[idx[lb]]
        return Y

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: list[str]):
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X shape mismatch")

        # нормализация
        self.mean_  = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-6
        Xn = ((X - self.mean_) / self.scale_).astype(np.float32)

        # веса классов (обратно-пропорциональные частоте)
        counts  = Counter(y)
        total_s = len(y)
        cw = np.array([
            total_s / (len(self.labels) * max(1, counts[lb]))
            for lb in self.labels
        ], dtype=np.float32)
        cw /= cw.mean()   # нормируем, чтобы средний вес = 1

        self._init_weights(Xn.shape[1])
        Y = self._smooth_one_hot(y, cw)
        n = float(len(Xn))

        # Adam state
        m_w = [np.zeros_like(w) for w in self.weights_]
        v_w = [np.zeros_like(w) for w in self.weights_]
        m_b = [np.zeros_like(b) for b in self.biases_]
        v_b = [np.zeros_like(b) for b in self.biases_]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        bs  = self.batch_size
        idx_arr = np.arange(len(Xn))
        rng = np.random.default_rng(42)

        for epoch in range(self.epochs):
            rng.shuffle(idx_arr)
            for start in range(0, len(Xn), bs):
                batch = idx_arr[start:start + bs]
                Xb, Yb = Xn[batch], Y[batch]

                acts, pres = self._forward(Xb, training=True)
                logits = acts[-1]
                probs  = self._softmax(logits)
                grad   = (probs - Yb) / max(1, len(batch))

                t += 1
                for i in reversed(range(len(self.weights_))):
                    a_prev = acts[i]
                    dW = a_prev.T @ grad
                    db = grad.sum(axis=0)
                    if i > 0:
                        grad = grad @ self.weights_[i].T * (pres[i - 1] > 0)

                    # Adam
                    m_w[i] = beta1 * m_w[i] + (1 - beta1) * dW
                    v_w[i] = beta2 * v_w[i] + (1 - beta2) * dW ** 2
                    m_b[i] = beta1 * m_b[i] + (1 - beta1) * db
                    v_b[i] = beta2 * v_b[i] + (1 - beta2) * db ** 2

                    mw_hat = m_w[i] / (1 - beta1 ** t)
                    vw_hat = v_w[i] / (1 - beta2 ** t)
                    mb_hat = m_b[i] / (1 - beta1 ** t)
                    vb_hat = v_b[i] / (1 - beta2 ** t)

                    self.weights_[i] -= (self.lr * mw_hat / (np.sqrt(vw_hat) + eps)).astype(np.float32)
                    self.biases_[i]  -= (self.lr * mb_hat / (np.sqrt(vb_hat) + eps)).astype(np.float32)

    # ── inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Возвращает вектор вероятностей для одного примера."""
        if self.mean_ is None:
            return np.ones(len(self.labels), dtype=np.float32) / len(self.labels)
        xn  = ((x - self.mean_) / self.scale_).astype(np.float32)
        acts, _ = self._forward(xn[None, :], training=False)
        return self._softmax(acts[-1])[0]

    def predict(self, x: np.ndarray) -> Prediction:
        probs  = self.predict_proba(x)
        idx    = int(np.argmax(probs))
        return Prediction(
            label=self.labels[idx],
            confidence=float(probs[idx]),
            probs={lb: float(probs[i]) for i, lb in enumerate(self.labels)},
        )

    # ── serialisation ─────────────────────────────────────────────────────────
    def export(self, X: np.ndarray, y: list[str]) -> dict:
        return {
            "labels":   np.array(self.labels, dtype=object),
            "hidden":   np.array(self.hidden, dtype=np.int64),
            "lr":       np.float32(self.lr),
            "epochs":   np.int64(self.epochs),
            "mean":     self.mean_.astype(np.float32),
            "scale":    self.scale_.astype(np.float32),
            "weights":  np.array(self.weights_, dtype=object),
            "biases":   np.array(self.biases_, dtype=object),
            "train_X":  X.astype(np.float32),
            "train_y":  np.array(y, dtype=object),
        }

    @classmethod
    def load(cls, path: Path) -> "NeuroMLP":
        with np.load(path, allow_pickle=True) as d:
            labels = [str(x) for x in d["labels"].tolist()]
            hidden = tuple(int(x) for x in d["hidden"].tolist()) if "hidden" in d else (256, 128, 64)
            lr     = float(d["lr"])     if "lr"     in d else 3e-4
            epochs = int(d["epochs"])   if "epochs" in d else 200
            m = cls(labels=labels, hidden=hidden, lr=lr, epochs=epochs)
            m.mean_    = np.asarray(d["mean"],  dtype=np.float32)
            m.scale_   = np.asarray(d["scale"], dtype=np.float32)
            m.weights_ = [np.asarray(w, dtype=np.float32) for w in d["weights"].tolist()]
            m.biases_  = [np.asarray(b, dtype=np.float32) for b in d["biases"].tolist()]
        return m


def find_saved_profile(name: str) -> tuple[Path, dict]:
    if not PROFILE_ROOT.exists():
        raise FileNotFoundError("Нет сохранённых профилей")
    name = name.strip()
    candidates = []
    for folder in PROFILE_ROOT.iterdir():
        if not folder.is_dir():
            continue
        mp = folder / "profile_model.npz"
        if not mp.exists():
            continue
        meta_p = folder / "profile_meta.json"
        meta   = {}
        if meta_p.exists():
            try: meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception: pass
        if name and str(meta.get("profile_name", "")).strip() != name:
            if not folder.name.endswith(f"_{name}"):
                continue
        candidates.append((folder, meta))
    if not candidates:
        raise FileNotFoundError(f"Профиль «{name}» не найден")
    latest = max(candidates, key=lambda t: t[0].stat().st_mtime)
    return latest[0] / "profile_model.npz", latest[1]


# ═══════════════════════════════════════════════════════════════════════════════
#  BLINK DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
def detect_double_blink(block: np.ndarray, sfreq: float) -> tuple[bool, float]:
    if block.size == 0 or block.shape[1] < max(16, int(sfreq * 0.25)):
        return False, 0.0
    x = block.astype(np.float32) - block.mean(axis=1, keepdims=True)
    signal = np.max(np.abs(x), axis=0)
    k = max(3, int(sfreq * 0.04)) | 1
    smooth = np.convolve(signal, np.ones(k, np.float32) / k, "same")
    med = np.median(smooth)
    mad = np.median(np.abs(smooth - med)) + 1e-6
    z   = (smooth - med) / (1.4826 * mad)
    rs  = max(0, block.shape[1] - int(BLINK_WIN_SEC * sfreq))
    z_r = z[rs:]
    if z_r.size == 0 or not np.any(z_r >= BLINK_THRESH_Z):
        return False, float(np.max(z_r, initial=0.0))
    min_w = max(1, int(BLINK_MIN_WIDTH_SEC * sfreq))
    max_w = max(min_w, int(BLINK_MAX_WIDTH_SEC * sfreq))
    min_g = max(1, int(BLINK_MIN_GAP_SEC  * sfreq))
    max_g = max(min_g, int(BLINK_MAX_GAP_SEC  * sfreq))
    base  = med + 1e-6
    segs: list[tuple[int,int,int,float,float]] = []
    st = None
    for i, a in enumerate(z_r >= BLINK_THRESH_Z):
        if a and st is None: st = i
        elif not a and st is not None:
            w = i - st
            if min_w <= w <= max_w:
                seg = z_r[st:i]
                pi  = int(np.argmax(seg))
                pz  = float(seg[pi])
                pr  = float((smooth[rs + st + pi] + 1e-6) / base)
                if pz >= BLINK_PEAK_Z and pr >= BLINK_MIN_PEAK_RATIO:
                    segs.append((st, i, st + pi, pz, pr))
            st = None
    if st is not None:
        w = len(z_r) - st
        if min_w <= w <= max_w:
            seg = z_r[st:]
            pi  = int(np.argmax(seg))
            pz  = float(seg[pi])
            pr  = float((smooth[rs + st + pi] + 1e-6) / base)
            if pz >= BLINK_PEAK_Z and pr >= BLINK_MIN_PEAK_RATIO:
                segs.append((st, len(z_r), st + pi, pz, pr))
    if len(segs) < 2:
        return False, float(np.max(z_r, initial=0.0))
    for a, b in zip(segs, segs[1:]):
        gap = b[2] - a[2]
        if min_g <= gap <= max_g:
            return True, max(a[3], b[3])
    return False, float(np.max(z_r, initial=0.0))


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADBAND CONTROLLER  (с graceful-disconnect и SDK fallback)
# ═══════════════════════════════════════════════════════════════════════════════
class HeadbandController:
    """Управляет нейрогарнитурой через CapsuleSDK.
    Если SDK недоступен — работает в симуляционном режиме."""

    def __init__(self, ui_q: "queue.Queue"):
        self.ui_q        = ui_q
        self.capsule     = None
        self.locator     = None
        self.device      = None
        self.mems        = None
        self._stop       = threading.Event()
        self._conn_ev    = threading.Event()
        self._list_ev    = threading.Event()
        self._seen_serials: set[str] = set()
        self._devices_by_key: dict[str, object] = {}
        self.device_infos: list = []
        self.sample_rate = SAMPLE_RATE
        self.eeg_buf: Optional[EEGBuffer]  = None
        self.mems_buf = MEMSBuffer(max_samples=int(SAMPLE_RATE * EEG_WIN_SEC * 4))
        self.connected   = False
        self._update_th: Optional[threading.Thread] = None
        self.target_serial_suffix = ""
        self.selected_index = 0

    def _put(self, *args):
        self.ui_q.put(args)

    def _wait_scan_window(self, seconds: float):
        deadline = time.monotonic() + max(0.1, float(seconds))
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            self._list_ev.wait(min(0.25, remaining))

    # ── scanning ──────────────────────────────────────────────────────────────
    def start_scan(self):
        if not _SDK_AVAILABLE:
            self._put("hb_status", "CapsuleSDK недоступен — режим симуляции")
            return
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _ensure_locator(self):
        if self.capsule is not None:
            return
        sys_name = platform.system().lower()
        lib_name = "CapsuleClient.dll" if sys_name.startswith("win") else "libCapsuleClient.dylib"
        for candidate in [APP_DIR / lib_name, APP_DIR / "CapsuleSDK" / lib_name]:
            if candidate.exists():
                self.capsule = Capsule(str(candidate))
                break
        if self.capsule is None:
            raise FileNotFoundError(f"Библиотека {lib_name} не найдена")
        self.locator = DeviceLocator(self.capsule.get_lib())
        self.locator.set_on_devices_list(self._on_device_list)

    def _scan_worker(self):
        try:
            self._ensure_locator()
            self.device_infos  = []
            self._seen_serials = set()
            self._devices_by_key = {}
            self._ensure_update_thread()
            dt = DeviceType.Any
            scan_seconds = SEARCH_TIMEOUT + ANY_SEARCH_EXTRA_SEC
            self._put("hb_status", f"Сканирую гарнитуры тип {int(dt)} в течение {int(scan_seconds)} сек…")
            self._list_ev.clear()
            self.locator.request_devices(device_type=dt, seconds_to_search=int(scan_seconds))
            self._wait_scan_window(scan_seconds + 1.0)
            rows = []
            for i, dev in enumerate(self.device_infos, 1):
                try: name = dev.get_name()
                except: name = "?"
                try: ser = dev.get_serial()
                except: ser = "?"
                rows.append(f"{i:02d}. {name} | {ser}")
            self._put("devices", rows)
            self._put("hb_debug", rows)
            if self.device_infos:
                self.selected_index = 0
                self._put("hb_select", 0)
            self._put("hb_status", f"Найдено гарнитур: {len(self.device_infos)}")
        except Exception as ex:
            self._put("hb_status", f"Ошибка поиска: {ex}")
            self._put("error", str(ex))

    def _on_device_list(self, locator, info, _fail):
        for i in range(len(info)):
            try:
                dev = info[i]
                s   = normalize_serial(dev.get_serial())
                if not s:
                    s = f"{dev.get_name()}|{dev.get_type()}"
            except Exception:
                continue
            if s in self._seen_serials:
                continue
            self._seen_serials.add(s)
            self._devices_by_key[s] = dev
            self.device_infos = list(self._devices_by_key.values())
        self._list_ev.set()

    # ── connection ────────────────────────────────────────────────────────────
    def connect(self, index: int):
        if not _SDK_AVAILABLE:
            # симуляция — просто создаём буфер и запускаем генератор
            self.eeg_buf = EEGBuffer(4, int(SAMPLE_RATE * EEG_WIN_SEC * 4))
            self.connected = True
            threading.Thread(target=self._sim_eeg_gen, daemon=True).start()
            self._put("hb_status", "Симуляция EEG активна")
            return
        if not self.device_infos:
            raise RuntimeError("Список устройств пуст. Сначала нажмите Поиск.")
        chosen = self.device_infos[index]
        self._conn_ev.clear()
        self._ensure_update_thread()
        self.device = Device(self.locator, chosen.get_serial(), self.locator.get_lib())
        self.device.set_on_connection_status_changed(self._on_conn_status)
        self.device.set_on_eeg(self._on_eeg)
        try:
            self.mems = MEMS(self.device, self.capsule.get_lib())
            self.mems.set_on_update(self._on_mems)
        except Exception:
            self.mems = None
        self.device.connect(bipolarChannels=False)
        if not self._conn_ev.wait(CONNECT_TIMEOUT):
            raise RuntimeError("Тайм-аут подключения к гарнитуре")
        self.device.start()
        try:
            sr = float(self.device.get_eeg_sample_rate())
            self.sample_rate = sr
        except Exception:
            pass
        self.connected = True
        self._put("hb_status", f"Подключено: {chosen.get_name()} | {chosen.get_serial()}")

    def _on_conn_status(self, device, status):
        try:
            sv = int(getattr(status, "value", status))
        except Exception:
            sv = -1
        if sv == DEVICE_CONNECTED:
            self._conn_ev.set()
        self._put("hb_status", f"Статус гарнитуры: {sv}")

    def _on_eeg(self, device, eeg: "EEGTimedData"):
        chn = eeg.get_channels_count()
        smp = eeg.get_samples_count()
        if smp <= 0 or chn <= 0:
            return
        if self.eeg_buf is None:
            try: sr = float(device.get_eeg_sample_rate())
            except: sr = SAMPLE_RATE
            self.sample_rate = sr
            self.eeg_buf = EEGBuffer(chn, int(sr * EEG_WIN_SEC * 4))
        block = np.zeros((chn, smp), dtype=np.float32)
        for i in range(smp):
            for c in range(chn):
                try: block[c, i] = eeg.get_processed_value(c, i)
                except: block[c, i] = eeg.get_raw_value(c, i)
        self.eeg_buf.append_block(block)

    def _on_mems(self, mems, md: "MEMSTimedData"):
        for i in range(len(md)):
            g = md.get_gyroscope(i)
            a = md.get_accelerometer(i)
            self.mems_buf.append(
                np.array([g.x, g.y, g.z], dtype=np.float32),
                np.array([a.x, a.y, a.z], dtype=np.float32),
            )

    def _ensure_update_thread(self):
        if self._update_th is None or not self._update_th.is_alive():
            self._stop.clear()
            self._update_th = threading.Thread(target=self._update_loop, daemon=True)
            self._update_th.start()

    def _update_loop(self):
        while not self._stop.is_set():
            try:
                if self.locator:
                    self.locator.update()
            except Exception:
                pass
            time.sleep(0.01)

    def _sim_eeg_gen(self):
        """Синтетический сигнал EEG для симуляции."""
        rng = np.random.default_rng(0)
        t   = 0.0
        while self.connected:
            chunk = 25  # ~100 Гц эффективно
            ts    = np.linspace(t, t + chunk / SAMPLE_RATE, chunk, endpoint=False)
            # альфа-ритм 10 Гц + шум
            sig = (
                5.0 * np.sin(2 * np.pi * 10 * ts) +
                2.0 * np.sin(2 * np.pi * 20 * ts) +
                rng.standard_normal(chunk).astype(np.float32)
            )
            block = np.tile(sig, (4, 1)).astype(np.float32)
            if self.eeg_buf is not None:
                self.eeg_buf.append_block(block)
            t += chunk / SAMPLE_RATE
            time.sleep(chunk / SAMPLE_RATE)

    def get_window(self, seconds: float):
        n = int(self.sample_rate * seconds)
        eeg  = self.eeg_buf.get_last(n) if self.eeg_buf else np.zeros((4, 0), dtype=np.float32)
        g, a = self.mems_buf.get_last(n)
        return eeg, g, a

    def disconnect(self):
        self.connected = False
        self._stop.set()
        try:
            if self.device:
                self.device.stop()
                self.device.disconnect()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARK BRIDGE  (плавное отключение + heartbeat)
# ═══════════════════════════════════════════════════════════════════════════════
class SharkBridge:
    """
    Обёртка над pc_manual_control для плавного управления акулой:
    - heartbeat каждые 0.35 с держит соединение живым
    - при обрыве — автоматически переподключается (без резкого стопа)
    - graceful shutdown: сначала release_all, потом закрыть BLE
    """

    def __init__(self):
        self.state: pc_manual_control.SharedState = pc_manual_control.SharedState()
        self.thread: Optional[threading.Thread]   = None
        self.lock         = threading.Lock()
        self.current_tag: Optional[int]  = None
        self.current_cmd  = "idle"
        self.last_send_at = 0.0
        self.last_change_at = 0.0
        self._hb_thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self) -> str:
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.state = pc_manual_control.SharedState()
                self.thread = pc_manual_control.start_ble_thread(self.state)
        deadline = time.time() + CONNECT_TIMEOUT
        while time.time() < deadline:
            snap = self.state.snapshot()
            if snap["connected"]:
                self._running = True
                if self._hb_thread is None or not self._hb_thread.is_alive():
                    self._hb_thread = threading.Thread(target=self._heartbeat, daemon=True)
                    self._hb_thread.start()
                return snap["device_text"]
            if snap["last_error"]:
                raise RuntimeError(snap["last_error"])
            time.sleep(0.05)
        raise TimeoutError("BLE timeout — акула не найдена")

    def _heartbeat(self):
        """Переотправляет текущую команду + следит за разрывами."""
        while self._running:
            time.sleep(SHARK_HEARTBEAT)
            with self.lock:
                snap = self.state.snapshot()
                # авто-реконнект при разрыве
                if not snap["connected"] and self._running:
                    try:
                        if self.thread is None or not self.thread.is_alive():
                            self.state = pc_manual_control.SharedState()
                            self.thread = pc_manual_control.start_ble_thread(self.state)
                    except Exception:
                        pass
                # resend текущей команды чтобы дирижабль не затормозил
                if self.current_tag is not None and snap["connected"]:
                    if (time.time() - self.last_send_at) >= BLE_RESEND_SEC:
                        self.state.queue_packet(
                            pc_manual_control.build_bluefruit_packet(self.current_tag, True)
                        )
                        self.last_send_at = time.time()

    def set_command(self, cmd: str):
        with self.lock:
            tag = COMMAND_TO_TAG.get(cmd)
            if tag == self.current_tag and cmd == self.current_cmd:
                return
            # плавный переход: сначала stop, потом новая команда
            if self.current_tag is not None:
                self.state.release_all()
                time.sleep(0.03)  # маленькая пауза между командами
            self.current_tag = tag
            self.current_cmd = cmd
            self.last_change_at = time.time()
            if tag is not None:
                self.state.press_tag(tag)
                self.last_send_at = time.time()

    def stop(self):
        with self.lock:
            if self.current_tag is not None:
                self.state.release_all()
            self.current_tag = None
            self.current_cmd = "idle"

    def is_connected(self) -> bool:
        return bool(self.state.snapshot()["connected"])

    def shutdown(self):
        self._running = False
        with self.lock:
            try: self.state.release_all()
            except Exception: pass
            try: self.state.set_running(False)
            except Exception: pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  2D SIMULATION SCENE
# ═══════════════════════════════════════════════════════════════════════════════
class SimPoint:
    def __init__(self, x: float, y: float):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.action = "idle"

    def step(self, action: str, dt: float):
        accel     = ACTION_TO_ACCEL.get(action, ACTION_TO_ACCEL["idle"])
        self.vel += accel * (dt * 2.6)
        self.vel *= max(0.0, 1.0 - 1.7 * dt)
        self.vel  = np.clip(self.vel, -0.85, 0.85)
        self.pos += self.vel * (dt * 1.3)
        for ax in (0, 1):
            if self.pos[ax] < 0.05:
                self.pos[ax]  = 0.05 + (0.05 - self.pos[ax]) * 0.35
                self.vel[ax]  = abs(self.vel[ax]) * 0.45
            elif self.pos[ax] > 0.95:
                self.pos[ax]  = 0.95 - (self.pos[ax] - 0.95) * 0.35
                self.vel[ax]  = -abs(self.vel[ax]) * 0.45
        self.action = action


class SimScene:
    PAD = 44

    def __init__(self, parent: tk.Widget):
        self.frame = tk.Frame(parent, bg="#07111f")
        self.frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.frame, bg="#07111f", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="Симуляция готова")
        tk.Label(self.frame, textvariable=self.status_var,
                 bg="#07111f", fg="#dbeafe", font=("Segoe UI", 10)).place(x=10, y=6)
        self.w, self.h = 900, 560
        self.canvas.bind("<Configure>", lambda e: setattr(self, 'w', e.width) or setattr(self, 'h', e.height))
        self.human = SimPoint(0.22, 0.55)
        self.ai    = SimPoint(0.22, 0.55)
        self.human_trail: deque = deque(maxlen=120)
        self.ai_trail:    deque = deque(maxlen=120)
        self.last_tick = time.time()
        self.mode = "idle"
        self.human_lbl = "idle"
        self.ai_lbl    = "idle"
        self.ai_conf   = 0.0
        self.err_rate  = 0.0
        self.samples   = 0
        self._tick()

    def update_state(self, *, mode, human_lbl, ai_lbl, ai_conf, err_rate, samples, hud=""):
        self.mode     = mode
        self.human_lbl = human_lbl
        self.ai_lbl    = ai_lbl
        self.ai_conf   = ai_conf
        self.err_rate  = err_rate
        self.samples   = samples
        self.status_var.set(hud)

    def reset(self):
        self.human = SimPoint(0.22, 0.55)
        self.ai    = SimPoint(0.22, 0.55)
        self.human_trail.clear()
        self.ai_trail.clear()

    def _xy(self, x, y):
        pad = self.PAD
        return pad + x * (self.w - pad * 2), pad + y * (self.h - pad * 2)

    def _draw_trail(self, trail, color):
        pts = []
        for p in trail:
            cx, cy = self._xy(*p)
            pts.extend([cx, cy])
        if len(pts) >= 4:
            self.canvas.create_line(*pts, fill=color, width=2, smooth=True)

    def _draw_point(self, p: SimPoint, lbl, fill, outline, glow=False):
        cx, cy = self._xy(*p.pos.tolist())
        r = 16
        if glow:
            self.canvas.create_oval(cx-30, cy-30, cx+30, cy+30, outline=outline, width=2)
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=fill, outline=outline, width=2)
        self.canvas.create_oval(cx-6, cy-6, cx+4, cy+4, fill="#ffffff", outline="")
        self.canvas.create_text(cx, cy-26, text=lbl.upper(), fill=outline, font=("Segoe UI", 9, "bold"))

    def _tick(self):
        now = time.time()
        dt  = min(0.05, max(0.01, now - self.last_tick))
        self.last_tick = now
        self.human.step(self.human.action, dt)
        self.ai.step(self.ai.action, dt)
        self.human_trail.append(tuple(float(v) for v in self.human.pos))
        self.ai_trail.append(tuple(float(v) for v in self.ai.pos))

        c = self.canvas
        c.delete("all")
        pad = self.PAD
        c.create_rectangle(0, 0, self.w, self.h, fill="#07111f", outline="")
        c.create_rectangle(pad, pad, self.w-pad, self.h-pad, fill="#0b1520", outline="#38bdf8", width=2)
        c.create_text(pad+10, pad+8, anchor="nw",
                      text="2D Симуляция  |  WASD / стрелки = движение  |  Space = стоп",
                      fill="#bfdbfe", font=("Segoe UI", 9, "bold"))
        for i in range(1, 4):
            x = pad + i * (self.w - pad*2) / 4
            y = pad + i * (self.h - pad*2) / 4
            c.create_line(x, pad, x, self.h-pad, fill="#1f2937", dash=(4,4))
            c.create_line(pad, y, self.w-pad, y, fill="#1f2937", dash=(4,4))

        self._draw_trail(self.human_trail, "#22c55e")
        self._draw_trail(self.ai_trail, "#f97316")
        self._draw_point(self.human, "HUMAN", "#0f172a", "#22c55e", glow=self.mode=="training")
        self._draw_point(self.ai, "AI", "#111827", "#f97316", glow=self.mode=="control")

        dx = float(self.human.pos[0] - self.ai.pos[0])
        dy = float(self.human.pos[1] - self.ai.pos[1])
        gap = (dx*dx + dy*dy) ** 0.5
        hud = [
            f"mode: {self.mode}",
            f"human: {self.human_lbl}",
            f"ai: {self.ai_lbl} ({self.ai_conf:.2f})",
            f"ошибка: {self.err_rate:.1%}",
            f"gap: {gap:.2f}",
            f"samples: {self.samples}",
        ]
        c.create_rectangle(8, 8, 310, 140, fill="#07111f", outline="#1f3b5b")
        c.create_text(16, 16, anchor="nw", fill="#e2e8f0", font=("Consolas", 10), text="\n".join(hud))

        c.after(33, self._tick)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
class NeuroAirshipApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NeuroAirship v2 — Shark AI Control")
        self.geometry("1400x900")
        self.minsize(1100, 760)
        self.configure(bg="#06101d")

        self.ui_q   = queue.Queue()
        self.hb     = HeadbandController(self.ui_q)
        self.shark  = SharkBridge()
        self.lock   = threading.Lock()

        # ── state ──
        self.model: Optional[NeuroMLP]     = None
        self.model_ready                   = False
        self.training_running              = False
        self.training_dirty                = False
        self.train_lock                    = threading.Lock()
        self.dataset_X: list[np.ndarray]   = []
        self.dataset_y: list[str]          = []
        self.sample_count                  = 0
        self.error_history: deque[int]     = deque(maxlen=60)
        self.last_pred: Optional[Prediction] = None
        self.capture_running               = False
        self.last_capture_at               = 0.0
        self.capture_interval              = 0.45   # чаще = больше данных

        # ── EEG voting buffer (ансамбль) ──
        self.vote_history: deque[str]      = deque(maxlen=5)

        # ── control state ──
        self.shark_control_on              = False
        self.blink_paused                  = False
        self.last_blink_toggle_at          = 0.0
        self.current_label                 = "idle"
        self.active_keys: set[str]         = set()
        self.key_order:   list[str]        = []
        self.closing                       = False
        self.device_infos: list            = []
        self.selected_dev_idx              = 0

        # ── vars ──
        self.profile_var    = tk.StringVar(value=PROFILE_DEFAULT)
        self.lr_var         = tk.StringVar(value="3e-4")
        self.epochs_var     = tk.StringVar(value="200")
        self.conf_var       = tk.StringVar(value="0.38")
        self.serial_var     = tk.StringVar(value="")
        self.blink_var      = tk.BooleanVar(value=True)

        self.v_hb_status    = tk.StringVar(value="Гарнитура: не подключена")
        self.v_shark_status = tk.StringVar(value="Акула: не подключена")
        self.v_human        = tk.StringVar(value="Человек: idle")
        self.v_ai           = tk.StringVar(value="ИИ: idle")
        self.v_model        = tk.StringVar(value="Модель: не обучена")
        self.v_train        = tk.StringVar(value="Обучение: ожидание")
        self.v_progress     = tk.StringVar(value="Готово к работе")
        self.v_samples      = tk.StringVar(value="Примеров: 0")
        self.v_error        = tk.StringVar(value="Ошибка: —")
        self.v_ctrl         = tk.StringVar(value="Управление: выкл")
        self.v_hint         = tk.StringVar(value="Сначала обучи ИИ, потом подключи акулу")
        self.dev_status_var = tk.StringVar(value="Устройства: не сканировались")

        self.scene_host = None
        self.dev_listbox: Optional[tk.Listbox] = None
        self._build_ui()
        self.scene = SimScene(self.scene_host)
        self._bind_keys()
        self.after(50, self._ui_tick)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ─────────────────────── UI BUILD ──────────────────────────────────────────
    def _build_ui(self):
        root = tk.Frame(self, bg="#06101d")
        root.pack(fill="both", expand=True, padx=14, pady=14)

        left = tk.Frame(root, bg="#06101d", width=340)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        right = tk.Frame(root, bg="#06101d")
        right.pack(side="right", fill="both", expand=True)

        # title
        tk.Label(left, text="🦈 NeuroAirship v2", bg="#06101d", fg="#f8fafc",
                 font=("Segoe UI", 18, "bold")).pack(anchor="w", pady=(0, 10))

        # ── Profile ──
        self._section(left, "ПРОФИЛЬ")
        self._row(left, "Профиль", self.profile_var)
        self._row(left, "Serial", self.serial_var)

        # ── Headband ──
        self._section(left, "НЕЙРОГАРНИТУРА")
        self._btn(left, "🔍 Найти гарнитуры", self.scan_headbands, "#1d4ed8")
        tk.Label(left, text="Найденные:", bg="#06101d", fg="#93c5fd", anchor="w").pack(fill="x", pady=(6,0))
        lf = tk.Frame(left, bg="#06101d")
        lf.pack(fill="x")
        self.dev_listbox = tk.Listbox(lf, height=5, bg="#0f172a", fg="#e5e7eb",
                                      selectbackground="#1d4ed8", relief="flat",
                                      font=("Consolas", 9))
        self.dev_listbox.pack(side="left", fill="x", expand=True)
        sb = tk.Scrollbar(lf, orient="vertical", command=self.dev_listbox.yview)
        sb.pack(side="right", fill="y")
        self.dev_listbox.configure(yscrollcommand=sb.set)
        self.dev_listbox.bind("<<ListboxSelect>>", self._on_dev_select)
        self.dev_listbox.bind("<Double-Button-1>", lambda e: self.connect_headband())
        tk.Label(left, textvariable=self.dev_status_var, bg="#06101d", fg="#94a3b8",
                 wraplength=320, justify="left").pack(fill="x")
        self._btn(left, "🔗 Подключить гарнитуру", self.connect_headband, "#2563eb")
        tk.Label(left, textvariable=self.v_hb_status, bg="#06101d", fg="#86efac",
                 wraplength=320, justify="left").pack(fill="x", pady=(2,6))

        # ── Training ──
        self._section(left, "ОБУЧЕНИЕ ИИ")
        self._row(left, "LR", self.lr_var)
        self._row(left, "Epochs", self.epochs_var)
        self._row(left, "Conf", self.conf_var)
        self._btn(left, "💾 Сохранить профиль", self.save_profile, "#0ea5e9")
        self._btn(left, "📂 Загрузить профиль", self.load_profile, "#0284c7")
        self._btn(left, "🗑 Очистить датасет", self.clear_dataset, "#7c3aed")
        self._btn(left, "🔄 Сбросить сцену", self.reset_scene, "#334155")

        # ── Shark ──
        self._section(left, "АКУЛА BLE")
        self._btn(left, "🦈 Подключить акулу", self.connect_shark, "#d97706")
        tk.Checkbutton(left, text="Двойное моргание = пауза/старт",
                       variable=self.blink_var, bg="#06101d", fg="#d1d5db",
                       selectcolor="#1c2230", activebackground="#06101d",
                       activeforeground="#fff", anchor="w").pack(fill="x", pady=(4,0))
        self._btn(left, "▶ Запустить ИИ-управление", self.start_shark_control, "#059669")
        self._btn(left, "■ Остановить управление", self.stop_shark_control, "#b91c1c")
        tk.Label(left, textvariable=self.v_shark_status, bg="#06101d", fg="#fbbf24",
                 wraplength=320, justify="left").pack(fill="x", pady=(2,2))
        tk.Label(left, textvariable=self.v_ctrl, bg="#06101d", fg="#e5e7eb",
                 wraplength=320, justify="left").pack(fill="x")

        # ── Status block ──
        self._section(left, "СТАТУС")
        for var in (self.v_human, self.v_ai, self.v_model, self.v_train,
                    self.v_samples, self.v_error, self.v_progress):
            tk.Label(left, textvariable=var, bg="#06101d", fg="#e5e7eb",
                     wraplength=320, justify="left", anchor="w").pack(fill="x", pady=1)
        tk.Label(left, textvariable=self.v_hint, bg="#06101d", fg="#93c5fd",
                 wraplength=320, justify="left").pack(fill="x", pady=(6,0))

        self.scene_host = tk.Frame(right, bg="#06101d")
        self.scene_host.pack(fill="both", expand=True)

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg="#06101d", fg="#38bdf8",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 2))
        tk.Frame(parent, height=1, bg="#1e3a5f").pack(fill="x", pady=(0, 4))

    def _row(self, parent, label, var):
        row = tk.Frame(parent, bg="#06101d")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, width=9, anchor="w", bg="#06101d", fg="#e5e7eb").pack(side="left")
        tk.Entry(row, textvariable=var, bg="#111827", fg="#fff",
                 insertbackground="white", relief="flat").pack(side="left", fill="x", expand=True)

    def _btn(self, parent, text, cmd, bg="#1e3a5f"):
        tk.Button(parent, text=text, command=cmd, bg=bg, fg="#fff",
                  relief="flat", padx=10, pady=6,
                  activebackground="#2563eb").pack(fill="x", pady=3)

    # ─────────────────────── KEYBOARD ──────────────────────────────────────────
    def _bind_keys(self):
        for k in ("w", "a", "s", "d", "Up", "Down", "Left", "Right", "space"):
            self.bind_all(f"<KeyPress-{k}>",   lambda e, kk=k: self._key_dn(kk))
            self.bind_all(f"<KeyRelease-{k}>", lambda e, kk=k: self._key_up(kk))

    def _key_dn(self, k):
        n = k.lower()
        if n not in self.active_keys:
            self.active_keys.add(n)
            self.key_order.append(n)
        self.current_label = self._resolve()

    def _key_up(self, k):
        n = k.lower()
        self.active_keys.discard(n)
        self.key_order = [x for x in self.key_order if x != n]
        self.current_label = self._resolve()

    def _resolve(self) -> str:
        for k in reversed(self.key_order):
            if k in KEY_TO_LABEL:
                return KEY_TO_LABEL[k]
        return "idle"

    # ─────────────────────── HEADBAND ──────────────────────────────────────────
    def scan_headbands(self):
        self.dev_status_var.set("Сканирую…")
        if self.dev_listbox:
            self.dev_listbox.delete(0, "end")
        self.hb.start_scan()

    def _on_dev_select(self, _e=None):
        if self.dev_listbox:
            sel = self.dev_listbox.curselection()
            if sel:
                self.selected_dev_idx = int(sel[0])

    def connect_headband(self):
        def worker():
            try:
                idx = self.selected_dev_idx
                self.hb.connect(idx)
                self.v_hint.set("Гарнитура подключена! Удерживай WASD для обучения.")
                # ждём появления первых данных
                for _ in range(50):
                    if self.hb.eeg_buf and len(self.hb.eeg_buf.samples) > 10:
                        break
                    time.sleep(0.1)
            except Exception as ex:
                self.ui_q.put(("error", str(ex)))
        threading.Thread(target=worker, daemon=True).start()

    # ─────────────────────── SHARK ─────────────────────────────────────────────
    def connect_shark(self):
        def worker():
            try:
                label = self.shark.connect()
                self.v_shark_status.set(f"Акула: {label}")
                self.ui_q.put(("progress", "Акула подключена — можно запускать управление"))
            except Exception as ex:
                self.v_shark_status.set(f"Акула: ошибка — {ex}")
                self.ui_q.put(("error", str(ex)))
        threading.Thread(target=worker, daemon=True).start()

    def start_shark_control(self):
        if not self.hb.connected:
            messagebox.showwarning("Гарнитура", "Сначала подключите гарнитуру.")
            return
        if not self.model_ready:
            messagebox.showwarning("ИИ", "Сначала обучите модель.")
            return
        if not self.shark.is_connected():
            messagebox.showwarning("Акула", "Сначала подключите акулу.")
            return
        self.shark_control_on = True
        self.blink_paused     = False
        self.last_blink_toggle_at = 0.0
        self.v_ctrl.set("Управление: ВКЛЮЧЕНО")
        self.after(CTRL_TICK_MS, self._ctrl_loop)

    def stop_shark_control(self):
        self.shark_control_on = False
        self.shark.stop()
        self.v_ctrl.set("Управление: выкл")

    def _predict_ai(self) -> tuple[str, float]:
        feat = self._capture_features()
        if feat is None:
            return "idle", 0.0
        mem_label, mem_conf = self._predict_from_memory(feat)
        if mem_label != "idle":
            self.last_pred = Prediction(mem_label, mem_conf, {lb: (mem_conf if lb == mem_label else 0.0) for lb in LABELS})
            return mem_label, mem_conf
        if not self.model_ready or self.model is None:
            return "idle", 0.0
        pred = self.model.predict(feat)
        self.last_pred = pred
        probs = pred.probs or {}
        best_dir = max(DIRECTION_LABELS, key=lambda label: float(probs.get(label, 0.0)))
        best_conf = float(probs.get(best_dir, 0.0))
        return best_dir, best_conf

    def _predict_from_memory(self, feat: np.ndarray) -> tuple[str, float]:
        with self.train_lock:
            if len(self.dataset_y) < MEMORY_PREDICT_MIN_SAMPLES:
                return "idle", 0.0
            pairs = [
                (self.dataset_X[i], self.dataset_y[i])
                for i in range(max(0, len(self.dataset_y) - MEMORY_PREDICT_MAX_SAMPLES), len(self.dataset_y))
                if self.dataset_y[i] in DIRECTION_LABELS
            ]
        if len(pairs) < MEMORY_PREDICT_MIN_SAMPLES:
            return "idle", 0.0

        x = feat.astype(np.float32, copy=False)
        x_norm = float(np.linalg.norm(x)) + 1e-6
        scored: list[tuple[float, str]] = []
        for sample, label in pairs:
            s = np.asarray(sample, dtype=np.float32)
            sim = float(np.dot(x, s) / (x_norm * (float(np.linalg.norm(s)) + 1e-6)))
            scored.append((sim, label))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:MEMORY_PREDICT_TOP_K]
        if not top:
            return "idle", 0.0

        weights = {label: 0.0 for label in DIRECTION_LABELS}
        for sim, label in top:
            weights[label] += max(0.0, sim)

        best_label = max(DIRECTION_LABELS, key=lambda label: weights[label])
        total = sum(weights.values()) + 1e-6
        best_conf = float(weights[best_label] / total)
        if best_conf < 0.34:
            return "idle", best_conf
        return best_label, best_conf

    def _ctrl_loop(self):
        if not self.shark_control_on or self.closing:
            return
        try:
            eeg, gyro, accel = self.hb.get_window(CTRL_WIN_SEC)
            if eeg.shape[1] < 8:
                self.after(CTRL_TICK_MS, self._ctrl_loop)
                return

            # ── двойное моргание ──
            if self.blink_var.get():
                detected, _ = detect_double_blink(eeg, self.hb.sample_rate)
                now = time.time()
                if detected and (now - self.last_blink_toggle_at) > BLINK_COOLDOWN_SEC:
                    self.last_blink_toggle_at = now
                    self.blink_paused = not self.blink_paused
                    self.shark.stop()
                    state_txt = "ПАУЗА (моргание)" if self.blink_paused else "возобновлено"
                    self.v_ctrl.set(f"Управление: {state_txt}")

            if self.blink_paused:
                self.after(CTRL_TICK_MS, self._ctrl_loop)
                return

            # ── ИИ предсказание ──
            conf_thresh = _safe_float(self.conf_var.get(), 0.38)
            stable, stable_conf = self._predict_ai()
            if stable_conf < conf_thresh:
                self.shark.stop()
                self.v_ai.set(f"ИИ: idle  conf={stable_conf:.2f}")
                self.v_ctrl.set("Управление: ожидание уверенного направления")
                self.after(CTRL_TICK_MS, self._ctrl_loop)
                return

            self.shark.set_command(stable)
            self.v_ai.set(f"ИИ: {stable}  conf={stable_conf:.2f}")
            self.v_ctrl.set(f"Управление: → {stable.upper()}")
        except Exception as ex:
            self.v_ctrl.set(f"Ошибка управления: {ex}")

        self.after(CTRL_TICK_MS, self._ctrl_loop)

    # ─────────────────────── TRAINING ──────────────────────────────────────────
    def _capture_features(self) -> Optional[np.ndarray]:
        eeg, gyro, accel = self.hb.get_window(CTRL_WIN_SEC)
        if eeg.size == 0 or eeg.shape[1] < 8:
            return None
        return extract_feature_vector(eeg, gyro, accel, self.hb.sample_rate)

    def _record_sample(self):
        label = self.current_label
        # не записываем idle — снижает шум
        if label == "idle":
            return
        if not self.hb.connected:
            return
        now = time.time()
        if (now - self.last_capture_at) < self.capture_interval:
            return
        if self.capture_running:
            return
        self.last_capture_at = now
        self.capture_running = True

        def worker():
            try:
                time.sleep(0.15)
                feat = self._capture_features()
                if feat is None:
                    return
                with self.train_lock:
                    self.dataset_X.append(feat)
                    self.dataset_y.append(label)
                    if len(self.dataset_X) > 1000:
                        self.dataset_X = self.dataset_X[-1000:]
                        self.dataset_y = self.dataset_y[-1000:]
                self.sample_count = len(self.dataset_y)
                self.v_samples.set(f"Примеров: {self.sample_count}")
                self.ui_q.put(("progress", f"записан {label}"))
                self._queue_train()
            except Exception as ex:
                self.ui_q.put(("error", str(ex)))
            finally:
                self.capture_running = False

        threading.Thread(target=worker, daemon=True).start()

    def _queue_train(self):
        # Обучаем при ≥12 примерах и хотя бы 2 разных классах
        with self.train_lock:
            if len(self.dataset_y) < 12:
                return
            if len(set(self.dataset_y)) < 2:
                return
        self.training_dirty = True
        if self.training_running:
            return

        def worker():
            while True:
                with self.train_lock:
                    self.training_running = True
                    self.training_dirty   = False
                    X = np.vstack(self.dataset_X[-TRAIN_RECENT_MAX_SAMPLES:]).astype(np.float32)
                    y = list(self.dataset_y[-TRAIN_RECENT_MAX_SAMPLES:])
                try:
                    lr     = _safe_float(self.lr_var.get(),    3e-4)
                    epochs = max(50, _safe_int(self.epochs_var.get(), 200))
                    m = NeuroMLP(labels=list(LABELS), hidden=(256, 128, 64),
                                 lr=lr, epochs=epochs, dropout=0.3, label_smooth=0.1)
                    m.fit(X, y)
                    with self.train_lock:
                        self.model       = m
                        self.model_ready = True
                    cnt = Counter(y)
                    self.ui_q.put(("model",    f"обучена | {len(y)} примеров | {dict(cnt)}"))
                    self.ui_q.put(("progress", f"обучение завершено на {len(y)} примерах"))
                    # автосохранение
                    self._auto_save(X, y, m)
                except Exception as ex:
                    self.ui_q.put(("error", str(ex)))
                    with self.train_lock:
                        self.training_running = False
                    break
                with self.train_lock:
                    if not self.training_dirty:
                        self.training_running = False
                        break

        threading.Thread(target=worker, daemon=True).start()

    def _auto_save(self, X, y, model: NeuroMLP):
        try:
            name = self.profile_var.get().strip() or PROFILE_DEFAULT
            root = PROFILE_ROOT / f"{_now_stamp()}_{name}"
            root.mkdir(parents=True, exist_ok=True)
            np.savez(root / "profile_model.npz", **model.export(X, y))
            meta = {
                "profile_name": name,
                "created_at":   _now_stamp(),
                "samples":      len(y),
                "labels":       list(LABELS),
            }
            (root / "profile_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def save_profile(self):
        if not self.dataset_X:
            messagebox.showinfo("Сохранить", "Нет данных для сохранения.")
            return
        if not self.model_ready or self.model is None:
            messagebox.showinfo("Сохранить", "Сначала обучите модель.")
            return
        X = np.vstack(self.dataset_X).astype(np.float32)
        y = list(self.dataset_y)
        self._auto_save(X, y, self.model)
        self.v_progress.set("Профиль сохранён")

    def load_profile(self):
        name = self.profile_var.get().strip() or PROFILE_DEFAULT
        try:
            path, meta = find_saved_profile(name)
            m = NeuroMLP.load(path)
            with self.train_lock:
                self.model       = m
                self.model_ready = True
            with np.load(path, allow_pickle=True) as d:
                if "train_X" in d and "train_y" in d:
                    self.dataset_X    = [np.asarray(r, dtype=np.float32) for r in d["train_X"].tolist()]
                    self.dataset_y    = [str(x) for x in d["train_y"].tolist()]
                    self.sample_count = len(self.dataset_y)
            self.v_model.set(f"Модель: загружена | {self.sample_count} примеров")
            self.v_samples.set(f"Примеров: {self.sample_count}")
            self.v_progress.set(f"Загружен профиль «{meta.get('profile_name', name)}»")
        except Exception as ex:
            messagebox.showerror("Загрузить", str(ex))

    def clear_dataset(self):
        with self.train_lock:
            self.dataset_X    = []
            self.dataset_y    = []
            self.sample_count = 0
            self.model        = None
            self.model_ready  = False
        self.error_history.clear()
        self.vote_history.clear()
        self.v_model.set("Модель: очищена")
        self.v_samples.set("Примеров: 0")
        self.v_progress.set("Датасет очищен")

    def reset_scene(self):
        self.scene.reset()
        self.current_label = "idle"
        self.active_keys.clear()
        self.key_order.clear()

    # ─────────────────────── UI TICK ──────────────────────────────────────────
    def _step_logic(self):
        # обновляем 2D сцену
        self.scene.human.action = self.current_label
        self._record_sample()

        ai_lbl, ai_conf = self._predict_ai()
        thresh = _safe_float(self.conf_var.get(), 0.38)
        if ai_conf < thresh:
            ai_lbl = "idle"
        if self.current_label != "idle" and self.last_pred is not None:
            self.error_history.append(0 if ai_lbl == self.current_label else 1)

        self.scene.ai.action = ai_lbl
        err = sum(self.error_history) / max(1, len(self.error_history))
        self.v_human.set(f"Человек: {self.current_label}")
        self.v_ai.set(f"ИИ: {ai_lbl}  conf={ai_conf:.2f}")
        self.v_error.set(f"Ошибка: {err:.1%}")
        self.v_train.set("Обучение: идёт" if self.training_running else "Обучение: ожидание")
        if not self.hb.connected:
            self.v_hint.set("Подключи гарнитуру и удерживай WASD/стрелки для обучения ИИ")
        elif not self.model_ready:
            self.v_hint.set("Обучение идёт... Нажимай разные направления по несколько секунд")
        else:
            self.v_hint.set("ИИ обучен! Подключи акулу и нажми «Запустить управление»")

        self.scene.update_state(
            mode="training" if self.training_running else ("control" if self.shark_control_on else "idle"),
            human_lbl=self.current_label,
            ai_lbl=ai_lbl,
            ai_conf=ai_conf,
            err_rate=err,
            samples=self.sample_count,
            hud=self.v_hint.get(),
        )

    def _ui_tick(self):
        if self.closing:
            return
        while True:
            try:
                evt = self.ui_q.get_nowait()
            except queue.Empty:
                break
            kind = evt[0]
            if kind == "hb_status":
                self.v_hb_status.set(f"Гарнитура: {evt[1]}")
            elif kind == "devices":
                rows = list(evt[1])
                if self.dev_listbox:
                    self.dev_listbox.delete(0, "end")
                    for r in rows:
                        self.dev_listbox.insert("end", r)
                self.dev_status_var.set(f"Найдено: {len([r for r in rows if not r.startswith('(')])}")
                self.device_infos = list(self.hb.device_infos)
            elif kind == "hb_select":
                self.selected_dev_idx = int(evt[1])
                if self.dev_listbox and self.dev_listbox.size() > self.selected_dev_idx:
                    self.dev_listbox.selection_clear(0, "end")
                    self.dev_listbox.selection_set(self.selected_dev_idx)
                    self.dev_listbox.see(self.selected_dev_idx)
            elif kind == "hb_debug":
                rows = list(evt[1])
                preview = "; ".join(rows[:8]) if rows else "nothing"
                if len(rows) > 8:
                    preview += f"; ... (+{len(rows) - 8})"
                self.dev_status_var.set(f"SDK found: {preview}")
            elif kind == "model":
                self.v_model.set(f"Модель: {evt[1]}")
            elif kind == "progress":
                self.v_progress.set(str(evt[1]))
            elif kind == "error":
                self.v_progress.set(f"Ошибка: {evt[1]}")
                messagebox.showerror("Ошибка", str(evt[1]))

        self._step_logic()
        self.after(50, self._ui_tick)

    def on_close(self):
        self.closing          = True
        self.shark_control_on = False
        self.shark.shutdown()
        self.hb.disconnect()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = NeuroAirshipApp()
    app.mainloop()


if __name__ == "__main__":
    main()
