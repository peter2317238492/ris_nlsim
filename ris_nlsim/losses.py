"""任务损失的唯一实现（DESIGN §7.3，INTEGRATION E）：numpy 与 torch 两套，数学逐式相同。

所有公开函数签名统一为 ``f(readings_db, labels, tau, reduction="mean", ...)``：
- readings_db: (B,K) 探头读数（dB / dBm），numpy 数组或 torch.Tensor；
- labels: (B,) 整数标签（numpy 或 torch 均可）；
- tau: 温度（dB）；
- reduction: "mean" → 标量（numpy 返回 float，torch 返回 0 维张量）；"none" → 逐样本 (B,)（INTEGRATION C：
  系统层把被标记自激的样本替换为常数，需要逐样本损失）。
输入为 torch.Tensor 时走 torch 路径（可自动微分），否则走 numpy 路径。

- 二分类（奇偶）：s = sgn(label)·(y1 − y2)，sgn = +1（label 1，奇）/ −1（label 0，偶）；
  ℓ_b = τ·softplus(−s/τ)，softplus(x) = log(1+e^x) 用 logaddexp(0,x) 数值稳定实现。
- 多分类：读数/τ 的 softmax 交叉熵，ℓ_b = logsumexp(y_b/τ) − y_{b,label}/τ。
- 聚焦：ℓ_b = −y_{b,0}（探头 0 的 dB 读数；labels、tau 忽略）。
- 阈值检测（单探头）：探头 0，阈值 t = 两类读数中位数的中点，s = sgn(label)·(y − t)，ℓ_b = τ·softplus(−s/τ)。
- loc3bit（INTEGRATION E，Song 2026 的 8 格定位）：K 个探头各自阈值判决，标签 c 的第 j 位 bit_j = (c>>j)&1；
  t_j = ½(median(y_j | bit_j=1) + median(y_j | bit_j=0))（"auto"）或给定 thresholds_db；
  s_bj = sgn(bit_j)·(y_bj − t_j)，ℓ_b = loss_weight·Σ_j τ·softplus(−s_bj/τ)。
"auto" 阈值（threshold 与 loc3bit）**不回传梯度**（INTEGRATION E：阈值是读数的标定量，不参与求导；审查 TR-4）。
"""
from __future__ import annotations

import math

import numpy as np


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _torch():
    import torch
    return torch


def _labels_np(labels) -> np.ndarray:
    if _is_torch(labels):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).astype(np.int64).ravel()


def _sign_np(labels) -> np.ndarray:
    # label 1（奇）→ +1，label 0（偶）→ −1
    return np.where(_labels_np(labels) != 0, 1.0, -1.0)


def _readings_np(y) -> np.ndarray:
    """读数的 numpy 副本（torch 输入 detach：用于不回传梯度的阈值标定）。"""
    if _is_torch(y):
        y = y.detach().cpu().numpy()
    return np.asarray(y, dtype=float)


def _reduce(per, reduction: str):
    if reduction == "mean":
        return per.mean() if _is_torch(per) else float(np.mean(per))
    if reduction == "none":
        return per
    raise ValueError(f"reduction 只能是 'mean' 或 'none'，得到 {reduction!r}")


def _bits_np(labels, n_bits: int) -> np.ndarray:
    """(B,) 标签 → (B,n_bits) 0/1，第 j 列 = (label >> j) & 1（INTEGRATION E）。"""
    lab = _labels_np(labels)
    return (lab[:, None] >> np.arange(n_bits)[None, :]) & 1


def _median_midpoint(y: np.ndarray, pos: np.ndarray, what: str) -> float:
    """t = ½(median(y | pos) + median(y | ~pos))；任一类为空时无定义 → ValueError。"""
    if not pos.any() or pos.all():
        raise ValueError(f"{what}：自动阈值需要两类样本都非空")
    return 0.5 * (float(np.median(y[pos])) + float(np.median(y[~pos])))


def threshold_value(readings_db, labels) -> float:
    """threshold 任务的自动阈值（探头 0，两类中位数中点），numpy 标量，不可微。"""
    y0 = _readings_np(readings_db)[:, 0]
    return _median_midpoint(y0, _labels_np(labels) != 0, "threshold")


def loc3bit_thresholds(readings_db, labels) -> np.ndarray:
    """loc3bit 的自动阈值 t_j（每探头一个，(K,) numpy，不可微）；tasks.LocalizationTask.predict 应复用它。"""
    y = _readings_np(readings_db)
    bits = _bits_np(labels, y.shape[1]).astype(bool)
    return np.array([_median_midpoint(y[:, j], bits[:, j], f"loc3bit 探头 {j}") for j in range(y.shape[1])])


def _loc3bit_t(readings_db, labels, thresholds_db) -> np.ndarray:
    if thresholds_db is None or (isinstance(thresholds_db, str) and thresholds_db == "auto"):
        return loc3bit_thresholds(readings_db, labels)
    t = np.asarray(thresholds_db, dtype=float).ravel()
    if t.size != np.shape(readings_db)[1]:
        raise ValueError(f"thresholds_db 长度 {t.size} 与探头数 {np.shape(readings_db)[1]} 不符")
    return t


# ---------------------------------------------------------------- numpy 实现（逐样本）
def _softplus_np(x):
    return np.logaddexp(0.0, x)


def _binary_np(y, labels, tau):
    y = np.asarray(y, dtype=float)
    s = _sign_np(labels) * (y[:, 0] - y[:, 1])
    return tau * _softplus_np(-s / tau)


def _multiclass_np(y, labels, tau):
    z = np.asarray(y, dtype=float) / tau
    m = np.max(z, axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.sum(np.exp(z - m), axis=1))
    lab = _labels_np(labels)
    return lse - z[np.arange(z.shape[0]), lab]


def _focus_np(y, labels, tau):
    return -np.asarray(y, dtype=float)[:, 0]


def _threshold_np(y, labels, tau):
    y0 = np.asarray(y, dtype=float)[:, 0]
    t = threshold_value(y, labels)
    s = _sign_np(labels) * (y0 - t)
    return tau * _softplus_np(-s / tau)


def _loc3bit_np(y, labels, tau, thresholds_db, loss_weight):
    y = np.asarray(y, dtype=float)
    t = _loc3bit_t(y, labels, thresholds_db)
    sgn = 2.0 * _bits_np(labels, y.shape[1]) - 1.0            # bit 1 → +1，bit 0 → −1
    s = sgn * (y - t[None, :])
    return loss_weight * np.sum(tau * _softplus_np(-s / tau), axis=1)


# ---------------------------------------------------------------- torch 实现（逐样本）
def _labels_t(labels, ref):
    torch = _torch()
    return torch.as_tensor(_labels_np(labels), device=ref.device)


def _const_t(x, ref):
    torch = _torch()
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=ref.dtype, device=ref.device)


def _softplus_t(x):
    # 与 numpy 同式：log(1+e^x) = logaddexp(0,x)（不用 F.softplus 的阈值近似）
    torch = _torch()
    return torch.logaddexp(torch.zeros_like(x), x)


def _binary_t(y, labels, tau):
    s = _const_t(_sign_np(labels), y) * (y[:, 0] - y[:, 1])
    return tau * _softplus_t(-s / tau)


def _multiclass_t(y, labels, tau):
    torch = _torch()
    z = y / tau
    lab = _labels_t(labels, y)
    lse = torch.logsumexp(z, dim=1)
    return lse - z[torch.arange(z.shape[0], device=y.device), lab]


def _focus_t(y, labels, tau):
    return -y[:, 0]


def _threshold_t(y, labels, tau):
    t = threshold_value(y, labels)                    # 由 detach 的读数算出：阈值不回传梯度
    s = _const_t(_sign_np(labels), y) * (y[:, 0] - t)
    return tau * _softplus_t(-s / tau)


def _loc3bit_t_loss(y, labels, tau, thresholds_db, loss_weight):
    t = _const_t(_loc3bit_t(y, labels, thresholds_db), y)   # 常量：不回传梯度
    sgn = _const_t(2.0 * _bits_np(labels, y.shape[1]) - 1.0, y)
    s = sgn * (y - t.unsqueeze(0))
    return loss_weight * (tau * _softplus_t(-s / tau)).sum(dim=1)


# ---------------------------------------------------------------- 公开接口
def binary_loss(readings_db, labels, tau: float = 1.0, reduction: str = "mean"):
    """二分类（奇偶）损失：mean τ·softplus(−s/τ)，s = sgn(label)(y1−y2)（DESIGN §7.3）。"""
    f = _binary_t if _is_torch(readings_db) else _binary_np
    return _reduce(f(readings_db, labels, tau), reduction)


def multiclass_loss(readings_db, labels, tau: float = 1.0, reduction: str = "mean"):
    """多分类损失：读数/τ 的 softmax 交叉熵（DESIGN §7.3）。"""
    f = _multiclass_t if _is_torch(readings_db) else _multiclass_np
    return _reduce(f(readings_db, labels, tau), reduction)


def focus_loss(readings_db, labels=None, tau: float = 1.0, reduction: str = "mean"):
    """聚焦损失：−mean(探头 0 读数 dB)（实验 F；labels、tau 忽略）。"""
    f = _focus_t if _is_torch(readings_db) else _focus_np
    return _reduce(f(readings_db, labels, tau), reduction)


def threshold_loss(readings_db, labels, tau: float = 1.0, reduction: str = "mean"):
    """阈值检测损失：探头 0，阈值取两类中位数中点（不回传梯度）的 logistic 判别（DESIGN §7.3）。"""
    f = _threshold_t if _is_torch(readings_db) else _threshold_np
    return _reduce(f(readings_db, labels, tau), reduction)


def loc3bit_loss(readings_db, labels, tau: float = 1.0, reduction: str = "mean", thresholds_db="auto",
                 loss_weight: float = 1.0):
    """8 格定位（3 探头 × 阈值 → 3-bit 格号）损失（INTEGRATION E）：
    mean_b loss_weight·Σ_j τ·softplus(−s_bj/τ)，s_bj = sgn(bit_j(label_b))·(y_bj − t_j)。
    thresholds_db："auto"/None → 每探头两类中位数中点（不回传梯度）；或长度 K 的固定阈值数组。"""
    f = _loc3bit_t_loss if _is_torch(readings_db) else _loc3bit_np
    return _reduce(f(readings_db, labels, tau, thresholds_db, float(loss_weight)), reduction)


LOSSES = {
    "binary": binary_loss,
    "multiclass": multiclass_loss,
    "focus": focus_loss,
    "threshold": threshold_loss,
    "loc3bit": loc3bit_loss,
}


def task_loss(readings_db, labels, task: str, tau: float = 1.0, reduction: str = "mean", **kwargs):
    """按任务名分派：'binary' | 'multiclass' | 'focus' | 'threshold' | 'loc3bit'。

    kwargs 原样传给对应损失（目前只有 loc3bit 接受 thresholds_db、loss_weight；其余任务传入会报 TypeError）。
    """
    try:
        fn = LOSSES[task]
    except KeyError:
        raise ValueError(f"未知任务损失 {task!r}，可选 {sorted(LOSSES)}") from None
    return fn(readings_db, labels, tau, reduction=reduction, **kwargs)


def chance_loss(task: str, n_outputs: int, tau: float = 1.0, thresholds_db="auto",
                loss_weight: float = 1.0) -> float | None:
    """"零信息"样本的逐样本损失：所有判决量恰在判决边界上（s = 0 / 各类读数相等）时的 ℓ_b。

    系统层把被标记可能自激的样本损失替换为 chance + osc_penalty（INTEGRATION C 写的 τ·log2 是二分类特例）：
    binary/threshold → τ·log2；multiclass（K 类，未乘 τ）→ log K；loc3bit（K 个比特）→ loss_weight·K·τ·log2；
    focus 无"判决边界"概念 → 返回 None（调用方保留原损失值但切断梯度）。thresholds_db 仅为接口对齐，不影响结果。
    """
    if task in ("binary", "threshold"):
        return tau * math.log(2.0)
    if task == "multiclass":
        return math.log(int(n_outputs))
    if task == "loc3bit":
        return float(loss_weight) * int(n_outputs) * tau * math.log(2.0)
    if task == "focus":
        return None
    raise ValueError(f"未知任务损失 {task!r}，可选 {sorted(LOSSES)}")
