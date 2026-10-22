"""读出与噪声：探头功率 → dBm 读数（方案 §4.6；DESIGN §4）。

流水线：功率(W) → dBm → +offset → (logdet) clip 到量程 → + N(0, σ_dB²)（ideal 不加）。
探头加性复高斯噪声加在复场 z 上（|·|² 之前），由 relay.probe_fields / 系统层负责；
本模块提供生成该噪声的工具 complex_awgn / Detector.probe_noise。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import dbm2w, w2dbm

_KINDS = ("ideal", "vna", "logdet")


def complex_awgn(rng: np.random.Generator, shape, power_w: float) -> np.ndarray:
    """圆对称复高斯噪声，E|n|² = power_w（实部、虚部方差各 power_w/2）。"""
    std = np.sqrt(power_w / 2.0)
    return std * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


@dataclass
class Detector:
    """探头检波器（方案 §4.6）。

    kind: "ideal"（无噪声）| "vna"（≈0.002 dB rms）| "logdet"（AD8317 类，0.1–0.5 dB，量程 −40…−10 dBm）
    """

    kind: str = "vna"
    sigma_db: float = 0.002
    probe_noise_dbm: float | None = None     # 探头加性复高斯噪声功率；None 关闭
    range_dbm: tuple = (-40.0, -10.0)        # logdet 量程
    offset_db: float = 0.0                   # 检波前增益/衰减

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"未知检波器类型 {self.kind!r}，可选 {_KINDS}")
        lo, hi = self.range_dbm
        if not lo < hi:
            raise ValueError("range_dbm 需满足 lo < hi")

    def calibrate(self, p_w_ref) -> None:
        """logdet：设 offset 使参考读数（无噪声）中位数 = 量程中点；vna / ideal：不变。"""
        if self.kind != "logdet":
            return
        mid = 0.5 * (self.range_dbm[0] + self.range_dbm[1])
        self.offset_db = float(mid - np.median(w2dbm(p_w_ref)))

    def read(self, p_w, rng: np.random.Generator | None = None) -> np.ndarray:
        """功率 (W) → dBm 读数，形状同 p_w。

        rng=None 时不加读数噪声（仍做 offset 与 clip），用于无噪声评估。
        """
        y = w2dbm(p_w) + self.offset_db
        if self.kind == "logdet":
            y = np.clip(y, self.range_dbm[0], self.range_dbm[1])
        if self.kind != "ideal" and rng is not None and self.sigma_db > 0:
            y = y + self.sigma_db * rng.standard_normal(np.shape(y))
        return np.asarray(y, dtype=float)

    def probe_noise_w(self) -> float:
        """探头加性噪声功率 (W)；关闭时为 0。"""
        return 0.0 if self.probe_noise_dbm is None else float(dbm2w(self.probe_noise_dbm))

    def probe_noise(self, shape, rng: np.random.Generator | None) -> np.ndarray | None:
        """探头处加性复高斯噪声样本（加到 z 上）；关闭或 rng=None 时返回 None。"""
        if self.probe_noise_dbm is None or rng is None:
            return None
        return complex_awgn(rng, shape, self.probe_noise_w())
