"""单元/天线幅度方向图（DESIGN §1.1）。

F(θ) = sqrt( d0 * ( max(cosθ,0)**q + 10**(backlobe_db/10) ) )，cosθ = n̂·d̂，
d̂ 为**从该节点指向另一节点**的单位向量；F² 即该方向上的方向性（线性）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DEFAULTS


@dataclass(frozen=True)
class Pattern:
    """cos^q 前半球 + 常数后瓣地板 的幅度方向图。

    - q=None：各向同性，F ≡ 1（此时 d0 必须为 None）。
    - d0=None：取 2(q+1)。由 ∮max(cosθ,0)^q dΩ = 2π/(q+1) 得
      ∮F² dΩ/(4π) = d0·(1/(2(q+1)) + 10^{bl/10}) = 1 + 2(q+1)·10^{bl/10}，
      后瓣极低时 → 1（方向性归一化）。
    - 显式 d0（如单元口径受限 d0 = 4πA/λ² = π）时不再归一化，平均增益 < 1 表示口径效率/遮挡。
    """

    q: float | None = None
    d0: float | None = None
    backlobe_db: float = -30.0

    def __post_init__(self):
        if self.q is None:
            if self.d0 is not None:
                raise ValueError("各向同性方向图(q=None)不能指定 d0")
            return
        if not np.isfinite(self.q) or self.q < 0:
            raise ValueError(f"q 必须为非负有限数，得到 {self.q}")
        if self.d0 is not None and not (np.isfinite(self.d0) and self.d0 > 0):
            raise ValueError(f"d0 必须为正有限数，得到 {self.d0}")
        if not np.isfinite(self.backlobe_db):
            raise ValueError("backlobe_db 必须有限")

    # ---- 派生量 ----
    @property
    def isotropic(self) -> bool:
        return self.q is None

    @property
    def d0_eff(self) -> float:
        """实际使用的 d0（各向同性返回 1）。"""
        if self.q is None:
            return 1.0
        return float(self.d0) if self.d0 is not None else 2.0 * (self.q + 1.0)

    @property
    def floor(self) -> float:
        """后瓣地板（线性功率，相对 d0）。"""
        return 10.0 ** (self.backlobe_db / 10.0)

    @property
    def peak_directivity(self) -> float:
        """视轴方向 F² = d0·(1 + floor)。"""
        if self.q is None:
            return 1.0
        return self.d0_eff * (1.0 + self.floor)

    def mean_gain(self) -> float:
        """解析的球面平均 ∮F²dΩ/(4π) = d0·(1/(2(q+1)) + floor)。"""
        if self.q is None:
            return 1.0
        return self.d0_eff * (0.5 / (self.q + 1.0) + self.floor)

    # ---- 求值 ----
    def amp_cos(self, cos: np.ndarray) -> np.ndarray:
        """由 cosθ 直接求幅度方向图 F（float64，与 cos 同形状）。"""
        cos = np.asarray(cos, dtype=np.float64)
        if self.q is None:
            return np.ones(cos.shape, dtype=np.float64)
        # 用 where 而非 max(cos,0)**q：保证 q=0 时后半球为 0（避免 0**0 = 1）
        front = np.where(cos > 0.0, np.abs(cos) ** self.q, 0.0)
        return np.sqrt(self.d0_eff * (front + self.floor))

    def amp(self, normals: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """幅度方向图 F(n̂, d̂)，normals/dirs 末维为 3，均应为单位向量。

        广播规则：若 normals 为 (N,3) 且 dirs 为 (N,M,3)（dirs 比 normals 多一维），
        则在 normals 倒数第二维插入长度 1 的轴 → 结果 (N,M)；其余情形按 numpy 常规广播。
        """
        n = np.asarray(normals, dtype=np.float64)
        d = np.asarray(dirs, dtype=np.float64)
        if n.shape[-1] != 3 or d.shape[-1] != 3:
            raise ValueError("normals 与 dirs 的最后一维必须为 3")
        if n.ndim >= 2 and d.ndim == n.ndim + 1:
            n = n[..., None, :]
        if self.q is None:
            return np.ones(np.broadcast_shapes(n.shape, d.shape)[:-1], dtype=np.float64)
        cos = np.sum(n * d, axis=-1)
        return self.amp_cos(cos)

    def gain(self, normals: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """方向性 F²。"""
        return self.amp(normals, dirs) ** 2


# ---- 默认方向图（DESIGN §1.1，参数取自 constants.DEFAULTS）----
ISOTROPIC = Pattern()
PO_BACKLOBE_DB = -60.0      # 金属板 PO 单元后瓣地板（DESIGN §1.1；DEFAULTS 中没有该项，故放这里）


def element_pattern() -> Pattern:
    """中继收/发单元、无源面单元：q=1, d0=π（(λ/2)² 口径 4πA/λ²），后瓣 −25 dB。"""
    return Pattern(q=DEFAULTS["elem_q"], d0=DEFAULTS["elem_d0"],
                   backlobe_db=DEFAULTS["elem_backlobe_db"])


def horn_pattern() -> Pattern:
    """馈源喇叭：q=10，d0 = 2(q+1) = 22（≈13.4 dBi）。"""
    return Pattern(q=DEFAULTS["horn_q"])


def probe_pattern() -> Pattern:
    """探头：q=1，d0 = 4。"""
    return Pattern(q=DEFAULTS["probe_q"])


def po_cell_pattern(cell_area: float, lam: float) -> Pattern:
    """金属板物理光学单元：q=1, d0 = 4πA_cell/λ²，后瓣 −60 dB（实验 F 参考）。"""
    return Pattern(q=1.0, d0=4.0 * np.pi * cell_area / lam**2, backlobe_db=PO_BACKLOBE_DB)
