"""PA 模型：线性 PA 与 Rapp(AM-AM) + Saleh(AM-PM)（方案 §4.4；DESIGN §3.1）。

单位约定：PA 输入 u、输出 g 均为 √W 复幅度，|·|² = 功率 (W)。
所有方法逐元素作用，接受任意形状的复数组（含标量）。
"""
from __future__ import annotations

import numpy as np

from .constants import DEFAULTS, dbm2w

_TINY = 1e-30          # |u| 小于它时按 u→0 极限处理（DESIGN §3.1）
_R_MAX = 1e150         # r 上限（先截断再平方），防止 AM-PM 公式溢出


class PA:
    """PA 基类。子类须实现 g / wirtinger / p1db_in_w。

    属性：gain_db（小信号功率增益 dB）、gain_v = 10**(gain_db/20)（小信号电压增益）。
    """

    gain_db: float = 0.0
    gain_v: float = 1.0
    is_linear: bool = False   # True 时不动点可直接线性求解（DESIGN §3.4）

    def g(self, u: np.ndarray) -> np.ndarray:
        """逐元素复传输 g(u)。"""
        raise NotImplementedError

    def wirtinger(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Wirtinger 导数 (a, b)：δg = a·δu + b·conj(δu)。"""
        raise NotImplementedError

    def p1db_in_w(self) -> float:
        """输入 1 dB 压缩点功率 (W)。"""
        raise NotImplementedError


class LinearPA(PA):
    """理想线性 PA：g = G_v·u（方案 §4.4 的对照 M1）。"""

    is_linear = True

    def __init__(self, gain_db: float = DEFAULTS["pa_gain_db"]):
        self.gain_db = float(gain_db)
        self.gain_v = 10.0 ** (self.gain_db / 20.0)

    def g(self, u):
        return self.gain_v * np.asarray(u, dtype=np.complex128)

    def wirtinger(self, u):
        shape = np.shape(u)
        a = np.full(shape, self.gain_v, dtype=np.complex128)
        b = np.zeros(shape, dtype=np.complex128)
        return a, b

    def p1db_in_w(self) -> float:
        return float("inf")

    def __repr__(self) -> str:
        return f"LinearPA(gain_db={self.gain_db})"


class RappPA(PA):
    """Rapp AM-AM + Saleh 型 AM-PM（方案 §4.4，AM-PM 自变量归一化到输入饱和幅度，DESIGN §3.1）。

    A_s = sqrt(Psat_out)，A_in = A_s/G_v，r = |u|/A_in
    |g| = G_v|u| / (1 + r^{2p})^{1/(2p)}
    ∠g = ∠u + Φ(r)，Φ(r) = α_φ r²/(1+β_φ r²)，α_φ = deg2rad(ampm_max_deg)·β_φ（Φ(∞) = ampm_max_deg）
    """

    def __init__(self, gain_db: float = DEFAULTS["pa_gain_db"],
                 psat_out_dbm: float = DEFAULTS["psat_out_dbm"],
                 p: float = DEFAULTS["rapp_p"],
                 ampm_max_deg: float = DEFAULTS["ampm_max_deg"],
                 ampm_beta: float = DEFAULTS["ampm_beta"],
                 ampm: bool = True):
        if p <= 0:
            raise ValueError("Rapp 平滑度 p 必须 > 0")
        if ampm_beta < 0:
            raise ValueError("ampm_beta 必须 >= 0")
        self.gain_db = float(gain_db)
        self.gain_v = 10.0 ** (self.gain_db / 20.0)
        self.psat_out_dbm = float(psat_out_dbm)
        self.psat_out_w = float(dbm2w(psat_out_dbm))
        self.A_s = float(np.sqrt(self.psat_out_w))      # 输出饱和幅度 (√W)
        self.A_in = self.A_s / self.gain_v              # 输入饱和幅度 (√W)
        self.p = float(p)
        self.ampm_max_deg = float(ampm_max_deg)
        self.ampm_beta = float(ampm_beta)
        self.beta_phi = self.ampm_beta
        self.alpha_phi = float(np.deg2rad(ampm_max_deg)) * self.ampm_beta
        self.ampm = bool(ampm)

    def __repr__(self) -> str:
        return (f"RappPA(gain_db={self.gain_db}, psat_out_dbm={self.psat_out_dbm}, p={self.p}, "
                f"ampm_max_deg={self.ampm_max_deg}, ampm_beta={self.ampm_beta}, ampm={self.ampm})")

    # ---- 内部：幅度/相位剖面 ----
    def _terms(self, rho: np.ndarray):
        """给定 ρ=|u|，返回 (c1, c2, Φ, ρΦ′)。

        c1 = (1+r^{2p})^{-1/(2p)}       ⇒ |g| = G_v ρ c1
        c2 = (1+r^{2p})^{-(1+2p)/(2p)}  ⇒ d|g|/dρ = G_v c2（Rapp 导数的闭式化简）
        用 log(1+r^{2p}) = logaddexp(0, 2p·ln r) 避免大 r 溢出。
        """
        r = rho / self.A_in
        with np.errstate(divide="ignore"):
            t = 2.0 * self.p * np.log(r)                # r=0 → -inf，logaddexp 给 0
        L = np.logaddexp(0.0, t)
        c1 = np.exp(-L / (2.0 * self.p))
        c2 = np.exp(-L * (1.0 + 2.0 * self.p) / (2.0 * self.p))
        if self.ampm and self.alpha_phi != 0.0:
            q = np.square(np.minimum(r, _R_MAX))
            x = self.beta_phi * q
            inv = 1.0 / (1.0 + x)
            phi = self.alpha_phi * q * inv              # Φ = α r²/(1+β r²)
            rho_dphi = 2.0 * self.alpha_phi * q * inv * inv   # ρ·dΦ/dρ = 2α r²/(1+β r²)²
        else:
            phi = np.zeros_like(r)
            rho_dphi = np.zeros_like(r)
        return c1, c2, phi, rho_dphi

    def amam(self, rho) -> np.ndarray:
        """AM-AM：|g| 作为 |u| 的函数（√W）。"""
        rho = np.abs(np.asarray(rho, dtype=float))
        c1, _, _, _ = self._terms(rho)
        return self.gain_v * rho * c1

    def ampm_rad(self, rho) -> np.ndarray:
        """AM-PM：Φ(|u|)（弧度）。"""
        rho = np.abs(np.asarray(rho, dtype=float))
        return self._terms(rho)[2]

    # ---- 公共接口 ----
    def g(self, u):
        u = np.asarray(u, dtype=np.complex128)
        c1, _, phi, _ = self._terms(np.abs(u))
        # g = G_v c1 e^{jΦ} u（ρ e^{j∠u} = u，无需除以 |u|）
        return self.gain_v * c1 * np.exp(1j * phi) * u

    def wirtinger(self, u):
        """DESIGN §3.1：g = h(ρ)·u/ρ，h = |g|e^{jΦ}
        a = ½(h′ + h/ρ)，b = ½ e^{2j∠u}(h′ − h/ρ)
        其中 h/ρ = G c1 e^{jΦ}，h′ = G (c2 + j c1 ρΦ′) e^{jΦ}。u→0：c1=c2=1、ρΦ′=0 ⇒ a=G_v、b=0。
        """
        u = np.asarray(u, dtype=np.complex128)
        rho = np.abs(u)
        c1, c2, phi, rho_dphi = self._terms(rho)
        common = 0.5 * self.gain_v * np.exp(1j * phi)
        a = common * (c2 + c1 + 1j * c1 * rho_dphi)
        small = rho < _TINY
        rot = np.where(small, 1.0 + 0.0j, (u / np.where(small, 1.0, rho)) ** 2)   # e^{2j∠u}
        b = common * rot * (c2 - c1 + 1j * c1 * rho_dphi)
        return a, b

    def p1db_in_w(self) -> float:
        """P1dB_in = A_in²·(10^{0.1p} − 1)^{1/p}（由 (1+r^{2p})^{1/(2p)} = 10^{1/20} 解出）。"""
        return float(self.A_in ** 2 * (10.0 ** (0.1 * self.p) - 1.0) ** (1.0 / self.p))
