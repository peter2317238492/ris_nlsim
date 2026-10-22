"""体素极化率与无源单元等效极化率（DESIGN §1.3，方案 §4.1）。

标量耦合偶极子模型：p = α·E_loc（α 单位 m³），偶极子辐射场 E = k²G·p。

能量守恒（单偶极子，入射平面波 E0，E_loc = E_inc）：
  σ_ext = −k·Im(p·conj(E_inc))/|E0|² = −k·Im(α)        （e^{jωt} 约定：有耗 ⇔ Im α < 0）
  σ_sca = k⁴|α|²/(4π)·|E_loc|²/|E0|²  = k⁴|α|²/(4π)    （|E|² = k⁴|p|²/(16π²R²) 在球面积分）
  σ_abs = σ_ext − σ_sca = k|α|²·(Im(1/α) − k³/(4π))
辐射修正 α = α0/(1 + j·k³α0/(4π)) ⇔ 1/α = 1/α0 + j·k³/(4π)
  ⇒ σ_abs = k|α|²·Im(1/α0)：无耗（α0 实数）时 σ_abs = 0（光学定理）；
     εr = ε′ − jε″（ε″>0）时 Im(α0)<0 ⇒ Im(1/α0)>0 ⇒ σ_abs > 0。
多偶极子同理：P_ext − P_sca = k·Σ_i |p_i|²·Im(1/α0_i)（耦合项因 G 的对称性相消）。

模型与连续极限（方案 §1 的标量 Lippmann–Schwinger：E^s = k²∫Gχ E^tot dV，χ = εr − 1）：
- "scalar_rr"（本实现的默认，见下方"默认模型的取舍"）：标量 LS 的一致离散。体素 i 的总场 E_i = E_loc,i + χ·S·E_i，
  自作用 S = k²∫_{等体积球} G dV ≈ k²a²/2 − j·k³V/(4π)（a = (3V/4π)^{1/3}，取到辐射项为止以严格保持光学定理），
  p_i = χV·E_i ⇒ α = χV / (1 − χk²a²/2 + jχk³V/(4π))，即 α0 = χV/(1 − χk²a²/2) 再做辐射修正。
  它就是标量格林函数下的"局域场（Clausius–Mossotti 型）修正 + 辐射修正"：矢量 CM 的 1/α0 = 1/(χV) + 1/(3V)
  来自矢量 G 的 δ 去极化项，标量 G 没有该项，静态自作用为 0，领头修正是动态项 −k²a²/(2V)。
  d → 0 时收敛到 εr 本身的标量 Mie 解（solver.mie_scalar_far_field；tests/test_solver.py 有回归测试）。
- "cm_rr"（DESIGN §1.3 字面默认）：标量方程里没有矢量格林函数的 δ 去极化项，CM 形式相当于把对比度换成
  χ_eff = 3(εr−1)/(εr+2) ≤ 3，网格加密后收敛到 ε_eff = 1 + χ_eff 的标量 Mie 解，而不是 εr 本身
  （eps4→2.5，eps10→3.25，水→3.89−0.014j，金属→4；见 tests/test_solver.py）。它在极小物体的矢量准静态极限里
  给出正确偶极矩，但在标量模型、ka≈1 的奇偶散射体上会压缩并打乱 εr 扫描（RQ2），且金属退化成 εr=4 的介质。
  即使对它自己的连续极限 Mie(ε_eff)，因缺 k²a²/2 自作用项，λ/10 时误差也有 13%~22%（εr ≥ 4；scalar_rr 对 εr ≤ 10 只有 0.3%~3%）。
- "metal"：cm/cm_rr/volume_rr 沿用 DESIGN 的 α0 = 3V（矢量 CM 的 εr→∞ 极限，标量模型下是启发式，连续极限是 εr=4 介质）；
  scalar_rr 取 χ→∞ 极限 α = V/(−k²a²/2 + jk³V/(4π))（体素内总场为 0，逼近软球 / Dirichlet 散射体，即方案 §4.1
  "金属用高对比度体素近似"）。

默认模型的取舍（PHY-1 / PHY-R2-1，偏离 DESIGN §1.3 字面的 model="cm_rr"）：DEFAULT_MODEL = "scalar_rr"。
方案 §4.1 要的是"带辐射修正、强散射时能量守恒的 CM 形式"，其目的（体素的局域场修正 + 光学定理）在标量模型里由
scalar_rr 实现；照搬矢量 CM 会让 polarizability()/voxels_to_nodes() 的默认路径（tasks 按 INTEGRATION §E 不传 model）
求解错误的介电常数。cm / cm_rr / volume_rr 仍可显式选用（对照实验时请同时报 effective_eps(·, model)）。

体素尺寸参考（λ/3 球，+z 平面波，DDA 远场 vs 标量 Mie 的相对误差；geometry.voxelize_sphere 已按格胞体积缩放格距）：
  名义体素 d：           λ/10   λ/15   λ/20   λ/25   λ/30   λ/40
  scalar_rr eps4         0.3%   0.4%   0.8%   0.7%   0.4%   0.2%
  scalar_rr eps10        2.9%   3.1%   3.6%   4.4%   3.2%   1.6%
  scalar_rr 水           25%    19%    8.3%   4.1%   4.2%   4.5%
  scalar_rr 金属(软球)   18%    13%    9.3%   6.1%   4.8%   4.4%
  cm_rr vs Mie(ε_eff)    13%    5.1%   3.6%   2.8%   1.9%   1.0%   （eps4；对 Mie(εr) 恒为 65%~70%）
经验判据：介质 |sqrt(εr)|·k·d ≲ 2 时误差 ≲ 5%（水 |m| ≈ 8.9 ⇒ d ≲ λ/28）；默认 voxel_d = λ/10 只对 εr ≤ 10 够用。

无源 / 金属板 PO 单元 α = λΓ/k² 的模型限制见 alpha_passive。
"""
from __future__ import annotations

import numpy as np

from .constants import MATERIALS
from .patterns import Pattern
from .scene import NodeSet

MODELS = ("cm", "cm_rr", "volume_rr", "scalar_rr")
# 默认模型（PHY-R2-1）：DESIGN §1.3 字面为 "cm_rr"，但它在标量模型里收敛到 ε_eff = 1 + 3(εr−1)/(εr+2) ≤ 4，
# 压缩 εr 扫描且金属退化成 εr=4 介质；scalar_rr 收敛到 εr 本身（模块 docstring"默认模型的取舍"）。
DEFAULT_MODEL = "scalar_rr"


def resolve_eps(eps_r) -> complex | str:
    """材料名 / 数值 → 复介电常数，或字符串 "metal"。

    约定 εr = ε′ − jε″（DESIGN §1，e^{jωt}）：Im εr > 0 是增益介质（物理约定 ε′ + jε″ 写反了），直接报错。
    """
    if isinstance(eps_r, str):
        if eps_r == "metal":
            return "metal"
        if eps_r not in MATERIALS:
            raise ValueError(f"未知材料 {eps_r!r}，可选 {list(MATERIALS)}")
        val = MATERIALS[eps_r]
        return val if isinstance(val, str) else resolve_eps(val)
    eps = complex(eps_r)
    if not (np.isfinite(eps.real) and np.isfinite(eps.imag)):
        raise ValueError(f"εr 必须有限，得到 {eps_r!r}")
    if eps.imag > 1e-12 * max(1.0, abs(eps)):        # 容许舍入噪声
        raise ValueError(f"εr = {eps} 的虚部 > 0：本项目用 e^{{jωt}} 工程约定 εr = ε′ − jε″"
                         f"（DESIGN §1，例如水 78 − 10j），请把虚部取负")
    return eps


def radiative_correction(alpha0, k: float):
    """辐射修正 α = α0/(1 + j·k³·α0/(4π))（e^{jωt}，DESIGN §1.3）。"""
    alpha0 = np.asarray(alpha0, dtype=np.complex128)
    return alpha0 / (1.0 + 1j * k**3 * alpha0 / (4.0 * np.pi))


def _scalar_rr(eps, V: np.ndarray, k: float) -> np.ndarray:
    """标量 LS 一致的体素极化率（模块 docstring）。

    α = χV / (1 − χ·k²a²/2 + j·χ·k³V/(4π))；金属（χ→∞）：α = V / (−k²a²/2 + j·k³V/(4π))。
    写成"分子 / 分母"而非 1/(1/α0 + …)，使 χ=0、V=0、实 χ 的静态共振（χk²a²/2 = 1）都无 1/0。
    """
    a2 = (3.0 * V / (4.0 * np.pi)) ** (2.0 / 3.0)          # a²
    rad = 1j * k**3 * V / (4.0 * np.pi)
    if eps == "metal":
        den = -0.5 * k**2 * a2 + rad
        out = np.zeros(np.shape(V), dtype=np.complex128)
        return np.divide(V, den, out=out, where=V > 0)            # V=0 → α=0（避免 0/0）
    chi = eps - 1.0
    return chi * V / (1.0 - chi * (0.5 * k**2 * a2) + chi * rad)


def polarizability(eps_r: complex | str, volume, k: float, model: str = DEFAULT_MODEL):
    """体素极化率（DESIGN §1.3，方案 §4.1）。

    - "cm"：α0 = 3V(εr−1)/(εr+2)（不做辐射修正）
    - "cm_rr"：CM 再辐射修正（DESIGN §1.3 字面默认；连续极限是 ε_eff 而非 εr，见 effective_eps）
    - "volume_rr"：α0 = (εr−1)V 再辐射修正
    - "scalar_rr"（默认 DEFAULT_MODEL）：α = χV/(1 − χk²a²/2 + jχk³V/(4π))，标量 LS 的一致离散
      （收敛到 εr 本身的标量 Mie；取舍见模块 docstring"默认模型的取舍"）
    - eps_r == "metal"：cm/cm_rr/volume_rr 用 α0 = 3V（CM 在 εr→∞ 的极限）；是否辐射修正由 model 是否以 "_rr"
      结尾决定（取舍：DESIGN 写"再做辐射修正"，这里对 "cm" 保持"无修正"的含义一致，"volume_rr" 对金属同样用 3V）；
      scalar_rr 用 χ→∞ 极限（软球）。
    volume 可为标量或数组；标量输入返回 complex。
    """
    if model not in MODELS:
        raise ValueError(f"未知模型 {model!r}，可选 {MODELS}")
    if not (np.isfinite(k) and k > 0):
        raise ValueError("k 必须为正")
    V = np.asarray(volume, dtype=np.float64)
    if np.any(V < 0):
        raise ValueError("volume 不能为负")
    eps = resolve_eps(eps_r)
    if model == "scalar_rr":
        alpha = _scalar_rr(eps, V, k)
    else:
        if eps == "metal":
            alpha0 = 3.0 * V + 0j
        elif model == "volume_rr":
            alpha0 = (eps - 1.0) * V
        else:
            if abs(eps + 2.0) < 1e-12:
                raise ValueError("εr = −2 处 Clausius–Mossotti 发散")
            alpha0 = 3.0 * V * (eps - 1.0) / (eps + 2.0)
        alpha = radiative_correction(alpha0, k) if model.endswith("_rr") else np.asarray(alpha0, np.complex128)
    return complex(alpha) if np.ndim(alpha) == 0 else alpha


def effective_eps(eps_r: complex | str, model: str = DEFAULT_MODEL) -> complex | str:
    """模型在网格加密极限下实际求解的介电常数 ε_eff（标量 LS 的 1 + χ_eff；见模块 docstring）。

    scalar_rr / volume_rr：ε_eff = εr；cm / cm_rr：ε_eff = 1 + 3(εr−1)/(εr+2)；
    金属：scalar_rr 为 "metal"（软球极限），其余模型为 4（CM 极限 χ_eff = 3）。
    选体素尺寸时用 |sqrt(ε_eff)|·k·d ≲ 2 作为 DDA 分辨率判据（模块 docstring 的误差表）。
    """
    if model not in MODELS:
        raise ValueError(f"未知模型 {model!r}，可选 {MODELS}")
    eps = resolve_eps(eps_r)
    if eps == "metal":
        # scalar_rr：χ→∞ 软球；cm/cm_rr/volume_rr 的金属都用 α0 = 3V（polarizability）⇒ χ_eff = 3。
        # （审查 R2-INT-1：born 物体改用 volume_rr 后，金属的 effective_eps 记录须为 4 而不是 "metal"）
        return "metal" if model == "scalar_rr" else 4.0 + 0j
    if model in ("scalar_rr", "volume_rr"):
        return eps
    return 1.0 + 3.0 * (eps - 1.0) / (eps + 2.0)


def alpha_passive(Gamma, k: float, lam: float):
    """无源 / 金属板单元的等效极化率 α = λΓ/k²（DESIGN §1.3）。

    推导：单元接收 v = E·λF_in/sqrt(4π)，乘 Γ 后作为源再辐射：E′ = Γv·sqrt(4π)F_out·G = λΓ F_in F_out G·E；
    与偶极子 E′ = k²α·F_in F_out G·E 对比得 k²α = λΓ。方向图在耦合函数里另乘。
    金属板 PO 单元 d0 = 4πA/λ²、Γ=1：k²αD = 4πA/λ，单站 RCS = 4πA²/λ²，与物理光学一致。

    模型限制（只做"接收 × Γ × 再辐射"的反射半空间模型）：没有辐射阻尼（1/α 无 jk³/(4π)·… 项），也没有前向
    阴影。因此单元不满足光学定理：|Γ|=1 时 σ_sca 与 Γ 的相位无关，但 σ_ext = −k·Im(α)·F_in·F_fwd 随相位变号
    （Γ=1 为 0，Γ=+j 为负即"增益"，Γ=−j 为正）。默认几何里面内耦合只经 −25 dB 后瓣地板，影响很小；
    若加入无源面正对 PO 板之类的无耗腔体，需要另加辐射阻尼项并做能量平衡验证。
    """
    out = lam * np.asarray(Gamma, dtype=np.complex128) / k**2
    return complex(out) if np.ndim(out) == 0 else out


def single_dipole_cross_sections(alpha, k: float) -> dict:
    """孤立偶极子在平面波下的截面 (m²)：σ_ext = −k·Im α，σ_sca = k⁴|α|²/(4π)，σ_abs = 差值。"""
    alpha = np.asarray(alpha, dtype=np.complex128)
    ext = -k * alpha.imag
    sca = k**4 * np.abs(alpha) ** 2 / (4.0 * np.pi)
    return {"sigma_ext": ext, "sigma_sca": sca, "sigma_abs": ext - sca}


def voxels_to_nodes(pos, vol_each: float, material, k: float, model: str = DEFAULT_MODEL,
                    group: int = 0) -> NodeSet:
    """体素 → 各向同性散射体 NodeSet（所有体素同材料同体积，group 统一编号）。"""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    alpha = polarizability(material, vol_each, k, model)
    return NodeSet(pos, None, Pattern(), alpha=np.full(len(pos), alpha, dtype=np.complex128),
                   group=np.full(len(pos), group, dtype=np.int64))
