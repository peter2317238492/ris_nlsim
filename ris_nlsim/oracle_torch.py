"""连续相位 + 自动微分 oracle（DESIGN §6.3，方案 §4.7；方案写 JAX，这里用 torch complex128/CPU）。

可微路径按 DESIGN 公式用 torch 独立重写（不调用 solver.py 的函数）：
- §2.3 eval_reduced（无源面相位 → 样本响应 H）；
- §3.1 Rapp + Saleh PA（AM-PM 自变量归一化到输入饱和幅度）；
- §3.2–3.3 中继权重与线性量 u_in、K、d、C；
- §3.4 工作点 2R×2R 实雅可比与环路增益。
测试把这里的 torch 路径与生产 numpy 管线（solver → relay）逐项对比（DESIGN §6.3：相对误差 < 1e-8）。

唯一的例外是**无梯度的不动点求解**：直接调用 relay.solve_fixed_point（用 pa.RappPA/LinearPA 按同一参数构造）。
强环路下不动点不唯一（稳定饱和根 / 不稳定近线性根 / 无 CW 稳态），选哪个吸引子取决于求解器细节；
共用同一求解器才能保证 torch 前向 == numpy 前向（u、收敛标志、可能自激标志逐样本一致，审查 TR-2）。

ReducedSample 一律按"鸭子类型"访问（属性 mode, n_rx, n_probe, n_tx, n_p, H, alpha_p_base, Cpp, Ep, Hrp,
Eo, Cop, Cpo, Lam, Hro），数组为 numpy。

不动点求导（隐函数定理）：先无梯度求得 u*，对**收敛**样本构造
    c = (I − J)⁻¹ (F(u*, θ) − u*)，  û = u* + (c − c.detach())   （实 2R 表示，J = ∂F/∂u|_{u*} 已 detach）
前向值恰为 u*（与 numpy 逐位相同），而 dû/dθ = (I − J)⁻¹ ∂F/∂θ 恰为精确的隐函数梯度。
未收敛样本 û = u*（最后迭代值，常量），读数梯度在 TorchSystem.forward 中切断（审查 TR-1）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import losses
from . import pa as np_pa
from . import relay as np_relay
from .constants import DEFAULTS

CDT = torch.complex128
RDT = torch.float64
TWO_PI = 2.0 * math.pi


# ============================================================ 工具
def _ct(x) -> torch.Tensor:
    """转 complex128 张量（numpy 或 torch 输入）。"""
    if isinstance(x, torch.Tensor):
        return x.to(CDT)
    return torch.as_tensor(np.asarray(x, dtype=np.complex128))


def _rt(x) -> torch.Tensor:
    """转 float64 张量。"""
    if isinstance(x, torch.Tensor):
        return x.to(RDT)
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def _expj(phi: torch.Tensor) -> torch.Tensor:
    """e^{jφ}，φ 为实张量（保持可微）。"""
    return torch.polar(torch.ones_like(phi), phi)


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(torch.zeros_like(x), x)


def to_real(z: torch.Tensor) -> torch.Tensor:
    """复 (…,R) → 实 (…,2R) = [Re; Im]。"""
    return torch.cat([z.real, z.imag], dim=-1)


def from_real(x: torch.Tensor) -> torch.Tensor:
    """实 (…,2R) → 复 (…,R)。"""
    r = x.shape[-1] // 2
    return torch.complex(x[..., :r], x[..., r:])


# ============================================================ §2.3 约化表示求值
@dataclass
class ReducedBatch:
    """把若干 ReducedSample 堆成批（born/intra 的物体维按最大 N_O 零填充，零填充在数学上精确无影响）。"""
    mode: str
    n_rx: int
    n_probe: int
    n_tx: int
    n_p: int
    H: torch.Tensor                       # (B,N_RCV,S)
    alpha_p_base: torch.Tensor | None = None   # (B,N_P)
    Cpp: torch.Tensor | None = None       # (B,N_P,N_P)
    Ep: torch.Tensor | None = None        # (B,N_P,S)
    Hrp: torch.Tensor | None = None       # (B,N_RCV,N_P)
    Eo: torch.Tensor | None = None        # (B,N_O,S)
    Cop: torch.Tensor | None = None       # (B,N_O,N_P)
    Cpo: torch.Tensor | None = None       # (B,N_P,N_O)
    Lam: torch.Tensor | None = None       # (B,N_O,N_O)
    Hro: torch.Tensor | None = None       # (B,N_RCV,N_O)

    def __len__(self) -> int:
        return self.H.shape[0]


def _pad2(x, rows: int, cols: int) -> torch.Tensor:
    out = torch.zeros((rows, cols), dtype=CDT)
    if x is not None and np.size(x) > 0:
        t = _ct(x)
        out[: t.shape[0], : t.shape[1]] = t
    return out


def stack_reduced(reduced: list) -> ReducedBatch:
    """把 ReducedSample 列表转成批量 torch 张量（各样本须同模式、同 N_RCV、S、N_P）。"""
    if len(reduced) == 0:
        raise ValueError("reduced 不能为空")
    r0 = reduced[0]
    mode = str(r0.mode)
    for rs in reduced:
        if str(rs.mode) != mode or int(rs.n_p) != int(r0.n_p):
            raise ValueError("所有样本的 mode 与 n_p 必须一致")
    n_p = int(r0.n_p)
    H = torch.stack([_ct(rs.H) for rs in reduced])
    rb = ReducedBatch(mode=mode, n_rx=int(r0.n_rx), n_probe=int(r0.n_probe), n_tx=int(r0.n_tx), n_p=n_p, H=H)
    if n_p == 0:
        return rb
    rb.alpha_p_base = torch.stack([_ct(np.asarray(rs.alpha_p_base).ravel()) for rs in reduced])
    rb.Cpp = torch.stack([_ct(rs.Cpp) for rs in reduced])
    rb.Ep = torch.stack([_ct(rs.Ep) for rs in reduced])
    rb.Hrp = torch.stack([_ct(rs.Hrp) for rs in reduced])
    if mode in ("born", "intra"):
        n_rcv, s = H.shape[1], H.shape[2]
        n_os = [0 if rs.Eo is None else int(np.shape(rs.Eo)[0]) for rs in reduced]
        n_o = max(n_os)
        rb.Eo = torch.stack([_pad2(rs.Eo, n_o, s) for rs in reduced])
        rb.Cop = torch.stack([_pad2(rs.Cop, n_o, n_p) for rs in reduced])
        rb.Cpo = torch.stack([_pad2(rs.Cpo, n_p, n_o) for rs in reduced])
        rb.Lam = torch.stack([_pad2(rs.Lam, n_o, n_o) for rs in reduced])
        rb.Hro = torch.stack([_pad2(rs.Hro, n_rcv, n_o) for rs in reduced])
    elif mode != "full":
        raise ValueError(f"未知物体模式 {mode!r}")
    return rb


def eval_reduced_batch(rb: ReducedBatch, passive_phase: torch.Tensor | None) -> torch.Tensor:
    """批量版 eval_reduced（DESIGN §2.3）。passive_phase: (N_P,) 或 (B,N_P) 弧度；返回 (B,N_RCV,S)。

    n_p>0 而 passive_phase=None 时按相位 0 处理（取舍：与"无源面存在但相位为 0"一致）。
    """
    if rb.n_p == 0:
        return rb.H
    if passive_phase is None:
        phi = torch.zeros(rb.n_p, dtype=RDT)
    else:
        phi = _rt(passive_phase)
    alpha = rb.alpha_p_base * _expj(phi)                      # α_P = α_base·e^{jφ}
    M = torch.diag_embed(1.0 / alpha) - rb.Cpp                # M = diag(1/α_P) − Cpp
    lu, piv = torch.linalg.lu_factor(M)
    p_bg = torch.linalg.lu_solve(lu, piv, rb.Ep)              # pP_bg = M⁻¹ Ep
    if rb.mode == "full":
        return rb.H + rb.Hrp @ p_bg                           # H + Hrp pP
    p_o = rb.Lam @ (rb.Eo + rb.Cop @ p_bg)                    # pO = Lam (Eo + Cop pP_bg)
    p_p = p_bg + torch.linalg.lu_solve(lu, piv, rb.Cpo @ p_o)  # pP = pP_bg + M⁻¹ Cpo pO
    return rb.H + rb.Hrp @ p_p + rb.Hro @ p_o


def eval_reduced_torch(rs, passive_phase: torch.Tensor | None) -> torch.Tensor:
    """与 solver.eval_reduced 完全同公式（DESIGN §2.3）；rs 为单个 ReducedSample，返回 (N_RCV,S)。"""
    return eval_reduced_batch(stack_reduced([rs]), passive_phase)[0]


# ============================================================ §3.1 PA
class TorchPA:
    """Rapp AM-AM + Saleh AM-PM（方案 §4.4，DESIGN §3.1），或线性 PA。

    A_s = sqrt(Psat_out)，A_in = A_s/G_v，r = |u|/A_in；
    |g| = G_v|u| / (1 + r^{2p})^{1/(2p)}，∠g = ∠u + Φ(r)，Φ = α_φ r²/(1+β_φ r²)，α_φ = deg2rad(ampm_max_deg)·β_φ。
    实现写成 g(u) = u·f(r²)，f = G_v (1+r^{2p})^{−1/(2p)} e^{jΦ}，避免 ∠u 在 u=0 的奇点。
    """

    def __init__(self, kind: str = "rapp", gain_db: float = DEFAULTS["pa_gain_db"],
                 psat_out_dbm: float = DEFAULTS["psat_out_dbm"], p: float = DEFAULTS["rapp_p"],
                 ampm_max_deg: float = DEFAULTS["ampm_max_deg"], ampm_beta: float = DEFAULTS["ampm_beta"],
                 ampm: bool = True):
        if kind not in ("rapp", "linear"):
            raise ValueError(f"未知 PA 类型 {kind!r}")
        self.kind = kind
        self.gain_db = float(gain_db)
        self.gain_v = 10.0 ** (self.gain_db / 20.0)
        self.psat_out_dbm = float(psat_out_dbm)
        self.a_s = math.sqrt(10.0 ** ((self.psat_out_dbm - 30.0) / 10.0))
        self.a_in = self.a_s / self.gain_v
        self.p = float(p)
        self.ampm_max_deg = float(ampm_max_deg)
        self.ampm = bool(ampm)
        self.beta_phi = float(ampm_beta)
        self.alpha_phi = math.radians(self.ampm_max_deg) * self.beta_phi if ampm else 0.0

    def to_numpy(self) -> np_pa.PA:
        """同参数的生产 numpy PA（pa.RappPA / pa.LinearPA），供无梯度不动点求解。"""
        if self.kind == "linear":
            return np_pa.LinearPA(self.gain_db)
        return np_pa.RappPA(gain_db=self.gain_db, psat_out_dbm=self.psat_out_dbm, p=self.p,
                            ampm_max_deg=self.ampm_max_deg, ampm_beta=self.beta_phi, ampm=self.ampm)

    _KEYS = ("gain_db", "psat_out_dbm", "p", "ampm_max_deg", "ampm_beta", "ampm")

    @classmethod
    def from_dict(cls, d: dict | None) -> "TorchPA":
        d = dict(d or {})
        kind = d.pop("type", d.pop("kind", "rapp"))
        bad = sorted(set(d) - set(cls._KEYS))
        if bad:
            raise ValueError(f"PA 字典含未知键 {bad}，允许 'type' 与 {list(cls._KEYS)}")
        if kind == "linear":                       # 线性 PA 只用增益
            d = {k: v for k, v in d.items() if k == "gain_db"}
        return cls(kind=kind, **d)

    def p1db_in_w(self) -> float:
        """P1dB_in = A_in²·(10^{0.1p} − 1)^{1/p}（线性 PA 为 inf）。"""
        if self.kind == "linear":
            return math.inf
        return self.a_in ** 2 * (10.0 ** (0.1 * self.p) - 1.0) ** (1.0 / self.p)

    def _r2(self, u):
        return (u.real ** 2 + u.imag ** 2) / self.a_in ** 2

    def _f(self, r2):
        mag = self.gain_v * (1.0 + r2 ** self.p) ** (-1.0 / (2.0 * self.p))
        phi = self.alpha_phi * r2 / (1.0 + self.beta_phi * r2)
        return torch.polar(mag, phi)

    def g(self, u: torch.Tensor) -> torch.Tensor:
        """逐元素复传输 g(u)。"""
        if self.kind == "linear":
            return self.gain_v * u
        return u * self._f(self._r2(u))

    def wirtinger(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(a, b)：δg = a·δu + b·conj(δu)。

        g = u f(r²)，r² = |u|²/A_in² ⇒ a = f + r² f′，b = f′ u²/A_in²。
        记 c = r² f′/f = −r^{2p}/(2(1+r^{2p})) + j α_φ r²/(1+β_φ r²)²，则 a = f(1+c)，b = f·c·e^{2j∠u}
        （与 DESIGN §3.1 的 h(r) 写法等价；|u|<1e-30 时 e^{2j∠u} 取 1，此时 c→0）。
        """
        if self.kind == "linear":
            return torch.full_like(u, self.gain_v), torch.zeros_like(u)
        r2 = self._r2(u)
        f = self._f(r2)
        q = r2 ** self.p
        c = torch.complex(-q / (2.0 * (1.0 + q)), self.alpha_phi * r2 / (1.0 + self.beta_phi * r2) ** 2)
        abs2 = u.real ** 2 + u.imag ** 2
        ok = abs2 > 1e-60
        safe = torch.where(ok, abs2, torch.ones_like(abs2))
        ph2 = torch.where(ok, u * u / safe, torch.ones_like(u))
        return f * (1.0 + c), f * c * ph2

    # ---- numpy 孪生版（与上面 torch 版逐式相同；供诊断与测试，无梯度不动点求解用 to_numpy() 的生产 PA）
    def _f_np(self, r2):
        mag = self.gain_v * (1.0 + r2 ** self.p) ** (-1.0 / (2.0 * self.p))
        return mag * np.exp(1j * (self.alpha_phi * r2 / (1.0 + self.beta_phi * r2)))

    def g_np(self, u: np.ndarray) -> np.ndarray:
        if self.kind == "linear":
            return self.gain_v * u
        return u * self._f_np((u.real ** 2 + u.imag ** 2) / self.a_in ** 2)

    def wirtinger_np(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "linear":
            return np.full(u.shape, self.gain_v, dtype=np.complex128), np.zeros(u.shape, dtype=np.complex128)
        abs2 = u.real ** 2 + u.imag ** 2
        r2 = abs2 / self.a_in ** 2
        f = self._f_np(r2)
        q = r2 ** self.p
        c = -q / (2.0 * (1.0 + q)) + 1j * (self.alpha_phi * r2 / (1.0 + self.beta_phi * r2) ** 2)
        ok = abs2 > 1e-60
        ph2 = np.ones(u.shape, dtype=np.complex128)
        ph2[ok] = u[ok] ** 2 / abs2[ok]
        return f * (1.0 + c), f * c * ph2


def make_pa(pa: dict | None) -> TorchPA:
    """由字典 {'type','gain_db','psat_out_dbm','p','ampm_max_deg','ampm_beta','ampm'} 构造 PA。"""
    return TorchPA.from_dict(pa)


# ============================================================ §3.2–3.3 中继线性量
def relay_weights_torch(phase_rx: torch.Tensor, phase_tx: torch.Tensor, loss_rx_db: float, loss_tx_db: float,
                        phase_err_rx=None, phase_err_tx=None) -> tuple[torch.Tensor, torch.Tensor]:
    """w = 10^{−loss_rx/20}/√N · e^{j(φ_rx+err_rx)}，ct 同理（DESIGN §3.2），N = 每块单元数。"""
    phase_rx, phase_tx = _rt(phase_rx), _rt(phase_tx)
    n = phase_rx.shape[-1]
    if phase_err_rx is not None:
        phase_rx = phase_rx + _rt(phase_err_rx)
    if phase_err_tx is not None:
        phase_tx = phase_tx + _rt(phase_err_tx)
    w = (10.0 ** (-loss_rx_db / 20.0) / math.sqrt(n)) * _expj(phase_rx)
    ct = (10.0 ** (-loss_tx_db / 20.0) / math.sqrt(n)) * _expj(phase_tx)
    return w, ct


def relay_linear_torch(H: torch.Tensor, w: torch.Tensor, ct: torch.Tensor, s_feed, kappa0: complex,
                       n_probes: int | None = None):
    """中继线性量（DESIGN §3.3，方案 §2）。H: (B,N_RCV,S)；w, ct: (R,N)；s_feed: 标量或 (B,)。

    u_in[b,r]  = s·Σ_n w[r,n] H[b, rx(r,n), 0]
    K[b,r,s′]  = Σ_{n,m} w[r,n] H[b, rx(r,n), 1+tx(s′,m)] ct[s′,m] + κ0 δ_{rs′}
    d[b,k]     = s·H[b, NRX+k, 0]
    C[b,k,s′]  = Σ_m H[b, NRX+k, 1+tx(s′,m)] ct[s′,m]
    返回 (u_in (B,R), K (B,R,R), d (B,K), C (B,K,R))。
    """
    B = H.shape[0]
    R, n = w.shape
    nrx = R * n
    k_pr = H.shape[1] - nrx if n_probes is None else int(n_probes)
    s = _rt(s_feed)
    s = s.reshape(-1, 1) if s.ndim > 0 else s
    h_rf = H[:, :nrx, 0].reshape(B, R, n)
    u_in = s * torch.einsum("rn,brn->br", w, h_rf)
    h_rt = H[:, :nrx, 1:1 + nrx].reshape(B, R, n, R, n)
    K = torch.einsum("rn,brnsm,sm->brs", w, h_rt, ct)
    K = K + complex(kappa0) * torch.eye(R, dtype=CDT)
    d = s * H[:, nrx:nrx + k_pr, 0]
    h_pt = H[:, nrx:nrx + k_pr, 1:1 + nrx].reshape(B, k_pr, R, n)
    C = torch.einsum("bksm,sm->bks", h_pt, ct)
    return u_in, K, d, C


# ============================================================ §3.4 不动点
def real_jacobian(K: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """F(u) = u_in + K g(u) 的 2R×2R 实雅可比（DESIGN §3.4），顺序 [Re u; Im u]。

    δF = P δu + Q conj(δu)，P = K diag(a)，Q = K diag(b) ⇒ J = [[Re(P+Q), −Im(P−Q)], [Im(P+Q), Re(P−Q)]]。
    """
    P = K * a.unsqueeze(-2)
    Q = K * b.unsqueeze(-2)
    top = torch.cat([(P + Q).real, -(P - Q).imag], dim=-1)
    bot = torch.cat([(P + Q).imag, (P - Q).real], dim=-1)
    return torch.cat([top, bot], dim=-2)


def spectral_radius(M: torch.Tensor) -> torch.Tensor:
    """批量谱半径 max|eig(M)|，M: (B,n,n)。"""
    return torch.linalg.eigvals(M).abs().amax(dim=-1)


def _F(u, u_in, K, pa: TorchPA):
    """F(u) = u_in + K g(u)（torch，可微）。"""
    return u_in + (K @ pa.g(u).unsqueeze(-1)).squeeze(-1)


def _solve_lin(A: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """批量解 A x = rhs（奇异时退化为伪逆）。"""
    try:
        return torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        return (torch.linalg.pinv(A) @ rhs.unsqueeze(-1)).squeeze(-1)


@dataclass
class FixedPointTorch:
    u: torch.Tensor                 # (B,R) 可微（收敛样本带隐函数梯度；未收敛样本 = 最后迭代值，常量）
    g: torch.Tensor                 # (B,R) = g(u)
    converged: np.ndarray           # (B,) bool（relay.solve_fixed_point 判定）
    iters: np.ndarray               # (B,) int
    rel_residual: np.ndarray        # (B,) ‖F(u*)−u*‖/(‖u_in‖+1e-30)
    loop_gain_ss: np.ndarray        # (B,) ρ(G_v K)
    loop_gain_op: np.ndarray        # (B,) 工作点实雅可比谱半径
    possibly_oscillating: np.ndarray  # (B,) bool = ~converged | (loop_gain_op ≥ 1)（与 relay 逐样本相同）
    J: torch.Tensor | None          # (B,2R,2R) 工作点实雅可比（detach）；线性 PA 为 None


def fixed_point_torch(u_in: torch.Tensor, K: torch.Tensor, pa: TorchPA, tol: float = 1e-12,
                      max_iter: int = 3000, beta0: float = 0.5, criterion: str = "cw") -> FixedPointTorch:
    """可微不动点 u = u_in + K g(u)（DESIGN §3.4，方案 §4.5）。

    无梯度部分（u*、收敛、ρ_ss、ρ_op、可能自激标志）一律取自生产求解器 relay.solve_fixed_point，
    保证与 numpy 前向逐样本选中同一吸引子（审查 TR-2）；criterion 为自激标记判据（relay.OSC_CRITERIA，审查 S0-0）。
    - 线性 PA：u = (I − G_v K)⁻¹ u_in 直接可微（标志仍取 relay：谐振奇异样本 converged=False）。
    - Rapp：收敛样本 û = u* + (c − c.detach())，c = (I − J)⁻¹(F(u*,θ) − u*) → 值为 u*、梯度为精确隐函数梯度；
      未收敛样本 û = u*（最后迭代值，不做 Newton 外推，审查 TR-1）。
    """
    B, R = u_in.shape
    ref = np_relay.solve_fixed_point(u_in.detach().numpy(), K.detach().numpy(), pa.to_numpy(),
                                     tol=tol, max_iter=max_iter, beta0=beta0, criterion=criterion)
    conv = np.asarray(ref.converged, dtype=bool).reshape(B)
    scale = np.linalg.norm(u_in.detach().numpy(), axis=-1) + 1e-30
    resid = np.full(B, np.nan) if ref.residual is None else np.asarray(ref.residual, dtype=float).reshape(B)
    common = dict(converged=conv, iters=np.asarray(ref.iters).reshape(B), rel_residual=resid / scale,
                  loop_gain_ss=np.asarray(ref.loop_gain_ss, dtype=float).reshape(B),
                  loop_gain_op=np.asarray(ref.loop_gain_op, dtype=float).reshape(B),
                  possibly_oscillating=np.asarray(ref.possibly_oscillating, dtype=bool).reshape(B))
    if pa.kind == "linear":
        A = torch.eye(R, dtype=CDT) - pa.gain_v * K
        u = _solve_lin(A, u_in)
        return FixedPointTorch(u=u, g=pa.g(u), J=None, **common)
    u_star = torch.as_tensor(np.asarray(ref.u, dtype=np.complex128).reshape(B, R))
    a, b = pa.wirtinger(u_star)
    J = real_jacobian(K.detach(), a, b)
    u = u_star
    idx = np.nonzero(conv)[0]
    if idx.size:
        it = torch.as_tensor(idx)
        us = u_star[it]
        res = _F(us, u_in[it], K[it], pa) - us                  # 依赖 θ（经 u_in、K），值 ≈ 0
        c = from_real(_solve_lin(torch.eye(2 * R, dtype=RDT) - J[it], to_real(res)))
        u = u_star.index_put((it,), us + (c - c.detach()))       # 只对收敛样本接入隐函数梯度
    return FixedPointTorch(u=u, g=pa.g(u), J=J, **common)


# ============================================================ 系统
_RELAY_KEYS = ("loss_rx_db", "loss_tx_db", "isolation_db", "kappa0_phase_deg", "phase_err_rx", "phase_err_tx")


class TorchSystem:
    """可微物理系统：样本 ReducedSample → (无源面) → 中继线性量 → PA 不动点 → 探头读数 → 任务损失。

    kind='relay'：可训练 'phase_rx' (R,N)、'phase_tx' (R,N)（若 n_p>0 另有 'phase_p'）。
    kind='passive'（M0）：可训练 'phase_p' (N_P,)；探头场 z = s_feed·√power_scale·H_total[N_RX+k, 0]。
    读数 = 10·log10(|z|² [W]) + 30（dBm，无噪声）。
    损失（与 numpy PhysicalSystem.loss(noisy=False) 同定义，INTEGRATION C；另加 DESIGN §6.3 的光滑自激惩罚）：
      L = mean_b ℓ_b                                               逐样本任务损失（losses.task_loss, reduction='none'）
          ℓ_b ← chance + osc_flag_penalty   若样本 b 被标记 possibly_oscillating（常量，无梯度；
                chance = losses.chance_loss：零裕量/等读数时的损失，二分类为 τ·log2；focus 无定义 → 保留原值但切断梯度）
        + null_weight·softplus(max_r nd_dB,r − target_dB)          null_ref 给出时（N_r>1 取各中继最大值）；
              nd_dB,r = 10log10(|u0_r|²/u0_ref_abs2)，u0 = 名义 s_feed 下空场景 H0 的 u_in，不含 power_scale
        + osc_penalty_weight·mean_b relu(ρ_b − 0.95)²              DESIGN §6.3；ρ_b 收敛样本取 ρ_op，
              未收敛样本取小信号 ρ(G_v K)（其读数已切断梯度）。所有惩罚按样本取平均，强度不随 B 变化（审查 TR-6）。
    pa 字典键：'type'('rapp'|'linear'), 'gain_db', 'psat_out_dbm', 'p', 'ampm_max_deg', 'ampm_beta', 'ampm'（未知键报错）；
    relay 字典键：'loss_rx_db', 'loss_tx_db', 'isolation_db', 'kappa0_phase_deg', 'phase_err_rx', 'phase_err_tx'
    （其余键如 nf_db 等与无噪声前向无关，忽略）。
    """

    def __init__(self, reduced: list, *, kind: str, s_feed: float, n_relays: int = 1, n_elem: int = 25,
                 pa: dict | None = None, relay: dict | None = None, labels: np.ndarray, task: str,
                 tau_db: float = 1.0, n_probes: int = 2, power_scale: np.ndarray | None = None,
                 null_ref: dict | None = None, osc_penalty_weight: float = 100.0,
                 osc_flag_penalty: float = 1.0, task_kwargs: dict | None = None, osc_criterion: str = "cw"):
        if kind not in ("relay", "passive"):
            raise ValueError(f"kind 只能是 'relay' 或 'passive'，得到 {kind!r}")
        self.rb = stack_reduced(list(reduced))
        self.B = len(self.rb)
        self.kind = kind
        self.s_feed = float(s_feed)
        self.n_relays = int(n_relays)
        self.n_elem = int(n_elem)
        self.n_rx = self.rb.n_rx
        self.n_p = self.rb.n_p
        self.n_probes = int(n_probes)
        if self.n_probes > self.rb.n_probe:
            raise ValueError("n_probes 超过样本中的探头数")
        if kind == "relay":
            nr = self.n_relays * self.n_elem
            if self.rb.n_rx != nr or self.rb.n_tx != nr:
                raise ValueError(f"中继单元数不符：n_rx={self.rb.n_rx}, n_tx={self.rb.n_tx}, 期望 {nr}")
        self.pa_cfg = dict(pa or {})
        self.pa = make_pa(self.pa_cfg)
        rc = {"loss_rx_db": DEFAULTS["loss_rx_db"], "loss_tx_db": DEFAULTS["loss_tx_db"],
              "isolation_db": DEFAULTS["isolation_db"], "kappa0_phase_deg": DEFAULTS["kappa0_phase_deg"],
              "phase_err_rx": None, "phase_err_tx": None}
        for k, v in (relay or {}).items():
            if k in _RELAY_KEYS:
                rc[k] = v
        self.relay_cfg = rc
        # 与 relay.RelayParams.kappa0 一致：端口隔离 × 收发网络损耗（DESIGN §3.3）
        self.kappa0 = (10.0 ** (float(rc["isolation_db"]) / 20.0) * np.exp(1j * np.deg2rad(rc["kappa0_phase_deg"]))
                       * 10.0 ** (-(float(rc["loss_rx_db"]) + float(rc["loss_tx_db"])) / 20.0))
        self.labels = np.asarray(labels)
        self.task = task
        self.tau = float(tau_db)
        self.task_kwargs = dict(task_kwargs or {})      # 例如 loc3bit 的 thresholds_db / loss_weight
        ps = np.ones(self.B) if power_scale is None else np.asarray(power_scale, dtype=float).ravel()
        if ps.size != self.B:
            raise ValueError("power_scale 长度须等于样本数")
        self.s_b = _rt(self.s_feed * np.sqrt(ps))                     # (B,) 每样本源幅度 √W
        self.null_ref = None
        if null_ref is not None and kind == "relay":                  # 无源面没有接收波束，零陷惩罚不适用
            self.null_ref = {"H0": _ct(null_ref["H0"]), "u0_ref_abs2": float(null_ref["u0_ref_abs2"]),
                             "target_db": float(null_ref.get("target_db", -20.0)),
                             "weight": float(null_ref.get("weight", 1.0))}
        self.osc_penalty_weight = float(osc_penalty_weight)
        self.osc_flag_penalty = float(osc_flag_penalty)   # INTEGRATION C 的 osc_penalty（被标记样本的附加损失）
        if osc_criterion not in np_relay.OSC_CRITERIA:
            raise ValueError(f"osc_criterion 只能是 {np_relay.OSC_CRITERIA}")
        self.osc_criterion = osc_criterion                # 自激标记判据（与 SystemConfig.osc_criterion 一致，审查 S0-0）
        # 不动点数值参数（可在构造后修改）
        self.fp_tol = 1e-12
        self.fp_max_iter = 3000
        self.fp_beta0 = 0.5
        shapes = {}
        if kind == "relay":
            shapes["phase_rx"] = (self.n_relays, self.n_elem)
            shapes["phase_tx"] = (self.n_relays, self.n_elem)
        if self.n_p > 0:
            shapes["phase_p"] = (self.n_p,)
        self.param_shapes = shapes

    # ---------------------------------------------------------------- 参数
    def init_phases(self, rng_or_seed=None) -> dict[str, torch.Tensor]:
        """均匀随机相位 U[0,2π)，按 param_shapes 的顺序抽取。"""
        rng = rng_or_seed if isinstance(rng_or_seed, np.random.Generator) else np.random.default_rng(rng_or_seed)
        return {k: torch.as_tensor(rng.uniform(0.0, TWO_PI, size=shp), dtype=RDT)
                for k, shp in self.param_shapes.items()}

    def as_phases(self, phases: dict) -> dict[str, torch.Tensor]:
        """numpy/torch 混合字典 → float64 张量字典（只保留本系统需要的键）。"""
        return {k: _rt(phases[k]) for k in self.param_shapes}

    # ---------------------------------------------------------------- 前向
    def weights(self, phases):
        rc = self.relay_cfg
        return relay_weights_torch(phases["phase_rx"], phases["phase_tx"], float(rc["loss_rx_db"]),
                                   float(rc["loss_tx_db"]), rc["phase_err_rx"], rc["phase_err_tx"])

    def H_total(self, phases) -> torch.Tensor:
        return eval_reduced_batch(self.rb, phases.get("phase_p") if self.n_p > 0 else None)

    @staticmethod
    def _dbm(z: torch.Tensor) -> torch.Tensor:
        p_w = torch.clamp(z.real ** 2 + z.imag ** 2, min=1e-300)
        return 10.0 * torch.log10(p_w) + 30.0

    def forward(self, phases: dict) -> dict:
        """完整前向，返回 z、readings_db、flags (B,) bool 以及中继中间量（u_in, K, d, C, fp, w, ct）。

        两种 kind 都返回 'flags'（无源面恒为 False）与 'fp'（无源面为 None），接口统一（审查 TR-7）。
        """
        H = self.H_total(phases)
        nrx, kp = self.n_rx, self.n_probes
        if self.kind == "passive":
            z = self.s_b.unsqueeze(-1) * H[:, nrx:nrx + kp, 0]
            return {"z": z, "readings_db": self._dbm(z), "H": H, "fp": None, "flags": np.zeros(self.B, dtype=bool)}
        w, ct = self.weights(phases)
        u_in, K, d, C = relay_linear_torch(H, w, ct, self.s_b, self.kappa0, n_probes=kp)
        fp = fixed_point_torch(u_in, K, self.pa, tol=self.fp_tol, max_iter=self.fp_max_iter, beta0=self.fp_beta0,
                               criterion=self.osc_criterion)
        z = d + (C @ fp.g.unsqueeze(-1)).squeeze(-1)                 # z = d + C g(u)
        if not fp.converged.all():
            # 取舍：不动点未收敛（CW 稳态不存在/可能自激）的样本，读数取最后迭代值 g(u*)（与 relay 相同），
            # 但切断梯度；这些样本只经自激惩罚（小信号环路增益）提供训练信号。
            bad = torch.as_tensor(~fp.converged).unsqueeze(-1)
            z = torch.where(bad, z.detach(), z)
        return {"z": z, "readings_db": self._dbm(z), "H": H, "w": w, "ct": ct,
                "u_in": u_in, "K": K, "d": d, "C": C, "fp": fp, "flags": np.asarray(fp.possibly_oscillating)}

    def readings_db(self, phases: dict) -> torch.Tensor:
        """无噪声读数 (B,K) dBm。"""
        return self.forward(phases)["readings_db"]

    # ---------------------------------------------------------------- 损失
    def _null_penalty(self, w: torch.Tensor) -> torch.Tensor:
        nr = self.null_ref
        R, n = w.shape
        h0 = nr["H0"][: R * n, 0].reshape(R, n)
        u0 = self.s_feed * (w * h0).sum(dim=-1)                      # u0_r = s Σ_n w[r,n] H0[rx(r,n),0]
        nd_db = 10.0 * torch.log10(torch.clamp(u0.real ** 2 + u0.imag ** 2, min=1e-300) / nr["u0_ref_abs2"])
        # INTEGRATION C：N_r>1 取各中继零陷深度的最大值（最差中继），再过一次 softplus
        return nr["weight"] * _softplus(nd_db.amax() - nr["target_db"])

    def _osc_penalty(self, out: dict) -> torch.Tensor:
        """weight·mean_b relu(ρ_b − 0.95)²（DESIGN §6.3；按样本平均，与任务损失同尺度）。

        ρ_b：不动点收敛的样本取工作点环路增益 ρ_op（2R×2R 实雅可比谱半径）；未收敛的样本（工作点无意义）
        取小信号环路增益 ρ(G_v K)（取舍，见 deviations）。只对 ρ ≥ 0.95 的样本带梯度重算谱半径
        （避免简并特征值处 eig 反传的 NaN）。
        """
        fp = out["fp"]
        if self.osc_penalty_weight == 0.0:
            return torch.zeros((), dtype=RDT)
        conv = np.asarray(fp.converged, dtype=bool)
        m_op = conv & (np.nan_to_num(fp.loop_gain_op, nan=np.inf) >= 0.95)
        m_ss = ~conv & (np.nan_to_num(fp.loop_gain_ss, nan=np.inf) >= 0.95)
        rhos = []
        if m_op.any():
            idx = torch.as_tensor(np.nonzero(m_op)[0])
            K = out["K"][idx]
            if self.pa.kind == "linear":
                rhos.append(spectral_radius(self.pa.gain_v * K))
            else:
                a, b = self.pa.wirtinger(fp.u[idx])
                rhos.append(spectral_radius(real_jacobian(K, a, b)))
        if m_ss.any():
            idx = torch.as_tensor(np.nonzero(m_ss)[0])
            rhos.append(spectral_radius(self.pa.gain_v * out["K"][idx]))
        if not rhos:
            return torch.zeros((), dtype=RDT)
        rho = torch.cat(rhos)
        return self.osc_penalty_weight * (torch.relu(rho - 0.95) ** 2).sum() / self.B

    def sample_losses(self, out: dict) -> torch.Tensor:
        """逐样本任务损失 (B,)；被标记可能自激的样本替换为 chance + osc_flag_penalty（INTEGRATION C，常量无梯度）。"""
        per = losses.task_loss(out["readings_db"], self.labels, self.task, self.tau, reduction="none",
                               **self.task_kwargs)
        flags = np.asarray(out["flags"], dtype=bool)
        if flags.any():
            ch = losses.chance_loss(self.task, self.n_probes, self.tau, **self.task_kwargs)
            base = per.detach() if ch is None else torch.full_like(per, ch)
            per = torch.where(torch.as_tensor(flags), base + self.osc_flag_penalty, per)
        return per

    def loss(self, phases: dict) -> torch.Tensor:
        out = self.forward(phases)
        total = self.sample_losses(out).mean()
        if self.kind == "relay":
            if self.null_ref is not None:
                total = total + self._null_penalty(out["w"])
            total = total + self._osc_penalty(out)
        return total


# ============================================================ oracle
def oracle_continuous(system, n_restarts: int = 8, steps: int = 1500, lr: float = 0.05, seed: int = 0,
                      init: dict | None = None) -> dict:
    """连续相位 oracle：Adam 多次随机重启，保留全程最优（DESIGN §6.3）。

    system：TorchSystem，或带 to_torch() 的 numpy 系统（PhysicalSystem）。后者若还有 evaluate_phases(phases, rng)，
            则按 DESIGN §6.3 用 numpy 系统对最优连续相位复评，结果放在 'eval'，并给出 'accuracy'（= eval['acc_clean']）
            与 'margin_db'。（INTEGRATION D 的 train.run_oracle_continuous 可传 system.to_torch() 自行复评，避免重复。）
    init：可选初始相位字典（numpy/torch），用于第 0 次重启；缺失的键随机初始化。
    返回 {'best_phases'/'phases': {键: numpy 数组 mod 2π}, 'best_loss'/'loss': float,
          'history': 每次重启的最终损失 list, 'restart_best': 每次重启内的最优损失 list[, 'eval', 'accuracy', 'margin_db']}。
    """
    ts = system if isinstance(system, TorchSystem) else system.to_torch()
    res = _optimize(ts, n_restarts, steps, lr, seed, init)
    res["best_phases"], res["best_loss"] = res["phases"], res["loss"]
    if ts is not system and hasattr(system, "evaluate_phases"):
        ev = system.evaluate_phases(res["phases"], np.random.default_rng(seed))
        res["eval"] = ev
        res["accuracy"] = ev.get("acc_clean") if isinstance(ev, dict) else None
        res["margin_db"] = ev.get("margin_db") if isinstance(ev, dict) else None
    return res


def _optimize(ts: TorchSystem, n_restarts, steps, lr, seed, init) -> dict:
    """Adam 多重启主循环；返回 {'phases', 'loss', 'history', 'restart_best'}。"""
    rng = np.random.default_rng(seed)
    if not ts.param_shapes:                        # 无可训练参数：只评估一次
        with torch.no_grad():
            lv = float(ts.loss({}))
        return {"phases": {}, "loss": lv, "history": [lv], "restart_best": [lv]}
    best_loss, best_ph = math.inf, None
    hist, rbest = [], []
    for r in range(max(int(n_restarts), 1)):
        ph = ts.init_phases(rng)
        if r == 0 and init is not None:
            for k in ph:
                if k in init:
                    ph[k] = _rt(init[k]).clone().reshape(ts.param_shapes[k])
        for v in ph.values():
            v.requires_grad_(True)
        opt = torch.optim.Adam(list(ph.values()), lr=lr)
        r_best = math.inf
        for _ in range(int(steps)):
            opt.zero_grad()
            L = ts.loss(ph)
            lv = float(L.detach())
            if not math.isfinite(lv):
                break
            if lv < r_best:
                r_best = lv
            if lv < best_loss:
                best_loss, best_ph = lv, {k: v.detach().clone() for k, v in ph.items()}
            if not L.requires_grad:                # 全部样本不收敛且无惩罚梯度：无从下降
                break
            L.backward()
            for v in ph.values():                  # 防御：个别 nan/inf 梯度分量置 0，避免相位被污染
                if v.grad is not None:
                    torch.nan_to_num_(v.grad, nan=0.0, posinf=0.0, neginf=0.0)
            opt.step()
        with torch.no_grad():
            lf = float(ts.loss(ph))
        if math.isfinite(lf):
            r_best = min(r_best, lf)
            if lf < best_loss:
                best_loss, best_ph = lf, {k: v.detach().clone() for k, v in ph.items()}
        hist.append(lf)
        rbest.append(r_best)
    if best_ph is None:
        best_ph = {k: v.detach().clone() for k, v in ph.items()}
    phases = {k: np.mod(v.numpy().astype(np.float64), TWO_PI) for k, v in best_ph.items()}
    return {"phases": phases, "loss": float(best_loss), "history": hist, "restart_best": rbest}
