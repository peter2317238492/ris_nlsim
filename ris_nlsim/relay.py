"""中继：电参数、相位权重、线性量 (u_in, K, d, C) 与 PA 不动点。

对应方案 §2（u、κ、d、c 的信号模型）、§4.3–4.5；实现契约 DESIGN §3.2–3.4。
本模块只处理原始响应矩阵 H，不依赖 scene / solver。

批量约定：所有函数接受任意前导批量维 (...)；DESIGN 中写作 (B, ·)。
无批量维的输入（如 H 为 (N_RCV,S)、u_in 为 (R,)）返回同样无批量维的结果。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DEFAULTS, KB, T0
from .pa import PA

_EPS = np.finfo(float).eps
_BETA_MIN = 1e-3          # 阻尼下限（DESIGN §3.4）
_GROW_STREAK = 10         # 连续有效步数（下降或增长型）达到它 → β 放大 1.5 倍
_NEWTON_SWITCH = 1e-6     # newton=True 时阻尼迭代到此相对残差即交给 Newton（提速，最终判据不变）
_NEWTON_GATE = 1e-4       # 只对残差 ≤ 1e-4·(‖u_in+n‖+‖u‖) 的样本做 Newton 精修
_NEWTON_MAX = 30          # Newton 最大步数
_LS_MAX = 12              # Newton 回溯线搜索最多折半次数
_FLOOR_ULP = 64.0         # 浮点地板：残差 ≤ 64·eps·(‖u_in+n‖+‖u‖) 也视为收敛
# 谐振/临界容差：σ_min(I − G_v K) ≤ 1e-9·(1+‖G_v K‖) 视为奇异（谐振，增强 > 180 dB，输入数据的舍入误差
# 已足以改变 1−ρ 的符号）；环路增益 ρ ≥ 1 − 1e-9 视为"≥ 1"（与 1 在数值上不可区分）。
_RES_RTOL = 1e-9
LOOP_GAIN_TOL = _RES_RTOL  # 公开：环路增益 ρ ≥ 1 − LOOP_GAIN_TOL 视为"≥ 1"（solve_fixed_point / system 共用）
# 自激判据（审查 S0-0；方案 §4.5 只规定 CW 稳态近似）：
#   "cw"（默认，DESIGN §3.4）：~converged | ρ_op ≥ 1 —— 只检验注入频率上的 CW 工作点；
#   "robust"：再加 ρ_rob ≥ 1 —— 对未知泄漏相位 / 带内其他频率的扰动也稳定（见 loop_gain_robust）；
#   "small_signal"：再加 ρ_ss = ρ(G_v K) ≥ 1 —— 无驱动（源关闭）时就会起振，最保守。
OSC_CRITERIA = ("cw", "robust", "small_signal")
# 停滞早停（逐样本，纯性能措施，非 DESIGN 条款）：每 _STALL_WINDOW 步做一次检查，用两条"纪录"判断：
#   起振：‖u‖ 创新高，超过此前检查点最高值的 _STALL_UGROW 倍（1.05/200 步 ⇔ 每步 > 2.4e-4；更慢的起振在
#         max_iter=3000 内本来也到不了饱和）。起振时残差纪录从本窗口重新起算——起振中残差先升后降，
#         回落过程必须算进展（审查 R2-RELAY-5 根因 (i)：原实现与全程最优残差比较，回落永远不算进展；
#         根因 (ii)：原 1.5 倍门限会截断 |Gκ|−1 < 0.002 的正常起振）。
#   进展：本窗口最小残差比纪录低点（自上次起振以来）再低 _STALL_GAIN 倍。
#   连续 _STALL_PATIENCE 个窗口既无进展又不在起振 → 判为停滞（无稳定 CW 稳态/极限环）→ 不再迭代。
#   用"纪录"而不是"与上一窗口比"：周期与窗口可公度的极限环（‖u‖、残差隔窗交替）不会被反复误判为进展/起振；
#   耐心 3 个窗口：AM-PM 下注入锁定的捕获过程中 ‖u‖ 与残差会来回摆动一两个窗口后才锁定。
# 停滞或耗尽预算仍未收敛的样本，newton=True 时再做一次救援 Newton（_newton_rescue，只接受稳定根）。
# 实测见 _scratch/fixer-relay/r2/（variants2.py / realistic.py / round1set.py），以"不早停 + 同样救援"为规格：
#   3441 个对抗样本（审查者近临界网格、AM-PM 近临界 R=1/2、弱注入随机族）标记翻转 0（原实现 36）；
#   第一轮 11520 个随机样本翻转 1（不早停时也要耗满 3000 步才收敛的样本）；P1dB 附近强注入的系统型样本
#   翻转 0，未收敛样本中位 1000 步停止（原 3000 步），总迭代步数约减半。
_STALL_WINDOW = 200
_STALL_GAIN = 1.1
_STALL_UGROW = 1.05
_STALL_PATIENCE = 3


# ---------------------------------------------------------------- §3.2 电参数与权重
@dataclass
class RelayParams:
    """中继电参数（DESIGN §3.2；默认值见 constants.DEFAULTS / DESIGN §8）。"""

    n_relays: int = 1
    n_elem: int = 25
    loss_rx_db: float = DEFAULTS["loss_rx_db"]       # 合路网络损耗
    loss_tx_db: float = DEFAULTS["loss_tx_db"]       # 功分网络损耗
    isolation_db: float = DEFAULTS["isolation_db"]   # 收发板端口隔离 |κ0|（方案 §4.3）
    # κ0 定义在收、发板的端口参考面（方案 §4.3 "收发板之间的直接耦合用隔离度 κ₀ 表示"）。PA 输出经功分网络
    # （loss_tx）到发射板端口，泄漏到接收板端口后再经合路网络（loss_rx）到 PA 输入，所以环路里出现的是
    # κ_eff = κ0·10^{−(loss_rx+loss_tx)/20}（审查 RELAY-1 的裁定）。默认 30 dB PA、−30 dB 隔离、各 2.5 dB 损耗
    # ⇒ 小信号环路增益 |G_v κ_eff| = −5 dB；方案 §5 的 −40/−25/−15 dB 扫描对应 −15/0/+10 dB。
    kappa0_phase_deg: float = DEFAULTS["kappa0_phase_deg"]   # 泄漏相位：未知硬件参数，默认 90°（不取最坏的同相 0°）
    phase_err_rx: np.ndarray | None = None           # (R,25) 弧度，固定通道相位误差
    phase_err_tx: np.ndarray | None = None
    nf_db: float = DEFAULTS["nf_db"]                 # PA 噪声系数
    noise_bw_hz: float = DEFAULTS["noise_bw_hz"]     # 等效噪声带宽
    temp_k: float = T0

    @property
    def kappa0_port(self) -> complex:
        """端口参考面的板间泄漏 κ0 = 10^{iso/20}·e^{jφ_κ}（即实测的收发隔离）。"""
        return complex(10.0 ** (self.isolation_db / 20.0) * np.exp(1j * np.deg2rad(self.kappa0_phase_deg)))

    @property
    def kappa0(self) -> complex:
        """环路里（PA 输出 → PA 输入）实际出现的泄漏 κ_eff = κ0_port·10^{−(loss_rx+loss_tx)/20}（DESIGN §3.3）。"""
        return self.kappa0_port * 10.0 ** (-(self.loss_rx_db + self.loss_tx_db) / 20.0)


def weights(phase_rx, phase_tx, params: RelayParams) -> tuple[np.ndarray, np.ndarray]:
    """接收合路权重 w 与发射功分系数 ct，均为 (R,N)（DESIGN §3.2）。

    w  = 10^{-loss_rx/20}/√N · e^{j(φ_rx + err_rx)}
    ct = 10^{-loss_tx/20}/√N · e^{j(φ_tx + err_tx)}
    （1/√N：无耗 N 路 Wilkinson 合路/功分的幅度系数）
    """
    phase_rx = np.asarray(phase_rx, dtype=float)
    phase_tx = np.asarray(phase_tx, dtype=float)
    n = params.n_elem
    shape = (params.n_relays, n)
    # 形状严格等于 (R,N)：避免 (1,N) 的误差/相位被静默广播到所有中继
    if phase_rx.shape != shape or phase_tx.shape != shape:
        raise ValueError(f"相位应为 (n_relays, n_elem) = {shape}，得到 {phase_rx.shape} / {phase_tx.shape}")
    err_rx = _phase_err(params.phase_err_rx, shape, "phase_err_rx")
    err_tx = _phase_err(params.phase_err_tx, shape, "phase_err_tx")
    amp_rx = 10.0 ** (-params.loss_rx_db / 20.0) / np.sqrt(n)
    amp_tx = 10.0 ** (-params.loss_tx_db / 20.0) / np.sqrt(n)
    w = amp_rx * np.exp(1j * (phase_rx + err_rx))
    ct = amp_tx * np.exp(1j * (phase_tx + err_tx))
    return w.astype(np.complex128), ct.astype(np.complex128)


def _phase_err(err, shape, name) -> np.ndarray | float:
    """通道相位误差：None → 0；否则形状必须恰为 (R,N)。"""
    if err is None:
        return 0.0
    err = np.asarray(err, dtype=float)
    if err.shape != shape:
        raise ValueError(f"{name} 应为 {shape}，得到 {err.shape}")
    return err


def bits_to_phase(bits, levels: int = 2) -> np.ndarray:
    """码字 → 相位（DESIGN §1）：φ = 2π·l/levels；1-bit 为 π·b，2-bit 为 (π/2)·l。"""
    if levels < 2:
        raise ValueError("levels 必须 >= 2")
    return (2.0 * np.pi / levels) * np.asarray(bits, dtype=float)


# ---------------------------------------------------------------- §3.3 线性量
@dataclass
class RelayLinear:
    """方案 §2 的线性量：u_in (B,R)、K (B,R,R)、d (B,K)、C (B,K,R)。"""

    u_in: np.ndarray
    K: np.ndarray
    d: np.ndarray
    C: np.ndarray


def relay_linear(H, w, ct, s_feed, params: RelayParams) -> RelayLinear:
    """由样本响应 H 计算中继线性量（DESIGN §3.3，方案 §2）。

    H: (N_RCV,S) 或 (...,N_RCV,S)；行 = [rx_0..rx_{RN-1}, probe_0..probe_{K-1}]，列 = [feed, tx_0..tx_{RN-1}]
    w, ct: (R,N)；s_feed: 标量或与批量维同形（每样本源幅度，√W）。
      u_in[b,r]   = s · Σ_n w[r,n] H[b, rx(r,n), 0]
      K[b,r,s']   = Σ_{n,m} w[r,n] H[b, rx(r,n), 1+tx(s',m)] ct[s',m] + κ0 δ_{rs'}
      d[b,k]      = s · H[b, NRX+k, 0]
      C[b,k,s']   = Σ_m H[b, NRX+k, 1+tx(s',m)] ct[s',m]
    R、N 取自 w 的形状（params.n_relays 仅作一致性检查）。
    """
    H = np.asarray(H, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)
    ct = np.asarray(ct, dtype=np.complex128)
    if w.ndim != 2 or w.shape != ct.shape:
        raise ValueError(f"w、ct 应为同形的 (R,N)，得到 {w.shape} / {ct.shape}")
    R, N = w.shape
    if params.n_relays != R:
        raise ValueError(f"params.n_relays={params.n_relays} 与权重的中继数 {R} 不一致")
    nrx = R * N
    if H.ndim < 2 or H.shape[-1] != 1 + nrx or H.shape[-2] < nrx:
        raise ValueError(f"H 形状 {H.shape} 与 R={R}, N={N} 不符（需要 S={1 + nrx}, N_RCV>={nrx}）")
    batch = H.shape[:-2]
    n_probe = H.shape[-2] - nrx
    s = np.asarray(s_feed)
    if s.ndim:
        s = np.broadcast_to(s, batch)[..., None]

    H_rx_feed = H[..., :nrx, 0].reshape(batch + (R, N))
    H_rx_tx = H[..., :nrx, 1:].reshape(batch + (R, N, R, N))
    H_pr_feed = H[..., nrx:, 0]
    H_pr_tx = H[..., nrx:, 1:].reshape(batch + (n_probe, R, N))

    u_in = s * np.einsum("rn,...rn->...r", w, H_rx_feed)
    K = np.einsum("rn,...rnsm,sm->...rs", w, H_rx_tx, ct) + params.kappa0 * np.eye(R)
    d = s * H_pr_feed
    C = np.einsum("...ksm,sm->...ks", H_pr_tx, ct)
    return RelayLinear(u_in=u_in, K=K, d=d, C=C)


# ---------------------------------------------------------------- §3.4 不动点
@dataclass
class FixedPointResult:
    """不动点解（DESIGN §3.4）。形状按批量维 (B,·)；residual 为附加诊断量 ‖F(u)−u‖。"""

    u: np.ndarray                     # (B,R) PA 输入
    g: np.ndarray                     # (B,R) PA 输出
    converged: np.ndarray             # (B,) bool
    iters: np.ndarray                 # (B,) int：阻尼迭代步数 + Newton 步数
    loop_gain_ss: np.ndarray          # (B,) 小信号环路增益 ρ(G_v K)
    loop_gain_op: np.ndarray          # (B,) 工作点实雅可比谱半径
    possibly_oscillating: np.ndarray  # (B,) bool = ~converged | (loop_gain_op >= 1)（criterion="cw"，见 OSC_CRITERIA）
    residual: np.ndarray | None = None
    loop_gain_robust: np.ndarray | None = None   # (B,) 相位稳健工作点环路增益上界（loop_gain_robust()）
    criterion: str = "cw"                        # possibly_oscillating 所用判据（OSC_CRITERIA）

    @property
    def ss_unstable(self) -> np.ndarray:
        """(B,) bool：小信号环路增益 ρ(G_v K) ≥ 1 − LOOP_GAIN_TOL（源关闭时会起振；审查 S0-0）。"""
        return gain_at_least_one(self.loop_gain_ss)

    @property
    def robust_unstable(self) -> np.ndarray:
        """(B,) bool：相位稳健环路增益 ≥ 1（某个泄漏相位 / 带内频率上可能起振）；未算时全 False。"""
        if self.loop_gain_robust is None:
            return np.zeros(np.shape(self.loop_gain_ss), dtype=bool)
        return gain_at_least_one(self.loop_gain_robust)


def gain_at_least_one(rho) -> np.ndarray:
    """ρ ≥ 1 − LOOP_GAIN_TOL（nan 也算，与 solve_fixed_point 的标记同一容差）。"""
    return ~(np.asarray(rho, dtype=float) < 1.0 - LOOP_GAIN_TOL)


def loop_gain_robust(u, K, pa: PA) -> np.ndarray:
    """相位稳健的工作点环路增益 ρ_rob = ‖K‖₂·max_r(|a_r| + |b_r|)，(a, b) = pa.wirtinger(u)（审查 S0-0）。

    CW 工作点 u* 处，频偏 Δ 的上、下边带扰动 (ε₊, ε̄₋) 一圈后为 diag(K(ω₀+Δ), K̄(ω₀−Δ))·[[a, b], [b̄, ā]]·(ε₊, ε̄₋)
    （R = 1；R > 1 时 a、b 为对角阵），其范数 ≤ ‖K‖·(|a|+|b|)。所以 ρ_rob < 1 ⇒ 只要带内 |K(ω)| ≤ ‖K‖，
    任何频率、任何泄漏相位上都不起振（小增益定理，充分条件）。R = 1 时它恰是"未知泄漏相位 φ_κ 取最坏值"时 CW 实雅可比
    谱半径的最大值（max_φ ρ(J_F(e^{jφ}K)) = |K|(|a|+|b|)），所以 ρ_op ≤ ρ_rob。
    线性 PA：a = G_v、b = 0 ⇒ ρ_rob = |G_v|·‖K‖₂ ≥ ρ_ss（R = 1 时相等）。深饱和 Rapp：|a|+|b| ≈ |g(u)|/|u|（大信号增益），
    强注入把 PA 压缩到大信号环路增益 < 1 时，即使 ρ_ss ≥ 1 也稳定（方案推论 4"PA 饱和把自激钳住"的强注入极限）。
    u: (...,R)，K: (...,R,R) → (...)。
    """
    u = np.asarray(u, dtype=np.complex128)
    K = np.asarray(K, dtype=np.complex128)
    if u.shape[-1] == 0:
        return np.zeros(u.shape[:-1])
    if getattr(pa, "is_linear", False):
        ab = np.full(u.shape, abs(complex(pa.gain_v)))
    else:
        a, b = pa.wirtinger(u)
        ab = np.abs(a) + np.abs(b)
    if K.shape[-1] == 1:
        kn = np.abs(K[..., 0, 0])
    else:
        kn = np.full(K.shape[:-2], np.nan)
        ok = np.isfinite(K).all(axis=(-2, -1))
        if ok.any():
            kn[ok] = np.linalg.norm(K[ok], ord=2, axis=(-2, -1))
    return kn * np.max(ab, axis=-1)


def real_jacobian(K, a, b) -> np.ndarray:
    """F(u) = u_in + K g(u) 的 2R×2R 实雅可比（方案 §4.5，DESIGN §3.4）。

    δF = A δu + B conj(δu)，A = K·diag(a)，B = K·diag(b)；实坐标 [Re u; Im u]：
      J = [[Re(A+B), −Im(A−B)], [Im(A+B), Re(A−B)]]
    K: (...,R,R)；a, b: (...,R) → (...,2R,2R)
    """
    K = np.asarray(K, dtype=np.complex128)
    A = K * np.asarray(a)[..., None, :]
    Bm = K * np.asarray(b)[..., None, :]
    P, M = A + Bm, A - Bm
    top = np.concatenate([P.real, -M.imag], axis=-1)
    bot = np.concatenate([P.imag, M.real], axis=-1)
    return np.concatenate([top, bot], axis=-2)


def loop_jacobian(u, K, pa: PA) -> np.ndarray:
    """在工作点 u 处组装实雅可比 J_F（供 oracle 隐函数求导等复用）。"""
    a, b = pa.wirtinger(u)
    return real_jacobian(K, a, b)


def _spectral_radius(M: np.ndarray) -> np.ndarray:
    """批量谱半径；含非有限元素的样本返回 nan。"""
    out = np.full(M.shape[:-2], np.nan)
    ok = np.isfinite(M).all(axis=(-2, -1))
    if ok.any():
        out[ok] = np.abs(np.linalg.eigvals(M[ok])).max(axis=-1)
    return out


def _near_resonant(GK: np.ndarray) -> np.ndarray:
    """批量判定 I − G_v K 数值奇异（谐振）：σ_min(I − GK) ≤ _RES_RTOL·(1 + ‖GK‖_F)。

    I − GK 奇异 ⇔ GK 有特征值 ≈ 1（同相正反馈，线性解 u → ∞）。尺度取被相减两项的量级 1 + ‖GK‖，
    而不是 σ_max(I − GK)：R = 1 时 σ_min ≡ σ_max，条件数判据永远看不出 1 − Gκ ≈ 1e-16。
    含非有限元素的样本也算奇异。
    """
    out = np.ones(GK.shape[:-2], dtype=bool)
    ok = np.isfinite(GK).all(axis=(-2, -1))
    if ok.any():
        G = GK[ok]
        smin = np.linalg.svd(np.eye(G.shape[-1]) - G, compute_uv=False)[..., -1]
        out[ok] = smin <= _RES_RTOL * (1.0 + np.linalg.norm(G, axis=(-2, -1)))
    return out


def _empty_result(batch, criterion: str = "cw") -> "FixedPointResult":
    """R = 0（无中继）时的平凡结果。"""
    z = np.zeros(batch + (0,), dtype=np.complex128)
    return FixedPointResult(u=z, g=z.copy(), converged=np.ones(batch, dtype=bool),
                            iters=np.zeros(batch, dtype=int), loop_gain_ss=np.zeros(batch),
                            loop_gain_op=np.zeros(batch), possibly_oscillating=np.zeros(batch, dtype=bool),
                            residual=np.zeros(batch), loop_gain_robust=np.zeros(batch), criterion=criterion)


def _batched_solve(M: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """批量解 M x = rhs（(B,n,n),(B,n)）；奇异样本返回 nan。"""
    try:
        return np.linalg.solve(M, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = np.full(rhs.shape, np.nan, dtype=np.result_type(M, rhs))
        for i in range(M.shape[0]):
            try:
                out[i] = np.linalg.solve(M[i], rhs[i])
            except np.linalg.LinAlgError:
                pass
        return out


class _FPProblem:
    """展平后的不动点问题 u = src + K g(u)，src = u_in + n，形状 (B,R)。"""

    def __init__(self, src: np.ndarray, K: np.ndarray, pa: PA):
        self.src, self.K, self.pa = src, K, pa
        self.src_norm = np.linalg.norm(src, axis=-1)

    def subset(self, idx) -> "_FPProblem":
        return _FPProblem(self.src[idx], self.K[idx], self.pa)

    def F(self, u, idx=None):
        src = self.src if idx is None else self.src[idx]
        K = self.K if idx is None else self.K[idx]
        return src + np.einsum("brs,bs->br", K, self.pa.g(u))

    def threshold(self, u, rtol, idx=None):
        """收敛门限：rtol·(‖u_in+n‖+1e-30)（DESIGN §3.4），并以浮点地板兜底（见 solve_fixed_point）。"""
        s = self.src_norm if idx is None else self.src_norm[idx]
        floor = _FLOOR_ULP * _EPS * (s + np.linalg.norm(u, axis=-1))
        return np.maximum(rtol * (s + 1e-30), floor)


def _damped_iterate(prob: _FPProblem, u, iters, max_iter, beta0, rtol, stalled=None):
    """逐样本自适应阻尼迭代 u ← (1−β)u + βF(u) = u + β r，r = F(u) − u（方案 §4.5，DESIGN §3.4）。

    - 残差下降：接受；连续 10 个有效步（下降或下述增长型）→ β = min(1, 1.5β)。
    - 残差上升：按线性化松弛 r′ ≈ μ r，μ = 1 − β + βλ，估计 μ ≈ Re⟨r, r′⟩/‖r‖²：
        μ ≤ 1（过冲/旋转，Re λ < 1，阻尼有效）→ 拒绝该步、β 减半（下限 1e-3；已在下限时仍接受）；
        μ > 1（Re λ > 1：正在离开不稳定不动点，如弱注入下小信号 |Gκ|>1 的起振，阻尼无益）→ 接受、β 不变。
      后一分支及"增长步计入连胜"是对 DESIGN "残差上升 → β 减半 / 连续 10 步下降" 的修正：
      否则起振过程中 β 塌到下限而停滞（见 deviations）。
    - 停滞早停（性能，非 DESIGN 条款，判据见 _STALL_* 常量处）：连续 _STALL_PATIENCE 个窗口
      残差未创纪录新低（_STALL_GAIN 倍）且 ‖u‖ 未创纪录新高（_STALL_UGROW 倍）→ 该样本停止迭代
      （无稳定 CW 稳态时不再耗满 max_iter）；stalled 非 None 时原地置 True。
    原地更新 u、iters（以及 stalled）。
    """
    B = u.shape[0]
    beta = np.full(B, float(beta0))
    streak = np.zeros(B, dtype=int)
    r = prob.F(u) - u
    res = np.linalg.norm(r, axis=-1)
    active = (res > prob.threshold(u, rtol)) & (iters < max_iter)
    win_min = res.copy()                                 # 本窗口最小残差
    res_rec = res.copy()                                 # 残差纪录低点（自上次 ‖u‖ 创新高以来）
    u_rec = np.linalg.norm(u, axis=-1)                   # ‖u‖ 纪录高点（检查点上）
    idle = np.zeros(B, dtype=int)                        # 连续"无进展且不在起振"的窗口数
    next_ck = iters + _STALL_WINDOW
    while active.any():
        idx = np.nonzero(active)[0]
        bi = beta[idx]
        r_i, res_i = r[idx], res[idx]
        un = u[idx] + bi[:, None] * r_i
        rn_vec = prob.F(un, idx) - un
        rn = np.linalg.norm(rn_vec, axis=-1)
        iters[idx] += 1
        finite = np.isfinite(rn)
        down = rn < res_i
        mu = np.real(np.sum(np.conj(r_i) * rn_vec, axis=-1)) / np.maximum(res_i ** 2, 1e-300)
        growth = ~down & finite & (mu > 1.0)
        overshoot = ~down & ~growth
        accept = finite & (down | growth | (bi <= _BETA_MIN))
        ia = idx[accept]
        u[ia], r[ia], res[ia] = un[accept], rn_vec[accept], rn[accept]
        io = idx[overshoot]
        beta[io] = np.maximum(beta[io] * 0.5, _BETA_MIN)
        streak[io] = 0
        ib = idx[down | growth]                          # 被接受的"有效步"都计入连胜
        streak[ib] += 1
        grow = ib[streak[ib] >= _GROW_STREAK]
        beta[grow] = np.minimum(1.0, 1.5 * beta[grow])
        streak[grow] = 0
        active[idx] = (res[idx] > prob.threshold(u[idx], rtol, idx)) & (iters[idx] < max_iter)
        # 停滞检测（逐样本，互不影响）
        win_min[idx] = np.minimum(win_min[idx], res[idx])
        due = idx[active[idx] & (iters[idx] >= next_ck[idx])]
        if due.size:
            un = np.linalg.norm(u[due], axis=-1)
            starting = un > _STALL_UGROW * u_rec[due]                # ‖u‖ 创新高：起振中
            progress = _STALL_GAIN * win_min[due] <= res_rec[due]    # 残差创新低：在收敛
            idle[due] = np.where(progress | starting, 0, idle[due] + 1)
            st = idle[due] >= _STALL_PATIENCE
            active[due[st]] = False
            if stalled is not None:
                stalled[due[st]] = True
            u_rec[due] = np.where(starting, un, u_rec[due])
            # 起振时残差纪录从本窗口重新起算（起振中残差先升后降，回落必须算进展）
            res_rec[due] = np.where(starting, win_min[due], np.minimum(res_rec[due], win_min[due]))
            win_min[due] = res[due]                      # 新窗口从当前残差起算
            next_ck[due] += _STALL_WINDOW
    return u


def _newton_polish(prob: _FPProblem, u, iters, gate_rel: float = _NEWTON_GATE, max_steps: int = _NEWTON_MAX):
    """Newton 精修：(I − J_F) δ = F(u) − u（实 2R 坐标），回溯线搜索，只接受降低残差的步。

    默认只对已接近收敛的样本（残差 ≤ gate_rel·(‖u_in+n‖+‖u‖)）启用：Newton 会收敛到任意根，
    从远处出发可能跳到不稳定根（例如小信号 |Gκ|>1 时的线性解），"精修"不应改变阻尼迭代选中的吸引子。
    gate_rel=inf 仅供 _newton_rescue 使用（它事后检验根的稳定性）。原地更新 u、iters。
    """
    R = u.shape[-1]
    eye = np.eye(2 * R)
    r = prob.F(u) - u
    res = np.linalg.norm(r, axis=-1)
    gate = gate_rel * (prob.src_norm + np.linalg.norm(u, axis=-1))
    active = np.isfinite(res) & (res > 0) & (res <= gate)
    for _ in range(max_steps):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        J = loop_jacobian(u[idx], prob.K[idx], prob.pa)
        rhs = np.concatenate([r[idx].real, r[idx].imag], axis=-1)
        delta = _batched_solve(eye - J, rhs)
        du = delta[:, :R] + 1j * delta[:, R:]
        pending = np.isfinite(du).all(axis=-1)
        improved = np.zeros(idx.size, dtype=bool)
        step = np.ones(idx.size)
        for _ls in range(_LS_MAX):
            p = np.nonzero(pending)[0]
            if p.size == 0:
                break
            gi = idx[p]
            ut = u[gi] + step[p, None] * du[p]
            rt = prob.F(ut, gi) - ut
            nt = np.linalg.norm(rt, axis=-1)
            ok = nt < res[gi]
            acc = gi[ok]
            u[acc], r[acc], res[acc] = ut[ok], rt[ok], nt[ok]
            improved[p[ok]] = True
            pending[p[ok]] = False
            step[p[~ok]] *= 0.5
        iters[idx[improved]] += 1
        active[idx[~improved]] = False
        active &= res > 0
    return u


def _newton_rescue(prob: _FPProblem, u, iters, tol) -> np.ndarray:
    """救援 Newton（审查 R2-RELAY-5，非 DESIGN 条款）：对阻尼迭代未收敛（停滞早停或耗尽 max_iter）的样本，
    从当前迭代点做不设门限、只接受降残差步的 Newton；仅当达到收敛判据且工作点稳定
    （loop_gain_op = ρ(J_F) < 1 − _RES_RTOL）时采用新根，否则保留迭代结果（仍判未收敛）。

    理由：临界附近（ρ(J_F) → 1）的稳定不动点阻尼迭代收敛极慢，在 max_iter 内到不了是预算假象而非自激；
    不稳定根（如 |Gκ|>1 时的小信号根）一律拒绝，"无稳定 CW 稳态"的判定不受影响。
    原地更新 u、iters（只累计被采用样本的 Newton 步数）；返回被采用的布尔掩码。
    """
    u_try = u.copy()
    it = np.zeros(u.shape[0], dtype=int)
    u_try = _newton_polish(prob, u_try, it, gate_rel=np.inf, max_steps=2 * _NEWTON_MAX)
    res = np.linalg.norm(prob.F(u_try) - u_try, axis=-1)
    ok = np.isfinite(u_try).all(axis=-1) & (res <= prob.threshold(u_try, tol))
    acc = np.zeros(u.shape[0], dtype=bool)
    if ok.any():
        lg = _spectral_radius(loop_jacobian(u_try[ok], prob.K[ok], prob.pa))
        acc[ok] = lg < 1.0 - _RES_RTOL
    u[acc] = u_try[acc]
    iters[acc] += it[acc]
    return acc


def solve_fixed_point(u_in, K, pa: PA, noise=None, tol: float = 1e-12, max_iter: int = 3000,
                      beta0: float = 0.5, newton: bool = True, criterion: str = "cw") -> FixedPointResult:
    """求解中继稳态 u = u_in + n + K g(u)（方案 §2、§4.5；DESIGN §3.4）。

    u_in: (...,R)；K: (...,R,R)；noise: (...,R) 或 None（PA 输入等效噪声）。
    - LinearPA：直接解 (I − G_v K) u = u_in + n，converged=True，loop_gain_op = loop_gain_ss；
      但 I − G_v K 数值奇异（σ_min ≤ 1e-9·(1+‖G_v K‖)，即环路恰在谐振上）时 converged=False：
      此时 np.linalg.solve 返回的天文数字不是可信稳态（DESIGN 未规定，见 deviations）。
    - 其他 PA：逐样本自适应阻尼迭代（初值 u_in+n）→ 可选 Newton 精修（仅对已接近收敛的样本，
      不降残差则保留迭代结果）；仍未达到 tol 的样本用剩余迭代预算继续阻尼迭代并再精修一次。
      明显停滞（连续 _STALL_PATIENCE=3 个检查窗口既无进展又不在起振、迭代在极限环上打转）的样本提前停止。
      newton=True 时，最终仍未收敛的样本（停滞或耗尽预算）再做一次救援 Newton：只有收敛到稳定根
      （ρ(J_F) < 1 − 1e-9）才采用，否则 converged=False（见 _newton_rescue）。newton=False 为纯阻尼迭代。
    收敛判据：‖F(u)−u‖ ≤ tol·(‖u_in+n‖+1e-30)，另加浮点地板 64·eps·(‖u_in+n‖+‖u‖)
    （环路放大 ≫1 时相对 ‖u_in‖ 的 1e-12 低于 float64 可达精度，见 deviations）。
    阻尼迭代可能收敛到无阻尼映射的不稳定不动点，因此"可能自激"用工作点实雅可比谱半径判定（CW 稳态近似）：
    possibly_oscillating = ~converged | (loop_gain_op ≥ 1 − 1e-9)（容差：与 1 数值上不可区分的 ρ 也算 ≥ 1）。
    criterion（审查 S0-0，OSC_CRITERIA）："cw"（默认，即上式）；"robust" 再或上 loop_gain_robust ≥ 1（任意泄漏相位 /
    带内频率稳定的充分条件）；"small_signal" 再或上 loop_gain_ss ≥ 1（源关闭即起振）。另两个量总会算出并返回
    （loop_gain_robust、ss_unstable / robust_unstable 属性），调用方也可事后自行组合。
    R = 0（无中继）：返回空的 u/g，converged=True，环路增益 0。
    """
    if criterion not in OSC_CRITERIA:
        raise ValueError(f"criterion 只能是 {OSC_CRITERIA}，得到 {criterion!r}")
    u_in = np.asarray(u_in, dtype=np.complex128)
    K = np.asarray(K, dtype=np.complex128)
    if u_in.ndim < 1:
        raise ValueError("u_in 至少为 (R,)")
    R = u_in.shape[-1]
    if K.ndim < 2 or K.shape[-2:] != (R, R):
        raise ValueError(f"K 形状 {K.shape} 与 R={R} 不符")
    src = u_in if noise is None else u_in + np.asarray(noise, dtype=np.complex128)
    batch = np.broadcast_shapes(src.shape[:-1], K.shape[:-2])
    if R == 0:
        return _empty_result(batch, criterion)
    src = np.broadcast_to(src, batch + (R,)).reshape(-1, R).copy()
    Kf = np.broadcast_to(K, batch + (R, R)).reshape(-1, R, R)
    prob = _FPProblem(src, Kf, pa)
    lg_ss = _spectral_radius(pa.gain_v * Kf)
    iters = np.zeros(src.shape[0], dtype=int)

    if getattr(pa, "is_linear", False):
        GK = pa.gain_v * Kf
        u = _batched_solve(np.eye(R) - GK, src)
        trusted = ~_near_resonant(GK)                    # 谐振（奇异）样本的解不可信
        lg_op = lg_ss.copy()
    else:
        u = src.copy()
        stalled = np.zeros(src.shape[0], dtype=bool)
        stop = max(tol, _NEWTON_SWITCH) if newton else tol
        u = _damped_iterate(prob, u, iters, max_iter, beta0, stop, stalled)
        if newton:
            u = _newton_polish(prob, u, iters)
            r_now = np.linalg.norm(prob.F(u) - u, axis=-1)
            bad = np.nonzero(~(r_now <= prob.threshold(u, tol)) & (iters < max_iter) & ~stalled)[0]
            if bad.size:
                sub = prob.subset(bad)
                u_b, it_b = u[bad], iters[bad]
                u_b = _damped_iterate(sub, u_b, it_b, max_iter, beta0, tol)
                u_b = _newton_polish(sub, u_b, it_b)
                u[bad], iters[bad] = u_b, it_b
            r_now = np.linalg.norm(prob.F(u) - u, axis=-1)
            left = np.nonzero(~(r_now <= prob.threshold(u, tol)))[0]
            if left.size:                                # 救援：停滞/耗尽预算的样本找稳定根
                u_l, it_l = u[left], iters[left]
                _newton_rescue(prob.subset(left), u_l, it_l, tol)
                u[left], iters[left] = u_l, it_l
        trusted = np.ones(src.shape[0], dtype=bool)
        lg_op = _spectral_radius(loop_jacobian(u, Kf, pa))

    g = pa.g(u)
    res = np.linalg.norm(src + np.einsum("brs,bs->br", Kf, g) - u, axis=-1)
    finite = np.isfinite(u).all(axis=-1) & np.isfinite(res)
    converged = finite & trusted & (res <= prob.threshold(u, tol))
    osc = ~converged | gain_at_least_one(lg_op)         # nan 的环路增益也标记
    lg_rob = loop_gain_robust(u, Kf, pa)
    if criterion == "robust":
        osc |= gain_at_least_one(lg_rob)
    elif criterion == "small_signal":
        osc |= gain_at_least_one(lg_ss)

    def rb(x, tail=()):
        return x.reshape(batch + tail)

    return FixedPointResult(u=rb(u, (R,)), g=rb(g, (R,)), converged=rb(converged), iters=rb(iters),
                            loop_gain_ss=rb(lg_ss), loop_gain_op=rb(lg_op),
                            possibly_oscillating=rb(osc), residual=rb(res), loop_gain_robust=rb(lg_rob),
                            criterion=criterion)


# ---------------------------------------------------------------- 探头场与噪声
def probe_fields(d, C, g, probe_noise=None) -> np.ndarray:
    """探头复场 z = d + C@g（+ 探头加性噪声）（方案 §2：y_k = |d_k + c_k g(u)|²）。

    d: (...,K)，C: (...,K,R)，g: (...,R) → z: (...,K)
    """
    d = np.asarray(d, dtype=np.complex128)
    z = d + np.einsum("...kr,...r->...k", np.asarray(C, dtype=np.complex128), np.asarray(g, dtype=np.complex128))
    if probe_noise is not None:
        z = z + np.asarray(probe_noise, dtype=np.complex128)
    return z


def pa_noise_power_w(params: RelayParams) -> float:
    """PA 输入等效噪声功率 k_B·T·B·F (W)（DESIGN §3.4；F 为线性噪声系数）。"""
    return float(KB * params.temp_k * params.noise_bw_hz * 10.0 ** (params.nf_db / 10.0))
