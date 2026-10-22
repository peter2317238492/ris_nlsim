"""中继线性量与 PA 不动点测试（DESIGN §3.2–3.4，方案 §2、§4.5、§6.1）。"""
import numpy as np
import pytest
from scipy.optimize import brentq

from ris_nlsim import relay
from ris_nlsim.constants import DEFAULTS, KB
from ris_nlsim.pa import LinearPA, RappPA
from ris_nlsim.relay import (RelayParams, bits_to_phase, loop_jacobian, pa_noise_power_w, probe_fields,
                             real_jacobian, relay_linear, solve_fixed_point, weights)


def crandn(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# ---------------------------------------------------------------- §3.2
def test_weights_and_bits_to_phase():
    rng = np.random.default_rng(0)
    R, N = 2, 25
    err_rx = rng.uniform(-0.1, 0.1, (R, N))
    params = RelayParams(n_relays=R, loss_rx_db=2.5, loss_tx_db=4.0, phase_err_rx=err_rx)
    bits_rx = rng.integers(0, 2, (R, N))
    bits_tx = rng.integers(0, 2, (R, N))
    w, ct = weights(bits_to_phase(bits_rx), bits_to_phase(bits_tx), params)
    assert w.shape == ct.shape == (R, N) and w.dtype == np.complex128
    assert np.allclose(np.abs(w), 10 ** (-2.5 / 20) / 5.0)
    assert np.allclose(np.abs(ct), 10 ** (-4.0 / 20) / 5.0)
    assert np.allclose(w / np.abs(w), np.exp(1j * (np.pi * bits_rx + err_rx)))
    assert np.allclose(ct / np.abs(ct), np.where(bits_tx == 1, -1.0, 1.0))   # 1-bit 权重 ±1
    # 2-bit：(π/2)·l
    assert np.allclose(bits_to_phase(np.arange(4), levels=4), np.pi / 2 * np.arange(4))
    # 合路无耗归一化：同相合成 N 个单位场 → |u|² = N·loss
    assert np.isclose(np.abs(np.sum(w[0] * np.exp(-1j * np.angle(w[0])))) ** 2, N * 10 ** (-0.25))
    with pytest.raises(ValueError):
        weights(np.zeros((1, 24)), np.zeros((1, 24)), RelayParams())


def test_weights_rejects_shape_mismatch():
    """回归（RELAY-4）：相位/通道误差的中继数必须与 n_relays 一致，不得静默广播。"""
    z1, z2 = np.zeros((1, 25)), np.zeros((2, 25))
    with pytest.raises(ValueError, match="phase_err_rx"):
        weights(z2, z2, RelayParams(n_relays=2, phase_err_rx=np.ones((1, 25))))
    with pytest.raises(ValueError, match="phase_err_tx"):
        weights(z1, z1, RelayParams(n_relays=1, phase_err_tx=np.ones((2, 25))))
    with pytest.raises(ValueError):
        weights(z2, z2, RelayParams(n_relays=1))            # 相位比 n_relays 多
    with pytest.raises(ValueError):
        weights(z1, z2, RelayParams(n_relays=2))            # rx 被广播
    with pytest.raises(ValueError):
        weights(np.zeros(25), np.zeros(25), RelayParams())  # 必须显式 (R,N)
    w, ct = weights(z2, z2, RelayParams(n_relays=2, phase_err_rx=np.ones((2, 25))))
    assert w.shape == ct.shape == (2, 25)


# ---------------------------------------------------------------- §3.3
def _relay_linear_loops(H, w, ct, s, kappa0):
    """逐元素显式循环参考实现（DESIGN §3.3 公式逐字翻译）。"""
    R, N = w.shape
    nrx = R * N
    n_probe = H.shape[0] - nrx
    u_in = np.zeros(R, complex)
    K = np.zeros((R, R), complex)
    d = np.zeros(n_probe, complex)
    C = np.zeros((n_probe, R), complex)
    for r in range(R):
        for n in range(N):
            u_in[r] += s * w[r, n] * H[r * N + n, 0]
    for r in range(R):
        for sp in range(R):
            for n in range(N):
                for m in range(N):
                    K[r, sp] += w[r, n] * H[r * N + n, 1 + sp * N + m] * ct[sp, m]
            if r == sp:
                K[r, sp] += kappa0
    for k in range(n_probe):
        d[k] = s * H[nrx + k, 0]
        for sp in range(R):
            for m in range(N):
                C[k, sp] += H[nrx + k, 1 + sp * N + m] * ct[sp, m]
    return u_in, K, d, C


@pytest.mark.parametrize("R", [1, 2])
def test_relay_linear_matches_loops(R):
    rng = np.random.default_rng(10 + R)
    N, n_probe, B = 25, 3, 3
    params = RelayParams(n_relays=R, isolation_db=-20.0, kappa0_phase_deg=37.0)
    kappa0_port = 10 ** (-1.0) * np.exp(1j * np.deg2rad(37.0))
    assert np.isclose(params.kappa0_port, kappa0_port)
    kappa0 = kappa0_port * 10 ** (-(params.loss_rx_db + params.loss_tx_db) / 20)   # 环路中乘收发网络损耗（DESIGN §3.3）
    assert np.isclose(params.kappa0, kappa0)
    w, ct = weights(rng.uniform(0, 2 * np.pi, (R, N)), rng.uniform(0, 2 * np.pi, (R, N)), params)
    H = crandn(rng, B, R * N + n_probe, 1 + R * N)
    s = 0.37

    # 批量
    lin = relay_linear(H, w, ct, s, params)
    assert lin.u_in.shape == (B, R) and lin.K.shape == (B, R, R)
    assert lin.d.shape == (B, n_probe) and lin.C.shape == (B, n_probe, R)
    for b in range(B):
        ref = _relay_linear_loops(H[b], w, ct, s, kappa0)
        for got, exp in zip((lin.u_in[b], lin.K[b], lin.d[b], lin.C[b]), ref):
            assert np.allclose(got, exp, rtol=1e-12, atol=1e-12)

    # 不批量：形状去掉批量维，数值一致
    lin1 = relay_linear(H[1], w, ct, s, params)
    assert lin1.u_in.shape == (R,) and lin1.K.shape == (R, R) and lin1.d.shape == (n_probe,)
    assert lin1.C.shape == (n_probe, R)
    assert np.allclose(lin1.K, lin.K[1]) and np.allclose(lin1.u_in, lin.u_in[1])

    # 每样本源幅度（功率抖动）
    s_b = np.array([0.1, 1.0, 2.5])
    lin_b = relay_linear(H, w, ct, s_b, params)
    assert np.allclose(lin_b.u_in, s_b[:, None] * lin.u_in / s)
    assert np.allclose(lin_b.d, s_b[:, None] * lin.d / s)
    assert np.allclose(lin_b.K, lin.K) and np.allclose(lin_b.C, lin.C)


def test_relay_linear_kappa0_only_on_diagonal():
    R, N = 2, 25
    params = RelayParams(n_relays=R, isolation_db=-30.0)
    w, ct = weights(np.zeros((R, N)), np.zeros((R, N)), params)
    H = np.zeros((R * N + 2, 1 + R * N), complex)    # 空响应 → K = κ0·I
    lin = relay_linear(H, w, ct, 1.0, params)
    assert np.allclose(lin.K, params.kappa0 * np.eye(R))
    with pytest.raises(ValueError):
        relay_linear(H[:, :-1], w, ct, 1.0, params)


# ---------------------------------------------------------------- §3.4 线性 PA 闭式
def test_linear_pa_closed_form_R1_and_R2():
    rng = np.random.default_rng(20)
    pa = LinearPA(30.0)
    G = pa.gain_v
    # R=1：u = u_in/(1 − G κ)
    kappa = 0.6 / G * np.exp(1j * rng.uniform(0, 2 * np.pi, 8))
    u_in = crandn(rng, 8, 1)
    res = solve_fixed_point(u_in, kappa[:, None, None], pa)
    assert np.allclose(res.u[:, 0], u_in[:, 0] / (1 - G * kappa), rtol=1e-13)
    assert np.allclose(res.g, G * res.u)
    assert res.converged.all() and not res.possibly_oscillating.any()
    # loop_gain_op（线性 PA）= |G κ|
    assert np.allclose(res.loop_gain_op, np.abs(G * kappa), rtol=1e-12)
    assert np.allclose(res.loop_gain_ss, np.abs(G * kappa), rtol=1e-12)
    # R=2：u = (I − G K)^{-1} u_in
    K = 0.3 / G * crandn(rng, 5, 2, 2)
    u2 = crandn(rng, 5, 2)
    res2 = solve_fixed_point(u2, K, pa)
    for b in range(5):
        assert np.allclose(res2.u[b], np.linalg.solve(np.eye(2) - G * K[b], u2[b]), rtol=1e-12)
        assert np.isclose(res2.loop_gain_ss[b], np.abs(np.linalg.eigvals(G * K[b])).max())


@pytest.mark.parametrize("gk", [1.0001, 1.5, 3.0])
def test_linear_relay_with_loop_gain_ge_1_flagged(gk):
    pa = LinearPA(30.0)
    phases = np.linspace(0.1, 6.0, 6)
    kappa = gk / pa.gain_v * np.exp(1j * phases)
    res = solve_fixed_point(np.ones((6, 1)), kappa[:, None, None], pa)
    assert np.allclose(res.loop_gain_ss, gk, rtol=1e-12)
    assert res.possibly_oscillating.all()
    ok = solve_fixed_point(np.ones((6, 1)), 0.99 * kappa[:, None, None] / gk, pa)
    assert not ok.possibly_oscillating.any()


def test_linear_relay_exact_resonance_flagged():
    """回归（RELAY-2/RELAY-1）：G_v κ0 恰为 +1（相差 1 ulp，例如 PA 30 dB + 隔离 −30 dB + 同相——即
    当前 DESIGN 默认值）时 I − G_v K 奇异，np.linalg.solve 给出 ~1e16 的"解"；必须判为未收敛并标记，
    而不是作为 +300 dB 的有效读数返回。"""
    pa = LinearPA(DEFAULTS["pa_gain_db"])
    params = RelayParams(isolation_db=-DEFAULTS["pa_gain_db"], kappa0_phase_deg=0.0, loss_rx_db=0.0, loss_tx_db=0.0)
    assert abs(pa.gain_v * params.kappa0 - 1.0) < 1e-15            # 1 ulp 量级的谐振
    G = pa.gain_v
    K = np.array([params.kappa0, 1.0 / G, 1.0 / G + 0j, 0.3 / G], complex)[:, None, None]
    u_in = np.array([1e-3, 1e-3, 0.0, 1e-3], complex)[:, None]       # 第 3 个：零注入也算谐振
    res = solve_fixed_point(u_in, K, pa)
    assert res.converged.tolist() == [False, False, False, True]
    assert res.possibly_oscillating.tolist() == [True, True, True, False]
    assert np.isclose(res.u[3, 0], 1e-3 / 0.7)
    # R=2：一个特征值恰为 1（谐振模式），另一个 0.3 → 仍是奇异，必须标记
    rng = np.random.default_rng(21)
    V = crandn(rng, 2, 2)
    K2 = V @ np.diag([1.0 / G, 0.3 / G]) @ np.linalg.inv(V)
    r2 = solve_fixed_point(crandn(rng, 2), K2, pa)
    assert not r2.converged and r2.possibly_oscillating
    # 接近但不在谐振上：|1 − Gκ| = 1e-3（+60 dB 增强）、|Gκ| < 1（cos φ > 0）是合法、良态的稳态，与闭式一致、不标记；
    # φ = 2.5 时 |Gκ| = |1 − 1e-3e^{jφ}| > 1 → 必须标记（但解本身良态、收敛）
    kap = (1 - 1e-3 * np.exp(1j * np.array([0.0, 1.0, -1.2, 2.5]))) / G
    ok = solve_fixed_point(np.full((4, 1), 1e-3 + 0j), kap[:, None, None], pa)
    assert ok.converged.all() and ok.possibly_oscillating.tolist() == [False, False, False, True]
    assert np.allclose(ok.u[:, 0], 1e-3 / (1 - G * kap), rtol=1e-9)
    # ρ 与 1 只差 1e-12（非同相，解良态）：数值上与 ρ ≥ 1 不可区分 → 标记，但解本身收敛
    edge = solve_fixed_point(np.ones((1, 1), complex), np.array([[[(1 - 1e-12) * 1j / G]]]), pa)
    assert edge.converged[0] and edge.possibly_oscillating[0]


def test_rapp_at_linear_resonance_is_clamped_not_flagged():
    """同一谐振环路（G_v κ0 = 1，无 AM-PM）换成 Rapp：饱和把增益拉到 < 1，稳态有限且稳定（方案 §2 推论 4）。
    实正注入下 r = r_in + c1(r)·r ⇒ r(1 − c1(r)) = r_in；工作点 loop_gain_op = max(c1, c2) = c1 < 1。"""
    pa = RappPA(ampm=False)
    params = RelayParams(isolation_db=-pa.gain_db, kappa0_phase_deg=0.0, loss_rx_db=0.0, loss_tx_db=0.0)
    r_in = np.array([1e-3, 0.1, 1.0])
    res = solve_fixed_point((r_in * pa.A_in)[:, None] + 0j, np.full((3, 1, 1), params.kappa0), pa)
    assert res.converged.all() and not res.possibly_oscillating.any()
    r_star = np.array([brentq(lambda r: r * (1 - _c1(r)) - ri, ri, 1e3) for ri in r_in])
    assert np.allclose(np.abs(res.u[:, 0]) / pa.A_in, r_star, rtol=1e-9)
    assert np.allclose(res.loop_gain_op, _c1(r_star), rtol=1e-8)
    lin = solve_fixed_point((r_in * pa.A_in)[:, None] + 0j, np.full((3, 1, 1), params.kappa0), LinearPA(pa.gain_db))
    assert lin.possibly_oscillating.all()


@pytest.mark.parametrize("pa", [LinearPA(30.0), RappPA()])
def test_zero_relays_returns_empty(pa):
    """回归（RELAY-4）：R = 0（无中继）不得以 reshape 错误崩溃。"""
    res = solve_fixed_point(np.zeros((3, 0), complex), np.zeros((3, 0, 0), complex), pa)
    assert res.u.shape == res.g.shape == (3, 0)
    assert res.converged.all() and not res.possibly_oscillating.any()
    assert np.array_equal(res.loop_gain_ss, np.zeros(3)) and np.array_equal(res.iters, np.zeros(3))
    one = solve_fixed_point(np.zeros(0, complex), np.zeros((0, 0), complex), pa)
    assert one.u.shape == (0,) and one.converged.shape == () and bool(one.converged)


# ---------------------------------------------------------------- §3.4 Rapp
def _residual(res, u_in, K, pa, noise=None):
    src = u_in if noise is None else u_in + noise
    F = src + np.einsum("...rs,...s->...r", K, pa.g(res.u))
    return np.linalg.norm(F - res.u, axis=-1), np.linalg.norm(src, axis=-1)


def _random_case(rng, B, R, pa, rho_range, backoff_range):
    """随机 K（小信号环路增益 ρ(G K) 落在 rho_range）与给定回退量的 u_in。"""
    K = crandn(rng, B, R, R)
    rho = np.abs(np.linalg.eigvals(pa.gain_v * K)).max(-1)
    K *= (rng.uniform(*rho_range, B) / rho)[:, None, None]
    bo = rng.uniform(*backoff_range, B)
    amp = np.sqrt(pa.p1db_in_w() * 10 ** (bo / 10))
    u_in = amp[:, None] * np.exp(1j * rng.uniform(0, 2 * np.pi, (B, R))) / np.sqrt(R)
    return u_in, K


@pytest.mark.parametrize("R", [1, 2])
def test_rapp_deep_backoff_equals_linear(R):
    rng = np.random.default_rng(30 + R)
    rapp = RappPA()
    u_in, K = _random_case(rng, 10, R, rapp, (0.1, 0.7), (-40, -40))
    nl = solve_fixed_point(u_in, K, rapp)
    li = solve_fixed_point(u_in, K, LinearPA(rapp.gain_db))
    assert nl.converged.all()
    rel = np.linalg.norm(nl.u - li.u, axis=-1) / np.linalg.norm(li.u, axis=-1)
    assert rel.max() < 1e-3


@pytest.mark.parametrize("R", [1, 2, 3])
def test_damped_vs_newton_agree_and_residual_tiny(R):
    rng = np.random.default_rng(40 + R)
    pa = RappPA(ampm=True)
    u_in, K = _random_case(rng, 24, R, pa, (0.1, 0.8), (-15, 15))
    noise = 1e-3 * np.sqrt(pa.p1db_in_w()) * crandn(rng, 24, R)
    rn = solve_fixed_point(u_in, K, pa, noise=noise, newton=True)
    rd = solve_fixed_point(u_in, K, pa, noise=noise, newton=False)
    assert rn.converged.all() and rd.converged.all()
    assert np.allclose(rn.u, rd.u, rtol=1e-9, atol=1e-12 * np.abs(u_in).max())
    for res in (rn, rd):
        r, s = _residual(res, u_in, K, pa, noise)
        assert np.all(r <= 1e-12 * s)
        assert np.allclose(res.residual, r, rtol=1e-6, atol=1e-30)
        assert np.allclose(res.g, pa.g(res.u))
    # Newton 精修显著减少迭代步数
    assert rn.iters.sum() < rd.iters.sum()
    # 噪声等价于加到 u_in 上
    alt = solve_fixed_point(u_in + noise, K, pa)
    assert np.allclose(alt.u, rn.u, rtol=1e-10)


def test_batch_equals_individual_and_unbatched_shapes():
    rng = np.random.default_rng(50)
    pa = RappPA()
    u_in, K = _random_case(rng, 6, 2, pa, (0.2, 0.9), (-10, 10))
    batch = solve_fixed_point(u_in, K, pa)
    for b in range(6):
        one = solve_fixed_point(u_in[b], K[b], pa)
        assert one.u.shape == (2,) and one.converged.shape == () and one.loop_gain_op.shape == ()
        assert np.allclose(one.u, batch.u[b], rtol=1e-10)
        assert np.isclose(one.loop_gain_op, batch.loop_gain_op[b], rtol=1e-8)
    # 任意前导批量维
    multi = solve_fixed_point(u_in.reshape(2, 3, 2), K.reshape(2, 3, 2, 2), pa)
    assert multi.u.shape == (2, 3, 2) and multi.converged.shape == (2, 3)
    assert np.allclose(multi.u.reshape(6, 2), batch.u, rtol=1e-10)


def _numerical_real_jacobian(u, K, pa, h=1e-7):
    """F(u) = K g(u) 的实雅可比（中心差分），实坐标 [Re u; Im u]。"""
    R = u.size
    J = np.zeros((2 * R, 2 * R))
    scale = h * max(np.abs(u).max(), 1e-30)

    def Fr(v):
        f = K @ pa.g(v)
        return np.concatenate([f.real, f.imag])

    for j in range(2 * R):
        e = np.zeros(R, complex)
        e[j % R] = scale if j < R else 1j * scale
        J[:, j] = (Fr(u + e) - Fr(u - e)) / (2 * scale)
    return J


def test_loop_gain_op_matches_numerical_jacobian():
    rng = np.random.default_rng(60)
    pa = RappPA(ampm=True, ampm_max_deg=20.0)
    u_in, K = _random_case(rng, 8, 2, pa, (0.3, 0.9), (-5, 10))
    res = solve_fixed_point(u_in, K, pa)
    assert res.converged.all()
    for b in range(8):
        Jn = _numerical_real_jacobian(res.u[b], K[b], pa)
        Ja = loop_jacobian(res.u[b], K[b], pa)
        assert np.allclose(Ja, Jn, rtol=1e-6, atol=1e-7 * np.abs(Jn).max())
        assert np.isclose(res.loop_gain_op[b], np.abs(np.linalg.eigvals(Jn)).max(), rtol=1e-6)


def test_real_jacobian_linear_case_spectrum():
    """b=0 时实雅可比的谱 = A 的谱 ∪ 共轭 → 谱半径相同。"""
    rng = np.random.default_rng(61)
    K = crandn(rng, 3, 3)
    a = crandn(rng, 3)
    J = real_jacobian(K, a, np.zeros(3))
    ev = np.sort_complex(np.linalg.eigvals(J))
    evA = np.linalg.eigvals(K * a[None, :])
    assert np.allclose(ev, np.sort_complex(np.concatenate([evA, evA.conj()])))


def _c1(r, p=2.0):
    return (1 + r ** (2 * p)) ** (-1 / (2 * p))


def test_saturated_rapp_with_small_signal_loop_gain_1p5(capsys):
    """|G_v κ| = 1.5：线性 PA 自激；Rapp 饱和把它钳在有限幅度（方案 §2 推论 4）。

    无 AM-PM、κ、u_in 实正时解析：r = r_in + 1.5·r·c1(r)；工作点实雅可比特征值 1.5·c2（幅度方向）、
    1.5·c1 = (r−r_in)/r（相位方向，注入锁定）→ loop_gain_op = 1.5·c1 < 1。
    """
    pa = RappPA(ampm=False)
    G, A_in = pa.gain_v, pa.A_in
    kappa = 1.5 / G
    r_in = np.array([0.02, 0.2, 3.0])           # 弱注入（自激被钳住）… 强驱动
    res = solve_fixed_point((r_in * A_in)[:, None], np.full((3, 1, 1), kappa + 0j), pa)
    lin = solve_fixed_point((r_in * A_in)[:, None], np.full((3, 1, 1), kappa + 0j), LinearPA(pa.gain_db))
    assert lin.possibly_oscillating.all()       # 线性对照：|Gκ| ≥ 1 被标记
    assert res.converged.all()
    r_star = np.array([brentq(lambda r: ri + 1.5 * r * _c1(r) - r, ri, ri + 10) for ri in r_in])
    assert np.allclose(np.abs(res.u[:, 0]) / A_in, r_star, rtol=1e-10)
    assert np.all(np.abs(res.g) > 0.6 * pa.A_s) and np.all(np.abs(res.g) < pa.A_s)
    assert np.allclose(res.loop_gain_ss, 1.5)
    assert np.allclose(res.loop_gain_op, 1.5 * _c1(r_star), rtol=1e-8)
    assert np.all(res.loop_gain_op < 1.0) and not res.possibly_oscillating.any()
    # 有 AM-PM：饱和处环路相位偏 Φ(r*)≈8°，u(1 − 1.5c1e^{jΦ}) = u_in。
    # 弱注入（r_in=0.02）锁不住：ρ>0.1 范围内 ρ|1 − 1.5c1e^{jΦ}| 的最小值 > r_in，唯一的根是小信号不稳定根
    # → 无稳定 CW 稳态，必须被标记；较强注入锁定，收敛到有限幅度，loop_gain_op 与数值雅可比一致。
    pa2 = RappPA(ampm=True)
    rho = np.linspace(0.1, 5, 4000)
    lhs = rho * np.abs(1 - 1.5 * _c1(rho) * np.exp(1j * pa2.ampm_rad(rho * A_in)))
    assert lhs.min() > r_in[0]
    res2 = solve_fixed_point((r_in * A_in)[:, None], np.full((3, 1, 1), kappa + 0j), pa2)
    assert res2.possibly_oscillating[0]
    assert res2.converged[1:].all() and np.all(np.isfinite(res2.loop_gain_op))
    for b in (1, 2):
        Jn = _numerical_real_jacobian(res2.u[b], np.array([[kappa + 0j]]), pa2)
        assert np.isclose(res2.loop_gain_op[b], np.abs(np.linalg.eigvals(Jn)).max(), rtol=1e-6)
        assert np.abs(res2.u[b, 0]) / A_in > 1.0                      # 饱和区的有限幅度
    with capsys.disabled():
        print(f"\n[|Gκ|=1.5 Rapp] r*={np.round(r_star, 4)}, loop_gain_op(noAMPM)={np.round(res.loop_gain_op, 4)}, "
              f"loop_gain_op(AMPM)={np.round(res2.loop_gain_op, 4)}, osc(AMPM)={res2.possibly_oscillating}")


def test_damped_iteration_can_land_on_unstable_point_and_is_flagged():
    """κG = −1.5（过冲负反馈）：阻尼迭代收敛，但弱驱动下工作点 loop_gain_op = 1.5·c1 ≥ 1 → 标记（DESIGN §3.4 注意）。"""
    pa = RappPA(ampm=False)
    kappa = -1.5 / pa.gain_v
    r_in = np.array([0.01, 10.0])
    res = solve_fixed_point((r_in * pa.A_in)[:, None], np.full((2, 1, 1), kappa + 0j), pa, newton=False)
    assert res.converged.all()
    r_star = np.array([brentq(lambda r: r * (1 + 1.5 * _c1(r)) - ri, 0, ri) for ri in r_in])
    assert np.allclose(res.u[:, 0].real / pa.A_in, r_star, rtol=1e-10)
    assert np.allclose(res.loop_gain_op, 1.5 * _c1(r_star), rtol=1e-8)
    assert res.possibly_oscillating.tolist() == [True, False]


def test_per_sample_adaptive_damping():
    """同一批中：κG=−5 需要小 β（β=0.5 发散），κG=+0.3 可用大 β；两者都要收敛到各自的解。

    逐样本 β 的可检验推论（回归 TEST-1）：一个样本的迭代轨迹与同批其他样本无关——
    κG=+0.3 单独求解与和 κG=−5 同批求解的步数、结果逐位相同；全批共享 β 的实现会被 κG=−5 拖慢。
    """
    pa = RappPA(ampm=False)
    G = pa.gain_v
    u0 = np.sqrt(pa.p1db_in_w() * 1e-4)          # −40 dB 回退，近线性
    K = np.array([-5.0 / G, 0.3 / G], complex)[:, None, None]
    res = solve_fixed_point(np.full((2, 1), u0 + 0j), K, pa, newton=False)
    assert res.converged.all()
    assert np.allclose(res.u[:, 0], u0 / (1 - G * K[:, 0, 0]), rtol=1e-3)
    for b in (0, 1):
        alone = solve_fixed_point(np.full((1, 1), u0 + 0j), K[b:b + 1], pa, newton=False)
        assert alone.iters[0] == res.iters[b]
        assert alone.u[0, 0] == res.u[b, 0]


def test_stalled_samples_stop_early_without_changing_outcome(monkeypatch):
    """回归（RELAY-3）：无稳定 CW 稳态（|Gκ|=1.5 + AM-PM + 弱注入，见上面的 |Gκ|=1.5 测试）的样本
    提前停止（不再耗满 max_iter），判定与不早停时完全一致；起振慢于一个检查窗口的正常样本不受影响。"""
    pa = RappPA()
    G = pa.gain_v
    u_bad = np.full((4, 1), 0.02 * pa.A_in + 0j)
    K_bad = np.full((4, 1, 1), 1.5 / G + 0j)
    fast = solve_fixed_point(u_bad, K_bad, pa)
    assert not fast.converged.any() and fast.possibly_oscillating.all()
    # 1 个起振窗口 + _STALL_PATIENCE 个停滞窗口
    assert np.all(fast.iters <= (relay._STALL_PATIENCE + 1) * relay._STALL_WINDOW)
    monkeypatch.setattr(relay, "_STALL_WINDOW", 10 ** 9)          # 关闭早停作对照
    slow = solve_fixed_point(u_bad, K_bad, pa)
    assert np.all(slow.iters == 3000) and not slow.converged.any() and slow.possibly_oscillating.all()
    monkeypatch.undo()

    # 慢起振：|Gκ|=1.03、r_in=1e-4、无 AM-PM，需要 > 2 个检查窗口才锁定，必须照常收敛到解析解
    pa0 = RappPA(ampm=False)
    r_in = 1e-4
    res = solve_fixed_point(np.array([[r_in * pa0.A_in + 0j]]), np.array([[[1.03 / G + 0j]]]), pa0, newton=False)
    assert res.converged[0] and res.iters[0] > 2 * relay._STALL_WINDOW
    r_star = brentq(lambda r: r_in + 1.03 * r * _c1(r) - r, r_in, 10)
    assert np.isclose(abs(res.u[0, 0]) / pa0.A_in, r_star, rtol=1e-10)


def test_stall_early_stop_agrees_with_full_budget(monkeypatch):
    """随机混合批（稳定/临界/自激，含 AM-PM 与弱注入）：早停与否的 converged、possibly_oscillating 与 u 完全一致。"""
    rng = np.random.default_rng(90)
    cases = []
    for R, ampm, rho, bo in [(1, True, (0.9, 2.0), (-15, 5)), (2, False, (0.8, 2.5), (-20, 10)),
                             (3, True, (0.5, 1.5), (-10, 10))]:
        pa = RappPA(ampm=ampm)
        u_in, K = _random_case(rng, 32, R, pa, rho, bo)
        u_in *= 10 ** rng.uniform(-4, 0, (32, 1))                     # 部分弱注入
        cases.append((u_in, K, pa))
    on = [solve_fixed_point(u, K, pa) for u, K, pa in cases]
    monkeypatch.setattr(relay, "_STALL_WINDOW", 10 ** 9)
    off = [solve_fixed_point(u, K, pa) for u, K, pa in cases]
    n_stopped = 0
    for a, b in zip(on, off):
        assert np.array_equal(a.converged, b.converged)
        assert np.array_equal(a.possibly_oscillating, b.possibly_oscillating)
        assert np.allclose(a.u[a.converged], b.u[b.converged], rtol=1e-10, atol=0)
        n_stopped += int(np.sum(a.iters < b.iters))
    assert n_stopped > 0                                              # 确实有样本被早停


def test_slow_startup_near_criticality_not_stopped_early(monkeypatch):
    """回归（R2-RELAY-5）：小信号 |Gκ| 略大于 1 + 弱注入的注入锁定稳态起振极慢（‖u‖ 每 200 步增长 < 1.5 倍，
    残差先升后降），旧早停把它们误判为停滞 → converged=False 并标记自激。

    无 AM-PM、实正 κ、u_in 时解析：r = r_in + gk·r·c1(r)（取 > 0 的饱和根）；
    工作点实雅可比特征值 gk·c2（幅度）、gk·c1 = 1 − r_in/r（相位，最大）→ loop_gain_op = 1 − r_in/r* < 1。
    审查者给出的例子 gk=1.002、r_in=1e-4 ⇒ r* = 0.3107。"""
    pa = RappPA(ampm=False)
    G, A_in = pa.gain_v, pa.A_in
    gk = np.array([1.001, 1.0015, 1.002, 1.002, 1.003, 1.005])
    r_in = np.array([1e-4, 3e-5, 1e-4, 1e-5, 3e-5, 1e-5])
    u_in, K = (r_in * A_in + 0j)[:, None], (gk / G + 0j)[:, None, None]
    res = solve_fixed_point(u_in, K, pa)
    r_star = np.array([brentq(lambda r: ri + g * r * _c1(r) - r, ri, 10) for g, ri in zip(gk, r_in)])
    assert np.isclose(r_star[2], 0.3107, atol=1e-4)
    assert res.converged.all() and not res.possibly_oscillating.any()
    assert np.allclose(np.abs(res.u[:, 0]) / A_in, r_star, rtol=1e-9)
    assert np.allclose(res.loop_gain_op, 1 - r_in / r_star, rtol=1e-9)
    # 早停不得改变结果：与关闭早停（同样带救援 Newton）逐位一致；纯阻尼迭代（newton=False）亦然
    nd = solve_fixed_point(u_in, K, pa, newton=False)
    monkeypatch.setattr(relay, "_STALL_WINDOW", 10 ** 9)
    off = solve_fixed_point(u_in, K, pa)
    off_nd = solve_fixed_point(u_in, K, pa, newton=False)
    assert np.array_equal(off.u, res.u) and np.array_equal(off.iters, res.iters)
    assert np.array_equal(off_nd.u, nd.u) and np.array_equal(off_nd.converged, nd.converged)


def test_stall_criterion_spares_all_slow_startups():
    """回归（R2-RELAY-5 根因 (i)(ii)）：直接检查停滞判定。审查者网格 |Gκ| ∈ [1.001, 1.01]（实正、无 AM-PM）×
    r_in ∈ [1e-7, 1e-2]：每个样本都有稳定饱和根，实值动力学只是起振慢、残差先升后降——一个都不许判停滞
    （旧判据在 newton=True 路径上误停 23/81）。用与 solve_fixed_point 相同的第一段参数（rtol = _NEWTON_SWITCH）。"""
    pa = RappPA(ampm=False)
    gk = np.array([1.001, 1.0015, 1.002, 1.0025, 1.003, 1.004, 1.005, 1.007, 1.01])
    r_in = np.array([1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2])
    GG, RR = np.meshgrid(gk, r_in, indexing="ij")
    u_in = (RR.ravel() * pa.A_in + 0j)[:, None]
    prob = relay._FPProblem(u_in.copy(), (GG.ravel() / pa.gain_v + 0j)[:, None, None], pa)
    u, it, stalled = u_in.copy(), np.zeros(u_in.shape[0], dtype=int), np.zeros(u_in.shape[0], dtype=bool)
    relay._damped_iterate(prob, u, it, 3000, 0.5, relay._NEWTON_SWITCH, stalled)
    assert not stalled.any(), list(zip(GG.ravel()[stalled], RR.ravel()[stalled]))


def test_ampm_locking_transient_not_stopped_early(monkeypatch):
    """回归（R2-RELAY-5，AM-PM）：|Gκ|≈1.003–1.008、环路相位略负、弱注入：注入锁定的捕获过程中 ‖u‖ 与残差
    来回摆动若干窗口后才锁定（旧判据在 400 步时就判停滞）。必须收敛、不标记，且等于独立求得的唯一稳定根。

    独立参考（R=1）：u(1 − gk·c1(r)e^{jΦ(r)}) = u_in ⇒ r·|1 − gk·c1(r)e^{jΦ(r)}| = r_in（r = |u|/A_in）；
    在细网格上找出全部根，用数值实雅可比判稳定性。"""
    pa = RappPA()
    G, A_in = pa.gain_v, pa.A_in
    cases = [(1.007728 - 0.047014j, 0.0040854), (1.003032 - 0.038933j, 0.00386), (1.004507 - 0.012533j, 0.0021682)]
    u_in = np.array([[ri * A_in + 0j] for _, ri in cases])
    K = np.array([[[g / G]] for g, _ in cases])
    res = solve_fixed_point(u_in, K, pa)
    assert res.converged.all() and not res.possibly_oscillating.any()
    rr = np.geomspace(1e-5, 10, 200001)
    for b, (g, ri) in enumerate(cases):
        loop = lambda r: g * _c1(r) * np.exp(1j * pa.ampm_rad(r * A_in))
        h = rr * np.abs(1 - loop(rr)) - ri
        roots = [brentq(lambda r: r * abs(1 - loop(r)) - ri, rr[i], rr[i + 1])
                 for i in np.nonzero(np.sign(h[:-1]) != np.sign(h[1:]))[0]]
        stable = []
        for r0 in roots:
            u0 = np.array([u_in[b, 0] / (1 - loop(r0))])
            rho = np.abs(np.linalg.eigvals(_numerical_real_jacobian(u0, K[b], pa))).max()
            if rho < 1:
                stable.append((u0[0], rho))
        assert len(stable) == 1                                         # 唯一稳定（饱和）根；其余根不稳定
        assert np.isclose(res.u[b, 0], stable[0][0], rtol=1e-8)
        assert np.isclose(res.loop_gain_op[b], stable[0][1], rtol=1e-5)
    monkeypatch.setattr(relay, "_STALL_WINDOW", 10 ** 9)
    off = solve_fixed_point(u_in, K, pa)
    assert np.array_equal(off.u, res.u) and np.array_equal(off.possibly_oscillating, res.possibly_oscillating)


def test_newton_rescue_accepts_only_stable_roots():
    """救援 Newton（R2-RELAY-5）：从任意迭代点出发，只采用"达到收敛判据且 ρ(J_F) < 1"的根。
    |Gκ|=1.5、无 AM-PM、实正注入有两个实根：小信号根 u_in/(1−1.5)（ρ ≈ 1.5，不稳定）与饱和根（稳定）。"""
    pa = RappPA(ampm=False)
    G, A_in = pa.gain_v, pa.A_in
    r_in = 0.02
    src = np.full((2, 1), r_in * A_in + 0j)
    prob = relay._FPProblem(src, np.full((2, 1, 1), 1.5 / G + 0j), pa)
    u_unstable = r_in * A_in / (1 - 1.5)
    r_star = brentq(lambda r: r_in + 1.5 * r * _c1(r) - r, r_in, 10)
    u = np.array([[0.9 * u_unstable], [0.8 * r_star * A_in]], dtype=complex)   # 分别靠近两个根
    u0, it = u.copy(), np.zeros(2, dtype=int)
    acc = relay._newton_rescue(prob, u, it, 1e-12)
    assert acc.tolist() == [False, True]
    assert u[0, 0] == u0[0, 0] and it[0] == 0                          # 被拒绝：保留原迭代点
    assert np.isclose(abs(u[1, 0]) / A_in, r_star, rtol=1e-10) and it[1] > 0
    # 端到端：迭代预算太小（max_iter=20）时阻尼迭代到不了 Newton 门限，救援仍能给出稳定稳态；
    # newton=False（纯阻尼迭代）不做救援 → 未收敛、标记
    r1 = solve_fixed_point(src[:1], prob.K[:1], pa, max_iter=20)
    assert r1.converged[0] and not r1.possibly_oscillating[0]
    assert np.isclose(abs(r1.u[0, 0]) / A_in, r_star, rtol=1e-10)
    r2 = solve_fixed_point(src[:1], prob.K[:1], pa, max_iter=20, newton=False)
    assert not r2.converged[0] and r2.possibly_oscillating[0]


def test_default_direct_loop_not_resonant():
    """跟踪（R2-RELAY-1）：默认参数下仅经板间直耦 κ0 的环路（空场景，H 的 rx←tx 块为 0）必须远离谐振：
    |G_v K| ≤ −3 dB，且线性对照 M1 的空场景样本不被标记。用 relay_linear 的实际 K（与 κ0 的参考面定义无关）。"""
    rng = np.random.default_rng(95)
    params = RelayParams()
    N, n_probe = params.n_elem, 2
    H = np.zeros((N + n_probe, 1 + N), complex)
    H[:, 0] = crandn(rng, N + n_probe)                                 # 馈源列；rx←tx 块为 0（空场景）
    H[N:, 1:] = crandn(rng, n_probe, N)
    w, ct = weights(bits_to_phase(rng.integers(0, 2, (1, N))), bits_to_phase(rng.integers(0, 2, (1, N))), params)
    lin = relay_linear(H, w, ct, 1.0, params)
    for pa in (LinearPA(DEFAULTS["pa_gain_db"]), RappPA()):
        assert np.abs(pa.gain_v * lin.K).max() <= 10 ** (-3 / 20)
    res = solve_fixed_point(lin.u_in, lin.K, LinearPA(DEFAULTS["pa_gain_db"]))
    assert res.converged and not res.possibly_oscillating


# ---------------------------------------------------------------- 探头场与噪声
def test_probe_fields_and_noise_power():
    rng = np.random.default_rng(70)
    d, C, g = crandn(rng, 4, 3), crandn(rng, 4, 3, 2), crandn(rng, 4, 2)
    z = probe_fields(d, C, g)
    for b in range(4):
        assert np.allclose(z[b], d[b] + C[b] @ g[b])
    n = crandn(rng, 4, 3)
    assert np.allclose(probe_fields(d, C, g, n), z + n)
    assert np.allclose(probe_fields(d[0], C[0], g[0]), z[0])
    p = RelayParams(nf_db=5.0, noise_bw_hz=1e3, temp_k=290.0)
    assert np.isclose(pa_noise_power_w(p), KB * 290.0 * 1e3 * 10 ** 0.5, rtol=1e-12)
    assert np.isclose(10 * np.log10(pa_noise_power_w(p) * 1e3), -174 + 30 + 5, atol=0.05)   # −139 dBm


# ---------------------------------------------------------------- 方案 §6.1 手算例
def _threshold_separates(mag, labels):
    """是否存在阈值 t 使 奇 > t ≥ 偶（或反向）。"""
    odd, even = mag[labels == 1], mag[labels == 0]
    return odd.min() > even.max() or even.min() > odd.max()


def test_scheme_6_1_worked_example_3bit_and_4bit():
    """方案 §6.1：u 只依赖 k=Σx_i，d(x)=d·k，|g(k)| = 0, 0.8A, A, A，取 cA = −2d
    ⇒ |z(k)| = |dk + c g(k)| = 0, 0.6|d|, 0, |d|，阈值 0.3|d| 分开 3 位奇偶；4 位延拓 z(4) = 2|d| 失败。"""
    d = 0.7 * np.exp(0.4j)
    A = 2.3
    c = -2 * d / A
    k = np.arange(5)
    g = A * np.array([0.0, 0.8, 1.0, 1.0, 1.0])
    z = d * k + c * g
    assert np.allclose(np.abs(z), np.abs(d) * np.array([0, 0.6, 0, 1, 2]))
    # 3 位：8 个输入按 k 映射
    x3 = np.arange(8)
    k3 = np.array([bin(i).count("1") for i in x3])
    lab3 = k3 % 2
    mag3 = np.abs(z[k3])
    assert _threshold_separates(mag3, lab3)
    t = 0.3 * np.abs(d)
    assert np.all((mag3 > t) == (lab3 == 1))
    # 4 位：同一个 c 失败
    k4 = np.array([bin(i).count("1") for i in range(16)])
    lab4 = k4 % 2
    assert not _threshold_separates(np.abs(z[k4]), lab4)
    # 更强：g(2..4) 同为饱和值 A 时，任意 c 都不行（z(2),z(3),z(4) 共线等距 + |·| 凸 + z(0)=0）
    rng = np.random.default_rng(80)
    cs = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)) * 3 * np.abs(d) / A
    zz = d * k[None, :] + cs[:, None] * g[None, :]
    mags = np.abs(zz[:, k4])
    assert not any(_threshold_separates(m, lab4) for m in mags)


@pytest.mark.parametrize("p", [2.0, 5.0])
def test_scheme_6_1_with_real_rapp_curve(p):
    """用真实 Rapp 曲线代替理想 |g| 剖面：u(k) = k·u1，选 u1 使 |g(u1)| = 0.8A_s，cA_s = −2d。
    3 位奇偶仍可由阈值分开（裕量随 p 变化），4 位仍失败。"""
    pa = RappPA(p=p, ampm=False)
    r1 = brentq(lambda r: r * _c1(r, p) - 0.8, 1e-3, 10)
    u = np.arange(5) * r1 * pa.A_in
    g = pa.g(u)
    assert np.isclose(np.abs(g[1]), 0.8 * pa.A_s)
    d = 1.0
    c = -2 * d / pa.A_s
    z = d * np.arange(5) + c * g
    k3 = np.array([bin(i).count("1") for i in range(8)])
    k4 = np.array([bin(i).count("1") for i in range(16)])
    assert _threshold_separates(np.abs(z[k3]), k3 % 2)
    assert not _threshold_separates(np.abs(z[k4]), k4 % 2)
    # 3 位：偶数最大（k=2 处的残余）< 奇数最小（0.6|d|）
    assert np.abs(z[2]) < 0.1 and np.isclose(np.abs(z[1]), 0.6)


# ============================================================ 审查 S0-0：自激判据（cw / robust / small_signal）
def test_loop_gain_robust_is_worst_case_over_leakage_phase_R1():
    """R = 1：ρ_rob = |K|(|a|+|b|) 恰是"泄漏相位取最坏值"时 CW 实雅可比谱半径的最大值，且 ρ_op ≤ ρ_rob；
    线性 PA：ρ_rob = ρ_ss。"""
    pa = RappPA()
    kappa = 10 ** (6.0 / 20) / pa.gain_v * np.exp(1j * np.pi / 2)        # |G_v κ| = +6 dB，φ = 90°
    inj = np.sqrt(pa.p1db_in_w() * 10 ** (np.array([-15.0, 0.0, 10.0, 20.0]) / 10)).astype(complex)
    K = np.full((4, 1, 1), kappa)
    res = solve_fixed_point(inj[:, None], K, pa)
    assert res.loop_gain_robust.shape == (4,) and res.criterion == "cw"
    assert np.all(res.loop_gain_op <= res.loop_gain_robust * (1 + 1e-12))
    a, b = pa.wirtinger(res.u)
    for i in range(4):
        M = real_jacobian(K[i], a[i], b[i])                       # 泄漏相位旋转 θ：J(e^{jθ}K) = Rot(θ)·M
        rho = [np.abs(np.linalg.eigvals(real_jacobian(K[i] * np.exp(1j * ph), a[i], b[i]))).max()
               for ph in np.linspace(0.0, 2 * np.pi, 1441)]
        assert max(rho) <= res.loop_gain_robust[i] * (1 + 1e-9)          # 上界
        # 取到上界的相位（峰极窄，网格可能错过）：M = UΣVᵀ，Rot(θ*) = V·D·Uᵀ（D = diag(1, ±1) 使其为旋转）
        U, sv, Vt = np.linalg.svd(M)
        D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        Rm = Vt.T @ D @ U.T
        th = np.arctan2(Rm[1, 0], Rm[0, 0])
        peak = np.abs(np.linalg.eigvals(real_jacobian(K[i] * np.exp(1j * th), a[i], b[i]))).max()
        assert peak == pytest.approx(res.loop_gain_robust[i], rel=1e-9) and sv[0] == pytest.approx(peak, rel=1e-9)
    assert np.allclose(relay.loop_gain_robust(res.u, K, pa), res.loop_gain_robust)
    lin = solve_fixed_point(inj[:, None], K * 0.5, LinearPA(pa.gain_db))
    assert np.allclose(lin.loop_gain_robust, lin.loop_gain_ss)
    # R = 2：上界 ‖K‖₂·max(|a|+|b|) ≥ ρ_op，形状正确
    rng = np.random.default_rng(0)
    K2 = 0.3 / pa.gain_v * crandn(rng, 3, 2, 2)
    r2 = solve_fixed_point(1e-3 * crandn(rng, 3, 2), K2, pa)
    assert r2.loop_gain_robust.shape == (3,) and np.all(r2.loop_gain_op <= r2.loop_gain_robust * (1 + 1e-12))


def test_osc_criteria_separate_three_regimes():
    """|G_v κ| = +6 dB、φ = 90°：
    弱注入 → 小信号不稳定根，三种判据都标记；+10 dB rel P1dB → CW 工作点稳定（ρ_op < 1，注入锁定）但 ρ_rob > 1
    （换个泄漏相位 / 频率会起振）→ 只有 robust / small_signal 标记；+20 dB → 深压缩，ρ_rob < 1 → 只有 small_signal 标记。"""
    pa = RappPA()
    kappa = 10 ** (6.0 / 20) / pa.gain_v * np.exp(1j * np.pi / 2)
    inj = np.sqrt(pa.p1db_in_w() * 10 ** (np.array([-15.0, 10.0, 20.0]) / 10)).astype(complex)[:, None]
    K = np.full((3, 1, 1), kappa)
    flags = {c: solve_fixed_point(inj, K, pa, criterion=c).possibly_oscillating for c in relay.OSC_CRITERIA}
    assert flags["cw"].tolist() == [True, False, False]
    assert flags["robust"].tolist() == [True, True, False]
    assert flags["small_signal"].tolist() == [True, True, True]
    res = solve_fixed_point(inj, K, pa)
    assert res.ss_unstable.all() and res.robust_unstable.tolist() == [True, True, False]
    assert np.array_equal(res.possibly_oscillating, flags["cw"])          # 默认 = DESIGN §3.4 的 CW 判据
    with pytest.raises(ValueError):
        solve_fixed_point(inj, K, pa, criterion="strict")
    assert relay.gain_at_least_one(np.array([1.0 - 1e-12, 0.999, np.nan])).tolist() == [True, False, True]
