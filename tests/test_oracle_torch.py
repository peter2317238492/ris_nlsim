"""oracle_torch.py（DESIGN §6.3）：torch 路径逐项对照独立的 numpy 参考实现与生产 numpy 管线
（solver → relay），并用有限差分验证隐函数梯度。"""
import itertools
import math
import types

import numpy as np
import pytest
import torch

from ris_nlsim import losses
from ris_nlsim import oracle_torch as ot
from ris_nlsim import pa as npa
from ris_nlsim import relay as nrl
from ris_nlsim import solver
from ris_nlsim.constants import F0, K0, LAMBDA, MATERIALS
from ris_nlsim.geometry import board_elements, make_relays, parity_slots, point_from, voxelize_sphere
from ris_nlsim.materials import alpha_passive, polarizability, voxels_to_nodes
from ris_nlsim.patterns import Pattern, element_pattern, horn_pattern, probe_pattern
from ris_nlsim.scene import NodeSet, Scene

N_EL = 25


# ============================================================ 数据构造
def _cn(rng, *shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)


def _ns(**kw):
    base = dict(alpha_p_base=None, Cpp=None, Ep=None, Hrp=None, Eo=None, Cop=None, Cpo=None, Lam=None, Hro=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def relay_samples(rng, B=4, R=1, K=2, sf=0.013, sk=0.01, sc=0.01, sd=0.002, n_p=0, mode="full"):
    """合成中继样本（n_p=0）：|u_in| ~ A_in（Rapp 默认参数下 P_in 在 P1dB 附近），|G_v K_free| ~ 0.2。"""
    nrx = R * N_EL
    S = 1 + nrx
    out = []
    for _ in range(B):
        H = np.zeros((nrx + K, S), complex)
        H[:nrx, 0] = sf * _cn(rng, nrx)
        H[:nrx, 1:] = sk * _cn(rng, nrx, nrx)
        H[nrx:, 0] = sd * _cn(rng, K)
        H[nrx:, 1:] = sc * _cn(rng, K, nrx)
        out.append(_ns(mode=mode, n_rx=nrx, n_probe=K, n_tx=nrx, n_p=n_p, H=H))
    return out


def reduced_sample(rng, mode, n_rcv=5, S=3, n_p=6, n_o=4):
    """随机 ReducedSample（数值条件良好即可，纯线性代数核对）。"""
    kw = dict(mode=mode, n_rx=0, n_probe=n_rcv, n_tx=S - 1, n_p=n_p, H=_cn(rng, n_rcv, S))
    if n_p > 0:
        kw.update(alpha_p_base=1.0 + 0.3 * _cn(rng, n_p), Cpp=0.15 * _cn(rng, n_p, n_p),
                  Ep=_cn(rng, n_p, S), Hrp=_cn(rng, n_rcv, n_p))
        if mode in ("born", "intra"):
            lam = np.diag(0.5 * _cn(rng, n_o)) if mode == "born" else 0.3 * _cn(rng, n_o, n_o)
            kw.update(Eo=_cn(rng, n_o, S), Cop=0.3 * _cn(rng, n_o, n_p), Cpo=0.3 * _cn(rng, n_p, n_o),
                      Lam=lam, Hro=_cn(rng, n_rcv, n_o))
    return _ns(**kw)


# ---- 真实小场景（L90 式几何，DESIGN §1.5）：用于对照生产 numpy 管线 solver → relay
def _facing_board(center, normal, n=3):
    """n×n 板（间距 λ/2），以 center 为中心、法向 normal（水平）。"""
    nrm = np.asarray(normal, float) / np.linalg.norm(normal)
    u = np.cross([0.0, 0.0, 1.0], nrm)
    u /= np.linalg.norm(u)
    v = np.array([0.0, 0.0, 1.0])
    corner = np.asarray(center) - 0.5 * n * (LAMBDA / 2) * (u + v)
    return board_elements(corner, u, v, n, n, LAMBDA / 2), nrm


def l90_scene(relay=True):
    """1 块中继（ψ=90°）、馈源 az −30°/1.8 m、探头 az 75°/105°/1.0 m、一个金属点环境、3×3 无源面（Γ=1）。"""
    rel = make_relays(1, 90.0)[0]
    fpos = point_from(rel.rx_center, -30.0, 0.0, 1.8)
    pp = np.array([point_from(rel.tx_center, a, 0.0, 1.0) for a in (75.0, 105.0)])
    ep = element_pattern()
    roi = rel.rx_center + np.array([0.5, 0.0, 0.0])
    env = [NodeSet([roi + [0.15, 0.35, 0.05]], None, Pattern(),
                   alpha=polarizability("metal", (LAMBDA / 5) ** 3, K0))]
    c = roi + np.array([-0.1, 0.45, 0.0])
    ppos, pn = _facing_board(c, roi - c)
    pas = NodeSet(ppos, pn, ep, alpha=alpha_passive(1.0, K0, LAMBDA))
    tx = [NodeSet(rel.tx_pos, rel.tx_normal, ep)] if relay else []
    rx = [NodeSet(rel.rx_pos, rel.rx_normal, ep)] if relay else []
    sc = Scene(F0, NodeSet(fpos, rel.rx_center - fpos, horn_pattern()), tx=tx, rx=rx,
               probes=NodeSet(pp, rel.tx_center - pp, probe_pattern()), env=env, passive=pas)
    return sc, roi


def parity_objects(roi, x):
    """奇偶槽位（沿 z，间距 λ/2）上直径 λ/3 的水球（体素 λ/10），x_j=1 表示存在；group = 槽号。"""
    slots = parity_slots(len(x), roi, LAMBDA / 2)
    sets = []
    for j, xj in enumerate(x):
        if xj:
            pos, v = voxelize_sphere(slots[j], LAMBDA / 3, LAMBDA / 10)
            sets.append(voxels_to_nodes(pos, v, MATERIALS["water"], K0, group=j))
    return NodeSet.concat(sets) if sets else NodeSet.empty()


# ============================================================ numpy 参考实现（照抄 DESIGN 公式）
def np_eval_reduced(rs, phi):
    """DESIGN §2.3。"""
    if rs.n_p == 0:
        return rs.H
    M = np.diag(1.0 / (rs.alpha_p_base * np.exp(1j * phi))) - rs.Cpp
    pP_bg = np.linalg.solve(M, rs.Ep)
    if rs.mode == "full":
        return rs.H + rs.Hrp @ pP_bg
    pO = rs.Lam @ (rs.Eo + rs.Cop @ pP_bg)
    pP = pP_bg + np.linalg.solve(M, rs.Cpo @ pO)
    return rs.H + rs.Hrp @ pP + rs.Hro @ pO


def np_rapp(u, gain_db=30.0, psat_out_dbm=20.0, p=2.0, ampm_max_deg=12.0, ampm_beta=1.0, ampm=True):
    """DESIGN §3.1：|g| 与 ∠g 分开写（与 torch 的 u·f(r²) 写法相互独立）。"""
    Gv = 10 ** (gain_db / 20)
    As = math.sqrt(10 ** ((psat_out_dbm - 30) / 10))
    Ain = As / Gv
    r = np.abs(u) / Ain
    mag = Gv * np.abs(u) / (1 + r ** (2 * p)) ** (1 / (2 * p))
    a_phi = np.deg2rad(ampm_max_deg) * ampm_beta if ampm else 0.0
    ph = np.angle(u) + a_phi * r ** 2 / (1 + ampm_beta * r ** 2)
    return mag * np.exp(1j * ph)


def np_relay_linear(H, phase_rx, phase_tx, s, loss_rx_db, loss_tx_db, kappa0, n_probes):
    """DESIGN §3.2–3.3，显式循环。"""
    R, n = phase_rx.shape
    w = 10 ** (-loss_rx_db / 20) / np.sqrt(n) * np.exp(1j * phase_rx)
    ct = 10 ** (-loss_tx_db / 20) / np.sqrt(n) * np.exp(1j * phase_tx)
    nrx = R * n
    u_in = np.zeros(R, complex)
    K = np.zeros((R, R), complex)
    d = np.zeros(n_probes, complex)
    C = np.zeros((n_probes, R), complex)
    for r in range(R):
        for i in range(n):
            u_in[r] += s * w[r, i] * H[r * n + i, 0]
            for sp in range(R):
                for m in range(n):
                    K[r, sp] += w[r, i] * H[r * n + i, 1 + sp * n + m] * ct[sp, m]
        K[r, r] += kappa0
    for k in range(n_probes):
        d[k] = s * H[nrx + k, 0]
        for sp in range(R):
            for m in range(n):
                C[k, sp] += H[nrx + k, 1 + sp * n + m] * ct[sp, m]
    return u_in, K, d, C


def np_fixed_point(u_in, K, g, beta=0.5, n_iter=3000):
    """最朴素的阻尼迭代 u ← (1−β)u + β(u_in + K g(u))。"""
    u = u_in.copy()
    for _ in range(n_iter):
        u = (1 - beta) * u + beta * (u_in + K @ g(u))
    return u


# ============================================================ PA
@pytest.mark.parametrize("p", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("ampm", [True, False])
def test_pa_matches_numpy_formula(p, ampm):
    rng = np.random.default_rng(0)
    pa = ot.TorchPA("rapp", gain_db=28.0, psat_out_dbm=17.0, p=p, ampm_max_deg=15.0, ampm_beta=0.7, ampm=ampm)
    mag = pa.a_in * 10 ** rng.uniform(-2, 1.5, size=200)      # −40 dB … +30 dB 相对输入饱和幅度
    u = mag * np.exp(1j * rng.uniform(0, 2 * np.pi, size=200))
    ref = np_rapp(u, 28.0, 17.0, p, 15.0, 0.7, ampm)
    got = pa.g(torch.as_tensor(u)).numpy()
    assert np.allclose(got, ref, rtol=1e-13, atol=0)
    assert np.allclose(pa.g_np(u), ref, rtol=1e-13, atol=0)           # 无梯度求解用的 numpy 孪生实现
    a_t, b_t = pa.wirtinger(torch.as_tensor(u))
    a_n, b_n = pa.wirtinger_np(u)
    assert np.allclose(a_n, a_t.numpy(), rtol=1e-13) and np.allclose(b_n, b_t.numpy(), rtol=1e-13, atol=1e-15)


def test_pa_p1db_ampm_limit_and_linear():
    for p in (1.0, 2.0, 5.0):
        pa = ot.TorchPA("rapp", p=p)
        u = torch.tensor([math.sqrt(pa.p1db_in_w()) + 0j], dtype=torch.complex128)
        gain_db = 20 * math.log10(abs(complex(pa.g(u)[0])) / abs(complex(u[0])))
        assert gain_db == pytest.approx(pa.gain_db - 1.0, abs=1e-10)       # P1dB 处恰好压缩 1 dB
    pa = ot.TorchPA("rapp", ampm_max_deg=12.0, ampm_beta=1.0)
    u = torch.tensor([1e4 * pa.a_in + 0j], dtype=torch.complex128)
    dphi = np.angle(complex(pa.g(u)[0]) / complex(u[0]))
    assert np.rad2deg(dphi) == pytest.approx(12.0, abs=1e-5)                # 深饱和 AM-PM 极限
    lin = ot.make_pa({"type": "linear", "gain_db": 25.0})
    u = torch.as_tensor(_cn(np.random.default_rng(1), 5))
    assert torch.allclose(lin.g(u), 10 ** (25 / 20) * u)
    assert lin.p1db_in_w() == math.inf


@pytest.mark.parametrize("p", [1.0, 2.0, 5.0])
def test_pa_wirtinger_matches_finite_difference(p):
    pa = ot.TorchPA("rapp", p=p, ampm_max_deg=20.0, ampm_beta=1.3)
    rng = np.random.default_rng(2)
    u = pa.a_in * 10 ** rng.uniform(-1.5, 1.0, size=50) * np.exp(1j * rng.uniform(0, 2 * np.pi, size=50))
    a, b = (t.numpy() for t in pa.wirtinger(torch.as_tensor(u)))
    h = 1e-6 * np.abs(u)
    g = lambda x: pa.g(torch.as_tensor(x)).numpy()
    dre = (g(u + h) - g(u - h)) / (2 * h)               # = a + b
    dim = (g(u + 1j * h) - g(u - 1j * h)) / (2j * h)     # = a − b
    assert np.allclose(dre, a + b, rtol=1e-7, atol=1e-7 * pa.gain_v)
    assert np.allclose(dim, a - b, rtol=1e-7, atol=1e-7 * pa.gain_v)
    a0, b0 = pa.wirtinger(torch.zeros(3, dtype=torch.complex128))   # u→0：a→G_v，b→0
    assert torch.allclose(a0, torch.full_like(a0, pa.gain_v)) and torch.all(b0 == 0)


# ============================================================ §2.3 eval_reduced
@pytest.mark.parametrize("mode", ["full", "born", "intra"])
@pytest.mark.parametrize("n_p", [0, 6])
def test_eval_reduced_torch_matches_numpy(mode, n_p):
    rng = np.random.default_rng(3)
    rs = reduced_sample(rng, mode, n_p=n_p)
    phi = rng.uniform(0, 2 * np.pi, size=n_p)
    ref = np_eval_reduced(rs, phi)
    got = ot.eval_reduced_torch(rs, torch.as_tensor(phi) if n_p else None).numpy()
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_eval_reduced_batch_padding_is_exact():
    rng = np.random.default_rng(4)
    rss = [reduced_sample(rng, "born", n_o=n_o) for n_o in (1, 4, 7)]
    phi = rng.uniform(0, 2 * np.pi, size=6)
    batch = ot.eval_reduced_batch(ot.stack_reduced(rss), torch.as_tensor(phi)).numpy()
    for b, rs in enumerate(rss):
        assert np.allclose(batch[b], np_eval_reduced(rs, phi), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("mode", ["full", "intra"])
def test_eval_reduced_torch_gradient(mode):
    rng = np.random.default_rng(5)
    rs = reduced_sample(rng, mode)
    phi0 = rng.uniform(0, 2 * np.pi, size=6)
    wts = _cn(rng, *rs.H.shape)
    f_np = lambda ph: float(np.real(np.sum(wts * np_eval_reduced(rs, ph))))
    phi = torch.tensor(phi0, requires_grad=True)
    torch.real(torch.sum(torch.as_tensor(wts) * ot.eval_reduced_torch(rs, phi))).backward()
    h = 1e-6
    fd = np.array([(f_np(phi0 + h * e) - f_np(phi0 - h * e)) / (2 * h) for e in np.eye(6)])
    assert np.allclose(phi.grad.numpy(), fd, rtol=1e-6, atol=1e-8)


# ============================================================ §3.3 中继线性量
def test_relay_linear_matches_explicit_loops():
    rng = np.random.default_rng(6)
    R, Kp = 2, 3
    rs = relay_samples(rng, B=3, R=R, K=Kp)
    H = np.stack([r.H for r in rs])
    prx, ptx = rng.uniform(0, 2 * np.pi, (R, N_EL)), rng.uniform(0, 2 * np.pi, (R, N_EL))
    s = np.array([0.7, 1.0, 1.9])
    kappa0 = 10 ** (-27 / 20) * np.exp(1j * 0.4)
    w, ct = ot.relay_weights_torch(torch.as_tensor(prx), torch.as_tensor(ptx), 2.0, 3.0)
    u_in, K, d, C = (t.numpy() for t in ot.relay_linear_torch(torch.as_tensor(H), w, ct, torch.as_tensor(s),
                                                               kappa0, n_probes=Kp))
    for b in range(3):
        ru, rK, rd, rC = np_relay_linear(H[b], prx, ptx, s[b], 2.0, 3.0, kappa0, Kp)
        assert np.allclose(u_in[b], ru, rtol=1e-13) and np.allclose(K[b], rK, rtol=1e-12)
        assert np.allclose(d[b], rd, rtol=1e-13) and np.allclose(C[b], rC, rtol=1e-12)


# ============================================================ §3.4 不动点
@pytest.mark.parametrize("R", [1, 2])
def test_fixed_point_matches_numpy_damped_iteration(R):
    rng = np.random.default_rng(7 + R)
    pa_kw = dict(gain_db=30.0, psat_out_dbm=20.0, p=2.0, ampm_max_deg=12.0, ampm_beta=1.0)
    pa = ot.TorchPA("rapp", **pa_kw)
    B = 5
    u_in = pa.a_in * 10 ** rng.uniform(-1, 1, (B, R)) * np.exp(1j * rng.uniform(0, 2 * np.pi, (B, R)))
    K = 0.6 / pa.gain_v * _cn(rng, B, R, R)                 # 小信号环路增益 ~0.6
    res = ot.fixed_point_torch(torch.as_tensor(u_in), torch.as_tensor(K), pa)
    assert res.converged.all()
    for b in range(B):
        ref = np_fixed_point(u_in[b], K[b], lambda x: np_rapp(x, **pa_kw))
        assert np.allclose(res.u[b].detach().numpy(), ref, rtol=1e-11, atol=0)
        assert np.allclose(res.g[b].detach().numpy(), np_rapp(ref, **pa_kw), rtol=1e-11)


def test_fixed_point_linear_closed_form_and_flag():
    pa = ot.TorchPA("linear", gain_db=30.0)
    u_in = torch.tensor([[1e-3 + 2e-3j], [3e-3 + 0j]], dtype=torch.complex128)
    kap = np.array([0.5 / pa.gain_v * np.exp(0.3j), 1.2 / pa.gain_v * np.exp(-1j)])
    K = torch.as_tensor(kap.reshape(2, 1, 1))
    res = ot.fixed_point_torch(u_in, K, pa)
    ref = u_in.numpy()[:, 0] / (1 - pa.gain_v * kap)       # u = u_in/(1 − G_v κ)
    assert np.allclose(res.u.detach().numpy()[:, 0], ref, rtol=1e-13)
    assert np.allclose(res.loop_gain_op, [0.5, 1.2], rtol=1e-12)
    assert list(res.possibly_oscillating) == [False, True]    # |G_v κ| ≥ 1 被标记为可能自激


def test_rapp_deep_backoff_matches_linear():
    rng = np.random.default_rng(8)
    rapp, lin = ot.TorchPA("rapp", ampm=True), ot.TorchPA("linear")
    u_in = torch.as_tensor(1e-4 * rapp.a_in * _cn(rng, 3, 2))   # −80 dB 回退
    K = torch.as_tensor(0.3 / rapp.gain_v * _cn(rng, 3, 2, 2))
    ur = ot.fixed_point_torch(u_in, K, rapp).u.detach().numpy()
    ul = ot.fixed_point_torch(u_in, K, lin).u.detach().numpy()
    assert np.allclose(ur, ul, rtol=1e-6)


def test_loop_gain_op_matches_numerical_jacobian():
    rng = np.random.default_rng(9)
    pa = ot.TorchPA("rapp", ampm_max_deg=25.0)
    R = 2
    u_in = pa.a_in * 3.0 * _cn(rng, 1, R)
    K = 0.9 / pa.gain_v * _cn(rng, 1, R, R)
    res = ot.fixed_point_torch(torch.as_tensor(u_in), torch.as_tensor(K), pa)
    u = res.u.detach().numpy()[0]
    F = lambda x: u_in[0] + K[0] @ np_rapp(x, ampm_max_deg=25.0)
    J = np.zeros((2 * R, 2 * R))
    h = 1e-7 * pa.a_in
    for j in range(2 * R):
        e = np.zeros(R, complex)
        e[j % R] = h if j < R else 1j * h
        dF = (F(u + e) - F(u - e)) / (2 * h)
        J[:, j] = np.concatenate([dF.real, dF.imag])
    assert np.allclose(res.J.numpy()[0], J, rtol=1e-6, atol=1e-8)
    assert res.loop_gain_op[0] == pytest.approx(np.max(np.abs(np.linalg.eigvals(J))), rel=1e-6)
    assert res.loop_gain_op[0] < res.loop_gain_ss[0]          # 饱和压低工作点环路增益


# ============================================================ TorchSystem
def _relay_system(rng, R, B=4, pa=None, isolation_db=-40.0, task="binary", **kw):
    rs = relay_samples(rng, B=B, R=R, K=2 if task != "multiclass" else 3)
    return ot.TorchSystem(rs, kind="relay", s_feed=1.0, n_relays=R, pa=pa or {"type": "rapp"},
                          relay={"isolation_db": isolation_db, "kappa0_phase_deg": 40.0, "loss_rx_db": 2.5,
                                 "loss_tx_db": 2.5}, labels=np.arange(B) % 2, task=task,
                          n_probes=2 if task != "multiclass" else 3, **kw), rs


def test_relay_readings_match_numpy_pipeline():
    rng = np.random.default_rng(10)
    R = 2
    perr = rng.normal(0, 0.2, (R, N_EL))
    rs = relay_samples(rng, B=3, R=R)
    ps = np.array([1.0, 0.5, 2.0])
    ts = ot.TorchSystem(rs, kind="relay", s_feed=0.8, n_relays=R, pa={"type": "rapp", "p": 3.0},
                        relay={"isolation_db": -35.0, "kappa0_phase_deg": 70.0, "phase_err_rx": perr},
                        labels=np.array([0, 1, 0]), task="binary", power_scale=ps)
    ph = {k: v.numpy() for k, v in ts.init_phases(3).items()}
    got = ts.readings_db(ts.as_phases(ph)).detach().numpy()
    kappa0 = 10 ** (-35 / 20) * np.exp(1j * np.deg2rad(70.0)) * 10 ** (-5.0 / 20)   # × 收发网络损耗 2.5+2.5 dB
    g = lambda x: np_rapp(x, p=3.0)
    for b in range(3):
        u_in, K, d, C = np_relay_linear(rs[b].H, ph["phase_rx"] + perr, ph["phase_tx"], 0.8 * np.sqrt(ps[b]),
                                        2.5, 2.5, kappa0, 2)
        u = np_fixed_point(u_in, K, g)
        z = d + C @ g(u)
        assert np.allclose(got[b], 10 * np.log10(np.abs(z) ** 2) + 30, rtol=0, atol=1e-9)


def test_passive_readings_and_gradient():
    rng = np.random.default_rng(11)
    rss = [reduced_sample(rng, "intra", n_rcv=2, S=1, n_p=8, n_o=n_o) for n_o in (2, 5)]
    ps = np.array([1.0, 3.0])
    ts = ot.TorchSystem(rss, kind="passive", s_feed=0.5, labels=np.array([0, 1]), task="binary", power_scale=ps)
    assert set(ts.param_shapes) == {"phase_p"}
    phi0 = rng.uniform(0, 2 * np.pi, 8)
    got = ts.readings_db({"phase_p": torch.as_tensor(phi0)}).numpy()
    ref = np.stack([10 * np.log10(np.abs(0.5 * np.sqrt(ps[b]) * np_eval_reduced(rss[b], phi0)[:2, 0]) ** 2) + 30
                    for b in range(2)])
    assert np.allclose(got, ref, rtol=0, atol=1e-10)
    f_np = lambda ph: losses.binary_loss(
        np.stack([10 * np.log10(np.abs(0.5 * np.sqrt(ps[b]) * np_eval_reduced(rss[b], ph)[:2, 0]) ** 2) + 30
                  for b in range(2)]), np.array([0, 1]), 1.0)
    phi = torch.tensor(phi0, requires_grad=True)
    ts.loss({"phase_p": phi}).backward()
    h = 1e-6
    fd = np.array([(f_np(phi0 + h * e) - f_np(phi0 - h * e)) / (2 * h) for e in np.eye(8)])
    assert np.allclose(phi.grad.numpy(), fd, rtol=1e-6, atol=1e-8)


def _flat_grad(fn, ph):
    ph = {k: v.clone().requires_grad_(True) for k, v in ph.items()}
    loss = fn(ph)
    loss.backward()
    return float(loss.detach()), {k: v.grad.numpy().copy() for k, v in ph.items()}


def _fd_check(ts, ph, n_check, rng, h=1e-6, rtol=1e-5, fn=None):
    """torch 梯度 vs 中心差分；fn(phases) → 标量张量，默认 ts.loss。"""
    fn = ts.loss if fn is None else fn
    f0, grad = _flat_grad(fn, ph)
    keys = list(ph)
    for _ in range(n_check):
        k = keys[rng.integers(len(keys))]
        idx = tuple(rng.integers(s) for s in ph[k].shape)
        hi = {kk: v.clone() for kk, v in ph.items()}
        lo = {kk: v.clone() for kk, v in ph.items()}
        hi[k][idx] += h
        lo[k][idx] -= h
        with torch.no_grad():
            fd = (float(fn(hi)) - float(fn(lo))) / (2 * h)
        g = grad[k][idx]
        assert abs(g - fd) <= rtol * max(abs(fd), 1e-3), (k, idx, g, fd)
    # 方向导数（同时扰动全部参数）
    dirs = {k: torch.as_tensor(rng.normal(size=v.shape)) for k, v in ph.items()}
    with torch.no_grad():
        fp = float(fn({k: ph[k] + h * dirs[k] for k in ph}))
        fm = float(fn({k: ph[k] - h * dirs[k] for k in ph}))
    dd_fd = (fp - fm) / (2 * h)
    dd = sum(float(np.sum(grad[k] * dirs[k].numpy())) for k in ph)
    assert abs(dd - dd_fd) <= rtol * max(abs(dd_fd), 1e-3)
    return grad


@pytest.mark.parametrize("R", [1, 2])
def test_implicit_gradient_matches_finite_difference(R):
    rng = np.random.default_rng(20 + R)
    ts, _ = _relay_system(rng, R, B=4, isolation_db=-31.0)     # G_v|κ_eff| = 30−5−31 = −6 dB + 自由空间回耦
    ph = ts.init_phases(rng)
    out = ts.forward(ph)
    assert np.all(out["fp"].loop_gain_op > 0.1)                  # 环路耦合不可忽略
    assert np.all(out["fp"].loop_gain_op < 0.95)                 # 无自激惩罚，纯隐函数梯度
    grad = _fd_check(ts, ph, n_check=12, rng=rng)
    # 对照：忽略环路（把 u* 当常数，只走 F 的显式依赖一次）得到的梯度明显不同 → 隐函数项确有贡献
    ph2 = {k: v.clone().requires_grad_(True) for k, v in ph.items()}
    o2 = ts.forward(ph2)
    u_star = o2["fp"].u.detach()
    g_naive = ts.pa.g(o2["u_in"] + (o2["K"] @ ts.pa.g(u_star).unsqueeze(-1)).squeeze(-1))
    z = o2["d"] + (o2["C"] @ g_naive.unsqueeze(-1)).squeeze(-1)
    losses.binary_loss(ts._dbm(z), ts.labels, ts.tau).backward()
    diff = np.linalg.norm(ph2["phase_rx"].grad.numpy() - grad["phase_rx"]) / np.linalg.norm(grad["phase_rx"])
    assert diff > 0.05


def test_gradient_with_null_and_osc_penalties():
    rng = np.random.default_rng(30)
    R = 1
    rs = relay_samples(rng, B=3, R=R)
    H0 = rs[0].H.copy()
    ts = ot.TorchSystem(rs, kind="relay", s_feed=1.0, n_relays=R, pa={"type": "linear", "gain_db": 30.0},
                        relay={"isolation_db": -25.0, "kappa0_phase_deg": 5.0}, labels=np.array([0, 1, 1]),
                        task="binary", null_ref={"H0": H0, "u0_ref_abs2": 1e-4, "target_db": -20.0, "weight": 2.0})
    ph = ts.init_phases(rng)
    out = ts.forward(ph)
    rho = out["fp"].loop_gain_op
    flags = out["flags"]
    assert np.any(rho >= 0.95)                                   # 自激惩罚处于激活状态
    assert flags.any() and (~flags).any()                        # 既有被标记样本也有正常样本
    # 损失数值与手算一致（INTEGRATION C：被标记样本 → τ·log2 + 1.0；惩罚按样本平均；零陷取中继最大值）
    w = out["w"].detach().numpy()
    u0 = (w[0] * H0[:N_EL, 0]).sum()
    nd = 10 * np.log10(abs(u0) ** 2 / 1e-4)
    per = losses.binary_loss(out["readings_db"].detach().numpy(), ts.labels, 1.0, reduction="none")
    per = np.where(flags, np.log(2.0) + 1.0, per)
    ref = per.mean() + 2.0 * np.logaddexp(0, nd + 20.0) + 100.0 * np.mean(np.maximum(rho - 0.95, 0) ** 2)
    with torch.no_grad():
        assert float(ts.loss(ph)) == pytest.approx(ref, rel=1e-12)
    _fd_check(ts, ph, n_check=8, rng=rng)


def test_strong_loop_unstable_fixed_point_flagged_and_gradient_exact():
    """G_v|κ0| ≈ +3 dB：生产求解器的阻尼迭代对部分样本收敛到不稳定不动点（ρ_op ≥ 1，DESIGN §3.4 注意事项）
    → 标记为可能自激，且标记、u 与 relay.solve_fixed_point 逐样本相同（TR-2）。
    该不动点处读数的隐函数梯度仍精确；被标记样本的任务损失为常数，损失梯度来自 ρ_op 惩罚（同样经隐函数求导）。"""
    for seed in range(55, 75):          # 选一个"既有收敛到不稳定根、又有稳定根"的随机实例（不依赖求解器细节）
        rng = np.random.default_rng(seed)
        ts, _ = _relay_system(rng, 1, B=6, isolation_db=-22.0)
        ph = ts.init_phases(rng)
        out = ts.forward(ph)
        fp = out["fp"]
        unstable = fp.converged & (fp.loop_gain_op >= 1.0)
        stable = fp.converged & (fp.loop_gain_op < 0.95)
        if unstable.any() and stable.any():
            break
    else:
        pytest.fail("找不到同时含稳定/不稳定收敛根的实例")
    assert np.array_equal(fp.possibly_oscillating[fp.converged], unstable[fp.converged])
    ref = nrl.solve_fixed_point(out["u_in"].detach().numpy(), out["K"].detach().numpy(), npa.RappPA())
    assert np.array_equal(ref.possibly_oscillating, fp.possibly_oscillating)
    assert np.array_equal(ref.u, fp.u.detach().numpy())        # 前向值恰为 numpy 的 u*
    conv = torch.as_tensor(fp.converged)
    wts = torch.as_tensor(rng.normal(size=(6, 2))) * conv[:, None]    # 只看收敛样本的读数梯度
    _fd_check(ts, ph, n_check=6, rng=rng, fn=lambda p: (wts * ts.readings_db(p)).sum())
    _fd_check(ts, ph, n_check=6, rng=rng)


def test_nonconverged_samples_detached_and_penalized_by_small_signal_gain():
    """G_v|κ0| ≈ +15 dB：AM-PM 下深饱和 CW 稳态一般不存在（环路相位条件不满足），不动点不收敛。
    读数取 relay 的最后迭代值 g(u*)（不做 Newton 外推，TR-1），梯度被切断；
    损失梯度只来自 100·mean_b relu(ρ(G_v K) − 0.95)²，与 numpy 有限差分一致。"""
    rng = np.random.default_rng(51)
    ts, rs = _relay_system(rng, 1, B=3, isolation_db=-15.0)
    ts.fp_max_iter = 300
    ph = ts.init_phases(rng)
    out = ts.forward(ph)
    assert not out["fp"].converged.any() and out["flags"].all()
    ref = nrl.solve_fixed_point(out["u_in"].detach().numpy(), out["K"].detach().numpy(), npa.RappPA(),
                                max_iter=300)
    assert not ref.converged.any()
    assert np.array_equal(out["fp"].u.detach().numpy(), ref.u)          # 最后迭代值本身
    z_ref = nrl.probe_fields(out["d"].detach().numpy(), out["C"].detach().numpy(), ref.g)
    assert np.allclose(out["z"].detach().numpy(), z_ref, rtol=1e-13, atol=0)
    ph_g = {k: v.clone().requires_grad_(True) for k, v in ph.items()}
    y = ts.readings_db(ph_g)
    if y.requires_grad:                                                  # 读数梯度整体切断（零或无梯度路径）
        y.sum().backward()
        assert all(v.grad is None or torch.all(v.grad == 0) for v in ph_g.values())
    _, grad = _flat_grad(ts.loss, ph)
    kappa0 = 10 ** (-15 / 20) * np.exp(1j * np.deg2rad(40.0)) * 10 ** (-5.0 / 20)   # × 收发网络损耗

    def pen(prx, ptx):
        tot = 0.0
        for b in range(3):
            _, K, _, _ = np_relay_linear(rs[b].H, prx, ptx, 1.0, 2.5, 2.5, kappa0, 2)
            rho = np.max(np.abs(np.linalg.eigvals(ts.pa.gain_v * K)))
            tot += 100.0 * max(rho - 0.95, 0.0) ** 2
        return tot / 3.0

    prx, ptx = ph["phase_rx"].numpy(), ph["phase_tx"].numpy()
    h = 1e-6
    for k, (i, j) in [("phase_rx", (0, 3)), ("phase_tx", (0, 17)), ("phase_rx", (0, 22))]:
        e = np.zeros_like(prx)
        e[i, j] = h
        fd = (pen(prx + e, ptx) - pen(prx - e, ptx)) / (2 * h) if k == "phase_rx" else \
             (pen(prx, ptx + e) - pen(prx, ptx - e)) / (2 * h)
        assert grad[k][i, j] == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_multiclass_rapp_gradient_R2():
    rng = np.random.default_rng(31)
    ts, _ = _relay_system(rng, 2, B=3, task="multiclass", isolation_db=-38.0)
    ts.labels = np.array([0, 1, 2])
    _fd_check(ts, ts.init_phases(rng), n_check=6, rng=rng)


def test_init_phases_shapes_and_determinism():
    rng = np.random.default_rng(12)
    ts, _ = _relay_system(rng, 2)
    a, b = ts.init_phases(5), ts.init_phases(np.random.default_rng(5))
    assert set(a) == {"phase_rx", "phase_tx"} and a["phase_rx"].shape == (2, N_EL)
    for k in a:
        assert torch.equal(a[k], b[k]) and a[k].dtype == torch.float64
        assert float(a[k].min()) >= 0 and float(a[k].max()) < 2 * np.pi


def test_constructor_validation():
    rng = np.random.default_rng(13)
    rs = relay_samples(rng, B=2, R=1)
    with pytest.raises(ValueError):
        ot.TorchSystem(rs, kind="relay", s_feed=1.0, n_relays=2, labels=np.zeros(2), task="binary")
    with pytest.raises(ValueError):
        ot.TorchSystem(rs, kind="bogus", s_feed=1.0, labels=np.zeros(2), task="binary")
    with pytest.raises(ValueError):                       # PA 字典拼错键（如 SystemConfig 的 rapp_p）要报错
        ot.TorchSystem(rs, kind="relay", s_feed=1.0, pa={"type": "rapp", "rapp_p": 2.0}, labels=np.zeros(2),
                       task="binary")
    ts = ot.TorchSystem(rs, kind="relay", s_feed=1.0, pa={"type": "linear", "gain_db": 20.0, "psat_out_dbm": 9.0},
                        labels=np.zeros(2), task="binary")
    assert ts.pa.kind == "linear" and ts.pa.gain_v == pytest.approx(10.0)


# ============================================================ oracle
def test_oracle_continuous_drives_loss_down():
    rng = np.random.default_rng(40)
    ts, _ = _relay_system(rng, 1, B=4)
    with torch.no_grad():
        init_losses = [float(ts.loss(ts.init_phases(np.random.default_rng(s)))) for s in range(4)]
    res = ot.oracle_continuous(ts, n_restarts=2, steps=120, lr=0.05, seed=0)
    assert set(res["phases"]) == {"phase_rx", "phase_tx"}
    assert len(res["history"]) == 2
    assert res["loss"] <= min(res["history"]) + 1e-12
    assert res["loss"] < 0.25 * np.median(init_losses)
    with torch.no_grad():
        relo = float(ts.loss(ts.as_phases(res["phases"])))      # 返回的相位复评得到同一损失（mod 2π 不变）
    assert relo == pytest.approx(res["loss"], rel=1e-9)
    y = ts.readings_db(ts.as_phases(res["phases"])).detach().numpy()
    sgn = np.where(ts.labels == 1, 1, -1)
    assert np.all(sgn * (y[:, 0] - y[:, 1]) > 0)                # 4 个样本全部分对


def test_oracle_continuous_passive_focus_reaches_closed_form():
    """无源面、Cpp = 0 时 z = s(H00 + Σ_n c_n e^{jφ_n})，c_n = Hrp[0,n] α_n Ep[n,0]；
    连续相位最优功率闭式为 s²(|H00| + Σ|c_n|)²，oracle 应达到（聚焦损失 = −dBm）。"""
    rng = np.random.default_rng(42)
    rs = reduced_sample(rng, "full", n_rcv=1, S=1, n_p=8)
    rs.Cpp = np.zeros((8, 8), complex)
    ts = ot.TorchSystem([rs], kind="passive", s_feed=0.3, labels=np.zeros(1), task="focus", n_probes=1)
    c = rs.Hrp[0] * rs.alpha_p_base * rs.Ep[:, 0]
    best_dbm = 10 * np.log10((0.3 * (abs(rs.H[0, 0]) + np.abs(c).sum())) ** 2) + 30
    res = ot.oracle_continuous(ts, n_restarts=2, steps=400, lr=0.05, seed=1)
    assert -res["loss"] == pytest.approx(best_dbm, abs=1e-6)
    assert -res["loss"] <= best_dbm + 1e-9


def test_oracle_continuous_respects_init():
    rng = np.random.default_rng(41)
    ts, _ = _relay_system(rng, 1, B=2)
    init = {k: v.numpy() for k, v in ts.init_phases(9).items()}
    res = ot.oracle_continuous(ts, n_restarts=1, steps=0, seed=0, init=init)
    for k in init:
        assert np.allclose(res["phases"][k], np.mod(init[k], 2 * np.pi))


# ============================================================ 对照生产 numpy 管线（DESIGN §6.3，审查 TR-2/TR-3）
@pytest.fixture(scope="module")
def l90_parity():
    """真实 L90 小场景 + 3 位奇偶的 8 个样本，三种物体模式的 ReducedSample（含环境与 3×3 无源面）。"""
    sc, roi = l90_scene()
    xs = list(itertools.product([0, 1], repeat=3))
    labels = np.array([sum(x) % 2 for x in xs])
    rss = {m: [solver.reduce_sample(sc, parity_objects(roi, x), m) for x in xs] for m in ("full", "born", "intra")}
    return rss, labels


def numpy_chain(rss, ph, s_b, prm, pa_np):
    """生产 numpy 管线：solver.eval_reduced → relay.weights/relay_linear → relay.solve_fixed_point → probe_fields。"""
    H = np.stack([solver.eval_reduced(r, ph["phase_p"]) for r in rss])
    w, ct = nrl.weights(ph["phase_rx"], ph["phase_tx"], prm)
    lin = nrl.relay_linear(H, w, ct, s_b, prm)
    fp = nrl.solve_fixed_point(lin.u_in, lin.K, pa_np)
    return nrl.probe_fields(lin.d, lin.C, fp.g), fp


# (s_feed, isolation_db, κ0 相位, AM-PM, 说明)：名义（P1dB 附近）/ 深饱和 / 强环路不收敛 / 收敛到不稳定根
# 隔离度比 κ0 改为端口参考面（×收发网络损耗 −5 dB）之前各加 5 dB，保持原有效环路增益
CHAIN_CASES = [(1.0, -25.0, 20.0, True, "nominal"), (30.0, -30.0, 70.0, False, "saturated"),
               (0.05, -17.0, 20.0, True, "nonconverged"), (1.0, -22.0, 180.0, True, "strong")]


@pytest.mark.parametrize("mode", ["full", "born", "intra"])
def test_relay_forward_matches_production_numpy_pipeline(l90_parity, mode):
    rss_all, labels = l90_parity
    rss = rss_all[mode]
    rng = np.random.default_rng(60)
    ps = rng.uniform(0.5, 2.0, size=len(rss))
    perr = rng.normal(0.0, 0.1, size=(1, N_EL))
    kinds = set()
    for s_feed, iso, kph, ampm, _ in CHAIN_CASES:
        ts = ot.TorchSystem(rss, kind="relay", s_feed=s_feed, pa={"type": "rapp", "ampm": ampm},
                            relay={"isolation_db": iso, "kappa0_phase_deg": kph, "phase_err_rx": perr},
                            labels=labels, task="binary", power_scale=ps)
        ph = {k: v.numpy() for k, v in ts.init_phases(rng).items()}
        assert set(ph) == {"phase_rx", "phase_tx", "phase_p"}
        with torch.no_grad():
            out = ts.forward(ts.as_phases(ph))
        prm = nrl.RelayParams(isolation_db=iso, kappa0_phase_deg=kph, phase_err_rx=perr)
        z, fp = numpy_chain(rss, ph, s_feed * np.sqrt(ps), prm, npa.RappPA(ampm=ampm))
        zt = out["z"].numpy()
        assert np.max(np.abs(zt - z) / np.abs(z)) < 1e-8
        assert np.allclose(out["readings_db"].numpy(), 10 * np.log10(np.abs(z) ** 2) + 30, rtol=0, atol=1e-8)
        assert np.array_equal(out["flags"], fp.possibly_oscillating)
        assert np.array_equal(out["fp"].converged, fp.converged)
        assert np.allclose(out["fp"].loop_gain_op, fp.loop_gain_op, rtol=1e-10)
        kinds.add(bool(fp.possibly_oscillating.any()))
    assert kinds == {False, True}                                   # 覆盖了正常与强环路（被标记）情形


def test_relay_with_passive_gradient_real_scene(l90_parity):
    """中继 + 无源面（phase_rx、phase_tx、phase_p 同时可训练）在真实场景上的梯度 vs 有限差分。"""
    rss_all, labels = l90_parity
    ts = ot.TorchSystem(rss_all["intra"], kind="relay", s_feed=1.0, pa={"type": "rapp"},
                        relay={"isolation_db": -30.0, "kappa0_phase_deg": 20.0}, labels=labels, task="binary")
    rng = np.random.default_rng(61)
    ph = ts.init_phases(rng)
    assert not ts.forward(ph)["flags"].any()
    _fd_check(ts, ph, n_check=8, rng=rng)


def test_passive_forward_matches_solver_real_scene():
    """M0（无中继、只有无源面）：z = s_b·H[N_RX+k, 0]，H 取自 solver.solve_scene（直接求解）。"""
    sc, roi = l90_scene(relay=False)
    xs = list(itertools.product([0, 1], repeat=3))
    rss = [solver.reduce_sample(sc, parity_objects(roi, x), "born") for x in xs]
    ps = np.linspace(0.5, 2.0, len(xs))
    ts = ot.TorchSystem(rss, kind="passive", s_feed=0.7, labels=np.array([sum(x) % 2 for x in xs]),
                        task="binary", power_scale=ps)
    phi = np.random.default_rng(62).uniform(0, 2 * np.pi, sc.n_passive)
    out = ts.forward({"phase_p": torch.as_tensor(phi)})
    assert out["flags"].dtype == bool and not out["flags"].any() and out["fp"] is None
    for b, x in enumerate(xs):
        H = solver.solve_scene(sc, parity_objects(roi, x), phi, "born")
        ref = 0.7 * np.sqrt(ps[b]) * H[sc.n_rx:sc.n_rx + 2, 0]
        assert np.max(np.abs(out["z"][b].detach().numpy() - ref) / np.abs(ref)) < 1e-8


def _p7_regime(rng, B=30):
    """审查 p6/p7：AM-PM 关、正实环路增益 G_v|κ| = 2–4、弱注入 |u_in| = −40 dB·A_in ⇒ 慢收敛的稳定饱和根。"""
    tp = ot.TorchPA("rapp", ampm=False)
    u_in = 1e-2 * tp.a_in * np.exp(1j * rng.uniform(0, 2 * np.pi, (B, 1)))
    K = rng.uniform(2, 4, (B, 1, 1)) / tp.gain_v + 0.05 / tp.gain_v * _cn(rng, B, 1, 1)
    return tp, npa.RappPA(ampm=False), u_in, K


def _p2_regime(rng, B=40, R=1):
    """审查 p2 'iso-24 weak uin'：随机 K、|κ0| = −24 dB、AM-PM 开、|u_in| = 1e-3·A_in。"""
    tp = ot.TorchPA("rapp")
    u_in = 1e-3 * tp.a_in * _cn(rng, B, R)
    K = 0.05 / tp.gain_v * _cn(rng, B, R, R) + 10 ** (-24 / 20) * np.exp(
        1j * rng.uniform(0, 2 * np.pi, (B, 1, 1))) * np.eye(R)
    return tp, npa.RappPA(), u_in, K


@pytest.mark.parametrize("regime", [_p7_regime, _p2_regime])
def test_fixed_point_selects_same_attractor_as_relay(regime):
    """强环路回归（TR-2）：不动点不唯一时，旧的 torch 内部求解器与 relay 选中不同的根（p7 判错收敛、
    p2 落到不稳定近线性根）。现在 u、g、收敛、ρ_op、可能自激标志与 relay.solve_fixed_point 逐样本相同。
    （只断言"与 relay 一致"，不断言选中哪个根：那是 relay 的契约。）"""
    tp, pnp, u_in, K = regime(np.random.default_rng(70))
    res = ot.fixed_point_torch(torch.as_tensor(u_in), torch.as_tensor(K), tp)
    ref = nrl.solve_fixed_point(u_in, K, pnp)
    assert ref.possibly_oscillating.any()                      # 确实处于强环路区
    assert np.array_equal(res.u.detach().numpy(), ref.u)
    assert np.array_equal(res.converged, ref.converged)
    assert np.array_equal(res.possibly_oscillating, ref.possibly_oscillating)
    assert np.array_equal(res.loop_gain_op, ref.loop_gain_op)
    assert np.array_equal(res.loop_gain_ss, ref.loop_gain_ss)
    assert np.allclose(res.g.detach().numpy(), ref.g, rtol=1e-13, atol=0)


# ============================================================ 训练目标定义（审查 TR-6/TR-7）
def test_loss_is_batch_size_invariant():
    """所有项按样本平均：把同一批样本复制一份，损失不变（旧实现的惩罚项按样本求和，随 B 增长）。"""
    rng = np.random.default_rng(80)
    rs = relay_samples(rng, B=3, R=1)
    kw = dict(kind="relay", s_feed=1.0, pa={"type": "linear", "gain_db": 30.0},
              relay={"isolation_db": -25.0, "kappa0_phase_deg": 5.0},
              null_ref={"H0": rs[0].H, "u0_ref_abs2": 1e-4, "target_db": -20.0, "weight": 2.0})
    ts1 = ot.TorchSystem(rs, labels=np.array([0, 1, 1]), task="binary", **kw)
    ts2 = ot.TorchSystem(rs + rs, labels=np.array([0, 1, 1, 0, 1, 1]), task="binary", **kw)
    ph = ts1.init_phases(rng)
    out = ts1.forward(ph)
    assert out["flags"].any() and np.any(out["fp"].loop_gain_op >= 0.95)   # 自激惩罚、标记替换均激活
    with torch.no_grad():
        assert float(ts2.loss(ph)) == pytest.approx(float(ts1.loss(ph)), rel=1e-13)


def test_null_penalty_takes_worst_relay():
    """N_r = 2：零陷惩罚 = weight·softplus(max_r nd_dB,r − target)（INTEGRATION C）。"""
    rng = np.random.default_rng(81)
    R = 2
    rs = relay_samples(rng, B=2, R=R)
    H0 = rs[0].H.copy()
    ts = ot.TorchSystem(rs, kind="relay", s_feed=0.5, n_relays=R, pa={"type": "rapp"},
                        relay={"isolation_db": -40.0}, labels=np.array([0, 1]), task="binary",
                        null_ref={"H0": H0, "u0_ref_abs2": 3e-5, "target_db": -15.0, "weight": 1.5},
                        osc_penalty_weight=0.0)
    ph = ts.init_phases(rng)
    out = ts.forward(ph)
    w = out["w"].detach().numpy()
    u0 = 0.5 * np.sum(w * H0[:R * N_EL, 0].reshape(R, N_EL), axis=1)
    nd = 10 * np.log10(np.abs(u0) ** 2 / 3e-5)
    assert abs(nd[0] - nd[1]) > 0.1
    ref = losses.binary_loss(out["readings_db"].detach().numpy(), ts.labels, 1.0) \
        + 1.5 * np.logaddexp(0.0, nd.max() + 15.0)
    with torch.no_grad():
        assert float(ts.loss(ph)) == pytest.approx(ref, rel=1e-12)
    _fd_check(ts, ph, n_check=6, rng=rng)


def test_loc3bit_task_loss_and_gradient():
    """loc3bit（3 探头 × 阈值 → 格号，INTEGRATION E）可在 TorchSystem 中使用：数值 = losses 的 numpy 实现；
    固定阈值 + loss_weight 时梯度与有限差分一致（auto 阈值是读数中位数的分段函数，只核对数值）。"""
    rng = np.random.default_rng(82)
    rs = relay_samples(rng, B=8, R=1, K=3)
    labels = np.arange(8)
    ts = ot.TorchSystem(rs, kind="relay", s_feed=1.0, pa={"type": "rapp"}, relay={"isolation_db": -40.0},
                        labels=labels, task="loc3bit", n_probes=3)
    ph = ts.init_phases(rng)
    with torch.no_grad():
        y = ts.readings_db(ph).numpy()
        assert float(ts.loss(ph)) == pytest.approx(losses.loc3bit_loss(y, labels, 1.0), rel=1e-12)
    t_fix = np.median(y, axis=0) + np.array([0.3, -0.2, 0.1])
    ts.task_kwargs = {"thresholds_db": t_fix, "loss_weight": 2.6}
    with torch.no_grad():
        assert float(ts.loss(ph)) == pytest.approx(
            losses.loc3bit_loss(y, labels, 1.0, thresholds_db=t_fix, loss_weight=2.6), rel=1e-12)
    _fd_check(ts, ph, n_check=6, rng=rng)


class _FakePhysicalSystem:
    """鸭子类型的 numpy 系统：只提供 to_torch() 与 evaluate_phases()（DESIGN §6.3 的 oracle_continuous(system)）。"""

    def __init__(self, ts):
        self.ts = ts
        self.calls = []

    def to_torch(self):
        return self.ts

    def evaluate_phases(self, phases, rng, n_noise=20):
        self.calls.append({k: np.array(v) for k, v in phases.items()})
        y = self.ts.readings_db(self.ts.as_phases(phases)).detach().numpy()
        sgn = np.where(self.ts.labels == 1, 1.0, -1.0)
        m = sgn * (y[:, 0] - y[:, 1])
        return {"acc_clean": float(np.mean(m > 0)), "margin_db": float(m.min())}


def test_oracle_continuous_interface_aliases_and_numpy_reevaluation():
    rng = np.random.default_rng(83)
    ts, _ = _relay_system(rng, 1, B=2)
    res = ot.oracle_continuous(ts, n_restarts=1, steps=3, seed=0)
    assert res["best_loss"] == res["loss"] and res["best_phases"] is res["phases"]
    assert "eval" not in res                                     # TorchSystem 本身没有 numpy 复评
    fake = _FakePhysicalSystem(ts)
    res2 = ot.oracle_continuous(fake, n_restarts=1, steps=3, seed=0)
    assert len(fake.calls) == 1
    for k in res2["best_phases"]:
        assert np.array_equal(fake.calls[0][k], res2["best_phases"][k])
    assert res2["accuracy"] == res2["eval"]["acc_clean"] and res2["margin_db"] == res2["eval"]["margin_db"]
    assert res2["best_loss"] == pytest.approx(res["best_loss"], rel=1e-12)   # 同 seed 同结果
