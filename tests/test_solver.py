"""solver.py：三种物体模式、约化表示、平面波散射（DESIGN §2.2–§2.3，方案 §4.1、§4.8）。

参考量全部独立构造：逐元素暴力公式 + 显式逆、闭式单偶极子、球面数值积分、互易交换、Walsh 手算。
"""
import numpy as np
import pytest

from ris_nlsim import solver
from ris_nlsim.constants import F0, K0, LAMBDA
from ris_nlsim.geometry import board_elements, make_relays, parity_slots, point_from, voxelize_sphere
from ris_nlsim.greens import src_to_rcv
from ris_nlsim.materials import alpha_passive, effective_eps, polarizability, voxels_to_nodes
from ris_nlsim.patterns import Pattern, element_pattern, horn_pattern, po_cell_pattern, probe_pattern
from ris_nlsim.scene import NodeSet, Scene
from ris_nlsim.solver import MODES, eval_reduced, reduce_sample, solve_scene

K, LAM = K0, LAMBDA


def rel(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# =========================================================== 场景工厂
def facing_board(center, normal, n=3):
    """n×n 板（间距 λ/2），以 center 为中心、法向 normal（水平），返回 (pos, normal)。"""
    nrm = unit(normal)
    u = unit(np.cross([0.0, 0.0, 1.0], nrm))
    v = np.array([0.0, 0.0, 1.0])
    corner = np.asarray(center) - 0.5 * n * (LAM / 2) * (u + v)
    return board_elements(corner, u, v, n, n, LAM / 2), nrm


def build_scene(n_relays=1, env=True, passive=True, **flags):
    """L90 式几何：馈源 az −30°/1.8 m，两探头 az 75°/105°/1.0 m，ROI = 接收板中心 + (0.5,0,0)。

    env = 一个金属点 + 一块 2×2 PO 金属板（带方向图）；passive = 3×3 无源面（Γ=1）。
    """
    rels = make_relays(max(n_relays, 1), 90.0)
    rx_c = np.mean([r.rx_center for r in rels], axis=0)
    tx_c = np.mean([r.tx_center for r in rels], axis=0)
    fpos = point_from(rx_c, -30.0, 0.0, 1.8)
    feeds = NodeSet(fpos, rx_c - fpos, horn_pattern())
    pp = np.array([point_from(tx_c, a, 0.0, 1.0) for a in (75.0, 105.0)])
    probes = NodeSet(pp, tx_c - pp, probe_pattern())
    ep = element_pattern()
    tx = [NodeSet(r.tx_pos, r.tx_normal, ep) for r in rels[:n_relays]]
    rx = [NodeSet(r.rx_pos, r.rx_normal, ep) for r in rels[:n_relays]]
    roi = rx_c + np.array([0.5, 0.0, 0.0])
    envs = []
    if env:
        pt = NodeSet([roi + [0.15, 0.35, 0.05]], None, Pattern(),
                     alpha=polarizability("metal", (LAM / 5) ** 3, K))
        ppos, pn = facing_board(roi + [0.3, -0.35, 0.0], [-1.0, 1.0, 0.0], n=2)   # 2×2，间距 λ/2
        plate = NodeSet(ppos, pn, po_cell_pattern((LAM / 2) ** 2, LAM), alpha=alpha_passive(-1.0, K, LAM))
        envs = [pt, plate]
    pas = NodeSet.empty()
    if passive:
        c = roi + np.array([-0.1, 0.45, 0.0])
        pos, nrm = facing_board(c, roi - c, n=3)
        pas = NodeSet(pos, nrm, ep, alpha=alpha_passive(1.0, K, LAM))
    return Scene(F0, feeds, tx=tx, rx=rx, probes=probes, env=envs, passive=pas, **flags), roi


def make_objects(roi, x, material="water", d=LAM / 10, spacing=LAM / 2):
    """奇偶槽位（沿 z，间距 spacing）上的 λ/3 球，x_j=1 表示第 j 个存在；group = 槽号。"""
    slots = parity_slots(len(x), roi, spacing)
    sets = []
    for j, xj in enumerate(x):
        if xj:
            pos, v = voxelize_sphere(slots[j], LAM / 3, d)
            sets.append(voxels_to_nodes(pos, v, material, K, group=j))
    return NodeSet.concat(sets) if sets else NodeSet.empty()


# =========================================================== 独立暴力参考
def bf_F(P, n, d):
    if P.q is None:
        return 1.0
    c = max(float(np.dot(n, d)), 0.0)
    d0 = P.d0 if P.d0 is not None else 2.0 * (P.q + 1.0)
    return np.sqrt(d0 * (c ** P.q + 10.0 ** (P.backlobe_db / 10.0)))


def bf_c(kind, a, b):
    """a、b = (pos, normal, pattern)；DESIGN §1.2 公式，a 行 b 列。"""
    R = np.linalg.norm(b[0] - a[0])
    dab = (b[0] - a[0]) / R
    G = np.exp(-1j * K * R) / (4 * np.pi * R)
    Fa, Fb = bf_F(a[2], a[1], dab), bf_F(b[2], b[1], -dab)
    return {"src_scat": np.sqrt(4 * np.pi) * Fa * Fb * G,
            "scat_scat": K**2 * G * Fa * Fb,
            "scat_rcv": LAM / np.sqrt(4 * np.pi) * K**2 * Fa * Fb * G,
            "src_rcv": LAM * Fa * Fb * G}[kind]


def nodes(sets):
    return [(s.pos[i], s.normal[i], s.pattern) for s in sets for i in range(len(s))]


def bf_solve(sc, objs, phase, mode):
    """逐元素组装 + 显式逆，按 DESIGN §2.2 的步骤字面实现。"""
    fw = sc.feed_weights
    feeds, tx = nodes([sc.feeds]), nodes(sc.tx)
    rcv, n_rx = nodes([*sc.rx, sc.probes]), sc.n_rx
    los = 0.0 if sc.block_direct_feed_probe else 10 ** (sc.skip_gain_db / 20)
    P = sc.passive
    aP = P.alpha * np.exp(1j * phase) if len(P) else np.zeros(0)
    O_nodes, B_nodes = nodes([objs]), nodes([*sc.env, P])
    aO = objs.alpha if len(objs) else np.zeros(0)
    aB = np.concatenate([*(e.alpha for e in sc.env), aP]) if B_nodes else np.zeros(0)
    scat = O_nodes + B_nodes
    alpha = np.concatenate([aO, aB])
    N, S = len(scat), 1 + len(tx)

    def src(targets, kind):
        out = np.zeros((len(targets), S), complex)
        for i, t in enumerate(targets):
            out[i, 0] = sum(fw[f] * bf_c(kind, t, feeds[f]) for f in range(len(feeds)))
            for m, s in enumerate(tx):
                out[i, 1 + m] = bf_c(kind, t, s)
        return out

    E0R = src(rcv, "src_rcv")
    E0R[n_rx:, 0] *= los
    if not sc.include_direct_tx_rx:
        E0R[:n_rx, 1:] = 0
    E0 = src(scat, "src_scat")
    C = np.array([[0 if i == j else bf_c("scat_scat", scat[i], scat[j]) for j in range(N)]
                  for i in range(N)], complex).reshape(N, N)
    CR = np.array([[bf_c("scat_rcv", r, s) for s in scat] for r in rcv], complex).reshape(len(rcv), N)
    if mode == "full":
        return E0R + CR @ np.linalg.inv(np.diag(1 / alpha) - C) @ E0
    o, b = np.arange(len(O_nodes)), np.arange(len(O_nodes), N)
    Ainv = np.linalg.inv(np.diag(1 / aB) - C[np.ix_(b, b)]) if len(b) else np.zeros((0, 0))
    pb_bg = Ainv @ E0[b]
    Eo = E0[o] + C[np.ix_(o, b)] @ pb_bg
    if mode == "born":
        po = aO[:, None] * Eo
    else:
        po = np.zeros_like(Eo)
        for g in np.unique(objs.group):
            ig = np.flatnonzero(objs.group == g)
            po[ig] = np.linalg.inv(np.diag(1 / aO[ig]) - C[np.ix_(ig, ig)]) @ Eo[ig]
    pb = pb_bg + Ainv @ C[np.ix_(b, o)] @ po
    return E0R + CR[:, b] @ pb + CR[:, o] @ po


def tiny_scene():
    """全部要素都有、但节点很少的场景：2 馈源（复权重）、2 块 2 单元中继、2 组 env、4 个无源单元。"""
    rng = np.random.default_rng(7)
    ep = element_pattern()
    feeds = NodeSet([[1.5, -0.9, 0.05], [1.4, -1.1, -0.1]], [[-1, 0.6, 0], [-1, 0.8, 0.1]], horn_pattern())
    tx = [NodeSet([[-0.05, 0.02, -0.03], [-0.09, 0.02, 0.03]], [0, 1, 0], ep),
          NodeSet([[-0.05, 0.02, 0.3], [-0.09, 0.02, 0.33]], [0, 1, 0], ep)]
    rx = [NodeSet([[0.0, -0.05, 0.0], [0.0, -0.09, 0.02]], [1, 0, 0], ep),
          NodeSet([[0.0, -0.05, 0.3], [0.0, -0.09, 0.28]], [1, 0, 0], ep)]
    probes = NodeSet([[0.2, 1.0, 0.0], [-0.3, 0.9, 0.1]], [[-0.2, -1, 0], [0.3, -1, 0]], probe_pattern())
    env = [NodeSet([[0.6, 0.3, 0.05]], None, Pattern(), alpha=3e-6 - 1e-6j),
           NodeSet([[0.7, -0.3, 0.0], [0.7, -0.26, 0.04]], [-1, 1, 0], Pattern(q=1.0, d0=2.0, backlobe_db=-60),
                   alpha=alpha_passive(-1.0, K, LAM))]
    ppos = np.array([[0.4, 0.4, 0.0], [0.44, 0.4, 0.0], [0.4, 0.4, 0.04], [0.44, 0.4, 0.04]])
    passive = NodeSet(ppos, [0, -1, 0], ep, alpha=alpha_passive(1.0, K, LAM))
    sc = Scene(F0, feeds, feed_weights=[1.0, 0.3 - 0.8j], tx=tx, rx=rx, probes=probes, env=env,
               passive=passive, skip_gain_db=-4.0, include_direct_tx_rx=True)
    opos = np.array([[0.5, 0.0, 0.0], [0.51, 0.0, 0.0], [0.5, 0.012, 0.0],
                     [0.5, 0.0, 0.05], [0.51, 0.01, 0.05], [0.5, 0.0, 0.06]])
    alpha_o = polarizability(78 - 10j, (LAM / 12) ** 3, K)
    objs = NodeSet(opos, None, Pattern(), alpha=alpha_o, group=[0, 0, 0, 1, 1, 1])
    phase = rng.uniform(0, 2 * np.pi, 4)
    return sc, objs, phase


# =========================================================== 正确性：暴力 / 闭式
@pytest.mark.parametrize("mode", MODES)
def test_matches_bruteforce(mode):
    sc, objs, phase = tiny_scene()
    H = solve_scene(sc, objs, phase, mode)
    ref = bf_solve(sc, objs, phase, mode)
    assert H.shape == (sc.n_rcv, sc.n_src) == (4 + 2, 1 + 4)
    assert rel(H, ref) < 1e-10
    # 散射部分本身也要对得上（去掉直达项后比较，避免被大直达项掩盖）
    H0 = bf_solve(sc, NodeSet.empty(), phase, "full")
    assert rel(H - H0, ref - H0) < 1e-9


def test_single_dipole_closed_form():
    """单个各向同性偶极子：H = LOS + scat_to_rcv·α·src_to_scat，三种模式相同。"""
    fpos, ppos, opos = np.array([1.0, -1.0, 0.0]), np.array([0.3, 1.2, 0.1]), np.array([0.5, 0.0, 0.0])
    feeds = NodeSet(fpos, opos - fpos, horn_pattern())
    probes = NodeSet(ppos, opos - ppos, probe_pattern())
    a = polarizability("eps4", (LAM / 6) ** 3, K)
    obj = NodeSet(opos, None, Pattern(), alpha=a, group=0)
    sc = Scene(F0, feeds, probes=probes)
    f, p, o = (fpos, unit(opos - fpos), horn_pattern()), (ppos, unit(opos - ppos), probe_pattern()), \
        (opos, np.zeros(3), Pattern())
    ref = bf_c("src_rcv", p, f) + bf_c("scat_rcv", p, o) * a * bf_c("src_scat", o, f)
    for mode in MODES:
        H = solve_scene(sc, obj, None, mode)
        assert H.shape == (1, 1)
        assert abs(H[0, 0] - ref) / abs(ref) < 1e-12


def test_two_dipoles_closed_form_full():
    """两个偶极子：full 的 2×2 逆矩阵闭式；born 忽略互耦。"""
    feeds = NodeSet([2.0, 0.0, 0.0], [-1, 0, 0], horn_pattern())
    probes = NodeSet([0.0, 2.0, 0.0], [0, -1, 0], probe_pattern())
    r1, r2 = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.03])
    a1, a2 = 2e-6 - 3e-7j, 1e-6
    objs = NodeSet([r1, r2], None, Pattern(), alpha=[a1, a2], group=[0, 1])
    sc = Scene(F0, feeds, probes=probes, block_direct_feed_probe=True)
    iso = Pattern()
    f, p = (feeds.pos[0], feeds.normal[0], feeds.pattern), (probes.pos[0], probes.normal[0], probes.pattern)
    o1, o2 = (r1, np.zeros(3), iso), (r2, np.zeros(3), iso)
    e = np.array([bf_c("src_scat", o1, f), bf_c("src_scat", o2, f)])
    c = bf_c("scat_scat", o1, o2)
    h = np.array([bf_c("scat_rcv", p, o1), bf_c("scat_rcv", p, o2)])
    det = 1 / (a1 * a2) - c * c
    p_full = np.array([e[0] / a2 + c * e[1], c * e[0] + e[1] / a1]) / det
    assert abs(solve_scene(sc, objs, None, "full")[0, 0] - h @ p_full) / abs(h @ p_full) < 1e-12
    p_born = np.array([a1, a2]) * e
    assert abs(solve_scene(sc, objs, None, "born")[0, 0] - h @ p_born) / abs(h @ p_born) < 1e-12
    # intra：两个体素分属不同物体 → 与 born 相同
    assert solve_scene(sc, objs, None, "intra")[0, 0] == pytest.approx(h @ p_born, rel=1e-12)


# =========================================================== 1) 互易
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("block", [False, True])
def test_reciprocity_feed_probe_swap(mode, block):
    sc, roi = build_scene(n_relays=1)
    objs = make_objects(roi, [1, 1, 0])
    phase = np.random.default_rng(3).uniform(0, 2 * np.pi, sc.n_passive)
    pa, pb = roi + np.array([1.2, -0.9, 0.2]), roi + np.array([-0.4, 1.1, -0.1])
    A = NodeSet(pa, roi - pa + [0.0, 0.1, 0.0], horn_pattern())
    B = NodeSet(pb, roi - pb + [0.05, 0.0, 0.0], probe_pattern())
    kw = dict(tx=sc.tx, rx=sc.rx, env=sc.env, passive=sc.passive, block_direct_feed_probe=block)
    s_ab = Scene(F0, A, probes=B, **kw)
    s_ba = Scene(F0, B, probes=A, **kw)
    h_ab = solve_scene(s_ab, objs, phase, mode)[s_ab.n_rx, 0]
    h_ba = solve_scene(s_ba, objs, phase, mode)[s_ba.n_rx, 0]
    assert abs(h_ab) > 0
    assert abs(h_ab - h_ba) / abs(h_ab) < 1e-10


@pytest.mark.parametrize("mode", MODES)
def test_reciprocity_tx_rx_element(mode):
    """发射单元 m → 接收单元 n（经全部散射体）= 把二者当作 馈源/探头 交换后的传输。"""
    sc, roi = build_scene(n_relays=1, block_direct_feed_probe=True)
    objs = make_objects(roi, [0, 1, 1])
    phase = np.random.default_rng(4).uniform(0, 2 * np.pi, sc.n_passive)
    H = solve_scene(sc, objs, phase, mode)
    m, n = 7, 18
    t, r = sc.tx[0].take([m]), sc.rx[0].take([n])
    kw = dict(env=sc.env, passive=sc.passive, block_direct_feed_probe=True)
    h_tr = solve_scene(Scene(F0, t, probes=r, **kw), objs, phase, mode)[0, 0]
    h_rt = solve_scene(Scene(F0, r, probes=t, **kw), objs, phase, mode)[0, 0]
    assert abs(H[n, 1 + m] - h_tr) / abs(h_tr) < 1e-10      # 列/行角色正确
    assert abs(h_tr - h_rt) / abs(h_tr) < 1e-10             # 互易


# =========================================================== 2) 光学定理
def _sphere(material, d=LAM / 10, model="cm_rr"):
    pos, v = voxelize_sphere([0.0, 0.0, 0.0], LAM / 3, d)
    return voxels_to_nodes(pos, v, material, K, model=model), v


@pytest.mark.parametrize("model", ["cm_rr", "scalar_rr"])
@pytest.mark.parametrize("kin", [(0, 0, 1), (1, -2, 0.5)])
def test_optical_theorem_lossless(kin, model):
    O, _ = _sphere("eps4", model=model)
    assert len(O) > 1
    p = solver.solve_objects_planewave(O, K, kin, "full")
    se, ss = solver.sigma_ext(p, O, K, kin), solver.sigma_sca(p, O, K)
    assert ss > 0
    assert abs(se - ss) / ss < 1e-8
    # intra（单个物体）= full
    np.testing.assert_allclose(solver.solve_objects_planewave(O, K, kin, "intra"), p, rtol=1e-12, atol=0)


@pytest.mark.parametrize("model", ["cm_rr", "scalar_rr"])
def test_optical_theorem_lossy_water(model):
    O, v = _sphere("water", model=model)
    kin = (0.3, 0.2, -1.0)
    p = solver.solve_objects_planewave(O, K, kin, "full")
    se, ss = solver.sigma_ext(p, O, K, kin), solver.sigma_sca(p, O, K)
    assert se > ss > 0
    # 吸收闭式（materials 文档）：σ_abs = k·Σ|p_i|²·Im(1/α0_i)，α0 为辐射修正前的极化率；
    # scalar_rr 的实自作用项 −k²a²/2 不改变 Im(1/α0) = Im(1/(χV))
    alpha0 = polarizability("water", v, K, model="cm") if model == "cm_rr" else (78 - 10j - 1) * v
    sabs = K * np.sum(np.abs(p) ** 2) * np.imag(1 / alpha0)
    assert abs((se - ss) - sabs) / sabs < 1e-8


def test_sigma_sca_matches_quadrature():
    """sinc 恒等式 vs 远场 |f|² 的球面数值积分（Gauss–Legendre × 均匀 φ）。"""
    O, _ = _sphere("eps4")
    p = solver.solve_objects_planewave(O, K, (1, 1, 0), "full")
    x, w = np.polynomial.legendre.leggauss(40)
    phi = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    st = np.sqrt(1 - x**2)
    dirs = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)),
                     np.outer(x, np.ones_like(phi))], axis=-1).reshape(-1, 3)
    f = solver.far_field_amplitude(p, O, K, dirs).reshape(len(x), len(phi))
    quad = np.sum(w[:, None] * np.abs(f) ** 2) * (2 * np.pi / len(phi))
    assert abs(quad - solver.sigma_sca(p, O, K)) / quad < 1e-8


def test_single_dipole_cross_sections():
    """单偶极子：σ_ext = −k·Im α，σ_sca = k⁴|α|²/(4π)（DESIGN §1.3）。"""
    a = polarizability(78 - 10j, (LAM / 8) ** 3, K)
    O = NodeSet([[0.1, 0.2, 0.3]], None, Pattern(), alpha=a)
    p = solver.solve_objects_planewave(O, K, (0, 1, 0))
    assert solver.sigma_ext(p, O, K, (0, 1, 0)) == pytest.approx(-K * a.imag, rel=1e-12)
    assert solver.sigma_sca(p, O, K) == pytest.approx(K**4 * abs(a) ** 2 / (4 * np.pi), rel=1e-12)


# =========================================================== 3) 弱对比 Born ≈ Full
def test_born_matches_full_weak_contrast():
    sc, roi = build_scene(n_relays=1)
    phase = np.random.default_rng(5).uniform(0, 2 * np.pi, sc.n_passive)
    d = {}
    for eps in (1.01, 4.0):
        objs = make_objects(roi, [1, 0, 0], material=eps)
        for mode in MODES:
            d[eps, mode] = solve_scene(sc, objs, phase, mode) - solve_scene(sc, None, phase, mode)
    assert np.linalg.norm(d[1.01, "full"]) > 0
    err_born = rel(d[1.01, "born"], d[1.01, "full"])
    assert err_born < 0.02
    assert rel(d[1.01, "intra"], d[1.01, "full"]) < err_born          # intra 多含体内多次散射
    assert rel(d[4.0, "born"], d[4.0, "full"]) > 0.1                   # 强对比时 Born 明显失效（测试有分辨力）


# =========================================================== 4) DWBA 仿射性（Walsh 手算）
def walsh_weight_ge2(vals):
    """vals: (8, ...)，x_j = (i>>j)&1；返回 W≥2 = Σ_{|S|≥2}|f̂(S)|² / Σ_{S≠∅}|f̂(S)|²（分量求和）。"""
    n = 3
    w = np.zeros(2**n)
    for S in range(2**n):
        sgn = np.array([(-1) ** bin(S & x).count("1") for x in range(2**n)], float)
        fhat = np.tensordot(sgn, vals, axes=1) / 2**n
        w[S] = np.sum(np.abs(fhat) ** 2)
    deg = np.array([bin(S).count("1") for S in range(2**n)])
    return w[deg >= 2].sum() / w[deg >= 1].sum()


def test_dwba_affine_in_object_presence():
    sc, roi = build_scene(n_relays=1)
    phase = np.random.default_rng(6).uniform(0, 2 * np.pi, sc.n_passive)
    xs = [[(i >> j) & 1 for j in range(3)] for i in range(8)]
    W = {}
    for mode in MODES:
        vals = np.array([solve_scene(sc, make_objects(roi, x), phase, mode) for x in xs])
        W[mode] = walsh_weight_ge2(vals)
    assert W["born"] < 1e-20
    assert W["intra"] < 1e-20                    # intra 也只对背景场响应，同样仿射
    assert W["full"] > 1e-4                      # 物体间多次散射 → 明显的高阶项


def test_parity_nonlinearity_stable_across_voxel_grids():
    """回归（PHY-R2-2）：奇偶任务的物体非线性指标（full 模式 H 的 Walsh W≥2，εr=4，默认模型）不应随体素网格跳变。

    旧体素化（格距 d 不缩放）：λ/10 4.83e-2 → λ/15 4.06e-2（台阶伪影 16%）；缩放格距后 λ/10~λ/30 都在 4.70e-2~4.78e-2。
    """
    sc, roi = build_scene(n_relays=1, env=False, passive=False)
    xs = [[(i >> j) & 1 for j in range(3)] for i in range(8)]
    w = []
    for nd in (10, 15, 20):
        vals = np.array([solve_scene(sc, make_objects(roi, x, material="eps4", d=LAM / nd), None, "full")
                         for x in xs])
        w.append(walsh_weight_ge2(vals))
    assert w[0] > 1e-2                                                   # 指标本身不小（测试有分辨力）
    assert (max(w) - min(w)) / max(w) < 0.04, w


# =========================================================== 5) 约化表示 == 直接求解
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("env", [False, True])
@pytest.mark.parametrize("passive", [False, True])
def test_reduced_matches_direct(mode, env, passive):
    sc, roi = build_scene(n_relays=2, env=env, passive=passive)
    objs = make_objects(roi, [1, 0, 1])
    rs = reduce_sample(sc, objs, mode)
    assert (rs.mode, rs.n_rx, rs.n_probe, rs.n_tx, rs.n_p) == (mode, 50, 2, 50, sc.n_passive)
    rng = np.random.default_rng(8)
    for _ in range(3):
        phase = rng.uniform(0, 2 * np.pi, sc.n_passive) if passive else None
        H = solve_scene(sc, objs, phase, mode)
        assert H.shape == (52, 51)
        assert rel(eval_reduced(rs, phase), H) < 1e-9
    if not passive:
        assert rs.Cpp is None and rs.Ep is None and rs.Hrp is None and rs.Eo is None and rs.Lam is None
    else:
        n_o = len(objs)
        assert rs.Cpp.shape == (9, 9) and rs.Ep.shape == (9, 51) and rs.Hrp.shape == (52, 9)
        if mode == "full":
            assert rs.Eo is None and rs.Lam is None
        else:
            assert rs.Eo.shape == (n_o, 51) and rs.Cop.shape == (n_o, 9) and rs.Cpo.shape == (9, n_o)
            assert rs.Lam.shape == (n_o, n_o) and rs.Hro.shape == (52, n_o)
            np.testing.assert_allclose(rs.Cpo, rs.Cop.T, rtol=1e-10, atol=1e-12 * np.abs(rs.Cop).max())
        np.testing.assert_allclose(rs.Cpp, rs.Cpp.T, rtol=1e-10, atol=1e-12 * np.abs(rs.Cpp).max())
        # 相位 None ≡ 全 0
        assert rel(eval_reduced(rs, None), solve_scene(sc, objs, np.zeros(sc.n_passive), mode)) < 1e-9


@pytest.mark.parametrize("mode", MODES)
def test_reduced_no_objects_and_no_relay(mode):
    sc, roi = build_scene(n_relays=0)
    phase = np.random.default_rng(9).uniform(0, 2 * np.pi, sc.n_passive)
    rs = reduce_sample(sc, None, mode)
    assert rs.n_tx == 0 and rs.n_rx == 0
    H = solve_scene(sc, NodeSet.empty(), phase, mode)
    assert H.shape == (2, 1)
    assert rel(eval_reduced(rs, phase), H) < 1e-9
    # 无物体时三种模式一致
    assert rel(solve_scene(sc, None, phase, "full"), H) < 1e-12


def test_eval_reduced_phase_checks():
    sc, roi = build_scene(n_relays=1, passive=False)
    rs = reduce_sample(sc, make_objects(roi, [1, 0, 0]), "full")
    with pytest.raises(ValueError):
        eval_reduced(rs, np.zeros(3))
    np.testing.assert_array_equal(eval_reduced(rs, None), rs.H)
    sc2, roi = build_scene(n_relays=1)
    rs2 = reduce_sample(sc2, None, "born")
    with pytest.raises(ValueError):
        eval_reduced(rs2, np.zeros(sc2.n_passive + 1))


# =========================================================== 6) 单体素物体：intra == born
def test_intra_equals_born_for_single_voxel_objects():
    sc, roi = build_scene(n_relays=1)
    rng = np.random.default_rng(10)
    pos = roi + 0.05 * rng.normal(size=(5, 3))
    a = polarizability("water", (LAM / 6) ** 3, K)
    objs = NodeSet(pos, None, Pattern(), alpha=a, group=np.arange(5))
    phase = rng.uniform(0, 2 * np.pi, sc.n_passive)
    Hb, Hi = solve_scene(sc, objs, phase, "born"), solve_scene(sc, objs, phase, "intra")
    assert rel(Hi, Hb) < 1e-13
    np.testing.assert_allclose(solver.object_operator(objs, K, "intra"), np.diag(objs.alpha), rtol=1e-14)
    assert rel(eval_reduced(reduce_sample(sc, objs, "intra"), phase), Hb) < 1e-9


def test_intra_single_group_equals_full_in_free_space():
    """没有背景时，所有体素同属一个物体：intra 的体内精确解就是 full。"""
    sc, roi = build_scene(n_relays=1, env=False, passive=False)
    objs = make_objects(roi, [1, 1, 0])
    one = NodeSet(objs.pos, None, Pattern(), alpha=objs.alpha, group=np.zeros(len(objs), int))
    assert rel(solve_scene(sc, one, None, "intra"), solve_scene(sc, one, None, "full")) < 1e-12
    nog = NodeSet(objs.pos, None, Pattern(), alpha=objs.alpha)      # group=None → 一个物体
    assert rel(solve_scene(sc, nog, None, "intra"), solve_scene(sc, one, None, "full")) < 1e-12


def test_object_operator_block_structure():
    _, roi = build_scene(n_relays=0)
    objs = make_objects(roi, [1, 1, 0])
    Lam = solver.object_operator(objs, K, "intra")
    g = objs.group
    assert np.all(Lam[np.ix_(g == 0, g == 1)] == 0) and np.all(Lam[np.ix_(g == 1, g == 0)] == 0)
    from ris_nlsim.greens import scat_to_scat
    sub = objs.take(np.flatnonzero(g == 0))
    A = np.diag(1 / sub.alpha) - scat_to_scat(sub, sub, K, same=True)
    np.testing.assert_allclose(Lam[np.ix_(g == 0, g == 0)] @ A, np.eye(len(sub)), atol=1e-9)
    np.testing.assert_array_equal(solver.object_operator(objs, K, "born"), np.diag(objs.alpha))
    with pytest.raises(ValueError):
        solver.object_operator(objs, K, "full")


# =========================================================== 7) 直达路径开关
@pytest.mark.parametrize("mode", ["full", "born"])
def test_direct_path_flags(mode):
    sc, roi = build_scene(n_relays=1)
    objs = make_objects(roi, [1, 0, 1])
    phase = np.random.default_rng(11).uniform(0, 2 * np.pi, sc.n_passive)

    def H(**flags):
        s = Scene(F0, sc.feeds, tx=sc.tx, rx=sc.rx, probes=sc.probes, env=sc.env, passive=sc.passive,
                  **flags)
        return solve_scene(s, objs, phase, mode)

    n_rx = sc.n_rx
    los = src_to_rcv(sc.probes, sc.feeds, K) @ sc.feed_weights
    H_open, H_blk = H(), H(block_direct_feed_probe=True)
    D = H_open - H_blk
    np.testing.assert_allclose(D[n_rx:, 0], los, rtol=1e-12)
    D[n_rx:, 0] = 0
    assert np.all(D == 0)                                            # 只去掉视距项，其余逐位不变
    # skip_gain_db 只缩放视距项
    for g in (-10.0, 6.0):
        Dg = H(skip_gain_db=g) - H_blk
        np.testing.assert_allclose(Dg[n_rx:, 0], 10 ** (g / 20) * los, rtol=1e-12)
        Dg[n_rx:, 0] = 0
        assert np.all(Dg == 0)
    assert np.all(H(block_direct_feed_probe=True, skip_gain_db=6.0) == H_blk)
    # include_direct_tx_rx 只开关 发射单元→接收单元 的自由空间项
    Dt = H(include_direct_tx_rx=True) - H_open
    np.testing.assert_allclose(Dt[:n_rx, 1:], src_to_rcv(sc.rx[0], sc.tx[0], K), rtol=1e-12)
    Dt[:n_rx, 1:] = 0
    assert np.all(Dt == 0)


# =========================================================== 多中继 / 馈源权重 / 退化输入
@pytest.mark.parametrize("mode", MODES)
def test_multi_relay_ordering(mode):
    """两块中继：列块 = 各中继发射单元，行块 = 各中继接收单元（发射/接收单元不参与散射，可逐块对照）。"""
    sc, roi = build_scene(n_relays=2)
    objs = make_objects(roi, [1, 1, 1])
    phase = np.random.default_rng(12).uniform(0, 2 * np.pi, sc.n_passive)
    H = solve_scene(sc, objs, phase, mode)
    for r in range(2):
        s1 = Scene(F0, sc.feeds, tx=[sc.tx[r]], rx=[sc.rx[r]], probes=sc.probes, env=sc.env,
                   passive=sc.passive)
        H1 = solve_scene(s1, objs, phase, mode)
        rows = np.r_[25 * r:25 * (r + 1), 50:52]
        cols = np.r_[0, 1 + 25 * r:1 + 25 * (r + 1)]
        assert rel(H[np.ix_(rows, cols)], H1) < 1e-12


@pytest.mark.parametrize("mode", MODES)
def test_feed_weights_linearity(mode):
    sc, roi = build_scene(n_relays=1)
    p2 = sc.feeds.pos[0] + np.array([0.0, -0.3, 0.1])
    feeds = NodeSet(np.vstack([sc.feeds.pos, p2]), np.vstack([sc.feeds.normal, sc.feeds.normal]),
                    horn_pattern())
    w = np.array([0.8 - 0.2j, -0.4 + 1.1j])
    objs = make_objects(roi, [0, 1, 1])
    phase = np.random.default_rng(13).uniform(0, 2 * np.pi, sc.n_passive)
    kw = dict(tx=sc.tx, rx=sc.rx, probes=sc.probes, env=sc.env, passive=sc.passive)
    H = solve_scene(Scene(F0, feeds, feed_weights=w, **kw), objs, phase, mode)
    Hs = [solve_scene(Scene(F0, feeds.take([i]), **kw), objs, phase, mode) for i in range(2)]
    assert rel(H[:, 0], w[0] * Hs[0][:, 0] + w[1] * Hs[1][:, 0]) < 1e-12
    assert rel(H[:, 1:], Hs[0][:, 1:]) < 1e-12                    # 发射单元列与馈源无关


def test_empty_everything():
    feeds = NodeSet([1.0, 0.0, 0.0], [-1, 0, 0], horn_pattern())
    probes = NodeSet([0.0, 1.0, 0.0], [0, -1, 0], probe_pattern())
    sc = Scene(F0, feeds, probes=probes)
    los = src_to_rcv(probes, feeds, K)
    for mode in MODES:
        np.testing.assert_allclose(solve_scene(sc, None, None, mode), los, rtol=1e-14)
        np.testing.assert_allclose(eval_reduced(reduce_sample(sc, None, mode), None), los, rtol=1e-14)
    sc0 = Scene(F0, NodeSet.empty())                                 # 连接收端都没有
    assert solve_scene(sc0, None, None, "full").shape == (0, 1)
    assert reduce_sample(sc0, None, "born").H.shape == (0, 1)


def test_input_validation():
    sc, roi = build_scene(n_relays=1)
    with pytest.raises(ValueError):
        solve_scene(sc, None, None, "dwba")
    with pytest.raises(ValueError):
        reduce_sample(sc, None, "Born")
    with pytest.raises(ValueError):
        solve_scene(sc, NodeSet(roi, None, Pattern()), None, "full")     # 物体缺 alpha
    with pytest.raises(ValueError):
        solve_scene(sc, None, np.zeros(sc.n_passive + 2), "full")


@pytest.mark.parametrize("mode", MODES)
def test_objects_must_differ_from_background_sets(mode):
    """回归（PHY-4）：物体 / 环境 / 无源面是同一 NodeSet 对象时，born/intra 曾把 C_BO 的该块当成"同一集合"
    （对角置 0）静默算错；现在所有模式、直接与约化求解都报错。"""
    sc, roi = build_scene(n_relays=1)
    for dup in (sc.env[0], sc.env[1], sc.passive):
        with pytest.raises(ValueError, match="同一个 NodeSet"):
            solve_scene(sc, dup, None, mode)
        with pytest.raises(ValueError, match="同一个 NodeSet"):
            reduce_sample(sc, dup, mode)
    # 无源面与环境是同一对象（约化表示的 C_XP 块同样会被静默置零）
    sc2 = Scene(F0, sc.feeds, tx=sc.tx, rx=sc.rx, probes=sc.probes, env=sc.env, passive=sc.env[1])
    objs = make_objects(roi, [1, 0, 0])
    with pytest.raises(ValueError):
        reduce_sample(sc2, objs, mode)
    with pytest.raises(ValueError):
        solve_scene(sc2, objs, None, mode)
    # 内容相同但对象不同：是两个重合的散射体，greens 按"跨集合 R<1e-9"报错（DESIGN §1.2）
    clone = NodeSet(sc.env[0].pos, None, Pattern(), alpha=sc.env[0].alpha)
    with pytest.raises(ValueError):
        solve_scene(sc, clone, None, mode)


@pytest.mark.parametrize("mode", MODES)
def test_complex_passive_phase_rejected(mode):
    """回归（PHY-6）：复数相位（误传权重 e^{jφ}）不得被静默转成 float（只剩 cos φ）。"""
    sc, roi = build_scene(n_relays=1)
    objs = make_objects(roi, [1, 0, 0])
    w = np.exp(1j * np.linspace(0, 2, sc.n_passive))
    with pytest.raises(TypeError):
        solve_scene(sc, objs, w, mode)
    rs = reduce_sample(sc, objs, mode)
    with pytest.raises(TypeError):
        eval_reduced(rs, w)
    # 实数相位（含整数数组）照常工作
    ph = np.arange(sc.n_passive)
    assert rel(eval_reduced(rs, ph), solve_scene(sc, objs, ph, mode)) < 1e-9


@pytest.mark.parametrize("mode", MODES)
def test_zero_alpha_passive_element(mode):
    """回归（PHY-5）：Γ=0 的无源单元（alpha_base=0）在约化求解里曾经 1/0 → NaN；
    现在 eval_reduced 有限、与直接求解一致，且等于把该单元删掉。"""
    sc, roi = build_scene(n_relays=1)
    P = sc.passive
    alpha = P.alpha.copy()
    alpha[3] = 0.0
    P0 = NodeSet(P.pos, P.normal, P.pattern, alpha=alpha)
    keep = np.flatnonzero(np.arange(len(P)) != 3)
    kw = dict(tx=sc.tx, rx=sc.rx, probes=sc.probes, env=sc.env)
    sc0 = Scene(F0, sc.feeds, passive=P0, **kw)
    sc_del = Scene(F0, sc.feeds, passive=P.take(keep), **kw)
    objs = make_objects(roi, [1, 0, 1])
    ph = np.random.default_rng(14).uniform(0, 2 * np.pi, len(P))
    Hd = solve_scene(sc0, objs, ph, mode)
    Hr = eval_reduced(reduce_sample(sc0, objs, mode), ph)
    assert np.all(np.isfinite(Hr))
    assert rel(Hr, Hd) < 1e-9
    assert rel(Hd, solve_scene(sc_del, objs, ph[keep], mode)) < 1e-12


def test_zero_alpha_node_is_inert():
    """α=0 的节点（不散射）不得产生 1/0，结果与去掉该节点相同。"""
    sc, roi = build_scene(n_relays=1)
    objs = make_objects(roi, [1, 0, 0])
    extra = NodeSet(roi + [0.0, 0.2, 0.0], None, Pattern(), alpha=0.0, group=5)
    with_zero = NodeSet.concat([objs, extra])
    for mode in MODES:
        H0 = solve_scene(sc, objs, None, mode)
        assert rel(solve_scene(sc, with_zero, None, mode), H0) < 1e-12


# =========================================================== 8) 体素收敛 + 标量 Mie 参考
def test_voxel_convergence_eps4_sphere():
    """体素收敛（方案 §4.8）：以标量 Mie 为独立参考，而不是相邻网格的自差分。

    回归（PHY-R2-2）：旧体素化保留格距 d、只改偶极子体积，λ/3 球在 λ/15、λ/20 的壳层台阶（N=81、171，
    理想 65.4、155）使格胞体积与偶极子体积不一致：scalar_rr εr=4 误差 12.6%、6.6%（λ/10 只有 1.0%），
    εr=10 达 25.6%、14.7%；cm_rr 对其连续极限 Mie(ε_eff) 在 λ/15 为 13.3%。格距按格胞体积缩放后误差随网格平稳/单调。
    """
    nds = (10, 12, 15, 20, 25)
    e4 = [_dda_vs_mie("eps4", nd, "scalar_rr") for nd in nds]          # 实测 0.3% 1.2% 0.4% 0.8% 0.7%
    e10 = [_dda_vs_mie("eps10", nd, "scalar_rr") for nd in nds]        # 实测 2.9% 5.0% 3.1% 3.6% 4.4%
    assert max(e4) < 0.015
    assert max(e10) < 0.06
    e_eff = effective_eps("eps4", "cm_rr")
    ecm = [_dda_vs_mie("eps4", nd, "cm_rr", mie_eps=e_eff) for nd in (10, 12, 15, 20, 25, 30)]
    assert all(b < a for a, b in zip(ecm, ecm[1:])), ecm               # 实测 12.9% 9.9% 5.1% 3.6% 2.8% 1.9%
    assert ecm[-1] < 0.025


def born_sphere_far_field(chi, radius, cos_th):
    """弱对比 Born 闭式：f = k²χ/(4π)·V·3(sin qa − qa·cos qa)/(qa)³，q = 2k·sin(θ/2) = k·sqrt(2(1−cosθ))。"""
    V = 4 * np.pi * radius**3 / 3
    qa = K * np.sqrt(2 * (1 - np.asarray(cos_th))) * radius
    with np.errstate(invalid="ignore", divide="ignore"):
        form = np.where(qa < 1e-6, 1.0, 3 * (np.sin(qa) - qa * np.cos(qa)) / qa**3)
    return K**2 * chi / (4 * np.pi) * V * form


def test_mie_reference_born_limit_and_optical_theorem():
    """Mie 参考本身的独立检验：弱对比 = Born 闭式；无耗时 σ_sca = σ_ext；软球系数 a_0 闭式；lmax 已收敛。"""
    ct = np.linspace(-1, 1, 7)
    a = LAM / 6
    chi = 1e-6
    f = solver.mie_scalar_far_field(1 + chi, a, K, ct)
    np.testing.assert_allclose(f, born_sphere_far_field(chi, a, ct), rtol=1e-5)
    for eps in [4.0, 10.0, "metal"]:
        al = solver.mie_scalar_coeffs(eps, a, K)
        ls = np.arange(len(al))
        s_sca = 4 * np.pi / K**2 * np.sum((2 * ls + 1) * np.abs(al) ** 2)
        s_ext = -4 * np.pi / K * np.imag(solver.mie_scalar_far_field(eps, a, K, 1.0))
        assert s_sca == pytest.approx(s_ext, rel=1e-10)
        assert abs(al[-1]) < 1e-10 * abs(al).max()                      # 级数已截断收敛
    al = solver.mie_scalar_coeffs("water", a, K)
    s_sca = 4 * np.pi / K**2 * np.sum((2 * np.arange(len(al)) + 1) * np.abs(al) ** 2)
    assert -4 * np.pi / K * np.imag(solver.mie_scalar_far_field("water", a, K, 1.0)) > 1.01 * s_sca
    x = K * a                                                            # 软球 l=0：a_0 = −j_0/h_0^(2)
    j0, y0 = np.sin(x) / x, -np.cos(x) / x
    assert solver.mie_scalar_coeffs("metal", a, K)[0] == pytest.approx(-j0 / (j0 - 1j * y0), rel=1e-13)


_MIE_DIRS = np.array([[0, 0, 1], [1, 0, 0], [0, 0, -1], [0, 1, 1]], float)


def _dda_vs_mie(material, nd, model, mie_eps=None, D=LAM / 3):
    """λ/3 球（体素 λ/nd）+z 平面波：DDA 远场 vs 标量 Mie 的相对误差（方案 §4.8 体素收敛的独立参考）。

    model=None 表示不传 model，走 voxels_to_nodes 的默认路径（tasks/system 用的就是它）。
    """
    pos, v = voxelize_sphere([0.0, 0.0, 0.0], D, LAM / nd)
    kw = {} if model is None else {"model": model}
    O = voxels_to_nodes(pos, v, material, K, **kw)
    p = solver.solve_objects_planewave(O, K, (0, 0, 1), "full")
    f = solver.far_field_amplitude(p, O, K, _MIE_DIRS)
    ref = solver.mie_scalar_far_field(material if mie_eps is None else mie_eps, D / 2, K,
                                      _MIE_DIRS[:, 2] / np.linalg.norm(_MIE_DIRS, axis=1))
    return rel(f, ref)


@pytest.mark.parametrize("model", ["cm_rr", "volume_rr", "scalar_rr"])
def test_dda_matches_mie_weak_contrast(model):
    assert _dda_vs_mie(1.01, 10, model) < 0.01


def test_scalar_rr_converges_to_mie():
    """回归（PHY-1/PHY-3）：scalar_rr 收敛到 εr 本身的标量 Mie 解（cm_rr 不会，见下一个测试）。

    实测（λ/3 球，格距缩放后的体素化）：eps4 λ/30 0.4%、λ/40 0.2%；eps10 λ/40 1.6%；
    水 λ/25~λ/40 约 4%~4.5%（λ/10 时 25%）；金属（χ→∞ 软球）λ/30 4.8%。
    """
    assert _dda_vs_mie("eps4", 40, "scalar_rr") < 0.005
    assert _dda_vs_mie("eps4", 30, "scalar_rr") < 0.01
    assert _dda_vs_mie("eps10", 40, "scalar_rr") < 0.025
    assert _dda_vs_mie("water", 30, "scalar_rr") < 0.06
    assert _dda_vs_mie("metal", 30, "scalar_rr") < 0.07                   # χ→∞：软球（Dirichlet）


def _mie_sigma_sca(material, radius):
    al = solver.mie_scalar_coeffs(material, radius, K)
    return 4 * np.pi / K**2 * np.sum((2 * np.arange(len(al)) + 1) * np.abs(al) ** 2)


def test_default_model_path_solves_true_contrast():
    """回归（PHY-R2-1）：不传 model 的默认路径（INTEGRATION §E 的 tasks 就这样调用）求解 εr 本身。

    旧默认 cm_rr 把对比度压成 χ_eff ≤ 3：金属 = εr=4 介质，λ/3 球的 σ_sca 排序变成 金属 > 水 > eps10 > eps4，
    而标量 Mie 是 eps10 > eps4 > 金属 ≈ 水 > eps1.5（eps10 接近体内共振；金属、水只差 12%，不比较二者顺序）。
    默认模型必须复现 Mie 的量值与介质间的排序。
    """
    mats = ["eps1.5", "eps4", "eps10", "water", "metal"]
    ref = np.array([_mie_sigma_sca(m, LAM / 6) for m in mats])
    pos, v = voxelize_sphere([0.0, 0.0, 0.0], LAM / 3, LAM / 20)
    dda, dda_cm = [], []
    for m in mats:
        for out, kw in [(dda, {}), (dda_cm, {"model": "cm_rr"})]:
            O = voxels_to_nodes(pos, v, m, K, **kw)
            out.append(solver.sigma_sca(solver.solve_objects_planewave(O, K, (0, 0, 1), "full"), O, K))
    dda, dda_cm = np.array(dda), np.array(dda_cm)
    diel = slice(0, 4)                                                    # eps1.5, eps4, eps10, water
    assert list(np.argsort(ref[diel])) == [0, 3, 1, 2]                    # Mie：eps1.5 < 水 < eps4 < eps10
    assert list(np.argsort(dda[diel])) == [0, 3, 1, 2]
    np.testing.assert_allclose(dda, ref, rtol=0.15)                       # 实测最大偏差：金属 11%
    assert list(np.argsort(dda_cm[diel])) == [0, 1, 2, 3]                 # 对照：cm_rr 把水排到 eps10 之上
    assert dda_cm[4] > dda_cm[2]                                          # cm_rr 金属（= εr 4）> eps10
    # 远场：默认 eps10 在默认 voxel_d = λ/10 下 < 5%；默认金属是软球，与 εr=4 介质相差甚远（cm_rr 金属恰是 εr=4）
    assert _dda_vs_mie("eps10", 10, None) < 0.05
    assert _dda_vs_mie("metal", 25, None) < 0.08
    assert _dda_vs_mie("metal", 25, None, mie_eps=4.0) > 0.5
    assert _dda_vs_mie("metal", 30, "cm_rr", mie_eps=4.0) < 0.04


def test_cm_rr_converges_to_effective_contrast():
    """cm_rr（DESIGN §1.3 字面默认，现为可选模型）的连续极限是 ε_eff = 1 + 3(εr−1)/(εr+2) 的标量 Mie，而不是 εr 本身（PHY-1）。"""
    e_eff = effective_eps("eps4", "cm_rr")
    assert _dda_vs_mie("eps4", 40, "cm_rr", mie_eps=e_eff) < 0.02
    assert _dda_vs_mie("eps4", 40, "cm_rr") > 0.5
    assert _dda_vs_mie("water", 40, "cm_rr", mie_eps=effective_eps("water", "cm_rr")) < 0.03
