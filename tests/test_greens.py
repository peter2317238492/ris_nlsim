"""greens 测试：Friis、互易、方向图方向约定、多偶极子能量守恒、边界情况（DESIGN §1.2）。"""
import numpy as np
import pytest

from ris_nlsim import greens
from ris_nlsim.constants import K0, LAMBDA
from ris_nlsim.greens import green, scat_to_rcv, scat_to_scat, src_to_rcv, src_to_scat
from ris_nlsim.materials import polarizability, resolve_eps
from ris_nlsim.patterns import Pattern
from ris_nlsim.scene import NodeSet

K = K0
LAM = LAMBDA
ISO = Pattern()
HORN = Pattern(q=10.0)
ELEM = Pattern(q=1.0, d0=np.pi, backlobe_db=-25.0)
FUNCS = [src_to_scat, scat_to_scat, scat_to_rcv, src_to_rcv]


def _rand_dirs(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _rand_set(rng, n, pattern, center=(0, 0, 0), spread=0.3, alpha=None):
    pos = np.asarray(center, float) + spread * rng.uniform(-1, 1, size=(n, 3))
    nrm = None if pattern.isotropic else _rand_dirs(rng, n)
    return NodeSet(pos, nrm, pattern, alpha=alpha)


# ---------------------------------------------------------------- 格林函数本身

def test_green_is_outgoing_helmholtz_solution():
    # 数值验证 ∇²G + k²G = 0（R≠0）以及外行波相位 e^{-jkR}（e^{jωt} 约定）
    r0 = np.array([0.31, -0.17, 0.22])
    h = 1e-4

    def G(r):
        return green(np.linalg.norm(r), K)

    lap = sum((G(r0 + h * e) - 2 * G(r0) + G(r0 - h * e)) / h**2 for e in np.eye(3))
    # 二阶中心差分截断误差 ~ (kh)²/12 ≈ 5e-6（相对）
    assert abs(lap + K**2 * G(r0)) < 1e-5 * abs(K**2 * G(r0))
    R = np.array([1.0, 1.0 + LAM / 4])
    g = green(R, K)
    assert np.angle(g[1] / g[0]) == pytest.approx(-np.pi / 2, abs=1e-12)  # 远离源相位滞后


# ---------------------------------------------------------------- Friis 与功率密度

@pytest.mark.parametrize("R", [0.7, 1.8, 5.0])
def test_friis_two_horns_facing(R):
    A = NodeSet([[0.0, 0.0, 0.0]], [1.0, 0.0, 0.0], HORN)
    B = NodeSet([[R, 0.0, 0.0]], [-1.0, 0.0, 0.0], HORN)
    D = 22.0 * (1 + 1e-3)                                 # 视轴方向 d0(1+floor)
    v = src_to_rcv(B, A, K)
    assert v.shape == (1, 1) and v.dtype == np.complex128
    assert abs(v[0, 0]) ** 2 == pytest.approx(D * D * (LAM / (4 * np.pi * R)) ** 2, rel=1e-13)


def test_friis_off_boresight_and_different_patterns():
    # A 喇叭朝 +x，B（探头 q=1，d0=4）位于 A 轴外 30°、法向偏离指向 A 的方向 20°
    th = np.deg2rad(30.0)
    R = 1.3
    A = NodeSet([[0.0, 0.0, 0.0]], [1.0, 0.0, 0.0], HORN)
    rB = R * np.array([np.cos(th), np.sin(th), 0.0])
    to_A = -rB / R
    ang = np.deg2rad(20.0)
    nB = np.array([to_A[0] * np.cos(ang) - to_A[1] * np.sin(ang),
                   to_A[0] * np.sin(ang) + to_A[1] * np.cos(ang), 0.0])
    PROBE = Pattern(q=1.0)
    B = NodeSet([rB], nB, PROBE)
    Da = 22.0 * (np.cos(th) ** 10 + 1e-3)
    Db = 4.0 * (np.cos(ang) + 1e-3)
    v = src_to_rcv(B, A, K)
    assert abs(v[0, 0]) ** 2 == pytest.approx(Da * Db * (LAM / (4 * np.pi * R)) ** 2, rel=1e-12)


def test_src_to_scat_power_density_and_direction_convention():
    # 源方向图在"源→散射体"方向求值；散射体方向图在"散射体→源"方向求值
    R = 2.0
    src = NodeSet([[0.0, 0.0, 0.0]], [1.0, 0.0, 0.0], HORN)
    front = NodeSet([[R, 0, 0]], None, ISO)
    back = NodeSet([[-R, 0, 0]], None, ISO)
    D0, Db = 22 * (1 + 1e-3), 22 * 1e-3
    # 1 W 源：|E|² = P·D/(4πR²)
    assert abs(src_to_scat(front, src, K)[0, 0]) ** 2 == pytest.approx(D0 / (4 * np.pi * R**2), rel=1e-13)
    assert abs(src_to_scat(back, src, K)[0, 0]) ** 2 == pytest.approx(Db / (4 * np.pi * R**2), rel=1e-13)
    # 有方向性的散射体：面朝源（法向 −x）得到视轴增益，背对源得到后瓣
    facing = NodeSet([[R, 0, 0]], [-1.0, 0, 0], ELEM)
    away = NodeSet([[R, 0, 0]], [1.0, 0, 0], ELEM)
    e_f = src_to_scat(facing, src, K)[0, 0]
    e_a = src_to_scat(away, src, K)[0, 0]
    assert abs(e_f) ** 2 == pytest.approx(D0 * np.pi * (1 + 10**-2.5) / (4 * np.pi * R**2), rel=1e-13)
    assert abs(e_a) ** 2 == pytest.approx(D0 * np.pi * 10**-2.5 / (4 * np.pi * R**2), rel=1e-13)


def test_scat_to_rcv_is_field_times_receive_aperture():
    # v = E·λF/sqrt(4π)：用位于接收端位置的各向同性虚拟散射体取偶极子场 E = k²G p F_b
    rng = np.random.default_rng(3)
    S = _rand_set(rng, 6, ELEM, spread=0.2)
    rc = _rand_set(rng, 4, Pattern(q=1.0), center=(1.0, 0.5, 0.0), spread=0.2)
    dummy = NodeSet(rc.pos, None, ISO)
    E = scat_to_scat(dummy, S, K)                       # (4,6)，已含 F_S
    diff = S.pos[None] - rc.pos[:, None]
    cos = np.einsum("abk,ak->ab", diff, rc.normal) / np.linalg.norm(diff, axis=-1)
    Fr = np.sqrt(4.0 * (np.maximum(cos, 0) + 1e-3))
    np.testing.assert_allclose(scat_to_rcv(rc, S, K), E * LAM * Fr / np.sqrt(4 * np.pi), rtol=1e-13)
    # 与 src_to_scat 的转置关系：scat_to_rcv(A,S) = (λk²/4π)·src_to_scat(S,A)ᵀ
    np.testing.assert_allclose(scat_to_rcv(rc, S, K),
                               LAM * K**2 / (4 * np.pi) * src_to_scat(S, rc, K).T, rtol=1e-13)


# ---------------------------------------------------------------- 互易

def test_src_to_rcv_reciprocal_between_antenna_sets():
    rng = np.random.default_rng(11)
    A = _rand_set(rng, 5, HORN, center=(0, 0, 0))
    B = _rand_set(rng, 3, Pattern(q=1.0), center=(1.5, 0.2, -0.3))
    np.testing.assert_allclose(src_to_rcv(A, B, K), src_to_rcv(B, A, K).T, rtol=1e-14, atol=0)


def test_scat_to_scat_symmetric_and_zero_diagonal():
    rng = np.random.default_rng(5)
    S = _rand_set(rng, 30, ELEM, spread=0.4)
    C = scat_to_scat(S, S, K)
    assert C.shape == (30, 30)
    assert np.all(np.diag(C) == 0)
    np.testing.assert_allclose(C, C.T, rtol=1e-14, atol=0)
    # 跨集合（不同方向图）：C(S1,S2) = C(S2,S1)ᵀ
    S2 = _rand_set(rng, 7, ISO, center=(0.9, 0, 0), spread=0.2)
    np.testing.assert_allclose(scat_to_scat(S, S2, K), scat_to_scat(S2, S, K).T, rtol=1e-14)
    # same 显式指定：不同对象但位置相同 → 同集合
    S_copy = NodeSet(S.pos, S.normal, S.pattern)
    np.testing.assert_array_equal(scat_to_scat(S, S_copy, K, same=True), C)
    with pytest.raises(ValueError):
        scat_to_scat(S, S2, K, same=True)


def _transmission(rx: NodeSet, tx: NodeSet, scat: list[NodeSet]) -> np.ndarray:
    """暴力求解 tx → rx 的总传输：直达 + 经全部散射体（精确多次散射）。"""
    blocks_C = [[scat_to_scat(si, sj, K) for sj in scat] for si in scat]
    C = np.block(blocks_C)
    alpha = np.concatenate([s.alpha for s in scat])
    E0 = np.vstack([src_to_scat(s, tx, K) for s in scat])
    Crs = np.hstack([scat_to_rcv(rx, s, K) for s in scat])
    p = np.linalg.solve(np.diag(1 / alpha) - C, E0)
    return src_to_rcv(rx, tx, K) + Crs @ p


def test_reciprocity_through_scatterers():
    # 两组天线 A、B（不同方向图），经各向同性体素 + 有方向性的无源单元（不同 α）：H_BA = H_ABᵀ
    rng = np.random.default_rng(21)
    k3 = K**3
    A = _rand_set(rng, 3, HORN, center=(-1.0, 0, 0), spread=0.2)
    B = _rand_set(rng, 4, Pattern(q=1.0), center=(1.0, 0.3, 0), spread=0.2)
    vox = _rand_set(rng, 25, ISO, spread=0.15,
                    alpha=(rng.uniform(0.5, 2.0, 25) * 4 * np.pi / k3) * np.exp(-0.3j))
    pas = _rand_set(rng, 16, ELEM, center=(0, 0.6, 0), spread=0.2,
                    alpha=LAM / K**2 * np.exp(1j * rng.uniform(0, 2 * np.pi, 16)))
    H_BA = _transmission(B, A, [vox, pas])
    H_AB = _transmission(A, B, [vox, pas])
    assert np.max(np.abs(H_BA)) > 0
    np.testing.assert_allclose(H_BA, H_AB.T, rtol=1e-10, atol=1e-14 * np.max(np.abs(H_BA)))


# ---------------------------------------------------------------- 多偶极子能量守恒

def _cluster_powers(material):
    """耦合偶极子团在平面波（+z 传播，E0=1）下的 (P_ext, P_sca, P_abs_closed_form)。"""
    rng = np.random.default_rng(2)
    d = LAM / 10
    idx = np.array([(i, j, l) for i in range(3) for j in range(3) for l in range(3)], float)
    pos = idx * d + rng.uniform(-0.15, 0.15, size=idx.shape) * d   # 扰动立方体，强耦合
    V = d**3
    alpha = polarizability(material, V, K, "cm_rr")
    S = NodeSet(pos, None, ISO, alpha=alpha)
    E_inc = np.exp(-1j * K * pos[:, 2])                      # e^{jωt}：+z 传播 e^{-jkz}
    p = np.linalg.solve(np.diag(1 / S.alpha) - scat_to_scat(S, S, K), E_inc)
    P_ext = -K * np.sum(np.imag(p * np.conj(E_inc)))
    Rij = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    sinc = np.sinc(K * Rij / np.pi)                          # sin(kR)/(kR)，对角 = 1
    P_sca = K**4 / (4 * np.pi) * np.real(np.conj(p) @ sinc @ p)
    eps = resolve_eps(material)
    alpha0 = 3 * V * (eps - 1) / (eps + 2) if eps != "metal" else 3 * V
    P_abs = K * np.sum(np.abs(p) ** 2) * np.imag(1 / alpha0)
    return P_ext, P_sca, P_abs


@pytest.mark.parametrize("material", ["eps1.5", "eps4", "eps10", "metal"])
def test_coupled_dipoles_lossless_conserve_energy(material):
    P_ext, P_sca, _ = _cluster_powers(material)
    assert P_sca > 0
    assert P_ext == pytest.approx(P_sca, rel=1e-10)


def test_coupled_dipoles_lossy_absorption_matches_closed_form():
    P_ext, P_sca, P_abs = _cluster_powers("water")
    assert P_abs > 0
    assert P_ext - P_sca == pytest.approx(P_abs, rel=1e-9)


# ---------------------------------------------------------------- 边界情况

@pytest.mark.parametrize("func", FUNCS)
def test_empty_sets_give_empty_matrices(func):
    rng = np.random.default_rng(0)
    S = _rand_set(rng, 4, ELEM)
    E = NodeSet.empty(ELEM)
    for a, b, shape in [(E, S, (0, 4)), (S, E, (4, 0)), (E, NodeSet.empty(), (0, 0))]:
        out = func(a, b, K)
        assert out.shape == shape and out.dtype == np.complex128


@pytest.mark.parametrize("func", FUNCS)
def test_coincident_cross_set_nodes_raise(func):
    a = NodeSet([[0.0, 0, 0], [0.1, 0, 0]], None, ISO)
    b = NodeSet([[0.5, 0, 0], [0.1 + 5e-10, 0, 0]], None, ISO)
    with pytest.raises(ValueError):
        func(a, b, K)
    ok = NodeSet([[0.5, 0, 0], [0.1 + 2e-9, 0, 0]], None, ISO)
    assert np.all(np.isfinite(func(a, ok, K)))


def test_same_set_duplicate_nodes_raise_and_single_node_ok():
    dup = NodeSet([[0.0, 0, 0], [0.0, 0, 0]], None, ISO)
    with pytest.raises(ValueError):
        scat_to_scat(dup, dup, K)
    one = NodeSet([[0.0, 0, 0]], None, ISO)
    assert scat_to_scat(one, one, K).tolist() == [[0j]]
    # 同一对象作为源与接收端（非散射）也会因自身距离为 0 报错
    with pytest.raises(ValueError):
        src_to_rcv(one, one, K)


def test_lam_keyword_consistency():
    rng = np.random.default_rng(4)
    a = _rand_set(rng, 3, HORN)
    b = _rand_set(rng, 2, ELEM, center=(2, 0, 0))
    np.testing.assert_array_equal(src_to_rcv(a, b, K, lam=LAM), src_to_rcv(a, b, K))
    with pytest.raises(ValueError):
        src_to_rcv(a, b, K, lam=1.1 * LAM)
    with pytest.raises(ValueError):
        scat_to_rcv(a, b, -K)


def test_vectorized_matches_scalar_loop():
    rng = np.random.default_rng(9)
    a = _rand_set(rng, 5, Pattern(q=2.0, d0=3.0, backlobe_db=-15), spread=0.5)
    b = _rand_set(rng, 4, ELEM, center=(1, 1, 0), spread=0.5)
    M = src_to_rcv(a, b, K)
    for i in range(5):
        for j in range(4):
            r = b.pos[j] - a.pos[i]
            R = np.linalg.norm(r)
            Fa = np.sqrt(3.0 * (max(a.normal[i] @ r / R, 0) ** 2 + 10**-1.5))
            Fb = np.sqrt(np.pi * (max(-b.normal[j] @ r / R, 0) + 10**-2.5))
            ref = LAM * Fa * Fb * np.exp(-1j * K * R) / (4 * np.pi * R)
            assert M[i, j] == pytest.approx(ref, rel=1e-13)
    assert greens.MIN_DIST == 1e-9


# ---------------------------------------------------------------- NodeSet / Scene（DESIGN §2.1）

def test_nodeset_basics_and_validation():
    ns = NodeSet([[0, 0, 0], [1, 0, 0]], [0, 0, 2.0], ELEM, alpha=1e-6, group=[0, 1])
    assert len(ns) == 2
    np.testing.assert_allclose(ns.normal, [[0, 0, 1], [0, 0, 1]])        # (3,) 广播并归一化
    assert ns.alpha.dtype == np.complex128 and ns.alpha.shape == (2,)
    assert ns.group.dtype == np.int64
    one = NodeSet([0.1, 0.2, 0.3], None, ISO)                            # 单点 (3,) → (1,3)
    assert one.pos.shape == (1, 3) and np.all(one.normal == 0)
    e = NodeSet.empty(HORN)
    assert len(e) == 0 and e.pos.shape == (0, 3) and e.pattern == HORN
    assert len(NodeSet.empty()) == 0 and NodeSet.empty().pattern.isotropic
    with pytest.raises(ValueError):
        NodeSet([[0, 0, 0]], None, HORN)                                 # 有方向性但无法向
    with pytest.raises(ValueError):
        NodeSet([[0, 0, 0], [1, 0, 0]], None, ISO, alpha=[1, 2, 3])
    sub = ns.take([1])
    assert len(sub) == 1 and sub.group[0] == 1


def test_nodeset_concat():
    a = NodeSet([[0, 0, 0]], [1, 0, 0], ELEM, alpha=1e-6, group=2)
    b = NodeSet([[1, 0, 0], [2, 0, 0]], [0, 1, 0], ELEM, alpha=[2e-6, 3e-6])
    c = NodeSet.concat([a, NodeSet.empty(ELEM), b])
    assert len(c) == 3
    np.testing.assert_allclose(c.alpha, [1e-6, 2e-6, 3e-6])
    np.testing.assert_array_equal(c.group, [2, -1, -1])                 # 缺 group 的填 -1
    np.testing.assert_allclose(c.normal[1:], [[0, 1, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        NodeSet.concat([a, NodeSet([[3, 0, 0]], [1, 0, 0], HORN)])      # 方向图不同
    with pytest.raises(ValueError):
        NodeSet.concat([a, NodeSet([[3, 0, 0]], [1, 0, 0], ELEM)])      # alpha 混合
    assert len(NodeSet.concat([])) == 0
    # 拼接后的耦合矩阵 = 分块耦合矩阵
    src = NodeSet([[0, 3.0, 0]], [0, -1, 0], HORN)
    np.testing.assert_allclose(src_to_scat(c, src, K),
                               np.vstack([src_to_scat(a, src, K), src_to_scat(b, src, K)]), rtol=1e-15)


def test_scene_defaults_and_counts():
    from ris_nlsim.constants import F0
    from ris_nlsim.scene import Scene
    feeds = NodeSet([[2, 0, 0], [2, 0.1, 0]], [-1, 0, 0], HORN)
    tx = [NodeSet(np.c_[np.zeros(25), np.arange(25) * 0.01, np.zeros(25)], [1, 0, 0], ELEM)]
    sc = Scene(F0, feeds, tx=tx, rx=tx[0], probes=NodeSet([[1, 1, 0]], [-1, -1, 0], Pattern(q=1.0)))
    np.testing.assert_array_equal(sc.feed_weights, [1, 1])
    assert sc.feed_weights.dtype == np.complex128
    assert isinstance(sc.rx, list) and sc.n_rx == 25 and sc.n_tx == 25
    assert sc.n_src == 26 and sc.n_rcv == 26 and sc.n_probe == 1
    assert sc.n_env == 0 and sc.n_passive == 0 and sc.env == []
    assert sc.k == pytest.approx(K) and sc.lam == pytest.approx(LAM)
    assert not sc.block_direct_feed_probe and sc.skip_gain_db == 0.0 and not sc.include_direct_tx_rx
    with pytest.raises(ValueError):
        Scene(F0, feeds, feed_weights=[1.0])
