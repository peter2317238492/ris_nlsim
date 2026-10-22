"""scene.py 的矩阵拼装辅助：与逐元素暴力公式（DESIGN §1.1–§1.2，独立重写）对照。"""
import numpy as np
import pytest

from ris_nlsim.constants import F0, K0, LAMBDA
from ris_nlsim.patterns import Pattern
from ris_nlsim.scene import (NodeSet, Scene, as_passive_phase, assemble_scat_to_rcv, assemble_scat_to_scat,
                             assemble_src_to_rcv, assemble_src_to_scat, check_scatterer_sets,
                             feed_probe_los_factor, passive_with_phase, receiver_sets, stack_alpha)

K = K0
LAM = LAMBDA


# ---------------- 独立的暴力参考（不调用 patterns / greens） ----------------
def bf_F(P: Pattern, n, d):
    """DESIGN §1.1：F = sqrt(d0·(max(cos,0)^q + 10^(bl/10)))，各向同性为 1。"""
    if P.q is None:
        return 1.0
    c = max(float(np.dot(n, d)), 0.0)
    d0 = P.d0 if P.d0 is not None else 2.0 * (P.q + 1.0)
    return np.sqrt(d0 * (c ** P.q + 10.0 ** (P.backlobe_db / 10.0)))


def bf_coupling(kind, ra, na, Pa, rb, nb, Pb):
    """DESIGN §1.2 四种耦合的逐元素公式（a 行、b 列）。"""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    R = np.linalg.norm(rb - ra)
    dab = (rb - ra) / R
    G = np.exp(-1j * K * R) / (4 * np.pi * R)
    Fa = bf_F(Pa, na, dab)
    Fb = bf_F(Pb, nb, -dab)
    if kind == "src_scat":
        return np.sqrt(4 * np.pi) * Fb * G * Fa
    if kind == "scat_scat":
        return K**2 * G * Fb * Fa
    if kind == "scat_rcv":
        return LAM / np.sqrt(4 * np.pi) * Fa * K**2 * G * Fb
    if kind == "src_rcv":
        return LAM * Fa * Fb * G
    raise ValueError(kind)


def bf_block(kind, A: list[NodeSet], B: list[NodeSet], zero_same=False):
    rows = [(s, i) for s in A for i in range(len(s))]
    cols = [(s, j) for s in B for j in range(len(s))]
    out = np.zeros((len(rows), len(cols)), complex)
    for r, (sa, i) in enumerate(rows):
        for c, (sb, j) in enumerate(cols):
            if zero_same and sa is sb and i == j:
                continue
            out[r, c] = bf_coupling(kind, sa.pos[i], sa.normal[i], sa.pattern,
                                    sb.pos[j], sb.normal[j], sb.pattern)
    return out


# ---------------- 场景工厂 ----------------
def rand_set(rng, n, center, pattern=Pattern(), alpha=True, spread=0.05):
    pos = np.asarray(center) + spread * rng.normal(size=(n, 3))
    nrm = None if pattern.isotropic else rng.normal(size=(n, 3))
    a = (1e-6 * (rng.normal(size=n) + 1j * rng.normal(size=n))) if alpha else None
    return NodeSet(pos, nrm, pattern, alpha=a)


@pytest.fixture
def small_scene():
    rng = np.random.default_rng(1)
    horn = Pattern(q=10.0)
    elem = Pattern(q=1.0, d0=np.pi, backlobe_db=-25.0)
    probe = Pattern(q=1.0)
    feeds = rand_set(rng, 2, [1.5, -0.8, 0.0], horn, alpha=False)
    tx = [rand_set(rng, 3, [0.0, 0.1, 0.0], elem, alpha=False),
          rand_set(rng, 2, [0.0, 0.1, 0.4], elem, alpha=False)]
    rx = [rand_set(rng, 2, [0.05, -0.1, 0.0], elem, alpha=False),
          rand_set(rng, 3, [0.05, -0.1, 0.4], elem, alpha=False)]
    probes = rand_set(rng, 2, [0.2, 1.0, 0.0], probe, alpha=False)
    sc = Scene(F0, feeds, feed_weights=[1.0 + 0.5j, -0.7j], tx=tx, rx=rx, probes=probes)
    return sc, rng


# ---------------- 测试 ----------------
def test_receiver_sets_order(small_scene):
    sc, _ = small_scene
    rs = receiver_sets(sc)
    assert rs[:2] == sc.rx and rs[2] is sc.probes
    assert sc.n_rcv == 2 + 3 + 2 and sc.n_src == 1 + 3 + 2


def test_stack_alpha_and_missing():
    a = NodeSet([[0, 0, 0], [0, 0, 1]], None, Pattern(), alpha=[1e-6, 2e-6j])
    b = NodeSet([[1, 0, 0]], None, Pattern(), alpha=3e-6)
    out = stack_alpha([a, NodeSet.empty(), b])
    np.testing.assert_array_equal(out, [1e-6, 2e-6j, 3e-6])
    assert out.dtype == np.complex128
    assert stack_alpha([]).shape == (0,)
    with pytest.raises(ValueError):
        stack_alpha([NodeSet([[0, 0, 0]], None, Pattern())])


def test_passive_with_phase():
    base = np.array([1e-5, 2e-5, 3e-5j])
    P = NodeSet(np.eye(3), np.eye(3), Pattern(q=1.0, d0=np.pi), alpha=base)
    sc = Scene(F0, NodeSet.empty(), passive=P)
    ph = np.array([0.0, np.pi / 2, np.pi])
    Pp = passive_with_phase(sc, ph)
    np.testing.assert_allclose(Pp.alpha, base * np.exp(1j * ph), rtol=0, atol=1e-20)
    np.testing.assert_array_equal(sc.passive.alpha, base)          # 原对象不变
    np.testing.assert_array_equal(passive_with_phase(sc, None).alpha, base)
    with pytest.raises(ValueError):
        passive_with_phase(sc, np.zeros(2))
    sc0 = Scene(F0, NodeSet.empty())
    assert len(passive_with_phase(sc0, None)) == 0
    assert len(passive_with_phase(sc0, np.zeros(0))) == 0
    with pytest.raises(ValueError):
        passive_with_phase(sc0, np.zeros(3))
    # 回归（PHY-6）：复数（例如误传权重 e^{jφ}）不得静默只保留实部
    with pytest.raises(TypeError):
        passive_with_phase(sc, np.exp(1j * ph))
    with pytest.raises(TypeError):
        as_passive_phase(np.zeros(3, dtype=complex), 3)        # 即使虚部为 0 也拒绝：类型本身说明传错了
    np.testing.assert_array_equal(as_passive_phase([0, 1, 2], 3), [0.0, 1.0, 2.0])
    assert as_passive_phase(None, 0) is None
    np.testing.assert_array_equal(as_passive_phase(None, 2), [0.0, 0.0])


def test_check_scatterer_sets():
    """回归（PHY-4）：物体 / 环境 / 无源面必须是不同对象；空集合不参与比较。"""
    rng = np.random.default_rng(3)
    e0, e1 = rand_set(rng, 2, [0.5, 0, 0]), rand_set(rng, 3, [0.7, 0, 0])
    P = rand_set(rng, 2, [0.6, 0.3, 0], Pattern(q=1.0, d0=np.pi))
    sc = Scene(F0, NodeSet.empty(), env=[e0, e1], passive=P)
    obj = rand_set(rng, 4, [0.4, 0.1, 0])
    check_scatterer_sets(sc, obj)
    check_scatterer_sets(sc, NodeSet.empty())
    for dup in (e0, e1, P):
        with pytest.raises(ValueError):
            check_scatterer_sets(sc, dup)
    with pytest.raises(ValueError):
        check_scatterer_sets(Scene(F0, NodeSet.empty(), env=[e0, e0]), obj)
    with pytest.raises(ValueError):
        check_scatterer_sets(Scene(F0, NodeSet.empty(), env=[e0], passive=e0), obj)
    # 空集合同一对象出现多次无害
    E = NodeSet.empty()
    check_scatterer_sets(Scene(F0, NodeSet.empty(), env=[E, E], passive=E), E)


def test_scat_to_scat_blocks_match_bruteforce():
    rng = np.random.default_rng(2)
    A = rand_set(rng, 4, [0, 0, 0])
    B = rand_set(rng, 3, [0.3, 0.1, 0], Pattern(q=2.0, d0=5.0, backlobe_db=-20.0))
    C = assemble_scat_to_scat([A, B], [A, B], K)
    ref = bf_block("scat_scat", [A, B], [A, B], zero_same=True)
    np.testing.assert_allclose(C, ref, rtol=1e-12, atol=0)
    assert np.all(np.diag(C) == 0)
    np.testing.assert_allclose(C, C.T, rtol=1e-13, atol=0)          # 互易：对称
    # 非方块（行、列集合不同）
    np.testing.assert_allclose(assemble_scat_to_scat([B], [A], K),
                               bf_block("scat_scat", [B], [A]), rtol=1e-12, atol=0)
    with pytest.raises(ValueError):
        assemble_scat_to_scat([A, A], [B], K)                        # 同一对象出现两次
    assert assemble_scat_to_scat([], [A], K).shape == (0, 4)


def test_src_to_scat_feed_combination_and_tx_columns(small_scene):
    sc, rng = small_scene
    scat = [rand_set(rng, 3, [0.5, 0, 0]), rand_set(rng, 2, [0.4, 0.3, 0], Pattern(q=1.0, d0=np.pi))]
    E0 = assemble_src_to_scat(sc, scat)
    assert E0.shape == (5, sc.n_src)
    ref_feed = bf_block("src_scat", scat, [sc.feeds]) @ sc.feed_weights
    np.testing.assert_allclose(E0[:, 0], ref_feed, rtol=1e-12, atol=0)
    np.testing.assert_allclose(E0[:, 1:], bf_block("src_scat", scat, sc.tx), rtol=1e-12, atol=0)
    # 发射单元列顺序：先中继 0 的 3 个，再中继 1 的 2 个
    np.testing.assert_allclose(E0[:, 4:6], bf_block("src_scat", scat, [sc.tx[1]]), rtol=1e-12)


def test_scat_to_rcv_rows(small_scene):
    sc, rng = small_scene
    scat = [rand_set(rng, 3, [0.5, 0, 0]), rand_set(rng, 2, [0.4, 0.3, 0], Pattern(q=1.0, d0=np.pi))]
    C = assemble_scat_to_rcv(sc, scat)
    ref = bf_block("scat_rcv", [*sc.rx, sc.probes], scat)
    np.testing.assert_allclose(C, ref, rtol=1e-12, atol=0)
    assert assemble_scat_to_rcv(sc, []).shape == (sc.n_rcv, 0)


@pytest.mark.parametrize("block,skip_db,incl", [(False, 0.0, False), (True, 0.0, False),
                                                (False, -7.0, True), (True, 5.0, True)])
def test_src_to_rcv_direct_terms(small_scene, block, skip_db, incl):
    sc, _ = small_scene
    sc.block_direct_feed_probe, sc.skip_gain_db, sc.include_direct_tx_rx = block, skip_db, incl
    H0 = assemble_src_to_rcv(sc)
    n_rx = sc.n_rx
    fac = 0.0 if block else 10 ** (skip_db / 20)
    assert feed_probe_los_factor(sc) == pytest.approx(fac)
    np.testing.assert_allclose(H0[:n_rx, 0], bf_block("src_rcv", sc.rx, [sc.feeds]) @ sc.feed_weights,
                               rtol=1e-12)
    np.testing.assert_allclose(H0[n_rx:, 0],
                               fac * (bf_block("src_rcv", [sc.probes], [sc.feeds]) @ sc.feed_weights),
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(H0[n_rx:, 1:], bf_block("src_rcv", [sc.probes], sc.tx), rtol=1e-12)
    if incl:
        np.testing.assert_allclose(H0[:n_rx, 1:], bf_block("src_rcv", sc.rx, sc.tx), rtol=1e-12)
    else:
        # 同一中继的 tx→rx 块为 0（交给 κ0）；不同中继之间的块仍包含（PHY-R2-4）
        ref = bf_block("src_rcv", sc.rx, sc.tx).copy()
        i0 = j0 = 0
        for a, b in zip(sc.rx, sc.tx):
            ref[i0:i0 + len(a), j0:j0 + len(b)] = 0.0
            i0 += len(a)
            j0 += len(b)
        np.testing.assert_allclose(H0[:n_rx, 1:], ref, rtol=1e-12, atol=0)


def test_empty_scene_shapes():
    sc = Scene(F0, NodeSet.empty())
    assert assemble_src_to_rcv(sc).shape == (0, 1)
    assert assemble_src_to_scat(sc, []).shape == (0, 1)
    s = NodeSet([[0, 0, 0]], None, Pattern(), alpha=1e-6)
    np.testing.assert_array_equal(assemble_src_to_scat(sc, [s]), np.zeros((1, 1)))
    probes = NodeSet([[1, 0, 0]], [-1, 0, 0], Pattern(q=1.0))
    sc2 = Scene(F0, NodeSet.empty(), probes=probes)              # 无馈源：第 0 列为 0
    np.testing.assert_array_equal(assemble_src_to_rcv(sc2), np.zeros((1, 1)))


def test_inter_relay_tx_rx_coupling_always_included():
    """PHY-R2-4：N_r>1 时，不同中继之间的 发射→接收 自由空间耦合始终在 H0 里；同一中继的块由 include_direct_tx_rx 控制。"""
    from ris_nlsim.geometry import make_relays
    from ris_nlsim.greens import src_to_rcv
    from ris_nlsim.patterns import Pattern
    from ris_nlsim.scene import NodeSet, Scene, assemble_src_to_rcv
    from ris_nlsim.constants import F0, wavenumber

    pat = Pattern(q=1.0, d0=np.pi, backlobe_db=-25.0)
    rel = make_relays(2, 0.0)
    rx = [NodeSet(r.rx_pos, r.rx_normal, pat) for r in rel]
    tx = [NodeSet(r.tx_pos, r.tx_normal, pat) for r in rel]
    feed = NodeSet(np.array([[2.0, -1.0, 0.0]]), np.array([[-1.0, 0.5, 0.0]]), Pattern(q=10.0))
    probes = NodeSet(np.array([[1.0, 1.0, 0.0]]), np.array([[-1.0, -1.0, 0.0]]), Pattern(q=1.0))
    k = wavenumber(F0)
    for direct in (False, True):
        sc = Scene(freq_hz=F0, feeds=feed, tx=tx, rx=rx, probes=probes, include_direct_tx_rx=direct)
        H0 = assemble_src_to_rcv(sc)
        n = len(rx[0])
        cross = H0[n:2 * n, 1:1 + n]                      # 中继 1 的 rx ← 中继 0 的 tx
        assert np.allclose(cross, src_to_rcv(rx[1], tx[0], k))
        same = H0[:n, 1:1 + n]                            # 中继 0 的 rx ← 中继 0 的 tx
        if direct:
            assert np.allclose(same, src_to_rcv(rx[0], tx[0], k))
        else:
            assert np.all(same == 0)
