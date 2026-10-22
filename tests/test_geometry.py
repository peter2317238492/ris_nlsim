"""geometry 测试：中继板几何、观测点、体素化、字母位图、奇偶槽位（DESIGN §1.4）。"""
import numpy as np
import pytest

from ris_nlsim.constants import LAMBDA, PITCH
from ris_nlsim.geometry import (LETTERS, board_elements, make_relay, make_relays, parity_slots,
                                point_from, voxelize_box, voxelize_letter, voxelize_sphere)

W = 5 * PITCH
GAP = 0.01


def _min_pair_dist(a, b):
    return np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1))


def _board_samples(corner_inner, u, n_s=401):
    """板在 xy 平面上的截线（z 方向两板相同）：从内边缘沿 u 取样 W 长度。"""
    t = np.linspace(0.0, W, n_s)
    return corner_inner[None, :2] + t[:, None] * u[None, :2]


# ---------------------------------------------------------------- 板单元

def test_board_elements_cell_centres():
    pos = board_elements([1.0, 2.0, 3.0], [0, 2.0, 0], [0, 0, 5.0], nu=3, nv=4, pitch=0.1)
    assert pos.shape == (12, 3)
    np.testing.assert_allclose(pos[0], [1.0, 2.05, 3.05])
    np.testing.assert_allclose(pos[1] - pos[0], [0, 0, 0.1])      # 索引 i*nv + j：j 沿 v
    np.testing.assert_allclose(pos[4] - pos[0], [0, 0.1, 0])      # i 沿 u
    np.testing.assert_allclose(pos[-1], [1.0, 2.25, 3.35])


@pytest.mark.parametrize("psi", [0.0, 37.0, 90.0, 135.0, 180.0])
def test_relay_counts_spacing_planarity(psi):
    g = make_relay(psi)
    for pos, nrm in [(g.rx_pos, g.rx_normal), (g.tx_pos, g.tx_normal)]:
        assert pos.shape == (25, 3)
        assert np.linalg.norm(nrm) == pytest.approx(1.0)
        # 板平面：(pos − center)·n = 0
        np.testing.assert_allclose((pos - pos.mean(0)) @ nrm, 0.0, atol=1e-14)
        # 最近邻间距 = pitch，且 5×5 网格（每个单元的最近邻恰为 pitch）
        D = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(25) * 1e9
        np.testing.assert_allclose(D.min(axis=1), PITCH, rtol=1e-12)
        grid = pos.reshape(5, 5, 3)
        np.testing.assert_allclose(np.linalg.norm(np.diff(grid, axis=0), axis=-1), PITCH, rtol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(np.diff(grid, axis=1), axis=-1), PITCH, rtol=1e-12)
    np.testing.assert_allclose(g.rx_center, g.rx_pos.mean(0))
    np.testing.assert_allclose(g.tx_center, g.tx_pos.mean(0))
    np.testing.assert_allclose(g.center, [0, 0, 0])
    assert g.fold_deg == psi


def test_relay_normals_for_canonical_folds():
    g0, g90, g180 = make_relay(0.0), make_relay(90.0), make_relay(180.0)
    for g in (g0, g90, g180):
        np.testing.assert_allclose(g.rx_normal, [1, 0, 0])
    np.testing.assert_allclose(g0.tx_normal, [1, 0, 0], atol=1e-15)       # 共面，两板都朝 +x
    np.testing.assert_allclose(g90.tx_normal, [0, 1, 0], atol=1e-15)      # L 形，发射板朝 +y
    np.testing.assert_allclose(g180.tx_normal, [-1, 0, 0], atol=1e-15)    # 背靠背
    assert np.all(g90.tx_pos[:, 0] < 0)                                    # L 形发射板位于 x<0 侧


def test_rx_board_matches_design_and_flat_tx_board():
    o = np.array([0.3, -0.2, 0.5])
    g = make_relay(0.0, origin=o)
    p = PITCH
    rel = g.rx_pos - o
    np.testing.assert_allclose(rel[:, 0], 0.0, atol=1e-15)                # 平面 x = origin.x
    assert rel[:, 1].min() == pytest.approx(-GAP / 2 - W + p / 2)
    assert rel[:, 1].max() == pytest.approx(-GAP / 2 - p / 2)
    assert rel[:, 2].min() == pytest.approx(-W / 2 + p / 2)
    assert rel[:, 2].max() == pytest.approx(W / 2 - p / 2)
    # ψ=0 时发射板严格按 DESIGN 公式：y ∈ [gap/2, gap/2 + W]，x = origin.x（与接收板共面）
    t = g.tx_pos - o
    np.testing.assert_allclose(t[:, 0], 0.0, atol=1e-15)
    assert t[:, 1].min() == pytest.approx(GAP / 2 + p / 2)
    assert t[:, 1].max() == pytest.approx(GAP / 2 + W - p / 2)
    # 内边缘间距恰为 gap：两板最近单元相距 gap + pitch
    assert _min_pair_dist(g.rx_pos, g.tx_pos) == pytest.approx(GAP + p, rel=1e-12)


def test_back_offset_zero_recovers_design_formula():
    for psi in [30.0, 90.0, 150.0]:
        g = make_relay(psi, back_offset=0.0)
        s = np.deg2rad(psi)
        e_t = np.array([-np.sin(s), np.cos(s), 0])
        corner = (GAP / 2) * e_t + np.array([0, 0, -W / 2])
        ref = board_elements(corner, e_t, [0, 0, 1], 5, 5, PITCH)
        np.testing.assert_allclose(g.tx_pos, ref, atol=1e-15)


@pytest.mark.parametrize("gap", [0.0, 0.01, 0.03])
def test_boards_never_overlap_and_gap_respected(gap):
    # 扫描折叠角：两块板（截线取样）最小距离 ≥ gap，且 ψ=0、180° 时恰为 gap；
    # 单元之间也至少相距 gap；发射板始终不在接收板前方（x ≤ origin.x）
    for psi in np.linspace(0.0, 180.0, 37):
        g = make_relay(psi, gap=gap)
        rx_line = _board_samples(g.rx_pos.mean(0) - (W / 2) * g.rx_u, g.rx_u)
        tx_line = _board_samples(g.tx_pos.mean(0) - (W / 2) * g.tx_u, g.tx_u)
        dmin = _min_pair_dist(np.c_[rx_line, np.zeros(len(rx_line))],
                              np.c_[tx_line, np.zeros(len(tx_line))])
        assert dmin >= gap - 1e-12
        # 内边缘距离闭式 gap·sqrt(1+sinψ)
        inner_rx = rx_line[0]
        inner_tx = tx_line[0]
        assert np.linalg.norm(inner_rx - inner_tx) == pytest.approx(
            gap * np.sqrt(1 + np.sin(np.deg2rad(psi))), abs=1e-12)
        if gap > 0:
            assert _min_pair_dist(g.rx_pos, g.tx_pos) >= gap - 1e-12
        assert np.all(g.tx_pos[:, 0] <= 1e-12)
    g180 = make_relay(180.0, gap=gap)
    if gap > 0:
        # 背靠背：两板平面相距 gap，单元一一正对
        np.testing.assert_allclose(g180.rx_pos - g180.tx_pos, np.tile([gap, 0, 0], (25, 1)), atol=1e-14)


def test_make_relay_validates_inputs():
    with pytest.raises(ValueError):
        make_relay(190.0)
    with pytest.raises(ValueError):
        make_relay(90.0, gap=-0.01)


def test_make_relays_stacked_and_centered():
    rs = make_relays(4, 90.0, spacing_z=0.30)
    assert len(rs) == 4
    zc = np.array([r.center[2] for r in rs])
    np.testing.assert_allclose(np.diff(zc), 0.30)
    assert zc.mean() == pytest.approx(0.0, abs=1e-15)
    allpos = np.vstack([np.vstack([r.rx_pos, r.tx_pos]) for r in rs])
    assert allpos.mean(0)[2] == pytest.approx(0.0, abs=1e-14)
    for i in range(3):          # 相邻中继的板不重叠
        assert _min_pair_dist(rs[i].rx_pos, rs[i + 1].rx_pos) >= 0.30 - W + PITCH - 1e-12
    single = make_relays(1, 0.0, origin=(1, 2, 3))
    np.testing.assert_allclose(single[0].center, [1, 2, 3])
    with pytest.raises(ValueError):
        make_relays(2, 90.0, spacing_z=0.1)


def test_point_from_angles():
    ref = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(point_from(ref, 0, 0, 2.0), ref + [2, 0, 0], atol=1e-15)
    np.testing.assert_allclose(point_from(ref, 90, 0, 2.0), ref + [0, 2, 0], atol=1e-15)
    np.testing.assert_allclose(point_from(ref, -30, 0, 1.8), ref + 1.8 * np.array(
        [np.cos(np.pi / 6), -0.5, 0]), atol=1e-15)
    np.testing.assert_allclose(point_from(ref, 123, 90, 1.0), ref + [0, 0, 1], atol=1e-15)
    assert np.linalg.norm(point_from(ref, 77, -20, 3.3) - ref) == pytest.approx(3.3)


# ---------------------------------------------------------------- 体素化

def _sphere_lattice_bruteforce(D, d):
    """独立参考：名义格距 d 下 |i|·d ≤ D/2 的全部整数格点 i（三重循环）。"""
    m = int(D / 2 / d) + 1
    rng_ = range(-m, m + 1)
    return {(a, b, c) for a in rng_ for b in rng_ for c in rng_
            if (a * a + b * b + c * c) * d * d <= (D / 2) ** 2 * (1 + 1e-12)}


@pytest.mark.parametrize("D,d", [(LAMBDA / 3, LAMBDA / 10), (LAMBDA / 3, LAMBDA / 12), (LAMBDA / 3, LAMBDA / 15),
                                 (LAMBDA / 3, LAMBDA / 20), (0.1, 0.01), (LAMBDA / 3, LAMBDA / 2),
                                 (LAMBDA / 3, LAMBDA / 8)])
def test_voxelize_sphere_volume_and_containment(D, d):
    c = np.array([0.4, -0.1, 0.2])
    pos, v = voxelize_sphere(c, D, d)
    n = len(pos)
    assert n >= 1
    assert n * v == pytest.approx(np.pi * D**3 / 6, rel=1e-14)          # 体积修正（DESIGN §1.4）
    np.testing.assert_allclose(pos.mean(0), c, atol=1e-14)               # 关于中心对称
    s = np.cbrt(v) / d                                                   # 实际格距 / 名义格距
    # 格距缩放后最外层可略超出球面：|pos − c| ≤ D/2·max(s, 1)
    assert np.all(np.linalg.norm(pos - c, axis=1) <= D / 2 * max(s, 1.0) * (1 + 1e-9))
    if n > 1:
        # 格胞体积 = 偶极子体积：最近邻间距恰为 vol_each^{1/3}（PHY-R2-2）
        D2 = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(n)
        assert D2.min() == pytest.approx(np.cbrt(v), rel=1e-12)
        # 选点规则不变：缩放回整数格后正是 |i|·d ≤ D/2 的全部格点（与三重循环暴力集合一致）
        idx = (pos - c) / np.cbrt(v)
        np.testing.assert_allclose(idx, np.round(idx), atol=1e-9)
        assert set(map(tuple, np.round(idx).astype(int))) == _sphere_lattice_bruteforce(D, d)
    if D / d >= 9.99:
        # 足够细时 N·d³ 逼近球体积 ⇒ 缩放因子接近 1
        assert n * d**3 == pytest.approx(np.pi * D**3 / 6, rel=0.05)
        assert abs(s - 1) < 0.02


def test_voxelize_sphere_cell_volume_equals_dipole_volume():
    """回归（PHY-R2-2）：旧实现保留格距 d、只把 vol_each 改成 πD³/(6N)，λ/3 球 λ/15 时 N=81（理想 65.4），
    格胞体积 d³ 与偶极子体积相差 19%，DDA 误差随网格非单调。现在任何 d 下 N·(最近邻间距)³ = πD³/6。"""
    D = LAMBDA / 3
    V = np.pi * D**3 / 6
    pos, v = voxelize_sphere([0, 0, 0], D, LAMBDA / 15)
    assert len(pos) == 81 and 81 * (LAMBDA / 15) ** 3 / V == pytest.approx(1.238, abs=1e-3)   # 台阶误差确实存在
    for nd in [6, 9, 10, 11, 12, 13, 15, 17, 20, 25]:
        pos, v = voxelize_sphere([0, 0, 0], D, LAMBDA / nd)
        nn = np.min(np.linalg.norm(pos[:, None] - pos[None], axis=-1) + 1e9 * np.eye(len(pos)), axis=1)
        np.testing.assert_allclose(nn, np.cbrt(v), rtol=1e-12)          # 每个体素的最近邻都是同一格距
        assert len(pos) * nn.min() ** 3 == pytest.approx(V, rel=1e-12)


def test_voxelize_sphere_tiny_gives_one_voxel():
    pos, v = voxelize_sphere([1, 2, 3], 1e-3, LAMBDA / 10)
    assert pos.shape == (1, 3)
    np.testing.assert_allclose(pos[0], [1, 2, 3])
    assert v == pytest.approx(np.pi * 1e-9 / 6)
    assert len(voxelize_sphere([0, 0, 0], LAMBDA / 3, LAMBDA / 10)[0]) == 19   # |i|²≤2.78 的格点


def test_voxelize_box():
    c = np.array([0.1, 0.2, 0.3])
    size = np.array([0.05, 0.02, 0.031])
    pos, v = voxelize_box(c, size, 0.01)
    assert len(pos) == 5 * 2 * 4                                         # ceil(size/d)
    assert len(pos) * v == pytest.approx(np.prod(size), rel=1e-14)
    np.testing.assert_allclose(pos.mean(0), c, atol=1e-15)
    assert np.all(np.abs(pos - c) <= size / 2 + 1e-15)
    pos1, v1 = voxelize_box(c, [1e-3, 1e-3, 1e-3], 0.01)
    assert len(pos1) == 1 and v1 == pytest.approx(1e-9)


# ---------------------------------------------------------------- 字母

def test_letter_bitmaps_distinct_7x5():
    assert set(LETTERS) == {"A", "B", "C"}
    for bm in LETTERS.values():
        assert bm.shape == (7, 5)
        assert set(np.unique(bm)) <= {0, 1}
        assert bm.sum() >= 10
    keys = list(LETTERS)
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.sum(LETTERS[keys[i]] != LETTERS[keys[j]]) >= 3
    with pytest.raises(ValueError):
        LETTERS["A"][0, 0] = 1                                            # 只读


@pytest.mark.parametrize("letter", ["A", "b", "C"])
def test_voxelize_letter(letter):
    c = np.array([0.5, 0.0, 0.1])
    u, v = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    pix, th, d = LAMBDA / 2, LAMBDA / 2, LAMBDA / 5
    pos, vol, ids = voxelize_letter(letter, c, u, v, pix, th, d)
    bm = LETTERS[letter.upper()]
    n_on = int(bm.sum())
    assert len(pos) == len(ids)
    assert len(np.unique(ids)) == n_on
    assert len(pos) * vol == pytest.approx(n_on * pix * pix * th, rel=1e-13)
    # 体素边长 ≤ d：λ/2 / (λ/5) = 2.5 → 每轴 3 格，每像素 27 个体素
    assert len(pos) == 27 * n_on
    # 每个像素的体素重心 = 像素中心；行 0 在 +v 一侧、列 0 在 −u 一侧；厚度方向 n = u×v = +x 居中
    for pid in np.unique(ids):
        r, col = divmod(int(pid), 5)
        assert bm[r, col] == 1
        cen = pos[ids == pid].mean(0)
        expect = c + (col - 2) * pix * u + (3 - r) * pix * v
        np.testing.assert_allclose(cen, expect, atol=1e-14)
    assert np.ptp(pos[:, 0]) == pytest.approx(th * 2 / 3, rel=1e-12)    # 3 层，间距 th/3
    with pytest.raises(ValueError):
        voxelize_letter("Z", c, u, v)
    with pytest.raises(ValueError):
        voxelize_letter("A", c, u, [0, 1, 1])


def test_parity_slots():
    c = np.array([0.5, 0.0, 0.0])
    s = parity_slots(3, c, LAMBDA / 2)
    np.testing.assert_allclose(s.mean(0), c, atol=1e-15)
    np.testing.assert_allclose(np.diff(s, axis=0), np.tile([0, 0, LAMBDA / 2], (2, 1)), atol=1e-15)
    s4 = parity_slots(4, c, 0.1, axis=(0, 2, 0))
    np.testing.assert_allclose(s4[:, 1], [-0.15, -0.05, 0.05, 0.15], atol=1e-15)
    np.testing.assert_allclose(parity_slots(1, c, 0.1), c[None])
