"""patterns.Pattern 测试：方向性归一化（球面数值积分）、广播、取值（DESIGN §1.1）。"""
import numpy as np
import pytest

from ris_nlsim.patterns import Pattern


def _frame(normal):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    t = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, t)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return n, e1, e2


def sphere_mean_gain(pat: Pattern, normal, n_mu=256, n_phi=64):
    """在以 normal 为极轴的坐标系里做球面求积：∮F²dΩ/(4π)。
    cosθ 用 Gauss–Legendre（在 cosθ=0 处分段，避开 max(cos,0) 的拐点），φ 用均匀梯形（周期函数精确）。
    """
    n, e1, e2 = _frame(normal)
    x, w = np.polynomial.legendre.leggauss(n_mu)
    mu = np.concatenate([(x - 1) / 2, (x + 1) / 2])            # [-1,0] ∪ [0,1]
    wm = np.concatenate([w / 2, w / 2])
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    MU, PHI = np.meshgrid(mu, phi, indexing="ij")
    s = np.sqrt(1 - MU**2)
    dirs = (MU[..., None] * n + s[..., None] * (np.cos(PHI)[..., None] * e1
                                                 + np.sin(PHI)[..., None] * e2))
    F = pat.amp(n, dirs.reshape(-1, 3)).reshape(MU.shape)
    integral = np.sum(wm[:, None] * F**2) * (2 * np.pi / n_phi)
    return integral / (4 * np.pi)


def test_isotropic_is_exactly_one():
    p = Pattern()
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(7, 11, 3))
    F = p.amp(rng.normal(size=(7, 3)), dirs)
    assert F.shape == (7, 11)
    assert np.all(F == 1.0)
    assert p.mean_gain() == 1.0 and p.peak_directivity == 1.0
    assert sphere_mean_gain(p, [0.3, -0.2, 0.9]) == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("q", [0.0, 1.0, 2.0, 10.0])
def test_default_d0_normalized_integer_q(q):
    # 后瓣极低、d0 默认 2(q+1) → 球面平均方向性 = 1（高精度，整数 q 时 GL 精确）
    p = Pattern(q=q, backlobe_db=-200.0)
    assert p.d0_eff == pytest.approx(2 * (q + 1))
    m = sphere_mean_gain(p, [0.2, 0.5, -0.8])
    assert m == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("q", [0.5, 2.5, 7.3])
def test_default_d0_normalized_fractional_q(q):
    p = Pattern(q=q, backlobe_db=-200.0)
    m = sphere_mean_gain(p, [1.0, 1.0, 1.0], n_mu=600)
    assert m == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("q,bl", [(1.0, -30.0), (10.0, -30.0), (1.0, -25.0)])
def test_mean_gain_with_backlobe_matches_closed_form(q, bl):
    p = Pattern(q=q, backlobe_db=bl)
    expected = 1.0 + 2 * (q + 1) * 10 ** (bl / 10)
    assert p.mean_gain() == pytest.approx(expected, rel=1e-14)
    assert sphere_mean_gain(p, [0, 0, 1]) == pytest.approx(expected, rel=1e-12)


def test_element_pattern_aperture_limited():
    # 中继单元：q=1, d0=π, 后瓣 −25 dB → 平均 = π(1/4 + 10^-2.5)（不归一化，口径受限）
    p = Pattern(q=1.0, d0=np.pi, backlobe_db=-25.0)
    exp = np.pi * (0.25 + 10 ** -2.5)
    assert p.mean_gain() == pytest.approx(exp, rel=1e-14)
    assert sphere_mean_gain(p, [0, 1, 0]) == pytest.approx(exp, rel=1e-12)


def test_values_boresight_back_and_side():
    p = Pattern(q=10.0)                      # 喇叭：d0 = 22
    n = np.array([0.0, 0.0, 1.0])
    fl = 1e-3
    assert p.amp(n, n) ** 2 == pytest.approx(22 * (1 + fl), rel=1e-14)
    assert p.amp(n, -n) ** 2 == pytest.approx(22 * fl, rel=1e-14)
    assert p.amp(n, [1.0, 0, 0]) ** 2 == pytest.approx(22 * fl, rel=1e-14)
    th = np.deg2rad(25.0)
    d = np.array([np.sin(th), 0, np.cos(th)])
    assert p.gain(n, d) == pytest.approx(22 * (np.cos(th) ** 10 + fl), rel=1e-13)
    # q=0：前半球常数、后半球只剩地板（不能出现 0**0 = 1）
    h = Pattern(q=0.0, backlobe_db=-40)
    assert h.gain(n, -n) == pytest.approx(2 * 1e-4)
    assert h.gain(n, n) == pytest.approx(2 * (1 + 1e-4))


def test_broadcast_shapes_and_values_against_loop():
    rng = np.random.default_rng(7)
    p = Pattern(q=3.0, d0=5.0, backlobe_db=-20.0)
    N, M = 4, 6
    nrm = rng.normal(size=(N, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    dirs = rng.normal(size=(N, M, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    F = p.amp(nrm, dirs)
    assert F.shape == (N, M)
    for i in range(N):
        for j in range(M):
            c = float(nrm[i] @ dirs[i, j])
            ref = np.sqrt(5.0 * (max(c, 0.0) ** 3 + 0.01))
            assert F[i, j] == pytest.approx(ref, rel=1e-14)
    # (3,) 与 (M,3) 常规广播；(N,3) 与 (N,3) 逐元素
    assert p.amp(nrm[0], dirs[0]).shape == (M,)
    assert p.amp(nrm, dirs[:, 0, :]).shape == (N,)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        Pattern(q=None, d0=3.0)
    with pytest.raises(ValueError):
        Pattern(q=-1.0)
    with pytest.raises(ValueError):
        Pattern(q=1.0, d0=0.0)


def test_hashable_and_equal():
    assert Pattern(q=1.0, d0=np.pi, backlobe_db=-25) == Pattern(q=1, d0=np.pi, backlobe_db=-25.0)
    assert len({Pattern(), Pattern(), Pattern(q=1.0)}) == 2


def test_default_pattern_factories():
    from ris_nlsim.constants import LAMBDA
    from ris_nlsim.patterns import (ISOTROPIC, element_pattern, horn_pattern, po_cell_pattern,
                                    probe_pattern)
    assert ISOTROPIC.isotropic
    e = element_pattern()
    assert (e.q, e.d0_eff, e.backlobe_db) == (1.0, pytest.approx(np.pi), -25.0)
    # 单元 d0 = 4πA/λ²，A = (λ/2)²
    assert e.d0_eff == pytest.approx(4 * np.pi * (LAMBDA / 2) ** 2 / LAMBDA**2)
    h = horn_pattern()
    assert h.d0_eff == pytest.approx(22.0)
    assert 10 * np.log10(h.peak_directivity) == pytest.approx(13.4, abs=0.05)
    assert probe_pattern().d0_eff == pytest.approx(4.0)
    po = po_cell_pattern((LAMBDA / 4) ** 2, LAMBDA)
    assert po.d0_eff == pytest.approx(np.pi / 4) and po.backlobe_db == -60.0
