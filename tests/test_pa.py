"""PA 模型测试（DESIGN §3.1，方案 §4.4）。"""
import numpy as np
import pytest
from scipy.optimize import brentq

from ris_nlsim.constants import dbm2w
from ris_nlsim.pa import LinearPA, RappPA


def _random_u(rng, pa, backoffs_db, n_per=40):
    """在若干回退量（相对 P1dB）附近随机取复输入。"""
    p1 = pa.p1db_in_w()
    out = []
    for bo in backoffs_db:
        amp = np.sqrt(p1 * 10 ** (bo / 10)) * rng.uniform(0.7, 1.3, n_per)
        out.append(amp * np.exp(1j * rng.uniform(-np.pi, np.pi, n_per)))
    return np.concatenate(out)


@pytest.mark.parametrize("ampm", [True, False])
@pytest.mark.parametrize("p", [1.0, 2.0, 5.0])
def test_wirtinger_vs_complex_finite_difference(ampm, p):
    rng = np.random.default_rng(1)
    pa = RappPA(p=p, ampm=ampm, ampm_max_deg=25.0, ampm_beta=0.7)
    u = _random_u(rng, pa, backoffs_db=[-40, -20, -10, -3, 0, 3, 10, 20, 30])
    a, b = pa.wirtinger(u)
    eps = 1e-6 * np.abs(u)
    # δu = ε：δg = (a+b)ε；δu = jε：δg = (a−b)jε（中心差分）
    d1 = (pa.g(u + eps) - pa.g(u - eps)) / (2 * eps)
    d2 = (pa.g(u + 1j * eps) - pa.g(u - 1j * eps)) / (2j * eps)
    a_fd, b_fd = 0.5 * (d1 + d2), 0.5 * (d1 - d2)
    scale = np.abs(a) + np.abs(b)
    assert np.max(np.abs(a - a_fd) / scale) < 1e-7
    assert np.max(np.abs(b - b_fd) / scale) < 1e-7


def test_wirtinger_small_signal_limit_and_zero():
    pa = RappPA()
    a, b = pa.wirtinger(np.array([0.0 + 0j, 1e-40 + 1e-40j, 1e-12 * pa.A_in]))
    assert np.allclose(a, pa.gain_v, rtol=1e-12)
    assert np.allclose(b, 0.0, atol=1e-12 * pa.gain_v)
    assert pa.g(0.0) == 0.0


@pytest.mark.parametrize("p", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("gain_db,psat", [(30.0, 20.0), (15.0, 5.0)])
def test_p1db_exactly_one_db_compression(p, gain_db, psat):
    pa = RappPA(gain_db=gain_db, psat_out_dbm=psat, p=p)
    p1 = pa.p1db_in_w()
    u = np.sqrt(p1) * np.exp(1j * np.linspace(0, 2 * np.pi, 7))
    comp_db = 20 * np.log10(np.abs(pa.g(u)) / (pa.gain_v * np.abs(u)))
    assert np.allclose(comp_db, -1.0, atol=1e-12)
    # 与独立数值求根一致
    f = lambda x: 20 * np.log10(pa.amam(np.sqrt(x)) / (pa.gain_v * np.sqrt(x))) + 1.0
    p_root = brentq(f, 1e-6 * p1, 1e3 * p1, xtol=1e-30, rtol=1e-14)
    assert np.isclose(p_root, p1, rtol=1e-10)


def test_deep_saturation_amplitude_and_ampm_limit():
    pa = RappPA(gain_db=30.0, psat_out_dbm=20.0, p=2.0, ampm_max_deg=12.0, ampm_beta=1.0)
    assert np.isclose(pa.A_s, np.sqrt(dbm2w(20.0)))
    u = pa.A_in * np.array([1e4, 1e6, 1e100, 1e200]) * np.exp(0.3j)
    with np.errstate(over="raise", invalid="raise", divide="raise"):   # 下溢（c2→0）是良性的
        g = pa.g(u)
        a, b = pa.wirtinger(u)
    assert np.allclose(np.abs(g), pa.A_s, rtol=1e-12)
    phase_deg = np.rad2deg(np.angle(g / u))
    assert np.allclose(phase_deg, 12.0, atol=1e-6)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    # 深饱和：|a| ≈ |b| ≈ A_s/(2|u|)（输出幅度恒定，只剩相位通道）
    assert np.allclose(np.abs(a[0]), pa.A_s / (2 * abs(u[0])), rtol=1e-6)
    assert np.allclose(np.abs(b[0]), pa.A_s / (2 * abs(u[0])), rtol=1e-6)


def test_ampm_formula_and_switch():
    pa = RappPA(ampm_max_deg=12.0, ampm_beta=2.0)
    r = np.array([0.0, 0.1, 0.5, 1.0, 3.0])
    alpha = np.deg2rad(12.0) * 2.0
    expect = alpha * r**2 / (1 + 2.0 * r**2)
    assert np.allclose(pa.ampm_rad(r * pa.A_in), expect, rtol=1e-13, atol=0)
    pa_off = RappPA(ampm=False)
    u = pa_off.A_in * np.array([0.1, 1.0, 10.0]) * np.exp(1j * 0.7)
    assert np.allclose(np.angle(pa_off.g(u) / u), 0.0, atol=1e-14)


def test_rapp_amam_closed_form():
    pa = RappPA(p=3.0, ampm=False)
    r = np.logspace(-3, 3, 50)
    expect = pa.gain_v * r * pa.A_in / (1 + r ** 6) ** (1 / 6)
    assert np.allclose(np.abs(pa.g(r * pa.A_in)), expect, rtol=1e-12)


def test_arbitrary_shapes():
    rng = np.random.default_rng(3)
    for pa in (RappPA(), LinearPA(30.0)):
        for shape in [(), (5,), (2, 3), (2, 3, 4)]:
            u = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 1e-2
            g = pa.g(u)
            a, b = pa.wirtinger(u)
            assert np.shape(g) == shape and np.shape(a) == shape and np.shape(b) == shape
            assert np.asarray(g).dtype == np.complex128
        # 逐元素：整体调用 == 展平逐个调用
        u = (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))) * 1e-2
        flat = np.array([pa.g(x) for x in u.ravel()]).reshape(3, 4)
        assert np.allclose(pa.g(u), flat, rtol=1e-14)


def test_linear_pa_trivial():
    pa = LinearPA(20.0)
    assert np.isclose(pa.gain_v, 10.0)
    u = np.array([[1 + 2j, -3j], [0.5, 0]])
    assert np.allclose(pa.g(u), 10.0 * u)
    a, b = pa.wirtinger(u)
    assert np.all(a == 10.0) and np.all(b == 0.0)
    assert pa.p1db_in_w() == np.inf
    assert pa.is_linear and not RappPA().is_linear


def test_small_signal_gain_matches_linear():
    """深回退下 Rapp ≈ 线性 PA（同 G_v）。"""
    rapp, lin = RappPA(), LinearPA(30.0)
    u = np.sqrt(rapp.p1db_in_w() * 1e-4) * np.exp(1j * np.linspace(0, 6, 9))   # -40 dB 回退
    assert np.allclose(rapp.g(u), lin.g(u), rtol=1e-4)
