"""指标测试（DESIGN §5，方案 §6.1、§7）。"""
import itertools

import numpy as np
import pytest

from ris_nlsim.metrics import (accuracy, binary_margin_db, null_depth_db, parity_labels, popcount,
                               required_snr_db, walsh_coeffs, walsh_degree_profile, walsh_weight_ge)


def _inputs(n):
    """x[i, j] = (i >> j) & 1（DESIGN §5 编码）。"""
    i = np.arange(2 ** n)
    return ((i[:, None] >> np.arange(n)[None, :]) & 1).astype(float)


def test_walsh_matches_bruteforce_definition():
    n = 4
    rng = np.random.default_rng(0)
    f = rng.standard_normal((2 ** n, 3)) + 1j * rng.standard_normal((2 ** n, 3))
    x = np.arange(2 ** n)
    ref = np.zeros_like(f)
    for S in range(2 ** n):
        sign = (-1.0) ** popcount(S & x)
        ref[S] = (sign[:, None] * f).sum(0) / 2 ** n
    assert np.allclose(walsh_coeffs(f, n), ref, atol=1e-14)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_walsh_round_trip(n):
    rng = np.random.default_rng(n)
    for shape in [(2 ** n,), (2 ** n, 4), (2 ** n, 2, 3)]:
        f = rng.standard_normal(shape)
        c = walsh_coeffs(f, n)
        assert c.shape == shape
        assert np.allclose(2 ** n * walsh_coeffs(c, n), f, atol=1e-13)
        # Parseval：Σ|f|² = 2^n Σ|f̂|²
        assert np.isclose((f ** 2).sum(), 2 ** n * (c ** 2).sum())
    fc = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    assert np.allclose(2 ** n * walsh_coeffs(walsh_coeffs(fc, n), n), fc)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_parity_has_full_top_degree_weight(n):
    lab = parity_labels(n)
    x = _inputs(n)
    assert np.array_equal(lab, x.sum(1).astype(int) % 2)
    assert walsh_weight_ge(lab, n, n) == pytest.approx(1.0, abs=1e-14)
    assert walsh_weight_ge((-1.0) ** lab, n, n) == pytest.approx(1.0, abs=1e-14)
    prof = walsh_degree_profile(lab, n)
    assert np.allclose(prof, np.eye(n)[n - 1])


def test_abs_affine_squared_has_no_degree3_weight():
    """方案 §6.1：Born 下 e(x) = e0 + Σx_i e_i（仿射）⇒ y = |w·e|² 为二次 ⇒ W≥3 = 0。"""
    rng = np.random.default_rng(1)
    for n in (3, 4, 5):
        x = _inputs(n)
        e0 = rng.standard_normal(6) + 1j * rng.standard_normal(6)
        E = rng.standard_normal((n, 6)) + 1j * rng.standard_normal((n, 6))
        e = e0[None, :] + x @ E
        w = rng.standard_normal((2, 6)) + 1j * rng.standard_normal((2, 6))
        y = np.abs(e @ w.T) ** 2                                   # (2^n, 2) 两个探头
        assert walsh_weight_ge(y, n, 3) < 1e-25
        assert walsh_weight_ge(y[:, 0] - y[:, 1], n, 3) < 1e-25
        assert walsh_weight_ge(e, n, 2) < 1e-25                    # 仿射场本身只有一阶
        assert walsh_weight_ge(y, n, 2) > 1e-3                     # 但平方律检波产生二阶
    # 对照：x0·x1·x2 的 W≥3 > 0
    x = _inputs(3)
    assert walsh_weight_ge(x[:, 0] * x[:, 1] * x[:, 2], 3, 3) > 0.1


def test_degree_profile_closed_form_and_sums_to_one():
    """f = x0 + x1·x2：x_i = (1 − χ_i)/2 ⇒ 非空系数 {0}:−1/2, {1}:−1/4, {2}:−1/4, {1,2}:+1/4
    ⇒ 一阶权重 (1/4+1/16+1/16)/(7/16) = 6/7，二阶 1/7。"""
    x = _inputs(3)
    f = x[:, 0] + x[:, 1] * x[:, 2]
    assert np.allclose(walsh_degree_profile(f, 3), [6 / 7, 1 / 7, 0.0])
    assert walsh_weight_ge(f, 3, 2) == pytest.approx(1 / 7)
    c = walsh_coeffs(f, 3)
    assert np.isclose(c[0], 0.5 + 0.25) and np.isclose(c[0b001], -0.5) and np.isclose(c[0b110], 0.25)
    rng = np.random.default_rng(2)
    for n in (2, 4, 6):
        prof = walsh_degree_profile(rng.standard_normal((2 ** n, 3)), n)
        assert prof.shape == (n,) and np.isclose(prof.sum(), 1.0)
        assert np.isclose(walsh_weight_ge(rng.standard_normal(2 ** n), n, 1), 1.0)


def test_constant_function_gives_zero():
    assert walsh_weight_ge(np.full(8, 3.0), 3, 1) == 0.0
    assert np.array_equal(walsh_degree_profile(np.full(16, 2.0 + 1j), 4), np.zeros(4))
    # 浮点噪声级的非常数部分也按"分母为 0"处理（回归 TEST-2：输入必须真的在 ulp 级上不同）。
    # f = 1 + a·parity：非常数能量 / 总能量 ≈ a²/4；门限 _DENOM_REL = 1e-30 ⇒ 边界 a ≈ 2e-15
    par = parity_labels(3).astype(float)
    f_ulp = 1.0 + 2.220446049250313e-16 * par           # 1 与 1+eps 两个不同的值
    assert np.unique(f_ulp).size == 2
    assert walsh_weight_ge(f_ulp, 3, 3) == 0.0
    assert np.array_equal(walsh_degree_profile(f_ulp, 3), np.zeros(3))
    f_real = 1.0 + 1e-12 * par                           # 远高于门限：是真实的 3 阶结构
    assert walsh_weight_ge(f_real, 3, 3) == pytest.approx(1.0, abs=1e-6)
    assert walsh_weight_ge(1.0 + 1e-14 * par, 3, 3) == pytest.approx(1.0, abs=1e-2)   # 门限之上一个量级
    with pytest.raises(ValueError):
        walsh_coeffs(np.zeros(7), 3)


def test_binary_margin_and_accuracy():
    labels = np.array([0, 1, 1, 0])
    y = np.array([[-30.0, -28.0], [-20.0, -25.0], [-10.0, -10.5], [-40.0, -39.0]])
    # sgn·(y1−y2) = [2, 5, 0.5, 1] → 裕量 0.5
    assert binary_margin_db(y, labels) == pytest.approx(0.5)
    y_bad = y.copy()
    y_bad[2] = [-11.0, -10.0]
    assert binary_margin_db(y_bad, labels) == pytest.approx(-1.0)
    pred = (y_bad[:, 0] > y_bad[:, 1]).astype(int)
    assert accuracy(pred, labels) == 0.75
    assert accuracy(labels, labels) == 1.0


def test_required_snr_interpolation():
    assert required_snr_db([0, 10, 20], [0.5, 0.8, 1.0]) == pytest.approx(15.0)
    # 未排序输入
    assert required_snr_db([20, 0, 10], [1.0, 0.5, 0.8]) == pytest.approx(15.0)
    # 达不到 → nan；最低点即达标 → 最低点
    assert np.isnan(required_snr_db([0, 10, 20], [0.5, 0.6, 0.85]))
    assert required_snr_db([0, 10, 20], [0.95, 0.97, 1.0]) == 0.0
    # 非单调：取"此后始终达标"的交点
    assert required_snr_db([0, 10, 20], [0.95, 0.5, 0.92]) == pytest.approx(10 + 0.4 / 0.42 * 10)
    assert required_snr_db([0, 10], [0.0, 1.0], target=0.75) == pytest.approx(7.5)


def test_null_depth_db():
    assert null_depth_db(0.1, 1.0) == pytest.approx(-20.0)
    assert null_depth_db(1j * 0.01, -1.0) == pytest.approx(-40.0)
    assert null_depth_db(np.array([0.1, 0.1]), np.array([1.0, 1.0])) == pytest.approx(-20.0)
    assert null_depth_db(0.0, 1.0) == -np.inf
    assert isinstance(null_depth_db(np.array([0.1, 0.2]), 1.0), float)


def test_null_depth_db_per_sample_and_per_relay():
    """回归（METRICS-1）：每样本 (B,) 对标量聚焦参考不得有 +10log10(B) 偏差；
    多中继取各中继深度的最大值（最浅零陷，INTEGRATION §C）。"""
    u0 = np.full(8, 0.1)
    assert null_depth_db(u0, 1.0) == pytest.approx(-20.0)            # 旧实现给 −10.97 dB
    nd = null_depth_db(u0, 1.0, reduce="none")
    assert nd.shape == (8,) and np.allclose(nd, -20.0)
    # 两个中继深度不同：最大值 = 较浅者
    assert null_depth_db(np.array([0.1, 0.01]), 1.0) == pytest.approx(-20.0)
    assert np.allclose(null_depth_db(np.array([0.1, 0.01]), np.array([1.0, 0.1]), reduce="none"), [-20.0, -20.0])
    # (B,R) 对每中继参考 (R,) 广播
    u = np.array([[0.1, 0.05], [0.01, 0.5]])
    ref = np.array([1.0, 5.0])
    exp = 20 * np.log10(np.abs(u) / ref)
    assert np.allclose(null_depth_db(u, ref, reduce="none"), exp)
    assert null_depth_db(u, ref) == pytest.approx(exp.max())
    assert isinstance(null_depth_db(0.1, 1.0, reduce="none"), float)
    with pytest.raises(ValueError):
        null_depth_db(np.ones(3), np.ones(2))                         # 形状不兼容
    with pytest.raises(ValueError):
        null_depth_db(0.1, 1.0, reduce="sum")


def test_popcount():
    vals = np.arange(64)
    assert np.array_equal(popcount(vals), [bin(v).count("1") for v in vals])
    big = np.array([2 ** 62 - 1, 2 ** 40 + 5])
    assert np.array_equal(popcount(big), [62, 3])
    # 回归（METRICS-2）：负数曾导致死循环（−1 >> 1 == −1），现在报错
    with pytest.raises(ValueError):
        popcount(-1)
    with pytest.raises(ValueError):
        popcount(np.array([3, -2]))


def test_exhaustive_parity_unsolvable_by_quadratic_threshold_small_case():
    """Minsky–Papert 的小规模检查：n=3 时任一 x 的二次多项式符号都不等于奇偶
    —— 等价于奇偶函数在 Walsh 基上全部权重位于 3 阶（与任何 ≤2 阶函数正交）。"""
    n = 3
    s = (-1.0) ** parity_labels(n)
    x = _inputs(n)
    feats = [np.ones(8)] + [x[:, i] for i in range(n)] + [x[:, i] * x[:, j] for i, j in itertools.combinations(range(n), 2)]
    Phi = np.stack(feats, 1)
    # 奇偶的 ±1 编码与所有 ≤2 阶特征正交
    assert np.allclose(Phi.T @ s, 0.0)
    # ⇒ 对任意系数 a，Σ_x s(x)·q(x) = 0，故 s·q 不可能处处 > 0
    rng = np.random.default_rng(3)
    a = rng.standard_normal((Phi.shape[1], 2000))
    assert not np.any(np.all(s[:, None] * (Phi @ a) > 0, axis=0))
