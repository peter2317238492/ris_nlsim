"""dspsa.py（DESIGN §6.1）：faithful 在 1-bit 上的退化、多电平收敛、subset 变体在 1-bit 上的效果、测量计数。"""
import itertools

import numpy as np
import pytest

from ris_nlsim.dspsa import DSPSAResult, decide, dspsa

P12 = 12
ALLX12 = np.array(list(itertools.product([0, 1], repeat=P12)))
SIGNS12 = (-1.0) ** ALLX12


class Recorder:
    """包装损失函数，记录每次被测量的码本（DSPSA 每迭代按 x+、x− 顺序测量）。"""

    def __init__(self, fn):
        self.fn = fn
        self.xs = []

    def __call__(self, x, rng):
        self.xs.append(np.array(x, copy=True))
        return self.fn(x, rng)


def _sym_loss(h):
    # L(x) = −|Σ (−1)^{x_n} h_n|²，满足 L(x) = L(1−x)
    return lambda x, rng: float(-np.abs(((-1.0) ** x) @ h) ** 2)


@pytest.mark.parametrize("a", ["auto", 0.5])
def test_faithful_1bit_is_degenerate_for_symmetric_loss(a):
    rng = np.random.default_rng(0)
    h = rng.normal(size=16) + 1j * rng.normal(size=16)
    rec = Recorder(_sym_loss(h))
    theta0 = rng.uniform(0, 1, size=16)
    res = dspsa(rec, np.full(16, 2), 300, np.random.default_rng(1), a=a, variant="faithful", theta0=theta0)
    # 每个梯度估计恒为 0：x± 互补 ⇒ L(x+) = L(x−)
    assert np.array_equal(res.history["loss_plus"], res.history["loss_minus"])
    assert np.all(res.history["grad_abs_mean"] == 0.0)
    # θ 从未移动、判决从未翻转
    assert np.array_equal(res.theta, theta0)
    assert res.history["n_flips"].sum() == 0
    if a == "auto":
        assert bool(res.history["a_auto_degenerate"][0])


def test_faithful_1bit_points_always_complementary_and_theta_independent():
    """任意（非对称）损失下，1-bit faithful 测量的两点总是一对互补码本，且 θ 移动也不改变这一点。"""
    rng = np.random.default_rng(2)
    c = rng.normal(size=10)
    rec = Recorder(lambda x, r: float(c @ x + 0.3 * (x[0] * x[1])))
    res = dspsa(rec, np.full(10, 2), 200, np.random.default_rng(3), variant="faithful")
    xs = np.array(rec.xs)
    assert xs.shape == (2 * 200 + 20, 10)
    xp, xm = xs[0::2], xs[1::2]
    assert np.all(xp + xm == 1)
    # θ 确实在动（损失非对称），但被测点仍然只是 {x, 1−x}：与当前判决无关
    assert np.ptp(res.history["grad_abs_mean"]) > 0


def test_faithful_4level_separable_quadratic_converges():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        p = 10
        t = rng.integers(0, 4, size=p)
        res = dspsa(lambda x, r: float(np.sum((x - t) ** 2)), np.full(p, 4), 1500, rng, variant="faithful")
        assert np.array_equal(res.bits, t)
        assert np.all((res.theta >= 0) & (res.theta <= 3))


def test_faithful_4level_noisy_quadratic_converges():
    rng = np.random.default_rng(11)
    p = 8
    t = rng.integers(0, 4, size=p)
    w = rng.uniform(0.5, 2.0, size=p)
    loss = lambda x, r: float(np.sum(w * (x - t) ** 2) + r.normal(0, 0.5))
    res = dspsa(loss, np.full(p, 4), 3000, rng, variant="faithful")
    assert np.array_equal(res.bits, t)


def _toy_problem(seed, h0_scale=2.0):
    """无全局符号对称的 1-bit 玩具：P(x) = |h0 + Σ (−1)^{x_n} h_n|²，|h0| 与 |h_n| 同量级；损失 = −10log10 P。"""
    rng = np.random.default_rng(1000 + seed)
    h = (rng.normal(size=P12) + 1j * rng.normal(size=P12)) / np.sqrt(2)
    h0 = h0_scale * np.exp(1j * rng.uniform(0, 2 * np.pi))
    pmax = float(np.max(np.abs(h0 + SIGNS12 @ h) ** 2))       # 穷举 4096 个码本
    power = lambda x: float(np.abs(h0 + ((-1.0) ** x) @ h) ** 2)
    return power, pmax


def test_subset_1bit_reaches_exhaustive_optimum():
    """subset DSPSA（默认 m = max(2, p//8)）p=12：每个问题 4 次独立运行取最好，应在穷举最优的 3% 以内；
    单次运行中位数也在 3% 以内。失败的运行是陷入 1-flip 局部最优（局部搜索固有现象；离线统计 24 个问题中
    约 1 个四次全陷），故允许 6 个问题中至多 1 个不达标。"""
    singles, n_bad = [], 0
    for seed in range(6):
        power, pmax = _toy_problem(seed)
        loss = lambda x, r: -10.0 * np.log10(power(x))
        runs = []
        for run in range(4):
            res = dspsa(loss, np.full(P12, 2), 2000, np.random.default_rng([seed, run]), variant="subset")
            runs.append(power(res.bits) / pmax)
        singles += runs
        n_bad += max(runs) < 0.97
    assert n_bad <= 1
    assert np.median(singles) >= 0.97


def test_subset_1bit_beats_faithful_on_symmetric_loss():
    rng = np.random.default_rng(5)
    h = rng.normal(size=P12) + 1j * rng.normal(size=P12)
    pw = np.abs(SIGNS12 @ h) ** 2
    loss = lambda x, r: -10.0 * np.log10(np.abs(((-1.0) ** x) @ h) ** 2)
    theta0 = rng.uniform(0, 1, size=P12)
    faithful = dspsa(loss, np.full(P12, 2), 1000, np.random.default_rng(6), variant="faithful", theta0=theta0)
    assert np.array_equal(faithful.theta, theta0)
    best = 0.0
    for run in range(3):
        sub = dspsa(loss, np.full(P12, 2), 1000, np.random.default_rng([7, run]), variant="subset", theta0=theta0)
        best = max(best, np.abs(((-1.0) ** sub.bits) @ h) ** 2 / pw.max())
    assert best >= 0.97


def test_subset_perturbs_only_subset_and_rest_follow_decision():
    p, m = 20, 3
    rec = Recorder(lambda x, r: float(np.sum(x)))
    theta0 = np.random.default_rng(8).uniform(0, 1, size=p)
    dspsa(rec, np.full(p, 2), 50, np.random.default_rng(9), a=1e-6, variant="subset", subset_size=m,
          theta0=theta0)
    xs = np.array(rec.xs)
    base = decide(theta0, np.full(p, 2))   # a 极小：判决不变
    for xp, xm in zip(xs[0::2], xs[1::2]):
        diff = np.nonzero(xp != xm)[0]
        assert diff.size == m                         # 子集内 x± 互补
        rest = np.setdiff1d(np.arange(p), diff)
        assert np.array_equal(xp[rest], base[rest]) and np.array_equal(xm[rest], base[rest])


@pytest.mark.parametrize("variant", ["faithful", "subset"])
def test_measurement_count_and_history(variant):
    calls = {"n": 0}

    def loss(x, r):
        calls["n"] += 1
        return float(np.sum((x - 1.0) ** 2) + r.normal(0, 0.1))

    lv = np.full(7, 3)
    res = dspsa(loss, lv, 123, np.random.default_rng(0), a="auto", variant=variant)
    assert isinstance(res, DSPSAResult)
    assert res.n_measurements == calls["n"] == 2 * 123 + 20
    calls["n"] = 0
    res2 = dspsa(loss, lv, 77, np.random.default_rng(0), a=0.2, variant=variant)
    assert res2.n_measurements == calls["n"] == 2 * 77
    for key in ("loss_plus", "loss_minus", "a_k", "n_flips"):
        assert res2.history[key].shape == (77,)
    # 增益序列 a_k = a/(1+k+A)^α
    k = np.arange(77)
    assert np.allclose(res2.history["a_k"], 0.2 / (1 + k + 400.0) ** 0.501)
    assert res2.bits.dtype.kind == "i" and np.all((res2.bits >= 0) & (res2.bits <= 2))


def test_callback_schedule_and_storage():
    seen = []

    def cb(k, theta):
        seen.append(k)
        return None if k == 50 else {"k": k, "mean": float(theta.mean())}

    res = dspsa(lambda x, r: float(np.sum(x)), np.full(5, 2), 120, np.random.default_rng(0), a=0.1,
                variant="subset", callback=cb, callback_every=50)
    assert seen == [0, 50, 100, 120]
    assert list(res.history["callback_iter"]) == [0, 100, 120]
    assert [d["k"] for d in res.history["callback"]] == [0, 100, 120]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("variant", ["faithful", "subset"])
def test_nonfinite_measurement_skips_update(bad, variant):
    """非有限测量（TR-5）：该次迭代不更新 θ。等价于把那一对测量换成相等值（ĝ = 0）——
    两次运行随机数消耗相同，最终 θ 必须逐位相同；θ 全程有限、判决合法，并计数。"""
    c = np.array([0.7, -1.2, 0.4, 2.0, -0.3, 0.9])
    lv = np.array([2, 2, 2, 4, 4, 3])

    def make(replace):
        calls = {"n": 0}

        def f(x, r):
            calls["n"] += 1
            if calls["n"] in (5, 6):                # 第 3 次迭代（k=2）的 x+、x−
                return replace(calls["n"])
            return float(c @ x)
        return f

    kw = dict(a=0.3, variant=variant, subset_size=6)
    res_bad = dspsa(make(lambda n: bad if n == 5 else 1.0), lv, 40, np.random.default_rng(3), **kw)
    res_tie = dspsa(make(lambda n: 1.0), lv, 40, np.random.default_rng(3), **kw)
    assert np.all(np.isfinite(res_bad.theta))
    assert np.array_equal(res_bad.theta, res_tie.theta)
    assert np.all((res_bad.bits >= 0) & (res_bad.bits < lv))
    assert res_bad.history["nonfinite"].tolist() == [k == 2 for k in range(40)]
    assert int(res_bad.history["n_nonfinite"][0]) == 1 and int(res_tie.history["n_nonfinite"][0]) == 0
    assert res_bad.n_measurements == 80


def test_auto_gain_ignores_nonfinite_calibration_estimates():
    calls = {"n": 0}

    def f(x, r):
        calls["n"] += 1
        return np.nan if calls["n"] == 1 else float(np.sum(x))

    # subset_size=1：ĝ = Δ² = 1 恒定 ⇒ 有限估计的 mean|ĝ| = 1，a = 0.3·(1+A)^α
    res = dspsa(f, np.full(6, 2), 120, np.random.default_rng(0), a="auto", variant="subset", subset_size=1)
    assert res.history["a"][0] == pytest.approx(0.3 * 401.0 ** 0.501, rel=1e-12)
    assert not bool(res.history["a_auto_degenerate"][0])
    assert int(res.history["n_nonfinite"][0]) == 1
    assert np.all(np.isfinite(res.theta)) and np.all(res.bits == 0)      # 最优 = 全 0


def test_projection_margin_is_1e10():
    """INTEGRATION G2：θ 截到 [0, L−1−1e-10]；上界处 π(θ) 仍取 L−2+½，判决为 L−1。"""
    lv = np.array([2, 4])
    res = dspsa(lambda x, r: -float(np.sum(x)), lv, 50, np.random.default_rng(0), a=50.0, variant="faithful",
                theta0=np.array([0.5, 2.5]))
    assert res.theta[1] == pytest.approx(3 - 1e-10, abs=1e-15)
    assert res.bits[1] == 3


def test_mixed_levels_and_bounds():
    lv = np.array([2, 4, 3, 2, 4])
    t = np.array([1, 3, 0, 0, 2])
    res = dspsa(lambda x, r: float(np.sum((x - t) ** 2)), lv, 1500, np.random.default_rng(4), variant="faithful")
    assert np.all(res.bits < lv) and np.all(res.theta <= lv - 1)
    # 1-bit 分量在 faithful 下不可学，只检查多电平分量
    multi = lv > 2
    assert np.array_equal(res.bits[multi], t[multi])


def test_auto_gain_is_the_auto_rule_and_public():
    """审查 S2-4：dspsa(a="auto") 的标定就是 auto_gain（同 rng 序列 → 同一 a、20 次测量）。"""
    from ris_nlsim.dspsa import auto_gain, dspsa

    def loss(x, rng):
        return float(np.sum((x - np.arange(x.size) % 2) ** 2) + 0.01 * rng.standard_normal())

    L = np.full(16, 2)
    r = dspsa(loss, L, 0, np.random.default_rng(3), variant="subset")
    rng = np.random.default_rng(3)
    theta = rng.uniform(0.0, 1.0, size=16)
    cal = auto_gain(loss, L, rng, theta=theta, variant="subset")
    assert cal["a"] == pytest.approx(float(r.history["a"][0]), rel=1e-12)
    assert cal["n_measurements"] == r.n_measurements == 20 and not cal["degenerate"]
    # 对称损失 + 1-bit faithful：退化 → mean|ĝ| := 1
    deg = auto_gain(lambda x, r_: float(np.sum(x == 1) * np.sum(x == 0)), L, np.random.default_rng(0),
                    variant="faithful")
    assert deg["degenerate"] and deg["a"] == pytest.approx(0.3 * 401 ** 0.501)
