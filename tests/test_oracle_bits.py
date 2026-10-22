"""oracle_bits.py（DESIGN §6.2）：在 p=10 的问题上与穷举最优比较。"""
import itertools

import numpy as np
import pytest

from ris_nlsim.oracle_bits import oracle_bits


def _exhaustive(loss, levels):
    best_x, best_f = None, np.inf
    for x in itertools.product(*[range(int(l)) for l in levels]):
        f = loss(np.array(x))
        if f < best_f:
            best_x, best_f = np.array(x), f
    return best_x, best_f


def _beam_problem(seed, p=10):
    # 1-bit 波束：−|h0 + Σ(−1)^{x_n} h_n|²（存在非全局的 1-flip 局部最优）
    rng = np.random.default_rng(seed)
    h = rng.normal(size=p) + 1j * rng.normal(size=p)
    h0 = 1.5 * (rng.normal() + 1j * rng.normal())
    return lambda x: float(-np.abs(h0 + ((-1.0) ** np.asarray(x)) @ h) ** 2)


def _qubo_problem(seed, p=10):
    # 稠密随机伪布尔二次函数 xᵀQx + cᵀx（典型的多局部最优地形）
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(p, p))
    c = rng.normal(size=p)
    return lambda x: float(np.asarray(x) @ Q @ np.asarray(x) + c @ np.asarray(x))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("make", [_beam_problem, _qubo_problem])
def test_finds_exhaustive_optimum_1bit_p10(make, seed):
    loss = make(seed)
    lv = np.full(10, 2)
    _, f_ex = _exhaustive(loss, lv)
    x, f = oracle_bits(loss, lv, np.random.default_rng(seed), n_restarts=8)
    assert x.dtype.kind == "i" and x.shape == (10,)
    assert f == pytest.approx(loss(x), abs=0, rel=0)       # 返回的损失与返回的码本一致
    assert f == pytest.approx(f_ex, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_finds_exhaustive_optimum_with_annealing(seed):
    loss = _qubo_problem(100 + seed)
    lv = np.full(10, 2)
    _, f_ex = _exhaustive(loss, lv)
    x, f = oracle_bits(loss, lv, np.random.default_rng(seed), n_restarts=4, anneal_steps=300)
    assert f == pytest.approx(f_ex, rel=1e-12, abs=1e-12)
    assert f == loss(x)


def test_multilevel_2bit_phases():
    # 2-bit 相位（4 电平，φ = (π/2)·l）的聚焦问题，p=6 → 4096 个码本
    rng = np.random.default_rng(7)
    p = 6
    h = rng.normal(size=p) + 1j * rng.normal(size=p)
    h0 = 0.8 * (rng.normal() + 1j * rng.normal())
    loss = lambda x: float(-np.abs(h0 + np.exp(0.5j * np.pi * np.asarray(x)) @ h) ** 2)
    lv = np.full(p, 4)
    _, f_ex = _exhaustive(loss, lv)
    x, f = oracle_bits(loss, lv, np.random.default_rng(0), n_restarts=8)
    assert np.all((x >= 0) & (x < 4))
    assert f == pytest.approx(f_ex, rel=1e-12)


def test_nan_first_evaluation_does_not_poison_restarts():
    """TR-5 回归（审查 p12）：首次评估为 nan 时，旧实现 best_f = nan，之后所有有限结果被丢弃。"""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return float("nan") if calls["n"] == 1 else float(np.sum(x))

    x, v = oracle_bits(f, np.full(6, 2), np.random.default_rng(0), n_restarts=4)
    assert v == 0.0 and np.array_equal(x, np.zeros(6))


@pytest.mark.parametrize("anneal_steps", [0, 200])
def test_nonfinite_region_treated_as_infinite(anneal_steps):
    """损失在部分码本上为 nan/inf（如数值奇异样本）：结果 = 有限区域上的穷举最优，且永不落在非有限区域。"""
    base = _qubo_problem(7)

    def loss(x):
        x = np.asarray(x)
        if x[0] == 1 and x[1] == 1:
            return float("nan")
        if x[2] == 1 and x[3] == 0:
            return float("inf")
        return base(x)

    lv = np.full(10, 2)
    _, f_ex = _exhaustive(lambda x: loss(x) if np.isfinite(loss(x)) else np.inf, lv)
    x, f = oracle_bits(loss, lv, np.random.default_rng(1), n_restarts=8, anneal_steps=anneal_steps)
    assert np.isfinite(f) and f == loss(x)
    assert f == pytest.approx(f_ex, rel=1e-12, abs=1e-12)


def test_all_nonfinite_returns_inf():
    x, f = oracle_bits(lambda x: float("nan"), np.full(5, 3), np.random.default_rng(2), n_restarts=2,
                       anneal_steps=20)
    assert f == np.inf and x.shape == (5,) and np.all((x >= 0) & (x < 3))


def test_single_restart_reaches_coordinate_local_optimum():
    loss = _qubo_problem(42)
    lv = np.full(10, 2)
    x, f = oracle_bits(loss, lv, np.random.default_rng(3), n_restarts=1)
    for i in range(10):                      # 任一单分量改动都不应更好
        y = x.copy()
        y[i] = 1 - y[i]
        assert loss(y) >= f
