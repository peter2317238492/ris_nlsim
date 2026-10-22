"""losses.py：numpy 与 torch 实现逐式一致，并与独立闭式核对（DESIGN §7.3）。"""
import math

import numpy as np
import pytest
import torch

from ris_nlsim import losses


def _data(rng, B=16, K=3):
    y = rng.normal(-40.0, 8.0, size=(B, K))
    return y


def test_binary_matches_closed_form():
    rng = np.random.default_rng(0)
    y = _data(rng, K=2)
    lab = rng.integers(0, 2, size=y.shape[0])
    tau = 0.7
    ref = 0.0
    for (y1, y2), l in zip(y, lab):
        s = (1 if l == 1 else -1) * (y1 - y2)
        ref += tau * math.log1p(math.exp(-s / tau)) if -s / tau < 700 else -s
    ref /= y.shape[0]
    assert losses.binary_loss(y, lab, tau) == pytest.approx(ref, rel=1e-12)


def test_binary_numerically_stable_extremes():
    y = np.array([[1000.0, 0.0], [0.0, 1000.0]])
    lab = np.array([1, 1])
    # 第 1 个样本 s=+1000 → ≈0；第 2 个 s=−1000 → ≈1000·τ/τ... = 1000
    v = losses.binary_loss(y, lab, 1.0)
    assert np.isfinite(v) and v == pytest.approx(500.0, rel=1e-12)
    vt = losses.binary_loss(torch.tensor(y), lab, 1.0)
    assert float(vt) == pytest.approx(500.0, rel=1e-12)


def test_multiclass_matches_closed_form():
    rng = np.random.default_rng(1)
    y = _data(rng, K=3)
    lab = rng.integers(0, 3, size=y.shape[0])
    tau = 2.0
    z = y / tau
    ref = np.mean([np.log(np.sum(np.exp(zi))) - zi[l] for zi, l in zip(z, lab)])
    assert losses.multiclass_loss(y, lab, tau) == pytest.approx(ref, rel=1e-12)


TASKS = [("binary", 2), ("multiclass", 3), ("focus", 2), ("threshold", 1), ("loc3bit", 3)]


def _labels(task, B, rng):
    n_cls = {"multiclass": 3, "loc3bit": 8}.get(task, 2)
    lab = np.arange(B) % n_cls
    rng.shuffle(lab)
    return lab


@pytest.mark.parametrize("task,K", TASKS)
def test_numpy_torch_identical(task, K):
    rng = np.random.default_rng(2)
    for B in (7, 8):            # 奇偶个样本（阈值任务中位数两种情形）
        y = _data(rng, B=B, K=K)
        lab = _labels(task, B, rng)
        for tau in (0.5, 1.0, 3.0):
            v_np = losses.task_loss(y, lab, task, tau)
            yt = torch.tensor(y, requires_grad=True)
            v_t = losses.task_loss(yt, torch.as_tensor(lab), task, tau)
            assert isinstance(v_np, float)
            assert float(v_t.detach()) == pytest.approx(v_np, rel=1e-13, abs=1e-13)
            v_t.backward()
            assert torch.isfinite(yt.grad).all()


@pytest.mark.parametrize("task,K", TASKS)
def test_reduction_none_is_per_sample(task, K):
    """reduction='none' 返回 (B,) 逐样本损失，其均值 = 'mean'；numpy 与 torch 逐样本一致（INTEGRATION C）。"""
    rng = np.random.default_rng(5)
    y = _data(rng, B=8, K=K)
    lab = _labels(task, 8, rng)
    per = losses.task_loss(y, lab, task, 0.8, reduction="none")
    assert isinstance(per, np.ndarray) and per.shape == (8,)
    assert np.mean(per) == pytest.approx(losses.task_loss(y, lab, task, 0.8), rel=1e-14)
    per_t = losses.task_loss(torch.tensor(y), lab, task, 0.8, reduction="none")
    assert tuple(per_t.shape) == (8,)
    assert np.allclose(per_t.numpy(), per, rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError):
        losses.task_loss(y, lab, task, 0.8, reduction="sum")


def test_focus_and_gradients():
    y = torch.tensor([[-10.0, 0.0], [-20.0, 5.0]], requires_grad=True)
    v = losses.focus_loss(y)
    assert float(v.detach()) == pytest.approx(15.0)
    v.backward()
    assert torch.allclose(y.grad, torch.tensor([[-0.5, 0.0], [-0.5, 0.0]], dtype=y.dtype))


def test_binary_gradient_matches_finite_difference():
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.5, size=(16, 2))      # y1−y2 ~ τ 量级，梯度 O(1)
    lab = rng.integers(0, 2, size=y.shape[0])
    yt = torch.tensor(y, requires_grad=True)
    losses.binary_loss(yt, lab, 1.3).backward()
    h = 1e-5
    for i, j in [(0, 0), (3, 1), (10, 0)]:
        yp, ym = y.copy(), y.copy()
        yp[i, j] += h
        ym[i, j] -= h
        fd = (losses.binary_loss(yp, lab, 1.3) - losses.binary_loss(ym, lab, 1.3)) / (2 * h)
        assert float(yt.grad[i, j]) == pytest.approx(fd, rel=1e-7, abs=1e-9)


def test_threshold_uses_median_midpoint():
    y = np.array([[0.0], [2.0], [10.0], [12.0], [14.0]])
    lab = np.array([0, 0, 1, 1, 1])
    t = 0.5 * (1.0 + 12.0)
    s = np.array([-1, -1, 1, 1, 1]) * (y[:, 0] - t)
    ref = np.mean(np.logaddexp(0, -s))
    assert losses.threshold_loss(y, lab, 1.0) == pytest.approx(ref, rel=1e-13)


def test_threshold_is_not_differentiated():
    """自动阈值不回传梯度（INTEGRATION E）：torch 梯度 = 把 t 当常数时的闭式梯度。"""
    y = np.array([[0.0], [2.0], [10.0], [12.0], [14.0]])
    lab = np.array([0, 0, 1, 1, 1])
    sgn = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    t = 0.5 * (1.0 + 12.0)
    s = sgn * (y[:, 0] - t)
    ref = -sgn / (1.0 + np.exp(s)) / 5.0                    # d/dy mean softplus(−s)，t 固定
    yt = torch.tensor(y, requires_grad=True)
    losses.threshold_loss(yt, lab, 1.0).backward()
    assert np.allclose(yt.grad.numpy()[:, 0], ref, rtol=1e-12)
    assert np.allclose(ref, [0.00030024, 0.00219739, -0.00586245, -0.00081403, -0.00011056], atol=1e-8)


def _loc3bit_ref(y, lab, tau, t=None, w=1.0):
    """独立的显式循环：比特 j = (c>>j)&1；t_j 为两类中位数中点（或给定）；ℓ_b = w·Σ_j τ·log(1+exp(−s_bj/τ))。"""
    B, K = y.shape
    bits = np.array([[(c >> j) & 1 for j in range(K)] for c in lab])
    if t is None:
        t = [0.5 * (np.median(y[bits[:, j] == 1, j]) + np.median(y[bits[:, j] == 0, j])) for j in range(K)]
    per = np.zeros(B)
    for b in range(B):
        for j in range(K):
            s = (1 if bits[b, j] else -1) * (y[b, j] - t[j])
            per[b] += w * tau * math.log1p(math.exp(-s / tau))
    return per


def test_loc3bit_matches_explicit_loops():
    rng = np.random.default_rng(6)
    y = rng.normal(-40.0, 2.0, size=(16, 3))
    lab = np.concatenate([np.arange(8), rng.integers(0, 8, size=8)])
    for tau, w in ((1.0, 1.0), (0.6, 2.6)):
        ref = _loc3bit_ref(y, lab, tau, w=w)
        assert losses.loc3bit_loss(y, lab, tau, loss_weight=w) == pytest.approx(ref.mean(), rel=1e-12)
        assert np.allclose(losses.loc3bit_loss(y, lab, tau, reduction="none", loss_weight=w), ref, rtol=1e-12)
    t_fix = np.array([-41.0, -40.0, -39.0])
    ref = _loc3bit_ref(y, lab, 1.0, t=t_fix, w=2.6)
    v = losses.task_loss(y, lab, "loc3bit", 1.0, thresholds_db=t_fix, loss_weight=2.6)
    assert v == pytest.approx(ref.mean(), rel=1e-12)
    assert np.allclose(losses.loc3bit_thresholds(y, lab),
                       [0.5 * (np.median(y[(lab >> j) & 1 == 1, j]) + np.median(y[(lab >> j) & 1 == 0, j]))
                        for j in range(3)])


def test_loc3bit_gradient_treats_thresholds_as_constants():
    rng = np.random.default_rng(7)
    y = rng.normal(-40.0, 1.5, size=(8, 3))
    lab = np.arange(8)
    t = losses.loc3bit_thresholds(y, lab)
    yt = torch.tensor(y, requires_grad=True)
    losses.loc3bit_loss(yt, lab, 0.7, loss_weight=1.3).backward()
    h = 1e-6
    for i, j in [(0, 0), (3, 1), (6, 2), (7, 0)]:
        yp, ym = y.copy(), y.copy()
        yp[i, j] += h
        ym[i, j] -= h
        fd = (losses.loc3bit_loss(yp, lab, 0.7, thresholds_db=t, loss_weight=1.3)
              - losses.loc3bit_loss(ym, lab, 0.7, thresholds_db=t, loss_weight=1.3)) / (2 * h)
        assert float(yt.grad[i, j]) == pytest.approx(fd, rel=1e-6, abs=1e-10)


def test_auto_threshold_needs_both_classes():
    y = np.zeros((4, 3))
    with pytest.raises(ValueError):
        losses.loc3bit_loss(y, np.array([1, 3, 5, 7]), 1.0)          # 比特 0 全为 1
    with pytest.raises(ValueError):
        losses.threshold_loss(np.zeros((3, 1)), np.array([1, 1, 1]), 1.0)
    with pytest.raises(ValueError):
        losses.loc3bit_loss(y, np.arange(4), 1.0, thresholds_db=[0.0, 1.0])   # 长度不符


@pytest.mark.parametrize("task,K", TASKS)
def test_chance_loss_is_loss_on_decision_boundary(task, K):
    """chance_loss = 读数恰在判决边界上（各读数相等 / y = t）时的逐样本损失；focus 无定义返回 None。"""
    lab = np.arange(8) % {"multiclass": 3, "loc3bit": 8}.get(task, 2)
    y = np.full((8, K), -37.5)
    kw = {"thresholds_db": np.full(K, -37.5), "loss_weight": 2.6} if task == "loc3bit" else {}
    ch = losses.chance_loss(task, K, 0.7, **kw)
    if task == "focus":
        assert ch is None
        return
    per = losses.task_loss(y, lab, task, 0.7, reduction="none", **kw)
    assert np.allclose(per, ch, rtol=1e-14)
    if task == "binary":
        assert ch == pytest.approx(0.7 * math.log(2.0))


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        losses.task_loss(np.zeros((2, 2)), np.zeros(2), "nope")
    with pytest.raises(ValueError):
        losses.chance_loss("nope", 2)
    with pytest.raises(TypeError):                                    # loc3bit 专用参数传给二分类
        losses.task_loss(np.zeros((2, 2)), np.array([0, 1]), "binary", loss_weight=2.0)
