"""读出与噪声测试（DESIGN §4，方案 §4.6）。"""
import numpy as np
import pytest

from ris_nlsim.constants import dbm2w, w2dbm
from ris_nlsim.readout import Detector, complex_awgn


def test_ideal_adds_no_noise():
    det = Detector(kind="ideal", sigma_db=0.5)
    p = dbm2w(np.array([-60.0, -30.0, 0.0, 10.0]))
    rng = np.random.default_rng(0)
    assert np.allclose(det.read(p, rng), [-60.0, -30.0, 0.0, 10.0], atol=1e-12)
    assert np.array_equal(det.read(p, rng), det.read(p, np.random.default_rng(99)))


@pytest.mark.parametrize("sigma", [0.002, 0.3])
def test_vna_noise_std_and_mean(sigma):
    det = Detector(kind="vna", sigma_db=sigma)
    p = np.full(200_000, dbm2w(-23.0))
    y = det.read(p, np.random.default_rng(1))
    err = y - (-23.0)
    assert abs(err.std() / sigma - 1) < 0.01
    assert abs(err.mean()) < 5 * sigma / np.sqrt(err.size)
    # vna 不做量程截断；rng=None → 无噪声
    assert np.isclose(det.read(dbm2w(-90.0)), -90.0)
    assert np.allclose(det.read(p[:3]), -23.0, atol=1e-12)


def test_logdet_clipping_to_range():
    det = Detector(kind="logdet", sigma_db=0.0, range_dbm=(-40.0, -10.0))
    dbm = np.linspace(-80, 20, 101)
    y = det.read(dbm2w(dbm), np.random.default_rng(2))
    assert y.min() == -40.0 and y.max() == -10.0
    inside = (dbm >= -40) & (dbm <= -10)
    assert np.allclose(y[inside], dbm[inside])
    # 先截断再加噪：远超量程的读数 = 边界 + N(0, σ²)
    det_n = Detector(kind="logdet", sigma_db=0.25)
    y_lo = det_n.read(np.full(100_000, dbm2w(-100.0)), np.random.default_rng(3))
    assert abs(y_lo.mean() + 40.0) < 0.01 and abs(y_lo.std() / 0.25 - 1) < 0.02
    # offset 在截断之前加
    det_o = Detector(kind="logdet", sigma_db=0.0, offset_db=15.0)
    assert det_o.read(dbm2w(-50.0)) == pytest.approx(-35.0)
    assert det_o.read(dbm2w(0.0)) == pytest.approx(-10.0)


def test_calibrate_centers_median():
    rng = np.random.default_rng(4)
    p_ref = dbm2w(rng.uniform(-75, -55, 101))
    det = Detector(kind="logdet", sigma_db=0.2, range_dbm=(-40.0, -10.0))
    det.calibrate(p_ref)
    assert np.isclose(np.median(det.read(p_ref)), -25.0)
    assert np.isclose(det.offset_db, -25.0 - np.median(w2dbm(p_ref)))
    # 重复标定不累加
    det.calibrate(p_ref)
    assert np.isclose(np.median(det.read(p_ref)), -25.0)
    # vna / ideal：不变
    for kind in ("vna", "ideal"):
        d = Detector(kind=kind, offset_db=3.0)
        d.calibrate(p_ref)
        assert d.offset_db == 3.0


def test_probe_noise_power_and_off():
    rng = np.random.default_rng(5)
    n = complex_awgn(rng, (400_000,), 2e-9)
    assert abs(np.mean(np.abs(n) ** 2) / 2e-9 - 1) < 0.01
    assert abs(np.mean(n.real ** 2) - np.mean(n.imag ** 2)) < 0.02 * 1e-9
    det = Detector(probe_noise_dbm=-60.0)
    z = det.probe_noise((300_000,), rng)
    assert z.shape == (300_000,) and z.dtype == np.complex128
    assert abs(np.mean(np.abs(z) ** 2) / dbm2w(-60.0) - 1) < 0.01
    assert Detector().probe_noise((3,), rng) is None
    assert det.probe_noise((3,), None) is None
    assert Detector().probe_noise_w() == 0.0


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        Detector(kind="power_meter")
