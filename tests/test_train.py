"""train.py（INTEGRATION D）：DSPSA 封装、1-bit oracle、连续 oracle 的冒烟与接口测试。"""
import numpy as np
import pytest

from ris_nlsim.constants import LAMBDA
from ris_nlsim.system import SystemConfig, make_system
from ris_nlsim.train import _variant, calibrate_dspsa_gain, run_dspsa, run_oracle_bits, run_oracle_continuous


@pytest.fixture(scope="module")
def m3_n2():
    return make_system(SystemConfig(condition="M3", object_mode="full", voxel_d=LAMBDA / 10), "parity", n=2)


def test_variant_auto(m3_n2):
    assert _variant(m3_n2, "auto") == "sign_acc"                     # INTEGRATION T：1-bit 新默认
    assert _variant(m3_n2, "subset") == "subset"
    s0 = m3_n2.derive(condition="M0", passive_levels=4)
    assert _variant(s0, "auto") == "faithful"
    assert _variant(m3_n2, "faithful") == "faithful"


def test_run_dspsa_reduces_clean_loss(m3_n2):
    rng = np.random.default_rng(0)
    res = run_dspsa(m3_n2, 300, rng, eval_every=100, n_noise_eval=3, variant="subset")
    assert res["variant"] == "subset"
    assert res["n_measurements"] == 2 * 300 + 20                   # a="auto" 的 20 次标定测量也计入
    curve = res["curve"]
    assert [c["iter"] for c in curve] == [0, 100, 200, 300]
    assert curve[-1]["loss_clean"] < curve[0]["loss_clean"]
    p = res["params"]
    assert p.shape == (50,) and set(np.unique(p)) <= {0, 1}
    assert res["eval"]["loss_clean"] == pytest.approx(m3_n2.loss(p, noisy=False), rel=1e-12)
    assert res["eval"]["loss_clean"] == pytest.approx(curve[-1]["loss_clean"], rel=1e-12)
    assert len(res["history"]["loss_plus"]) == 300 and res["n_flagged_measurements"] == 0
    assert res["objective"] == "measurable" and all("loss_meas_clean" in c for c in curve)


def test_run_oracle_bits_returns_valid_codebook(m3_n2):
    rng = np.random.default_rng(1)
    p0 = m3_n2.random_params(np.random.default_rng(2))
    res = run_oracle_bits(m3_n2, rng, n_restarts=1, max_sweeps=2, n_noise_eval=2)
    p = res["params"]
    assert p.shape == (50,) and p.dtype == np.int64 and set(np.unique(p)) <= {0, 1}
    assert res["loss"] == pytest.approx(m3_n2.loss(p, noisy=False), rel=1e-12)
    assert res["loss"] <= m3_n2.loss(p0, noisy=False) or res["loss"] < 1.0
    assert res["eval"]["loss_clean"] == pytest.approx(res["loss"], rel=1e-12)
    assert res["n_evals"] > 50


def test_run_oracle_continuous_smoke(m3_n2):
    res = run_oracle_continuous(m3_n2, n_restarts=1, steps=5, lr=0.05, seed=0, n_noise_eval=2)
    ph = res["phases"]
    assert set(ph) == {"phase_rx", "phase_tx"} and ph["phase_rx"].shape == (1, 25)
    # numpy 复评的无噪声损失 = torch 优化记录的最优损失（同一定义）
    assert res["eval"]["loss_clean"] == pytest.approx(res["loss"], rel=1e-8)
    assert "acc_clean" in res["eval"] and res["objective"] == "model"


def test_dspsa_optimises_measurable_objective_only():
    """审查 INT-SEM-3：DSPSA 的损失不含模型内部环路谱半径的光滑自激惩罚（探头读数测不到），oracle 用模型目标。
    |G κ_eff| = +20 dB：每个码本全部样本被标记 → 可测目标恒为 chance + osc_penalty，模型目标再加 100·mean relu(ρ−0.95)²。"""
    s = make_system(SystemConfig(condition="M1", object_mode="full", isolation_db=-5.0, voxel_d=LAMBDA / 10),
                    "parity", n=2)
    base = np.log(2.0) + s.cfg.osc_penalty
    p = s.random_params(np.random.default_rng(3))
    m = s.forward(s.phases_from_params(p), noisy=False)
    assert m.flags.all()
    assert s.loss_from(m, model_penalties=False) == pytest.approx(base, rel=1e-12)
    assert s.loss_from(m) > base + 10.0
    res = run_dspsa(s, 20, np.random.default_rng(4), eval_every=10, n_noise_eval=2, variant="subset")
    assert res["objective"] == "measurable"
    assert np.allclose(res["history"]["loss_plus"], base, rtol=1e-12)
    assert np.allclose(res["history"]["loss_minus"], base, rtol=1e-12)
    for c in res["curve"]:
        assert c["loss_meas_clean"] == pytest.approx(base, rel=1e-12) and c["loss_clean"] > base + 10.0
    assert res["eval"]["loss_meas_clean"] == pytest.approx(base, rel=1e-12)
    ob = run_oracle_bits(s, np.random.default_rng(5), n_restarts=1, max_sweeps=1, n_noise_eval=0)
    assert ob["objective"] == "model" and ob["loss"] == pytest.approx(s.loss(ob["params"], noisy=False), rel=1e-12)
    assert ob["loss"] > base + 10.0


# ============================================================ 审查 S0-2 / S1-0 / S2-4 增补
def test_run_dspsa_pocket_is_best_measured_and_does_not_change_iteration(m3_n2):
    """S1-0：pocket = 全部测量（含 a="auto" 的 20 次标定）中测到的可测目标最小的码本；DSPSA 迭代与 params/eval 不变。"""
    r0 = run_dspsa(m3_n2, 60, np.random.default_rng(7), eval_every=20, n_noise_eval=0, variant="subset")
    r1 = run_dspsa(m3_n2, 60, np.random.default_rng(7), eval_every=20, n_noise_eval=0, pocket=True, variant="subset")
    assert np.array_equal(r0["params"], r1["params"]) and r0["eval"]["loss_clean"] == r1["eval"]["loss_clean"]
    assert np.array_equal(r0["history"]["loss_plus"], r1["history"]["loss_plus"]) and "pocket" not in r0
    pk = r1["pocket"]
    meas = np.concatenate([r1["history"]["loss_plus"], r1["history"]["loss_minus"]])
    assert pk["loss_measured"] <= meas.min() + 1e-15 and 1 <= pk["at_measurement"] <= r1["n_measurements"]
    assert pk["params"].shape == (50,) and pk["eval"]["params"] == [int(v) for v in pk["params"]]
    # 无噪声复评与测量值只差 VNA 级噪声
    assert pk["eval"]["loss_meas_clean"] == pytest.approx(pk["loss_measured"], abs=0.05)
    cur = r1["curve"]
    assert all("pocket_loss_meas_clean" in c for c in cur)
    vals = [c["pocket_loss_meas_clean"] for c in cur]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))          # best-so-far 单调不增（同一无噪声目标）
    assert cur[-1]["pocket_loss_meas_clean"] == pytest.approx(pk["eval"]["loss_meas_clean"], rel=1e-12)


def test_calibrate_dspsa_gain_matches_auto():
    """S2-4：calibrate_dspsa_gain 与 a="auto" 同一规则（同 rng 序列 → 同一 a），可把参考条件的 a 固定传给别的条件。"""
    s = make_system(SystemConfig(condition="M1", object_mode="born", voxel_d=LAMBDA / 10), "parity", n=2)
    cal = calibrate_dspsa_gain(s, np.random.default_rng(11))
    res = run_dspsa(s, 3, np.random.default_rng(11), eval_every=3, n_noise_eval=0, variant="subset")
    assert cal["a"] == pytest.approx(res["a"], rel=1e-12) and cal["n_measurements"] == 20
    assert cal["variant"] == "subset"                                # auto → sign_acc 无需标定，按 subset 扰动标定
    s4 = s.derive(condition="M4")
    r4 = run_dspsa(s4, 3, np.random.default_rng(12), a=cal["a"], eval_every=3, n_noise_eval=0, variant="subset")
    assert r4["a"] == cal["a"] and r4["n_measurements"] == 6
    c2 = calibrate_dspsa_gain(s, np.random.default_rng(1), n_starts=3)
    assert c2["n_measurements"] == 60 and np.isfinite(c2["a"]) and c2["a"] > 0


def test_oracle_objectives_margin_and_accuracy():
    """S0-2：objective="accuracy" 最大化判对数（M0 + Born 3 位奇偶仍不能全对，门 G1），"margin" 最大化最小裕量；
    正对照：1 位有/无检测两者都全对。目标值与 evaluate 一致。"""
    from ris_nlsim.train import oracle_objective

    s3 = make_system(SystemConfig(condition="M0", object_mode="born", voxel_d=LAMBDA / 10), "parity", n=3)
    ra = run_oracle_bits(s3, np.random.default_rng(1), n_restarts=2, n_noise_eval=0, objective="accuracy")
    rm = run_oracle_bits(s3, np.random.default_rng(1), n_restarts=2, n_noise_eval=0, objective="margin")
    for r in (ra, rm):
        assert r["eval"]["all_correct"] is False and r["eval"]["acc_clean"] < 1.0
    assert ra["objective"] == "accuracy" and rm["objective"] == "margin"
    assert -rm["loss"] == pytest.approx(rm["eval"]["margin_db"], rel=1e-12) and rm["eval"]["margin_db"] <= 0.0
    n_ok = ra["eval"]["acc_clean"] * s3.B
    assert np.floor(-ra["loss"] + 0.5) == pytest.approx(n_ok)              # 判对数 + (−½, ½) 的平局项
    s1 = make_system(SystemConfig(condition="M0", object_mode="born", voxel_d=LAMBDA / 10), "parity", n=1)
    for obj in ("accuracy", "margin"):
        r = run_oracle_bits(s1, np.random.default_rng(2), n_restarts=1, n_noise_eval=0, objective=obj)
        assert r["eval"]["all_correct"] is True
    with pytest.raises(ValueError):
        oracle_objective(s1, "hinge")
    # 被标记样本按判错计（|G κ_eff| = +20 dB 的 M1：全部标记 → 判对 0、裕量 −FLAG_MARGIN_DB）
    sf = make_system(SystemConfig(condition="M1", object_mode="full", isolation_db=-5.0, voxel_d=LAMBDA / 10),
                     "parity", n=2)
    x = sf.random_params(np.random.default_rng(3))
    from ris_nlsim.train import FLAG_MARGIN_DB
    assert oracle_objective(sf, "margin")(x) == pytest.approx(FLAG_MARGIN_DB)
    assert oracle_objective(sf, "accuracy")(x) == pytest.approx(0.5)
