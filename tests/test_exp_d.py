"""实验 D（exp_d_feedback）冒烟测试：名义环路增益、κ0 相位、类间距定义、三种自激判据、临界点剔除、端到端微型配置。"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ris_nlsim.experiments import common
from ris_nlsim.experiments import exp_d_feedback as D
from ris_nlsim.system import SystemConfig, make_system
from ris_nlsim.train import run_dspsa


def test_nominal_loop_gain_and_kappa_phase():
    assert D.nominal_loop_gain_db(SystemConfig(isolation_db=-25.0)) == pytest.approx(0.0)
    assert D.nominal_loop_gain_db(SystemConfig(isolation_db=-40.0)) == pytest.approx(-15.0)
    assert D.nominal_loop_gain_db(SystemConfig(isolation_db=-15.0)) == pytest.approx(10.0)
    a, b = D.kappa_phase_for_seed(0), D.kappa_phase_for_seed(1)
    assert 0.0 <= a < 360.0 and a != b and a == D.kappa_phase_for_seed(0)


def test_default_grid_avoids_critical_point():
    """审查 R-D2：默认网格按名义环路增益取点，避开 |Gκ_eff| = 1（−25 dB，x = 000 的 ρ_ss 恰为 1）。"""
    isos = D.default_isolations()
    assert isos == pytest.approx([-40.0, -28.0, -26.0, -24.0, -15.0])
    gks = [D.nominal_loop_gain_db(SystemConfig(preset="flat0", isolation_db=v)) for v in isos]
    assert gks == pytest.approx(list(D.DEFAULT_LOOP_GAINS_DB))
    assert not any(D.is_critical(g) for g in gks)
    assert D.is_critical(0.0) and D.regime(0.0) == "critical"
    assert D.regime(-1.0) == "sub" and D.regime(1.0) == "super"
    cfg = D.build_config(D.parse_args(["--quick"]))
    assert cfg["isolations"] == pytest.approx(isos) and -25.0 not in cfg["isolations"]


def test_class_separation_definition():
    class _S:
        labels = np.array([0, 1, 1, 0])
    y = np.array([[0.0, 1.0], [2.0, 0.0], [3.0, 0.0], [0.0, 0.5]])          # Δy = −1, 2, 3, −0.5
    out = D.class_separation(_S(), y, np.array([False, False, True, False]))
    assert out["sep_class_db"] == pytest.approx(2.0 - (-0.75))
    assert out["sep_signed_mean_db"] == pytest.approx((1.0 + 2.0 + 0.5) / 3)
    none = D.class_separation(_S(), y, np.array([False, True, True, False]))
    assert none["sep_class_db"] is None


def test_criterion_flags_are_nested_unions():
    """审查 R-D1：robust = cw | ρ_rob ≥ 1，small_signal = cw | ρ_ss ≥ 1（relay.OSC_CRITERIA 的定义）。"""
    m = SimpleNamespace(flags=np.array([False, False, True, False]),
                        robust_unstable=np.array([True, False, False, False]),
                        ss_unstable=np.array([True, True, False, False]))
    f = D.criterion_flags(m)
    assert f["cw"].tolist() == [False, False, True, False]
    assert f["robust"].tolist() == [True, False, True, False]
    assert f["small_signal"].tolist() == [True, True, True, False]
    runs = D.criterion_runs({"oscillating_run": False, "robust_unstable_run": True, "ss_unstable_run": True,
                             "acc_clean": 0.75})
    assert runs == {"osc_robust_run": True, "osc_ss_run": True, "acc_clean_valid_robust": None}
    assert D.criterion_runs({"oscillating_run": False, "robust_unstable_run": False, "ss_unstable_run": True,
                             "acc_clean": 0.75})["acc_clean_valid_robust"] == 0.75


def test_measurement_counter_matches_train_and_criteria_are_ordered():
    """_CriteriaCounter 统计的 cw 计数 = run_dspsa 的 n_flagged_measurements；robust / ss 计数 ≥ cw（超集）。
    近临界（名义 +1 dB）的 M4：小信号环路 > 1，ss 判据下每次测量都被标记。"""
    cfg = SystemConfig(preset="flat0", object_mode="full", roi_offset=(0.8, 0.0, 0.0), isolation_db=-24.0,
                       kappa0_phase_deg=D.kappa_phase_for_seed(0), condition="M4")
    s = make_system(cfg, "parity", n=3)
    proxy = D._CriteriaCounter(s)
    res = run_dspsa(proxy, 6, np.random.default_rng(0), eval_every=3, n_noise_eval=1)
    c = proxy.counts
    assert c["n"] == res["n_measurements"] and c["cw"] == res["n_flagged_measurements"]
    assert c["cw"] <= c["robust"] <= c["n"] and c["cw"] <= c["small_signal"] == c["n"]
    # 随机码本诊断三种判据同样嵌套
    rd = D.random_codebook_diagnostics(s, np.random.default_rng(1), 3)
    assert rd["rand_osc_frac"] <= rd["rand_osc_frac_robust"] <= rd["rand_osc_frac_ss"] == 1.0


def test_paired_osc_drops_critical_rows_and_reports_all_criteria():
    base = {"dist": 0.8, "seed": 0, "kappa_phase_deg": None}
    rows = []
    for iso, gk in ((-25.0, 0.0), (-24.0, 1.0)):
        rows.append(dict(base, iso=iso, gk_nominal_db=gk, critical=D.is_critical(gk), cond="M1", oscillating_run=True,
                         osc_robust_run=True, osc_ss_run=True, rand_osc_frac=1.0, rand_osc_frac_robust=1.0,
                         rand_osc_frac_ss=1.0, flagged_meas_frac=1.0, flagged_meas_frac_robust=1.0,
                         flagged_meas_frac_ss=1.0))
        rows.append(dict(base, iso=iso, gk_nominal_db=gk, critical=D.is_critical(gk), cond="M4", oscillating_run=False,
                         osc_robust_run=True, osc_ss_run=True, rand_osc_frac=0.25, rand_osc_frac_robust=0.5,
                         rand_osc_frac_ss=1.0, flagged_meas_frac=0.1, flagged_meas_frac_robust=0.4,
                         flagged_meas_frac_ss=1.0))
    pt = D.paired_osc(rows)
    assert len(pt) == 1 and pt[0]["iso"] == -24.0 and pt[0]["regime"] == "super"
    p = pt[0]
    assert (p["d_osc"], p["d_osc_robust"], p["d_osc_ss"]) == (-1.0, 0.0, 0.0)
    assert p["d_rand_osc"] == pytest.approx(-0.75) and p["d_rand_osc_robust"] == pytest.approx(-0.5)
    assert p["d_rand_osc_ss"] == 0.0 and p["d_flagged_meas_frac_robust"] == pytest.approx(-0.6)
    txt = D.osc_verdict_text(pt, {})
    assert "rob" in txt and "super" in txt


def test_tiny_end_to_end(tmp_path):
    args = D.parse_args(["--quick", "--seeds", "1", "--isolations", "-40", "-25", "-15", "--distances", "0.8",
                         "--conditions", "M1", "M4", "--n-iter", "4", "--n-rand", "1", "--scan-phases", "0",
                         "--oracle-restarts", "1"])
    cfg = D.build_config(args)
    cfg["n_noise"] = 1
    cfg["oracle_sweeps"] = 1
    res = D.run(cfg, workers=1, out=str(tmp_path))
    assert not [r for r in res["rows"] if "error" in r]
    rows = [r for r in res["rows"] if r.get("kind") != "scan"]
    scan = [r for r in res["rows"] if r.get("kind") == "scan"]
    assert len(rows) == 6 and len(scan) == 3 * 2                     # 隔离 × 条件 × 扫描相位（1 个）
    for name in ("runs.jsonl", "summary.csv", "summary.md", "config.json", "paired.csv", "phase_scan.csv",
                 "oscillation_vs_loopgain.png", "separation_vs_loopgain.png", "accuracy_valid.png",
                 "kappa_phase_scan.png"):
        assert os.path.exists(os.path.join(tmp_path, name)), name
    for r in scan:
        assert r["kappa_phase_deg"] == 0.0 and r["rand_enh_db_med"] is not None
        assert r["rand_osc_frac"] <= r["rand_osc_frac_robust"] <= r["rand_osc_frac_ss"]
        if r["iso"] == -15.0 and r["cond"] == "M1":
            assert r["rand_osc_frac"] == 1.0 and r["rand_sens_db"] is None
        if r["iso"] == -40.0:
            assert r["rand_osc_frac"] == 0.0 and r["rand_sens_db"] > 0 and r["rand_rel_du"] > 0
    by = {(r["iso"], r["cond"]): r for r in rows}
    lin_hi = by[(-15.0, "M1")]
    # |G κ_eff| = +10 dB：线性中继每个码本都被标记（ρ ≥ 1），三种判据下 DSPSA 的每次测量都含被标记样本
    assert lin_hi["oscillating_run"] and lin_hi["flagged_meas_frac"] == 1.0 and lin_hi["rand_osc_frac"] == 1.0
    assert lin_hi["osc_robust_run"] and lin_hi["osc_ss_run"] and lin_hi["flagged_meas_frac_robust"] == 1.0
    assert lin_hi["acc_clean_valid"] is None and lin_hi["loop_gain_ss_max"] > 1.0
    lo = by[(-40.0, "M1")]
    assert not lo["oscillating_run"] and lo["loop_gain_ss_max"] < 1.0 and lo["acc_clean_valid"] is not None
    for r in rows:
        assert r["kappa_phase_deg"] == pytest.approx(D.kappa_phase_for_seed(0))
        # 单中继：Gershgorin 上界 = ρ_ss，|det(I − G K)| = σ_min(I − G K)；线性中继 ρ_rob = ρ_ss
        assert r["gershgorin_max"] == pytest.approx(r["loop_gain_ss_max"], rel=1e-9)
        assert r["det_min"] == pytest.approx(r["one_minus_gk_min"], rel=1e-9)
        if r["cond"] == "M1":
            assert r["loop_gain_robust_max"] == pytest.approx(r["loop_gain_ss_max"], rel=1e-9)
        assert r["voxel_d"] > 0 and r["p_feed_dbm"] is not None and r["gk_obj_db"] is not None
        # 审查 R-D1：三种判据的记录齐全且嵌套
        for k in ("robust_unstable_run", "n_robust_unstable", "ss_unstable_run", "n_ss_unstable",
                  "loop_gain_robust_max", "osc_criterion"):
            assert k in r, k
        assert r["osc_criterion"] == "cw"
        assert r["osc_robust_run"] == bool(r["oscillating_run"] or r["robust_unstable_run"])
        assert r["osc_ss_run"] == bool(r["oscillating_run"] or r["ss_unstable_run"])
        assert r["flagged_meas_frac"] <= r["flagged_meas_frac_robust"] <= 1.0
        assert r["flagged_meas_frac"] <= r["flagged_meas_frac_ss"] <= 1.0
        assert r["flagged_meas_frac"] == pytest.approx(r["n_flagged_measurements"] / r["n_measurements"])
        # 审查 R-D3：准确率目标 oracle 码本
        assert r["orc_objective"] == "accuracy" and r["orc_acc_clean"] is not None and len(r["orc_params"]) == 50
        # 审查 R-D2：−25 dB 标记为临界
        assert r["critical"] == (r["iso"] == -25.0)
    assert by[(-40.0, "M4")]["pa_in_rel_p1db_db_mean"] is not None
    assert all(p["iso"] != -25.0 for p in res["paired"])             # 临界行不参与配对
    md = open(os.path.join(tmp_path, "summary.md"), encoding="utf-8").read()
    assert "|Gκ| nom" in md and "cw/rob/ss" in md and "‡" in md and "rob 判据" in md
