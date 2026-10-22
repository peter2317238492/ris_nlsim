"""实验 E（exp_e_trainability）冒烟测试：90% 判据（chance 参照与可检验性）、共享增益、1-bit 取反对称、端到端微型配置。"""
import os

import numpy as np
import pytest

from ris_nlsim.experiments import common
from ris_nlsim.experiments import exp_e_trainability as E
from ris_nlsim.train import calibrate_dspsa_gain


def test_iters_to_fraction():
    it = [0, 50, 100, 150]
    assert E.iters_to_fraction(it, [2.0, 1.5, 1.05, 1.0], 2.0, 1.0) == (100, False, 1.0)   # 0.95 ≥ 0.9
    k, cens, gap = E.iters_to_fraction(it, [2.0, 1.9, 1.8, 1.5], 2.0, 1.0)
    assert k is None and cens and gap == 1.0
    assert E.iters_to_fraction(it, [1.0, 1.0, 1.0, 1.0], 1.0, 1.2)[:2] == (0, False)       # 参照点已不差于 oracle
    assert E.iters_to_fraction(it, [2.0, 0.5, 0.5, 0.5], 2.0, 1.0)[0] == 50               # 优于 oracle 也算达到


def test_iters_to_settle_and_condition_labels():
    it = [0, 50, 100, 150]
    L = [2.0, 1.0, 1.9, 1.0]                                    # 50 处首达，100 处又退回 → 150 才"达到并保持"
    assert E.iters_to_fraction(it, L, 2.0, 1.0)[0] == 50
    assert E.iters_to_settle(it, L, 2.0, 1.0) == (150, False)
    assert E.iters_to_settle(it, [2.0, 1.0, 1.0, 1.9], 2.0, 1.0) == (None, True)
    assert E.parse_condition("Rapp@+20") == {"condition": "M4", "backoff_db": 20.0}
    assert E.cond_backoff_db("M2") == -10.0 and E.cond_backoff_db("M1@M3") == 0.0
    assert E.cond_backoff_db("Rapp@+20") == 20.0


def test_chance_referenced_criterion_is_not_fooled_by_bias_removal():
    """审查 R-E2 回归：奇偶的 L0 ≈ 3（y1 − y2 常数偏置），L* ≈ chance。原 L0 参照判据在第一个曲线点就"达到 90%"，
    chance 参照判据判为不可检验；oracle 明显优于 chance 时才计迭代数，且以 chance 为参照。"""
    it = [0, 10, 20, 30]
    chance = float(np.log(2.0))
    L = [2.96, 0.70, 0.697, 0.695]                              # 偏置拉平后停在 chance 平台
    assert E.iters_to_fraction(it, L, L[0], 0.691)[0] == 10     # 旧判据：第一个点就"达到"
    pc = E.progress_criterion(it, L, L[0], 0.691, chance, n_iter=30)
    assert pc["testable"] is False and pc["iters_to_90"] is None and pc["reached_90"] is None
    assert pc["ref_kind"] == "chance" and pc["below_chance"] is False
    # 可检验：L* = 0.4 < chance − 0.05；DSPSA 只降到 0.6 → 删失
    pc = E.progress_criterion(it, [2.9, 0.8, 0.65, 0.6], 2.9, 0.4, chance, n_iter=30)
    assert pc["testable"] and pc["censored"] and pc["iters_to_90_lb"] == 30 and pc["below_chance"]
    # 降到 chance − 0.9·(chance − L*) 以下才算达到
    need = chance - 0.9 * (chance - 0.4)
    pc = E.progress_criterion(it, [2.9, 0.8, need - 1e-9, need - 1e-9], 2.9, 0.4, chance, n_iter=30)
    assert pc["iters_to_90"] == 20 and pc["settled_90"] and pc["iters_to_90_settled"] == 20
    # focus（无 chance）：以 L0 为参照
    pf = E.progress_criterion(it, [20.0, 12.0, 5.0, 4.0], 20.0, 3.0, None, n_iter=30)
    assert pf["ref_kind"] == "L0" and pf["testable"] and pf["iters_to_90"] == 30
    assert E.progress_criterion(it, [20.0, 20.0, 20.0, 20.0], 20.0, 19.5, None, n_iter=30)["testable"] is False
    # 准确率判据
    ac = E.accuracy_criterion(it, [0.33, 0.33, 0.67, 1.0], 1.0, 1 / 3, n_iter=30)
    assert ac["acc_testable"] and ac["acc_iters_to_90"] == 30
    assert not E.accuracy_criterion(it, [0.5] * 4, 0.55, 0.5, n_iter=30)["acc_testable"]


def test_shared_gain_jobs_use_one_value_per_task_seed():
    """审查 R-E1：shared 模式下同一 (任务, 种子) 的所有条件拿到同一个 a；auto 模式为 "auto"。"""
    cfg = E.build_config(E.parse_args(["--quick", "--seeds", "2", "--tasks", "parity3", "focus", "--variant", "subset"]))
    assert cfg["gain"] == "shared" and cfg["gain_ref"] == "M3"
    cal = E.make_cal_jobs(cfg)
    assert sorted((j["task"], j["seed"]) for j in cal) == [("focus", 0), ("focus", 1), ("parity3", 0), ("parity3", 1)]
    gains = {(t, s): 1.0 + 10 * s + (0.5 if t == "focus" else 0.0) for t in ("parity3", "focus") for s in range(2)}
    jobs = E.make_jobs(cfg, gains)
    for j in jobs:
        assert j["a"] == gains[(j["task"], j["seed"])]
    cfg_a = E.build_config(E.parse_args(["--quick", "--gain", "auto", "--variant", "subset"]))
    assert E.make_cal_jobs(cfg_a) == [] and all(j["a"] == "auto" for j in E.make_jobs(cfg_a))


def test_cal_job_matches_calibrate_dspsa_gain_on_reference():
    job = {"kind": "cal", "task": "focus", "seed": 0, "gain_ref": "M3", "n_cal_starts": 1}
    r = E.run_e_cal_job(job)
    ref = calibrate_dspsa_gain(E.make_e_system("focus", "M3"), common.seed_rng(E.EXP, "cal", "focus", 0), n_starts=1,
                               variant="subset")
    assert r["a"] == pytest.approx(ref["a"]) and r["kind"] == "cal"


@pytest.mark.parametrize("cond", ["M4", "M1@M3"])
def test_relay_complement_symmetry_is_exact(cond):
    """全部 50 比特取反：u → −u、C → −C ⇒ z 不变 ⇒ L(x) = L(1−x)（1-bit faithful DSPSA 的 x± 损失相等）。"""
    s = E.make_e_system("parity3", cond)
    assert E.flip_symmetry(s, np.random.default_rng(0), n=3) < 1e-12


def test_tiny_end_to_end(tmp_path):
    args = E.parse_args(["--quick", "--seeds", "1", "--tasks", "parity3", "focus", "--conditions", "M3", "M1@M3",
                         "--n-iter", "20", "--oracle-restarts", "1", "--faithful-conds", "M3", "--variant", "subset"])
    cfg = E.build_config(args)
    cfg.update(oracle_sweeps=1, n_land=2, n_noise=2, n_cal_starts=1)
    res = E.run(cfg, workers=1, out=str(tmp_path))
    rows = res["rows"]
    assert not [r for r in rows + res["cal_rows"] if "error" in r] and len(rows) == 4 and len(res["cal_rows"]) == 2
    for name in ("runs.jsonl", "summary.csv", "summary.md", "config.json", "learning_curves.png",
                 "faithful_vs_subset.png"):
        assert os.path.exists(os.path.join(tmp_path, name)), name
    a_of = {r["task"]: r["a"] for r in res["cal_rows"]}
    by = {(r["task"], r["cond"]): r for r in rows}
    for r in rows:
        assert [c["iter"] for c in r["curve"]] == [0, 10, 20]                       # eval_every = 10
        assert r["a_mode"] == "shared" and r["a"] == pytest.approx(a_of[r["task"]])  # 审查 R-E1：同一个 a
        assert r["L0"] == r["curve"][0]["loss_meas_clean"] and r["oracle_loss_meas"] is not None
        assert r["Lstar"] <= min(v for v in (r["oracle_loss_meas"], r["oracle_acc_obj_loss_meas"]) if v is not None)
        assert r["flip_sym_max_abs_dL"] < 1e-12
        assert r["p_feed_dbm"] is not None and r["pa_in_rel_p1db_db_mean"] is not None
        if r["testable"]:
            assert r["iters_to_90_lb"] in (0, 10, 20) and r["iters_to_90_settled_lb"] >= r["iters_to_90_lb"]
        else:
            assert r["iters_to_90"] is None and r["reached_90"] is None
    p = by[("parity3", "M3")]
    assert p["chance"] == pytest.approx(np.log(2.0)) and p["ref_kind"] == "chance"
    assert p["oracle_acc_obj_acc"] is not None and p["voxel_d"] > 0
    f = by[("focus", "M3")]
    assert f["chance"] is None and f["ref_kind"] == "L0" and f["oracle_acc_obj_loss_meas"] is None
    assert "faithful_curve" in f and "faithful_curve" not in by[("focus", "M1@M3")]
    assert f["faithful_a"] == pytest.approx(f["a"])                                  # faithful 用同一个 a
    assert f["faithful_curve"][0]["loss_meas_clean"] == pytest.approx(f["L0"])     # 同种子 → 同初始码本
    # faithful：x± 互补 ⇒ |L(x+) − L(x−)| 只剩测量噪声（VNA 0.002 dB 级），远小于 subset
    assert f["faithful_dL_pm_med"] < 1e-2 < f["dL_pm_med"]
    md = open(os.path.join(tmp_path, "summary.md"), encoding="utf-8").read()
    assert "shared" in md and "chance" in md and "可检验" in md
