"""实验 A（experiments/exp_a_parity.py）：条件标签解析、单 run job、门 G1 自检、端到端小网格（进程内，< 30 s）。"""
import json
import os

import numpy as np
import pytest

from ris_nlsim.experiments import exp_a_parity as A


def test_parse_condition_labels():
    assert A.parse_condition("M0") == {"condition": "M0", "objects_see_relay_tx": True}
    assert A.parse_condition("M4")["condition"] == "M4"
    c = A.parse_condition("M1@M4")
    assert c["condition"] == "M1" and c["backoff_db"] == 10.0
    c = A.parse_condition("M1b@M3")
    assert c["condition"] == "M1b" and c["backoff_db"] == 0.0
    c = A.parse_condition("M1@M2:noTx")
    assert c["backoff_db"] == -10.0 and c["objects_see_relay_tx"] is False
    assert A.parse_condition("M1@-3")["backoff_db"] == -3.0
    with pytest.raises(ValueError):
        A.parse_condition("M3@M4")
    with pytest.raises(ValueError):
        A.parse_condition("M7")
    assert A.gain_pairs(["M0", "M1@M3", "M3", "M4", "M1b@M4", "M1@M3:noTx", "M3:noTx"]) == [
        ("M3", "M1@M3"), ("M4", "M1b@M4"), ("M3:noTx", "M1@M3:noTx")]


def _job(**kw):
    j = {"exp": "A", "preset": "L90", "n": 3, "cond": "M0", "mode": "born", "material": "water", "n_relays": 1,
         "spacing_lam": 0.5, "seed": 0, "dspsa_iters": 20, "eval_every": 10, "n_noise_eval": 2,
         "oracle_restarts": 1, "oracle_sweeps": 2, "cont": False, "cont_restarts": 1, "cont_steps": 3, "n_random": 2}
    j.update(kw)
    return j


def test_run_job_m0_born_is_affine_and_cannot_solve_3bit_parity():
    """门 G1：M0 + Born 3 位奇偶——y1−y2（线性功率）对 x 二次（W≥3 ≈ 0），任何训练器都不能全对。"""
    row = A.run_job(_job(cond="M0", cont=True))
    assert row["sys"]["label"] == "M0" and row["sys"]["object_model"] == "volume_rr"
    assert row["sys"]["lambda_over_voxel_d"] > 20                      # 水自动体素 ≈ λ/28
    for tr in ("dspsa", "oracle", "cont"):
        assert row[tr]["all_correct"] is False
        assert row[tr]["eval"]["acc_clean"] < 1.0
    for src in ("random", "dspsa", "oracle", "cont"):
        w = row["w"][src]
        assert w["y_W3"] < 1e-20 and w["obj_W2"] < 1e-20 and w["pa_W3"] is None
    assert row["dspsa"]["n_measurements"] == 2 * 20          # 默认 sign_acc（INTEGRATION T）：无 a="auto" 标定测量
    assert row["dspsa"]["variant"] == "sign_acc" and row["dspsa_pocket"] is None
    g1 = A.g1_check([row], ["dspsa", "oracle", "cont"])
    assert g1["pass"] and g1["checked"] == 1


def test_run_job_m1_pa_contribution_is_zero_and_m3_records_operating_point():
    r1 = A.run_job(_job(cond="M1@M3:noTx", n=2))
    assert r1["sys"]["objects_see_relay_tx"] is False
    for src in ("random", "dspsa", "oracle"):
        assert abs(r1["w"][src]["pa_W2"]) < 1e-12                       # 线性 PA：W(g) = W(u)
        assert r1["w"][src]["K_relvar"] == pytest.approx(0.0, abs=1e-20)   # 物体不在环路里
    r3 = A.run_job(_job(cond="M4", n=2, mode="born"))
    ev = r3["dspsa"]["eval"]
    assert ev["backoff_db"] == 10.0 and r3["sys"]["backoff_db"] == 10.0
    assert ev["pa_in_rel_p1db_db_mean"] is not None and np.isfinite(ev["pa_in_rel_p1db_db_mean"])
    assert r3["sys"]["p_feed_dbm"] == pytest.approx(
        A.run_job(_job(cond="M1@M4", n=2, dspsa_iters=2, oracle_sweeps=1))["sys"]["p_feed_dbm"], abs=1e-9)


def test_main_end_to_end_tiny_grid(tmp_path):
    out = str(tmp_path / "expA")
    A.main(["--n", "2", "--conditions", "M1@M3", "M3", "--modes", "born", "--seeds", "2", "--dspsa-iters", "10",
            "--oracle-restarts", "1", "--oracle-sweeps", "1", "--cont-restarts", "1", "--cont-steps", "3",
            "--n-random", "1", "--no-g1-controls", "--design", "factorial", "--materials", "water", "--n-relays", "1",
            "--workers", "1", "--out", out])
    for f in ("runs.jsonl", "summary.csv", "summary.md", "config.json", "gain.csv",
              "ladder_water_Nr1_sp0.5.png", "w3_decomposition_water_Nr1_sp0.5.png"):
        assert os.path.exists(os.path.join(out, f)), f
    rows = [json.loads(line) for line in open(os.path.join(out, "runs.jsonl"), encoding="utf-8")]
    assert len(rows) == 4 and not any("error" in r for r in rows)
    md = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    for s in ("预测表", "非线性收益", "W≥3 分解", "实际工作点", "| 2 | M3 | M1@M3 | born |"):
        assert s in md, s
    # 重新汇总可复现同一文件集
    A.main(["--resummarize", out])
    assert os.path.exists(os.path.join(out, "gain.csv"))


def test_cfg_overrides_parse_and_reach_system_config():
    over = A.parse_cfg_overrides(["bg_cancel=null", "isolation_db=-40", "pa_noise=false", "voxel_d=none"])
    assert over == {"bg_cancel": "null", "isolation_db": -40, "pa_noise": False, "voxel_d": None}
    for bad in (["condition=M3"], ["nonexistent=1"], ["isolation_db"]):
        with pytest.raises(ValueError):
            A.parse_cfg_overrides(bad)
    s = A.get_system(_job(n=2, cond="M3", cfg_over={"isolation_db": -40.0}))
    assert s.cfg.isolation_db == -40.0 and A.get_system(_job(n=2, cond="M3")).cfg.isolation_db == -30.0


# ------------------------------------------------------------ 审查 Q-1：1-bit oracle = accuracy 目标 + polish
def test_oracle_is_accuracy_objective_not_softplus_loss():
    """M0 + full、n = 2（种子 0）：默认 softplus 损失 oracle 的 acc_clean = 0.25 低于随机（审查 S0-2），
    实验 A 的 "oracle" 列必须用 accuracy 目标（INTEGRATION G-A / B3），loss-oracle 只作诊断。"""
    row = A.run_job(_job(cond="M0", n=2, mode="full", oracle_restarts=8, oracle_sweeps=30, n_random=1))
    o, ol = row["oracle"], row["oracle_loss"]
    assert o["objective"] == "accuracy+polish" and ol["objective"] == "model"
    assert o["acc_clean"] >= 0.5 - 1e-12                       # 不低于随机
    assert o["acc_clean"] > ol["acc_clean"]                    # loss-oracle 低于随机的伪影不再进入 oracle / best 列
    assert row["best"]["acc_clean_valid"] >= o["acc_clean_valid"]
    assert o["stage1_acc_clean"] is not None and o["acc_clean"] >= o["stage1_acc_clean"] - 1e-12


def test_polish_min_margin_is_lexicographically_monotone():
    s = A.get_system(_job(cond="M1@M3", n=2, mode="born"))
    rng = np.random.default_rng(3)
    for _ in range(2):
        x0 = s.random_params(rng)
        m = s.forward(s.phases_from_params(x0), noisy=False)
        sm0 = s.task.sample_margins_db(s.clean_readings(m), s.labels)
        k0 = (int(np.sum(sm0 > 0)), float(np.min(sm0)))
        p = A.polish_min_margin(s, x0, np.random.default_rng(1), max_sweeps=3)
        assert (p["n_ok"], p["min_margin_db"]) >= k0
        m1 = s.forward(s.phases_from_params(p["params"]), noisy=False)
        sm1 = s.task.sample_margins_db(s.clean_readings(m1), s.labels)
        assert p["min_margin_db"] == pytest.approx(float(np.min(sm1)))
        assert p["n_ok"] == int(np.sum(sm1 > 0))


def test_solved_requires_margin_above_three_sigma():
    s = A.get_system(_job(cond="M1@M3", n=2))
    thr = A.solved_threshold_db(s)
    assert thr == pytest.approx(3 * 0.002)                     # VNA σ = 0.002 dB
    assert not A.is_solved({"all_correct": True, "margin_db_unflagged": 0.001}, thr)   # 审查 Q-1 的 0.001 dB 全对
    assert A.is_solved({"all_correct": True, "margin_db_unflagged": 0.01}, thr)
    assert not A.is_solved({"all_correct": False, "margin_db_unflagged": 1.0}, thr)
    assert not A.is_solved({"all_correct": True, "margin_db_unflagged": None}, thr)


def test_g1_check_counts_margin_oracle_and_positive_margin():
    base = {"n": 3, "cond": "M0", "mode": "born", "seed": 0, "w": {}}
    ok = dict(base, oracle_margin={"all_correct": False, "margin_db_unflagged": -0.07, "n_flagged": 0,
                                   "acc_clean": 0.375})
    assert A.g1_check([ok], ["oracle_margin"])["violations"] == []
    bad = dict(base, oracle_margin={"all_correct": False, "margin_db_unflagged": 0.01, "n_flagged": 0,
                                    "acc_clean": 1.0})
    g = A.g1_check([bad], ["oracle_margin"])
    assert len(g["violations"]) == 1 and not g["pass"]
    assert A.is_g1_cell("M0", "born", 3) and A.is_g1_cell("M1@M3:noTx", "born", 3)
    assert not A.is_g1_cell("M0", "full", 3) and not A.is_g1_cell("M0", "born", 2) and not A.is_g1_cell("M3", "born", 3)


# ------------------------------------------------------------ 审查 Q-2：DSPSA pocket、平台诊断、E 判据与收益扣留
def test_run_job_reports_dspsa_pocket_and_training_diagnostics():
    row = A.run_job(_job(cond="M1@M3", n=2, dspsa_iters=40, dspsa_variant="subset"))      # 旧变体的 pocket 诊断
    d, pk = row["dspsa"], row["dspsa_pocket"]
    assert pk is not None and pk["objective"] == "measurable (pocket)"
    assert pk["at_measurement"] is not None and 1 <= pk["at_measurement"] <= d["n_measurements"]
    for t in (d, pk):
        assert isinstance(t["plateau"], bool) and isinstance(t["reach90"], bool)
        assert t["trained"] == (t["reach90"] and not t["plateau"])
    e = row["e_ref"]
    assert e["L0"] == pytest.approx(d["curve"][0]["loss_meas_clean"])
    assert e["Lstar"] == pytest.approx(min(row["oracle"]["loss_meas_clean"], row["oracle_loss"]["loss_meas_clean"]))
    assert e["chance"] == pytest.approx(np.log(2.0))


def test_e_reach_criterion():
    assert A.e_reach(1.0, 0.55, 0.5) is True                   # (1 − 0.55) = 0.45 ≥ 0.9·0.5
    assert A.e_reach(1.0, 0.56, 0.5) is False
    assert A.e_reach(0.6, 0.7, 0.65) is True                   # oracle 不比初始码本好
    assert A.e_reach(None, 0.5, 0.4) is None and A.e_reach(1.0, 0.5, None) is None


def test_gain_table_withholds_untrained_dspsa_rows():
    """DSPSA / pocket 的收益只在两个格点都训练到位（≥ 80% run）时报告，否则"未训练好"；oracle 列照常（审查 Q-2）。"""
    def cell(cond, trained_mean):
        t = {"n": 3, "cond": cond, "mode": "full", "material": "water", "n_relays": 1, "spacing_lam": 0.5}
        for tr in A.DSPSA_TRAINERS:
            t.update({f"{tr}.trained_mean": trained_mean, f"{tr}.trained_n": 5, f"{tr}.reach90_mean": 1.0,
                      f"{tr}.reach90_n": 5})
        t.update({"oracle.pa_in_rel_p1db_db_mean_mean": -3.0, "oracle.pa_in_rel_p1db_db_mean_n": 5,
                  "oracle.pa_in_delta_rel_p1db_db_max_mean": -25.0, "oracle.pa_in_delta_rel_p1db_db_max_n": 5})
        return t

    g = {"n": 3, "gain_of": "M3", "vs": "M1@M3", "mode": "full", "material": "water", "n_relays": 1, "spacing_lam": 0.5}
    for tr in ("dspsa", "dspsa_pocket", "oracle"):
        g.update({f"{tr}.d_acc_mean": 0.2, f"{tr}.d_acc_n": 5, f"{tr}.d_acc_lo": 0.1, f"{tr}.d_acc_hi": 0.3,
                  f"{tr}.d_solved_mean": 0.4, f"{tr}.d_solved_n": 5, f"{tr}.d_solved_lo": 0.0,
                  f"{tr}.d_solved_hi": 0.8})
    gt = [dict(g)]
    A.annotate_gain_table(gt, [cell("M3", 1.0), cell("M1@M3", 0.4)])
    md = A.gain_table_md(gt, ["dspsa", "dspsa_pocket", "oracle", "best"])
    assert md.count("未训练好") == 2 and "+0.20 [+0.10,+0.30]" in md       # 只有 oracle 列给出数值
    assert gt[0]["drive_oracle_db"] == -3.0 and gt[0]["obj_drive_oracle_db"] == -25.0
    gt = [dict(g)]
    A.annotate_gain_table(gt, [cell("M3", 1.0), cell("M1@M3", 0.8)])
    assert "未训练好" not in A.gain_table_md(gt, ["dspsa", "dspsa_pocket", "oracle"])


# ------------------------------------------------------------ 审查 Q-3 / Q-4：显式回退的 Rapp 条件、功率扫描块、M1b@M3
def test_rapp_condition_with_explicit_backoff_and_power_sweep_block():
    c = A.parse_condition("M3@+20")
    assert c["condition"] == "M3" and c["backoff_db"] == 20.0
    assert A.condition_backoff_db("M4") == 10.0 and A.condition_backoff_db("M1@M4") == 10.0
    assert A.condition_backoff_db("M3@-5") == -5.0 and A.is_rapp("M3@+20") and not A.is_rapp("M1@+20")
    with pytest.raises(ValueError):
        A.parse_condition("M3@M4")
    with pytest.raises(ValueError):
        A.parse_condition("M0@+20")
    # 按回退值配对（不依赖写法）："M3@+10" 与 "M1@M4" 同源功率
    assert A.gain_pairs(["M3@+20", "M1@+20", "M1b@+20", "M3@+10", "M1@M4"]) == [
        ("M3@+20", "M1@+20"), ("M3@+20", "M1b@+20"), ("M3@+10", "M1@M4")]
    args = A.parse_args(["--quick", "--seeds", "1", "--scenario", "baseline"])
    assert "M1b@M3" in args.conditions and args.power_sweep == list(A.POWER_SWEEP_QUICK)    # 审查 Q-4
    assert ("M3", "M1b@M3") in A.gain_pairs(args.conditions)
    jobs = A.build_jobs(args)
    ps = {(j["n"], j["cond"], j["mode"]) for j in jobs if "@+" in j["cond"]}
    assert ps == {(3, f"{c}@+{b:g}", m) for b in A.POWER_SWEEP_QUICK for c in ("M3", "M1") for m in ("intra", "full")}
    # 没有 intra/full 时不加扫描块
    args = A.parse_args(["--n", "3", "--modes", "born", "--seeds", "1", "--power-sweep", "20"])
    assert not any("@+" in j["cond"] for j in A.build_jobs(args))


def test_power_sweep_condition_drives_pa_harder_than_nominal():
    """M3@+20：同一几何下源功率比 M3 高 20 dB，与 M1@+20 同源功率；记录的回退 = +20。"""
    s3 = A.get_system(_job(cond="M3", n=3, mode="intra"))
    s20 = A.get_system(_job(cond="M3@+20", n=3, mode="intra"))
    s1 = A.get_system(_job(cond="M1@+20", n=3, mode="intra"))
    assert s20.backoff_db == 20.0 and A.system_info(s20)["backoff_db"] == 20.0
    assert 10 * np.log10(s20.p_feed_w / s3.p_feed_w) == pytest.approx(20.0, abs=1e-9)
    assert s1.p_feed_w == pytest.approx(s20.p_feed_w, rel=1e-12)
