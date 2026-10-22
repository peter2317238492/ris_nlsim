"""INTEGRATION S：SystemConfig 的馈源 / 探头几何覆盖、命名场景 SCENARIOS / apply_scenario，以及实验 A / B / C / D / E 的 --scenario、
robust 判据、Rapp@ 条件与 cont-plus 连续 oracle。"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ris_nlsim.constants import DEFAULTS, LAMBDA
from ris_nlsim.geometry import point_from
from ris_nlsim.system import (SCENARIOS, SystemConfig, _scene_key, apply_scenario, build_scene, make_system,
                              scene_geometry)


def test_feed_and_probe_overrides_reach_geometry_and_scene():
    p = DEFAULTS["presets"]["L90"]
    base = scene_geometry(SystemConfig(), 3)
    cfg = SystemConfig(feed_az_deg=-45.0, feed_dist=1.2, probe_az_deg=[45, 75, 60], probe_dist=0.8)
    g = scene_geometry(cfg, 3)
    assert np.allclose(g["feed_pos"], point_from(g["rx_c"], -45.0, 0.0, 1.2))
    assert g["probe_az_deg"] == [45.0, 75.0, 60.0]
    for i, az in enumerate((45.0, 75.0, 60.0)):
        assert np.allclose(g["probe_pos"][i], point_from(g["tx_c"], az, 0.0, 0.8))
    assert np.allclose(base["feed_pos"], point_from(base["rx_c"], p["feed_az_deg"], 0.0, p["feed_dist"]))
    assert base["probe_az_deg"] == list(p["probe_az_deg"])                  # None = 预设值，预设本身不被修改
    assert cfg.probe_az_deg == (45.0, 75.0, 60.0) and json.dumps(cfg.to_dict())    # JSON 友好
    scene, info = build_scene(cfg, 2)
    assert np.allclose(info["probe_pos"], g["probe_pos"][:2]) and np.allclose(info["feed_pos"], g["feed_pos"])
    assert _scene_key(cfg) != _scene_key(SystemConfig())                    # derive 不复用旧几何的样本响应
    for bad in (dict(feed_dist=0.0), dict(probe_dist=-1.0), dict(probe_az_deg=[])):
        with pytest.raises(ValueError):
            SystemConfig(**bad)
    with pytest.raises(ValueError):
        scene_geometry(SystemConfig(probe_az_deg=[45.0]), 2)              # 覆盖列表也限制探头数


def test_probe_override_changes_readings_and_derive_rebuilds():
    s = make_system(SystemConfig(condition="M1", object_mode="born", voxel_d=LAMBDA / 6), "parity", n=1)
    s2 = s.derive(probe_az_deg=(45.0, 75.0, 60.0))
    x = s.random_params(np.random.default_rng(0))
    y1 = s.forward(s.phases_from_params(x), noisy=False).readings_dbm
    y2 = s2.forward(s2.phases_from_params(x), noisy=False).readings_dbm
    assert not np.allclose(y1, y2)
    assert np.allclose(s2.info["probe_pos"][0], point_from(s2.info["tx_c"], 45.0, 0.0, DEFAULTS["presets"]["L90"]["probe_dist"]))


def test_named_scenarios_and_apply_scenario():
    assert set(SCENARIOS) == {"baseline", "informative", "informative_probe"}
    c = apply_scenario(SystemConfig(condition="M1", object_mode="full"), "informative")
    assert c.roi_offset == (0.35, 0.0, 0.0) and c.block_direct and c.condition == "M1" and c.object_mode == "full"
    cp = apply_scenario(c, "informative_probe")
    assert cp.probe_az_deg == (45.0, 75.0, 60.0) and cp.block_direct
    b = apply_scenario(cp, "baseline")                                       # baseline 复位到预设默认几何
    assert b.roi_offset is None and not b.block_direct and b.probe_az_deg is None
    d = apply_scenario({"condition": "M3"}, "informative")
    assert d["condition"] == "M3" and d["block_direct"] is True and SystemConfig(**d).roi_offset == (0.35, 0.0, 0.0)
    with pytest.raises(ValueError):
        apply_scenario({}, "nope")


def test_exp_a_quick_defaults_to_informative_with_required_conditions():
    from ris_nlsim.experiments import exp_a_parity as A

    a = A.parse_args(["--quick"])
    assert a.scenario == "informative" and a.modes == ["full"] and a.cont_plus and a.oracle_cont
    assert {"M0", "M1", "M1b@+25", "Rapp@+20", "Rapp@+25", "Rapp@+30", "M3", "M4"} <= set(a.conditions)
    assert a.cfg_over == {"roi_offset": (0.35, 0.0, 0.0), "block_direct": True} and a.power_sweep == []
    jobs = A.build_jobs(a)
    assert {(j["cond"], j["mode"]) for j in jobs if j["mode"] == "born"} == {("M0", "born"), ("M1@M3:noTx", "born")}
    assert all(j["scenario"] == "informative" and j["cont_plus"] for j in jobs)
    c = A.parse_condition("Rapp@+25")
    assert c["condition"] == "M4" and c["backoff_db"] == 25.0 and A.is_rapp("Rapp@+25")
    assert A.condition_backoff_db("Rapp@-3") == -3.0
    with pytest.raises(ValueError):
        A.parse_condition("Rapp@M4")
    # 非 baseline 场景：没有同回退 M1 的 Rapp 条件与 M1 配对；baseline 保持原配对规则
    pairs = A.gain_pairs(a.conditions, any_power_m1=True)
    assert ("Rapp@+20", "M1") in pairs and ("Rapp@+25", "M1b@+25") in pairs and ("Rapp@+20", "M1b@+25") not in pairs
    assert ("Rapp@+20", "M1") not in A.gain_pairs(a.conditions)
    b = A.parse_args(["--quick", "--scenario", "baseline"])
    assert b.cfg_over == {} and b.modes == ["born", "intra", "full"] and not b.cont_plus
    assert A.parse_args([]).scenario == "baseline"


def test_exp_a_robust_criterion_and_gain_table_headline():
    from ris_nlsim.experiments import exp_a_parity as A

    assert A.is_robust({"all_correct": True, "margin_db_unflagged": 0.5})
    assert not A.is_robust({"all_correct": True, "margin_db_unflagged": 0.49})
    assert not A.is_robust({"all_correct": False, "margin_db_unflagged": 3.0})
    g = {"n": 3, "gain_of": "Rapp@+25", "vs": "M1", "mode": "full", "material": "water", "n_relays": 1,
         "spacing_lam": 0.5}
    for f, v in (("d_robust", 0.6), ("d_solved", 0.8), ("d_acc", 0.1)):
        g.update({f"cont.{f}_mean": v, f"cont.{f}_n": 5, f"cont.{f}_lo": v - 0.1, f"cont.{f}_hi": v + 0.1})
    md = A.gain_table_md([g], ["cont"])
    assert "Δrobust 连续 oracle" in md.splitlines()[0] and "+0.60 [+0.50,+0.70]" in md


def test_exp_a_run_job_cont_plus_tiny():
    from ris_nlsim.experiments import exp_a_parity as A

    job = {"exp": "A", "preset": "L90", "n": 2, "cond": "Rapp@+25", "mode": "born", "material": "water",
           "n_relays": 1, "spacing_lam": 0.5, "seed": 0, "dspsa_iters": 6, "eval_every": 3, "n_noise_eval": 2,
           "oracle_restarts": 1, "oracle_sweeps": 1, "polish_sweeps": 1, "cont": True, "cont_restarts": 1,
           "cont_steps": 3, "cont_plus": True, "cont_init_restarts": 1, "cont_refine_steps": 2, "n_random": 1,
           "scenario": "informative", "cfg_over": A.scenario_overrides("informative")}
    row = A.run_job(job)
    s = A.get_system(job)
    assert s.cfg.block_direct and s.cfg.roi_offset == (0.35, 0.0, 0.0) and row["scenario"] == "informative"
    c = row["cont"]
    assert c["cont_source"] in ("cont", "cinit", "crefine") and set(c["cont_candidates"]) == {"cont", "cinit", "crefine"}
    for tr in ("dspsa", "oracle", "cont"):
        assert isinstance(row[tr]["robust"], bool) and "robust_valid" in row[tr]
    assert row["dspsa"]["train_objective"] == "accuracy_first" and isinstance(row["dspsa"]["trained"], bool)
    assert A.margin_key(s, s.phases_from_params(row["oracle"]["eval"]["params"]), None)[0] == round(
        row["oracle"]["acc_clean"] * s.B)


def test_exp_b_and_e_scenario_flags():
    from ris_nlsim.experiments import exp_b_snr as B
    from ris_nlsim.experiments import exp_e_trainability as E

    a = B.parse_args(["--quick"])
    assert a.scenario == "informative" and a.modes == ["full"] and a.cfg_over["block_direct"] and a.cont_plus
    assert {"M0", "M1", "M1b@+25", "Rapp@+25"} <= set(a.conditions)
    assert ("Rapp@+25", "M1") in B.rq3_pairs(a.conditions, any_power_m1=True)
    assert all(j["scenario"] == "informative" for j in B.build_jobs(a))
    assert B.rq3_pairs(B.ALL_CONDITIONS) == list(B.RQ3_PAIRS)
    b = B.parse_args(["--quick", "--scenario", "baseline"])
    assert b.cfg_over == {} and b.modes == ["born"] and b.conditions == list(B.ALL_CONDITIONS)
    cfg = E.build_config(E.parse_args(["--quick", "--scenario", "informative"]))
    assert cfg["scenario"] == "informative" and E.make_cal_jobs(cfg) == []          # sign_acc 无需增益标定
    assert all(j["scenario"] == "informative" and j["variant"] == "auto" for j in E.make_jobs(cfg))
    s = E.make_e_system("focus", "M3", scenario="informative")
    assert s.cfg.block_direct and s.cfg.roi_offset == (0.35, 0.0, 0.0)
    r = E.run_e_job({"task": "focus", "cond": "M3", "seed": 0, "n_iter": 6, "oracle_restarts": 1, "oracle_sweeps": 1,
                     "faithful_conds": [], "n_land": 2, "n_noise": 1, "eval_every": 3, "scenario": "informative"})
    assert r["a_mode"] == "sign" and r["variant"] == "sign_acc" and r["n_measurements"] == 12
    assert r["meas_offset"] == 0 and [c["iter"] for c in r["curve"]] == [0, 3, 6]


def test_exp_c_scenario_flag_reuses_apply_scenario(tmp_path):
    from ris_nlsim.experiments import exp_c_invariance as C
    from ris_nlsim.system import roi_center

    for argv in (["--quick"], []):                                           # 默认 baseline（quick / full 都不改）
        cfg = C.build_config(C.parse_args(argv))
        assert cfg["scenario"] == "baseline" and all(j["scenario"] == "baseline" for j in C.make_jobs(cfg))
    job = {"task": "letter", "cond": "Rapp@+20", "scenario": "baseline"}
    assert C.make_c_cfg(job) == SystemConfig(preset="L90", object_mode="full", condition="M4", backoff_db=20.0)
    ci = C.make_c_cfg(dict(job, scenario="informative"))
    assert ci == apply_scenario(C.make_c_cfg(job), "informative")
    assert ci.block_direct and ci.roi_offset == (0.35, 0.0, 0.0) and ci.condition == "M4" and ci.backoff_db == 20.0
    assert C.make_c_cfg(dict(job, scenario="informative_probe")).probe_az_deg == (45.0, 75.0, 60.0)
    assert roi_center(C.make_c_cfg(job))[0] - roi_center(ci)[0] == pytest.approx(0.15)
    cfg = C.build_config(C.parse_args(["--quick", "--scenario", "informative", "--seeds", "1", "--tasks", "letter",
                                       "--conditions", "M3", "--jitters", "none", "--trainers", "oracle",
                                       "--oracle-restarts", "1"]))
    assert cfg["scenario"] == "informative" and all(j["scenario"] == "informative" for j in C.make_jobs(cfg))
    cfg.update(oracle_sweeps=1, n_noise=1)
    res = C.run(cfg, workers=1, out=str(tmp_path))
    assert not [r for r in res["rows"] if "error" in r]
    assert [r["scenario"] for r in res["rows"]] == ["informative"]
    assert "场景 informative" in res["markdown"] and "block_direct" in res["markdown"]
    with pytest.raises(ValueError):
        C.build_config(SimpleNamespace(**{**vars(C.parse_args(["--quick"])), "scenario": "nope"}))


def test_exp_d_scenario_keeps_flat0_geometry_and_only_switches_direct_path():
    from ris_nlsim.experiments import exp_d_feedback as D
    from ris_nlsim.scene import feed_probe_los_factor

    assert D.scenario_fields("baseline") == {"block_direct": False, "skip_gain_db": 0.0}
    assert D.scenario_fields("informative") == {"block_direct": True, "skip_gain_db": 0.0}
    assert D.scenario_fields("informative_probe") == D.scenario_fields("informative")   # 探头方位是 L90 专用，不沿用
    cfg = D.build_config(D.parse_args(["--quick"]))
    assert cfg["scenario"] == "baseline" and cfg["scenario_fields"]["block_direct"] is False
    cfg = D.build_config(D.parse_args(["--quick", "--scenario", "informative"]))
    assert cfg["scenario"] == "informative" and cfg["scenario_fields"]["block_direct"] is True
    assert all(j["scenario"] == "informative" for j in D.make_jobs(cfg))
    job = {"dist": 0.25, "iso": -40.0, "cond": "M3", "scenario": "baseline"}
    base = D._make_cfg(job, 30.0)
    assert base == SystemConfig(preset="flat0", object_mode="full", roi_offset=(0.25, 0.0, 0.0), isolation_db=-40.0,
                                kappa0_phase_deg=30.0, osc_criterion="cw", condition="M3")   # baseline = 改动前
    info = D._make_cfg(dict(job, scenario="informative_probe"), 30.0)
    assert info.preset == "flat0" and info.roi_offset == (0.25, 0.0, 0.0) and info.block_direct
    assert info.probe_az_deg is None and info.isolation_db == -40.0
    # 实测：flat0 下视距项被遮挡、读数改变；视距项不进环路 → 环路诊断不变
    sb, si = make_system(base, "parity", n=3), make_system(info, "parity", n=3)
    assert feed_probe_los_factor(sb.scene) == 1.0 and feed_probe_los_factor(si.scene) == 0.0
    x = sb.random_params(np.random.default_rng(0))
    mb, mi = (s.forward(s.phases_from_params(x), noisy=False) for s in (sb, si))
    assert not np.allclose(sb.clean_readings(mb), si.clean_readings(mi))
    rb, ri = (D.random_codebook_diagnostics(s, np.random.default_rng(1), 2) for s in (sb, si))
    for k in ("rand_loop_gain_ss_med", "rand_loop_gain_rob_med", "rand_osc_frac", "rand_rel_du"):
        assert rb[k] == pytest.approx(ri[k]), k
    assert rb["rand_sens_db"] != pytest.approx(ri["rand_sens_db"])
    row = D.run_d_scan_job({"kind": "scan", "dist": 0.8, "iso": -40.0, "cond": "M1", "kappa_phase_deg": 0.0,
                            "n_rand": 1, "object_mode": "full", "scenario": "informative"})
    assert row["scenario"] == "informative"
    md = D.summary_markdown([], [], dict(cfg, loop_gains_db=[]), [])
    assert "场景 informative" in md and "block_direct" in md


def test_exp_a_accepts_metal_for_informative_full_grid():
    from ris_nlsim.experiments import exp_a_parity as A

    a = A.parse_args(["--scenario", "informative", "--n", "2", "3", "--modes", "full", "intra", "--materials", "water",
                      "metal", "--conditions", *A.INFORMATIVE_QUICK_CONDITIONS, "--design", "factorial",
                      "--n-relays", "1", "--no-power-sweep", "--seeds", "1"])
    assert a.materials == ["water", "metal"] and a.cont_plus and a.oracle_loss and a.dspsa_iters == 4000
    jobs = A.build_jobs(a)
    main = {(j["n"], j["cond"], j["mode"], j["material"]) for j in jobs if j["mode"] != "born"}
    assert len(main) == 2 * len(A.INFORMATIVE_QUICK_CONDITIONS) * 2 * 2
    assert {j["material"] for j in jobs if j["mode"] == "born"} == {"water"}          # 门 G1 对照只用水
    s = A.get_system(next(j for j in jobs if j["material"] == "metal" and j["mode"] == "full"))
    assert s.task.material == "metal" and s.object_model == "scalar_rr" and s.cfg.block_direct


def test_relay_levels_multibit_phase_mapping():
    """relay_levels：多比特中继移相器。电平 l → 相位 2πl/L；聚焦码本在 L=4 下与 1-bit 的接收相位一致；
    同一组相位用 1-bit 与 4 电平表示时前向结果相同。"""
    import numpy as np
    from ris_nlsim.system import SystemConfig, make_system

    s2 = make_system(SystemConfig(condition="M3", object_mode="born"), "parity", n=2)
    s4 = make_system(SystemConfig(condition="M3", object_mode="born", relay_levels=4), "parity", n=2)
    assert np.all(s4.levels == 4) and np.all(s2.levels == 2)
    assert s4.p_feed_w == s2.p_feed_w                    # 标定与电平数无关
    rng = np.random.default_rng(3)
    b = rng.integers(0, 2, s2.n_params)
    ph2 = s2.phases_from_params(b)
    ph4 = s4.phases_from_params(2 * b)                  # 1-bit 的 π ↔ 4 电平的 2
    for k in ph2:
        assert np.allclose(ph2[k], ph4[k])
    y2 = s2.forward(ph2, noisy=False).p_w
    y4 = s4.forward(ph4, noisy=False).p_w
    assert np.allclose(y2, y4)
    f4 = s4.focus_params()
    assert f4.min() >= 0 and f4.max() < 4
    rx2 = s2.focus_params().reshape(1, 2, -1)[0, 0]
    rx4 = f4.reshape(1, 2, -1)[0, 0]
    assert np.array_equal(rx4, 2 * rx2)
