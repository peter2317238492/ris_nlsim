"""system.py（DESIGN §7，INTEGRATION A/B/B2/C/G2）：场景构建、源功率标定、条件语义、torch/numpy 一致、
DWBA 仿射性（门 G1 的仿真侧）、自激处理、噪声、Bussgang、各任务前向。"""
import dataclasses
import time

import numpy as np
import pytest
import torch

from ris_nlsim import losses, solver
from ris_nlsim.constants import DEFAULTS, K0, LAMBDA, w2dbm
from ris_nlsim.materials import alpha_passive
from ris_nlsim.metrics import walsh_weight_ge
from ris_nlsim.pa import LinearPA, RappPA
from ris_nlsim.relay import RelayParams, relay_linear, weights
from ris_nlsim.system import (ComplexLinearPA, PhysicalSystem, SystemConfig, _best_codebook,
                              build_scene, compress_reduced, make_system, make_task, quantized_conjugate,
                              reference_feed_power)
from ris_nlsim.tasks import FocusTask, ParityTask

VOX = LAMBDA / 10          # 水球 λ/10（19 体素/球）：测试用小规模；物理默认自动体素见 slow 测试
N_TX = 25


def _sys(cond="M3", mode="full", n=3, **kw):
    return make_system(SystemConfig(condition=cond, object_mode=mode, voxel_d=VOX, **kw), "parity", n=n)


@pytest.fixture(scope="module")
def m3_full():
    return _sys("M3", "full")


@pytest.fixture(scope="module")
def m3_born():
    return _sys("M3", "born")


@pytest.fixture(scope="module")
def m0_born():
    return _sys("M0", "born")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _az(deg):
    return np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg)), 0.0])


# ============================================================ A. 场景构建
def test_build_scene_L90_geometry():
    cfg = SystemConfig(preset="L90", condition="M3")
    sc, info = build_scene(cfg, 2)
    rel = info["relays"][0]
    assert info["fold_deg"] == 90.0
    rx_c, tx_c = rel.rx_center, rel.tx_center
    assert np.allclose(info["rx_c"], rx_c) and np.allclose(info["tx_c"], tx_c)
    assert np.allclose(rel.rx_normal, [1, 0, 0]) and np.allclose(rel.tx_normal, [0, 1, 0], atol=1e-15)
    # 馈源：接收板中心 az −30°、1.8 m，喇叭瞄准接收板中心
    assert np.allclose(sc.feeds.pos[0], rx_c + 1.8 * _az(-30.0))
    assert np.allclose(sc.feeds.normal[0], _unit(rx_c - sc.feeds.pos[0]))
    assert sc.feeds.pattern.q == DEFAULTS["horn_q"]
    # 探头：发射板中心 az 75° / 105°、1.0 m，法向指向发射板中心
    for i, a in enumerate((75.0, 105.0)):
        assert np.allclose(sc.probes.pos[i], tx_c + 1.0 * _az(a))
        assert np.allclose(sc.probes.normal[i], _unit(tx_c - sc.probes.pos[i]))
    # ROI = 接收板中心 + (0.5, 0, 0)
    assert np.allclose(info["roi_center"], rx_c + [0.5, 0, 0])
    # 节点数：中继 25 rx + 25 tx，无无源面；S = 26，N_RCV = 27
    assert (sc.n_rx, sc.n_tx, sc.n_passive, sc.n_src, sc.n_rcv) == (25, 25, 0, 26, 27)
    assert np.allclose(sc.rx[0].pos, rel.rx_pos) and np.allclose(sc.tx[0].pos, rel.tx_pos)
    assert not sc.include_direct_tx_rx and not sc.block_direct_feed_probe
    # 3 探头时第三个在 90°
    sc3, _ = build_scene(cfg, 3)
    assert np.allclose(sc3.probes.pos[2], tx_c + _az(90.0))
    with pytest.raises(ValueError):
        build_scene(cfg, 4)


def test_build_scene_M0_passive_nodes():
    cfg = SystemConfig(condition="M0")
    sc, info = build_scene(cfg, 2)
    rel = info["relays"][0]
    assert (sc.n_rx, sc.n_tx, sc.n_passive, sc.n_src) == (0, 0, 50, 1)
    assert np.allclose(sc.passive.pos[:25], rel.rx_pos) and np.allclose(sc.passive.pos[25:], rel.tx_pos)
    assert np.allclose(sc.passive.normal[:25], [1, 0, 0]) and np.allclose(sc.passive.normal[25:], [0, 1, 0])
    assert np.allclose(sc.passive.alpha, alpha_passive(1.0, K0, LAMBDA))
    assert sc.passive.pattern.q == DEFAULTS["elem_q"] and sc.passive.pattern.d0 == DEFAULTS["elem_d0"]
    # 同一 cfg 强制中继版本（标定用）
    sc_r, _ = build_scene(cfg, 2, relay=True)
    assert (sc_r.n_rx, sc_r.n_passive) == (25, 0)


def test_build_scene_two_relays():
    for cond, (nrx, npas) in (("M3", (50, 0)), ("M0", (0, 100))):
        sc, info = build_scene(SystemConfig(condition=cond, n_relays=2), 2)
        rels = info["relays"]
        assert len(rels) == 2 and (sc.n_rx, sc.n_passive) == (nrx, npas)
        assert np.isclose(rels[1].center[2] - rels[0].center[2], 0.30)
        rx_c = np.mean([r.rx_center for r in rels], axis=0)
        assert np.allclose(info["rx_c"], rx_c) and abs(rx_c[2]) < 1e-12
        assert np.allclose(sc.feeds.pos[0], rx_c + 1.8 * _az(-30.0))
        assert np.allclose(info["roi_center"], rx_c + [0.5, 0, 0])
        if cond == "M0":    # 参数顺序：中继 0 rx、tx，中继 1 rx、tx
            assert np.allclose(sc.passive.pos[50:75], rels[1].rx_pos)
            assert np.allclose(sc.passive.pos[75:], rels[1].tx_pos)
        else:
            assert [len(s) for s in sc.rx] == [25, 25] and [len(s) for s in sc.tx] == [25, 25]


def test_build_scene_flat0_and_overrides():
    cfg = SystemConfig(preset="flat0", condition="M3", block_direct=True, skip_gain_db=-3.0)
    sc, info = build_scene(cfg, 3)
    rel = info["relays"][0]
    assert info["fold_deg"] == 0.0
    assert np.allclose(rel.tx_normal, [1, 0, 0]) and np.allclose(rel.rx_normal, [1, 0, 0])
    assert np.allclose(info["c"], [0, 0, 0])
    assert np.allclose(info["roi_center"], [0.4, 0, 0])           # 中继中心（合页）+ (0.4,0,0)：物体在收、发波束中
    assert np.allclose(sc.feeds.pos[0], rel.rx_center + 1.8 * _az(-30.0))
    for i, a in enumerate((20.0, 40.0, 60.0)):
        assert np.allclose(sc.probes.pos[i], rel.tx_center + _az(a))
    assert sc.block_direct_feed_probe and sc.skip_gain_db == -3.0
    _, info2 = build_scene(dataclasses.replace(cfg, roi_offset=(0.8, 0.0, 0.0)), 2)
    assert np.allclose(info2["roi_center"], [0.8, 0, 0])
    _, info3 = build_scene(dataclasses.replace(cfg, fold_deg=45.0), 2)
    assert info3["fold_deg"] == 45.0


def test_config_validation_and_labels():
    with pytest.raises(ValueError):
        SystemConfig(condition="M5")
    with pytest.raises(ValueError):
        SystemConfig(m1_gain="foo")
    assert SystemConfig(condition="M1").label == "M1"
    assert SystemConfig(condition="M1", m1_gain="bussgang").label == "M1b"
    assert SystemConfig(condition="M1b").label == "M1b"
    d = SystemConfig(roi_offset=(0.1, 0, 0)).to_dict()
    assert d["roi_offset"] == [0.1, 0.0, 0.0] and d["condition"] == "M3"


# ============================================================ B. 源功率标定
def test_quantized_conjugate_and_best_codebook():
    rng = np.random.default_rng(0)
    v = (rng.normal(size=(2, 25)) + 1j * rng.normal(size=(2, 25)))
    cand = quantized_conjugate(v)
    assert cand.shape == (16, 2, 25)
    theta = 2 * np.pi * 3 / 16
    assert np.array_equal(cand[3], (np.cos(-np.angle(v) + theta) < 0).astype(int))
    bits, s = _best_codebook(v, 0.3, None)
    allv = 0.3 * np.sum(np.exp(1j * np.pi * cand) * v[None], axis=-1)
    assert np.allclose(np.abs(s), np.abs(allv).max(axis=0))
    # 量化共轭远好于随机码本（1-bit 相干合成的 2/π 规律）
    rnd = [np.abs(0.3 * np.sum(np.exp(1j * np.pi * rng.integers(0, 2, 25)) * v[0])) for _ in range(200)]
    assert np.abs(s[0]) > np.max(rnd)


def test_calibration_open_loop_focus_equals_p1db():
    """M3：聚焦码本在空场景的开环 |u_in|² == P1dB_in；M2/M4 恰差 −10/+10 dB；M0 与 M3 同功率。"""
    sysm = {c: PhysicalSystem(SystemConfig(condition=c), FocusTask(1)) for c in ("M2", "M3", "M4", "M0", "M1")}
    s3 = sysm["M3"]
    p1db = RappPA().p1db_in_w()
    assert s3.p1db_in_w == pytest.approx(p1db, rel=1e-15)
    m = s3.forward(s3.phases_from_params(s3.focus_params()), noisy=False)
    assert abs(m.u_in[0, 0]) ** 2 == pytest.approx(p1db, rel=1e-9)
    # 与 INTEGRATION B 的函数接口一致
    p_feed, p_ref = reference_feed_power(SystemConfig(condition="M3"))
    assert p_feed == pytest.approx(s3.p_feed_w, rel=1e-12) and p_feed * p_ref == pytest.approx(p1db, rel=1e-12)
    # 聚焦码本 = 16 个偏移中 |u0| 最大者
    v = s3.H0[:25, 0]
    amp = 10 ** (-2.5 / 20) / 5
    allu = np.abs(amp * np.sum(np.exp(1j * np.pi * quantized_conjugate(v)) * v, axis=-1))
    assert abs(m.u_in[0, 0]) == pytest.approx(s3.s_feed * allu.max(), rel=1e-12)
    # 回退量
    assert sysm["M2"].p_feed_w / s3.p_feed_w == pytest.approx(0.1, rel=1e-12)
    assert sysm["M4"].p_feed_w / s3.p_feed_w == pytest.approx(10.0, rel=1e-12)
    assert sysm["M0"].p_feed_w == pytest.approx(s3.p_feed_w, rel=1e-12)
    assert sysm["M1"].p_feed_w == pytest.approx(s3.p_feed_w, rel=1e-12)
    s1b = PhysicalSystem(SystemConfig(condition="M1", backoff_db=10.0), FocusTask(1))
    assert s1b.p_feed_w == pytest.approx(sysm["M4"].p_feed_w, rel=1e-12)
    for c, bo in (("M2", -10.0), ("M4", 10.0)):
        s = sysm[c]
        mc = s.forward(s.phases_from_params(s.focus_params()), noisy=False)
        assert 10 * np.log10(abs(mc.u_in[0, 0]) ** 2 / p1db) == pytest.approx(bo, abs=1e-9)
    # 标定是开环量；evaluate 报告实际闭环工作点（κ 反馈下与开环不同）
    ev = s3.evaluate(s3.focus_params(), np.random.default_rng(0), n_noise=2)
    assert ev["pa_in_open_rel_p1db_db"]["max"] == pytest.approx(0.0, abs=1e-8)
    assert abs(ev["pa_in_rel_p1db_db"]["max"]) > 0.05
    assert ev["p_feed_dbm"] == pytest.approx(float(w2dbm(s3.p_feed_w)))


def test_focus_params_beat_random_codebooks():
    rng = np.random.default_rng(1)
    for cond in ("M3", "M0"):
        s = PhysicalSystem(SystemConfig(condition=cond), FocusTask(1))
        y_f = s.forward(s.phases_from_params(s.focus_params()), noisy=False).readings_dbm[0, 0]
        y_r = [s.forward(s.phases_from_params(s.random_params(rng)), noisy=False).readings_dbm[0, 0]
               for _ in range(30)]
        assert y_f > max(y_r)


# ============================================================ C. 前向一致性
def test_relay_forward_matches_core_pipeline(m3_full):
    """系统前向 = solver.reduce_sample → relay_linear(s_b) → solve_fixed_point → probe_fields（逐样本独立重算）。"""
    s = m3_full
    rng = np.random.default_rng(2)
    p = s.random_params(rng)
    ph = s.phases_from_params(p)
    m = s.forward(ph, noisy=False)
    prm = RelayParams(isolation_db=s.cfg.isolation_db, kappa0_phase_deg=s.cfg.kappa0_phase_deg)
    w, ct = weights(ph["phase_rx"], ph["phase_tx"], prm)
    from ris_nlsim.relay import probe_fields, solve_fixed_point
    for b in (0, 5, 7):
        H = solver.solve_scene(s.scene, s.task.samples[b].objects, None, "full")
        lin = relay_linear(H, w, ct, s.s_feed, prm)
        fp = solve_fixed_point(lin.u_in[None], lin.K[None], RappPA())
        z = probe_fields(lin.d[None], lin.C[None], fp.g)[0]
        assert np.allclose(m.z[b], z, rtol=1e-9, atol=0)
    assert m.readings_dbm.shape == (8, 2) and m.u.shape == (8, 1) and not m.flags.any()
    assert np.allclose(m.readings_dbm, 10 * np.log10(np.abs(m.z) ** 2) + 30)


@pytest.mark.parametrize("mode", ["full", "born", "intra"])
def test_passive_batch_matches_eval_reduced_and_solve_scene(mode):
    s = _sys("M0", mode, n=2)
    rng = np.random.default_rng(3)
    phase = s.phases_from_params(s.random_params(rng))["phase_p"]
    Hb = s._pb.eval(phase, np.arange(s.B))
    for b, smp in enumerate(s.task.samples):
        rs = solver.reduce_sample(s.scene, smp.objects, mode)
        H_ref = solver.eval_reduced(rs, phase)
        assert np.allclose(Hb[b], H_ref, rtol=1e-10, atol=0)
        assert np.allclose(solver.eval_reduced(compress_reduced(rs), phase), H_ref, rtol=1e-10, atol=0)
    H_direct = solver.solve_scene(s.scene, s.task.samples[3].objects, phase, mode)
    assert np.allclose(Hb[3], H_direct, rtol=1e-9, atol=0)
    if mode != "full":
        assert s._pb.shared                                      # 背景 Cpp 只含环境：各样本相同
        assert s._rs[3].Lam.shape == (50 + 2, 50 + 2)            # 物体维压缩到 N_P + N_RCV


@pytest.mark.parametrize("cond,mode,extra", [("M3", "full", {}), ("M3", "born", {}), ("M0", "born", {}),
                                             ("M0", "full", {}), ("M4", "intra", {}), ("M1", "full", {}),
                                             ("M1b", "full", {}), ("M3", "full", {"n_relays": 2}),
                                             ("M3", "full", {"bg_cancel": "null"})])
def test_torch_numpy_consistency(cond, mode, extra):
    """system.to_torch()：无噪声读数一致（< 1e-6 dB），无标记时损失一致。"""
    s = _sys(cond, mode, n=3 if not extra else 2, **extra)
    ts = s.to_torch()
    rng = np.random.default_rng(4)
    for _ in range(3):
        ph = s.phases_from_params(s.random_params(rng))
        m = s.forward(ph, noisy=False)
        with torch.no_grad():
            y_t = ts.readings_db(ts.as_phases(ph)).numpy()
            l_t = float(ts.loss(ts.as_phases(ph)))
        assert np.max(np.abs(y_t - m.readings_dbm)) < 1e-6
        assert not m.flags.any()
        assert l_t == pytest.approx(s.loss_from(m), rel=1e-9, abs=1e-12)


def test_null_penalty_matches_definition(m3_full):
    s_null = m3_full.derive(bg_cancel="null", null_target_db=-20.0, null_weight=2.0)
    assert s_null._rs is m3_full._rs                         # 场景未变：复用缓存
    ph = m3_full.phases_from_params(m3_full.random_params(np.random.default_rng(5)))
    nd = s_null.null_depth_db(ph)
    w, _ = weights(ph["phase_rx"], ph["phase_tx"], s_null.params)
    u0 = s_null.s_feed * np.sum(w[0] * s_null.H0[:25, 0])
    assert nd == pytest.approx(10 * np.log10(abs(u0) ** 2 / s_null.u0_focus_abs2), rel=1e-12)
    diff = s_null.loss_phases(ph) - m3_full.loss_phases(ph)
    assert diff == pytest.approx(2.0 * np.logaddexp(0.0, nd + 20.0), rel=1e-12)
    # 聚焦码本的零陷深度 = 0 dB（定义）
    assert s_null.null_depth_db(s_null.phases_from_params(s_null.focus_params())) == pytest.approx(0.0, abs=1e-9)


def test_derive_reuses_samples_and_matches_fresh_build(m3_full):
    s4 = m3_full.derive(condition="M4")
    assert s4._rs is m3_full._rs and s4.label == "M4"
    fresh = _sys("M4", "full")
    ph = s4.phases_from_params(s4.random_params(np.random.default_rng(6)))
    assert np.allclose(s4.forward(ph, noisy=False).readings_dbm, fresh.forward(ph, noisy=False).readings_dbm,
                       rtol=0, atol=1e-12)
    s0 = m3_full.derive(condition="M0")
    assert s0._rs is not m3_full._rs and s0.is_passive


def test_condition_pa_semantics(m3_full):
    s1 = m3_full.derive(condition="M1")
    assert isinstance(s1.pa, LinearPA) and s1.pa.gain_v == pytest.approx(RappPA().gain_v)
    assert isinstance(m3_full.pa, RappPA) and m3_full.pa.ampm
    s2 = m3_full.derive(condition="M2", ampm=False)
    assert isinstance(s2.pa, RappPA) and not s2.pa.ampm
    s0 = m3_full.derive(condition="M0")
    assert s0.pa is None and s0.n_params == 50 and np.all(s0.levels == 2)
    assert m3_full.n_params == 50 and np.all(m3_full.levels == 2)
    s0b = m3_full.derive(condition="M0", passive_levels=4)
    assert np.all(s0b.levels == 4)
    ph = s0b.phases_from_params(np.arange(50) % 4)
    assert np.allclose(ph["phase_p"], (np.pi / 2) * (np.arange(50) % 4))
    ph3 = m3_full.phases_from_params(np.r_[np.ones(25, int), np.zeros(25, int)])
    assert np.allclose(ph3["phase_rx"], np.pi) and np.allclose(ph3["phase_tx"], 0.0)
    with pytest.raises(ValueError):
        m3_full.phases_from_params(np.full(50, 2))
    # M1 的环路是线性解：loop_gain_op = loop_gain_ss
    m = s1.forward(s1.phases_from_params(s1.random_params(np.random.default_rng(7))), noisy=False)
    assert np.allclose(m.loop_gain_op, m.loop_gain_ss)


# ============================================================ G1：DWBA 仿射性
def test_m0_born_parity_is_affine_walsh_ge3_vanishes(m0_born):
    """门 G1 自检：M0 + Born 下探头场对 x 仿射 ⇒ y1−y2（线性功率）是二次多项式，W≥3 < 1e-20（多个随机码本）。"""
    s = m0_born
    rng = np.random.default_rng(8)
    for _ in range(5):
        ph = s.phases_from_params(s.random_params(rng))
        m = s.forward(ph, noisy=False)
        ylin = m.p_w[:, 0] - m.p_w[:, 1]
        assert walsh_weight_ge(ylin, 3, 3) < 1e-20
        assert walsh_weight_ge(m.z, 3, 2) < 1e-20                    # z 仿射：二阶以上权重也为 0
        assert walsh_weight_ge(ylin, 3, 2) > 1e-6                    # 二阶项（平方律检波）存在
        prof = s.nonlinearity_profile(ph)
        assert prof["y_diff_lin"]["W3"] < 1e-20 and prof["e"]["W2"] < 1e-20


def test_m0_full_parity_has_high_order_terms():
    s = _sys("M0", "full")
    rng = np.random.default_rng(9)
    for _ in range(3):
        m = s.forward(s.phases_from_params(s.random_params(rng)), noisy=False)
        assert walsh_weight_ge(m.p_w[:, 0] - m.p_w[:, 1], 3, 3) > 1e-12


def test_relay_profile_keys(m3_born):
    prof = m3_born.nonlinearity_profile(m3_born.phases_from_params(m3_born.random_params(np.random.default_rng(10))))
    for k in ("e", "u_in", "u", "g", "z", "y_diff_lin", "y_diff_db"):
        assert k in prof and 0.0 <= prof[k]["W3"] <= prof[k]["W2"] <= 1.0 + 1e-12
        assert len(prof[k]["profile"]) == 3
    # Born：接收场与开环 u_in 对 x 仿射；PA 不动点引入高阶项
    assert prof["e"]["W2"] < 1e-20 and prof["u_in"]["W2"] < 1e-20
    assert prof["g"]["W3"] > 1e-12
    avg = m3_born.nonlinearity_profile_random(np.random.default_rng(11), n_codebooks=2)
    assert set(avg) >= {"e", "u", "g", "y_diff_lin"}
    with pytest.raises(TypeError):
        PhysicalSystem(SystemConfig(), FocusTask(1)).nonlinearity_profile({})


# ============================================================ 自激处理
def test_flagged_samples_use_chance_plus_penalty():
    s = _sys("M1", "full", n=2, isolation_db=-5.0, osc_smooth_weight=0.0)    # |G_v κ_eff| = 30−5−5 = +20 dB
    p = s.random_params(np.random.default_rng(12))
    m = s.forward(s.phases_from_params(p), noisy=False)
    assert m.flags.all() and np.all(m.loop_gain_ss > 1.0)
    assert s.loss(p, noisy=False) == pytest.approx(np.log(2.0) + 1.0, rel=1e-12)
    ev = s.evaluate(p, np.random.default_rng(0), n_noise=3)
    assert ev["oscillating_run"] and ev["n_flagged"] == s.B and ev["acc_clean"] == 0.0
    assert ev["loop_gain_ss_max"] > 1.0 and ev["gershgorin_max"] >= ev["loop_gain_ss_max"] - 1e-12
    # 带光滑自激惩罚（DESIGN §6.3）：再加 w·mean relu(ρ_ss − 0.95)²；torch 同定义
    s2 = s.derive(osc_smooth_weight=100.0)
    rho = m.loop_gain_ss
    want = np.log(2.0) + 1.0 + 100.0 * np.mean(np.maximum(rho - 0.95, 0) ** 2)
    assert s2.loss(p, noisy=False) == pytest.approx(want, rel=1e-12)
    with torch.no_grad():
        ts = s2.to_torch()
        assert float(ts.loss(ts.as_phases(s2.phases_from_params(p)))) == pytest.approx(want, rel=1e-9)


def test_partial_flags_replace_only_flagged_samples(m3_full):
    s = m3_full
    m = s.forward(s.phases_from_params(s.random_params(np.random.default_rng(13))), noisy=False)
    per = s.sample_losses(m)
    assert np.allclose(per, losses.binary_loss(m.readings_dbm, s.labels, 1.0, reduction="none"))
    flags = np.zeros(s.B, bool)
    flags[[1, 6]] = True
    mf = dataclasses.replace(m, flags=flags)
    per_f = s.sample_losses(mf)
    assert np.allclose(per_f[~flags], per[~flags])
    assert np.allclose(per_f[flags], np.log(2.0) + s.cfg.osc_penalty)
    acc_f = s._accuracy(m.readings_dbm, flags, s.labels)          # 被标记样本一律算错
    pred = s.task.predict(m.readings_dbm)
    assert acc_f == pytest.approx(np.mean((pred == s.labels) & ~flags))


def test_g2_loop_diagnostics(m3_full):
    ev = m3_full.evaluate(m3_full.random_params(np.random.default_rng(14)), np.random.default_rng(0), n_noise=2)
    # 端口隔离 −30 dB、收发网络损耗 5 dB、30 dB PA ⇒ 小信号环路增益 ≈ −5 dB（物体/自由空间回耦是小修正）
    assert ev["loop_gain_ss_max"] == pytest.approx(10 ** (-5 / 20), rel=0.05)
    assert ev["gershgorin_max"] >= ev["loop_gain_ss_max"] - 1e-12              # R=1 时相等
    assert ev["det_min"] == pytest.approx(abs(1 - 10 ** (-5 / 20) * 1j), rel=0.05)   # κ0 相位 90°
    assert ev["loop_gain_op_max"] <= ev["loop_gain_ss_max"] + 1e-12
    for key in ("acc_clean", "all_correct", "margin_db", "acc_noisy", "oscillating_run", "n_flagged",
                "pa_in_rel_p1db_db", "null_depth_db", "loss_clean", "p_feed_dbm", "voxel_d", "effective_eps"):
        assert key in ev
    assert ev["voxel_d"] == pytest.approx(VOX) and ev["effective_eps"] == "78-10j"


# ============================================================ 噪声
def test_noise_vna_scatter_and_determinism(m3_full):
    s = m3_full
    ph = s.phases_from_params(s.random_params(np.random.default_rng(15)))
    c1 = s.forward(ph, noisy=False).readings_dbm
    c2 = s.forward(ph, noisy=False).readings_dbm
    assert np.array_equal(c1, c2)
    with pytest.raises(ValueError):
        s.forward(ph, noisy=True)                                  # 有噪声必须显式 rng
    reps = 300
    m = s.forward(ph, np.random.default_rng(16), noisy=True, sample_idx=np.tile(np.arange(s.B), reps))
    dy = m.readings_dbm - np.tile(c1, (reps, 1))
    assert np.std(dy) == pytest.approx(0.002, rel=0.1)            # VNA 级 0.002 dB（PA 噪声 −139 dBm 可忽略）
    assert abs(np.mean(dy)) < 3e-4
    m_a = s.forward(ph, np.random.default_rng(17), noisy=True)
    m_b = s.forward(ph, np.random.default_rng(17), noisy=True)
    assert np.array_equal(m_a.readings_dbm, m_b.readings_dbm)


def test_probe_noise_and_logdet(m3_full):
    ph = m3_full.phases_from_params(m3_full.random_params(np.random.default_rng(18)))
    clean = m3_full.forward(ph, noisy=False)
    s_pn = m3_full.derive(probe_noise_dbm=float(np.max(clean.readings_dbm)) - 3.0, detector="ideal")
    m = s_pn.forward(ph, np.random.default_rng(19), noisy=True, sample_idx=np.tile(np.arange(8), 50))
    assert np.std(m.readings_dbm - np.tile(clean.readings_dbm, (50, 1))) > 0.5
    s_ld = m3_full.derive(detector="logdet")
    assert s_ld.detector.sigma_db == 0.3 and s_ld.detector.offset_db != 0.0
    lo, hi = s_ld.detector.range_dbm
    m2 = s_ld.forward(ph, np.random.default_rng(20), noisy=True)
    assert np.all(m2.readings_dbm > lo - 2.0) and np.all(m2.readings_dbm < hi + 2.0)
    assert np.allclose(s_ld.forward(ph, noisy=False).readings_dbm, clean.readings_dbm)   # 无噪声 = 理想 dBm


def test_power_jitter_scales_feed_per_sample(m3_full):
    """meta["power_db"] → 每样本有效源幅度 s_b（INTEGRATION C）：M1 线性中继下 z 按 √power 缩放。"""
    cfg = SystemConfig(condition="M1", voxel_d=VOX)
    task = make_task(cfg, "parity", n=2)
    task.samples[2].meta["power_db"] = 6.0
    s = PhysicalSystem(cfg, task)
    ref = PhysicalSystem(cfg, make_task(cfg, "parity", n=2))
    ph = s.phases_from_params(s.random_params(np.random.default_rng(21)))
    z, z0 = s.forward(ph, noisy=False).z, ref.forward(ph, noisy=False).z
    assert np.allclose(z[2], z0[2] * 10 ** (6 / 20), rtol=1e-10)
    assert np.allclose(z[[0, 1, 3]], z0[[0, 1, 3]], rtol=1e-12)


# ============================================================ M1b（Bussgang）
def test_bussgang_gain_below_small_signal_under_saturation():
    s = _sys("M1", "full", n=2, m1_gain="bussgang", backoff_db=10.0)          # 与 M4 同功率
    assert s.label == "M1b" and isinstance(s.pa, ComplexLinearPA)
    G_v = RappPA().gain_v
    assert abs(s.bussgang_gain) < G_v
    assert abs(s.bussgang_gain) > G_v / 10.0
    ev = s.evaluate(s.random_params(np.random.default_rng(22)), np.random.default_rng(0), n_noise=2)
    assert ev["condition"] == "M1b" and ev["bussgang_gain_db"] < 30.0
    # 深回退下 Bussgang 增益 → 小信号增益
    s_lin = _sys("M1", "full", n=2, m1_gain="bussgang", backoff_db=-25.0)
    assert 20 * np.log10(abs(s_lin.bussgang_gain)) == pytest.approx(30.0, abs=0.1)
    # 定义核对：用同一组码本与 Rapp 不动点重算
    m4 = _sys("M4", "full", n=2)
    rng = np.random.default_rng(0)
    num, den = 0j, 0.0
    for _ in range(16):
        m = m4.forward(m4.phases_from_params(rng.integers(0, 2, 50)), noisy=False)
        num += np.sum(m.g * np.conj(m.u))
        den += np.sum(np.abs(m.u) ** 2)
    assert s.bussgang_gain == pytest.approx(num / den, rel=1e-9)


# ============================================================ 其他任务
def test_letter_localization_focus_systems_forward():
    cfg = SystemConfig(condition="M3", object_mode="full")
    rng = np.random.default_rng(23)
    for kind, kw, K in (("letter", {}, 3), ("loc3bit", {"voxel_d": VOX}, 3), ("focus", {}, 1)):
        s = make_system(cfg, kind, **kw)
        assert s.n_probes == K
        p = s.random_params(rng)
        m = s.forward(s.phases_from_params(p), noisy=False)
        assert m.readings_dbm.shape == (s.B, K) and np.all(np.isfinite(m.readings_dbm))
        ev = s.evaluate(p, rng, n_noise=2)
        assert np.isfinite(ev["loss_clean"])
        if kind == "loc3bit":
            assert len(ev["bit_acc_clean"]) == 3
            pred = s.task.predict(m.readings_dbm)
            assert pred.min() >= 0 and pred.max() <= 7
        if kind == "letter":
            assert ev["material"] == "metal" and ev["effective_eps"] == "metal"
        if kind == "focus":
            assert ev["acc_clean"] is None and ev["loss_clean"] == pytest.approx(-m.readings_dbm[0, 0])
    # M0 字母（金属薄板，full 模式）
    s0 = make_system(SystemConfig(condition="M0"), "letter")
    m0 = s0.forward(s0.phases_from_params(s0.random_params(rng)), noisy=False)
    assert m0.readings_dbm.shape == (3, 3)


def test_forward_speed_parity(m3_full, m0_born):
    """性能：B=8 奇偶前向（无噪声）。M0 / M1 为批量线性代数（~0.2 ms）；Rapp 中继受 relay.solve_fixed_point 的
    逐步阻尼迭代约束（~2–3 ms）。阈值宽松，只防止数量级退化（例如 BLAS 多线程导致的 100× 变慢）。"""
    s1 = m3_full.derive(condition="M1")
    out = {}
    for name, s in (("M0", m0_born), ("M1", s1), ("M3", m3_full)):
        ph = s.phases_from_params(s.random_params(np.random.default_rng(24)))
        s.forward(ph, noisy=False)
        ts = []
        for _ in range(15):
            t0 = time.perf_counter()
            s.forward(ph, noisy=False)
            ts.append(time.perf_counter() - t0)
        out[name] = np.median(ts)
    assert out["M0"] < 0.01 and out["M1"] < 0.01 and out["M3"] < 0.03, out


@pytest.mark.slow
def test_build_time_water_auto_voxel_full():
    """L90、n=3、水（自动体素 ≈ λ/28，~437 体素/球）、full 模式：构建 ≪ 30 s，前向仍为毫秒级。"""
    t0 = time.perf_counter()
    s = make_system(SystemConfig(condition="M3", object_mode="full"), "parity", n=3)
    build = time.perf_counter() - t0
    assert s.task.n_vox_per_object > 400 and build < 30.0
    t0 = time.perf_counter()
    s0 = make_system(SystemConfig(condition="M0", object_mode="full"), "parity", n=3)
    assert time.perf_counter() - t0 < 30.0
    ph = s.phases_from_params(s.random_params(np.random.default_rng(25)))
    s.forward(ph, noisy=False)
    t0 = time.perf_counter()
    s.forward(ph, noisy=False)
    assert time.perf_counter() - t0 < 0.05
    m0 = s0.forward(s0.phases_from_params(s0.random_params(np.random.default_rng(26))), noisy=False)
    assert walsh_weight_ge(m0.p_w[:, 0] - m0.p_w[:, 1], 3, 3) > 1e-12
    s0b = s0.derive(object_mode="born")
    m0b = s0b.forward(s0b.phases_from_params(s0b.random_params(np.random.default_rng(27))), noisy=False)
    assert walsh_weight_ge(m0b.p_w[:, 0] - m0b.p_w[:, 1], 3, 3) < 1e-20


# ============================================================ 审查 INT-PHYS-1：物体在中继发射波束中
def test_relay_born_loop_terms_and_objects_see_relay_tx_switch():
    """flat0 + Born：物体被中继发射场照射 → K(x)、C(x) 依赖 x，M1（线性 PA）的 y 已含三阶以上 Walsh 项；
    增益匹配线性参考对 M1 恒等（pa_excess = 0）；objects_see_relay_tx=False 恢复"Born + 线性中继 = 二次"。"""
    s = _sys("M1", "born", preset="flat0")
    rng = np.random.default_rng(30)
    phs = [s.phases_from_params(s.random_params(rng)) for _ in range(3)]
    for ph in phs:
        prof = s.nonlinearity_profile(ph)
        assert prof["e"]["W2"] < 1e-20 and prof["u_in"]["W2"] < 1e-20          # 馈源列仍对 x 仿射
        assert prof["K"]["W2"] < 1e-20 and prof["C"]["W2"] < 1e-20            # K、C 本身对 x 仿射……
        assert prof["K"]["rel_var"] > 1e-4 and prof["C"]["rel_var"] > 1e-4    # ……但依赖 x（物体在发射波束中）
        assert prof["y_diff_lin"]["W3"] > 1e-8                                  # u = u_in/(1 − G K(x)) 是有理函数
        for k in ("u", "g", "z", "y_diff_lin", "y_diff_db"):
            assert prof[f"pa_excess_{k}"]["W3"] == 0.0 and prof[f"pa_excess_{k}"]["W2"] == 0.0
        assert not prof["linref_flags_any"]
    s_off = s.derive(objects_see_relay_tx=False)
    assert s_off._rs is not s._rs and s_off.task is s.task
    for rs, rs0 in zip(s_off._rs, s._rs):
        assert np.array_equal(rs.H[:, 1:], s_off.H0[:, 1:]) and np.array_equal(rs.H[:, 0], rs0.H[:, 0])
    for ph in phs:
        prof = s_off.nonlinearity_profile(ph)
        assert prof["K"]["rel_var"] == 0.0 and prof["C"]["rel_var"] == 0.0
        assert prof["u"]["W2"] < 1e-20 and prof["y_diff_lin"]["W3"] < 1e-20  # 二次：3 位奇偶不可解的前提
    # 开关关闭时 M3 的高阶项全部来自 PA：线性参考是二次的
    s3 = s_off.derive(condition="M3")
    assert s3._rs is s_off._rs
    prof3 = s3.nonlinearity_profile(phs[0])
    assert prof3["y_diff_lin_linref"]["W3"] < 1e-20 and prof3["g_linref"]["W2"] < 1e-20
    assert prof3["pa_excess_g"]["W3"] > 1e-14 and prof3["pa_excess_y_diff_lin"]["W3"] > 1e-14   # 浮点噪声 ~1e-30
    avg = s3.nonlinearity_profile_random(np.random.default_rng(31), n_codebooks=2)
    assert avg["K"]["rel_var"] == 0.0 and "W3" in avg["pa_excess_g"]
    # torch 前向使用同一组（替换了发射列的）样本响应
    ts = s3.to_torch()
    with torch.no_grad():
        y_t = ts.readings_db(ts.as_phases(phs[1])).numpy()
    assert np.max(np.abs(y_t - s3.forward(phs[1], noisy=False).readings_dbm)) < 1e-6
    ev = s3.evaluate_phases(phs[1], np.random.default_rng(0), n_noise=0)
    assert ev["objects_see_relay_tx"] is False and s3.describe()["objects_see_relay_tx"] is False


# ============================================================ 审查 INT-SEM-2 / INT-NUM-2：derive 与任务
def test_derive_rebuilds_task_when_object_fields_change(m3_born):
    ph = m3_born.phases_from_params(m3_born.random_params(np.random.default_rng(32)))
    for change in (dict(roi_offset=(0.8, 0.0, 0.0)), dict(material="eps4"), dict(voxel_d=LAMBDA / 12),
                   dict(preset="flat0")):
        d = m3_born.derive(**change)
        fresh = make_system(dataclasses.replace(m3_born.cfg, **change), "parity", n=3)
        assert d.task is not m3_born.task and d._rs is not m3_born._rs
        assert np.array_equal(d.forward(ph, noisy=False).readings_dbm, fresh.forward(ph, noisy=False).readings_dbm)
        assert np.allclose(d.task.slots.mean(axis=0), d.info["roi_center"])
        assert np.allclose(d.describe()["roi_center"], d.task.slots.mean(axis=0))
        assert d.task.material == fresh.task.material and d.task.voxel_d == fresh.task.voxel_d
    # 不改物体的字段：同一任务、复用缓存
    d4 = m3_born.derive(condition="M4", isolation_db=-25.0)
    assert d4.task is m3_born.task and d4._rs is m3_born._rs
    # 手工构造的任务无法重建 → 报错（而不是静默沿用旧物体）；born + 默认 scalar_rr 的手工任务会被警告（审查 R2-INT-1）
    with pytest.warns(RuntimeWarning, match="R2-INT-1"):
        manual = PhysicalSystem(m3_born.cfg, ParityTask(2, m3_born.info["roi_center"], voxel_d=VOX))
    assert manual.object_model == "scalar_rr" and m3_born.object_model == "volume_rr"
    for change in (dict(roi_offset=(0.8, 0.0, 0.0)), dict(material="eps4"), dict(freq_hz=3.0e9)):
        with pytest.raises(ValueError):
            manual.derive(**change)
    with pytest.warns(RuntimeWarning, match="R2-INT-1"):
        assert manual.derive(condition="M1")._rs is manual._rs
    # make_task 显式给出的材料不随 cfg.material 变化
    s_e = make_system(SystemConfig(condition="M3", object_mode="born", voxel_d=VOX), "parity", n=2, material="eps4")
    assert s_e.derive(material="metal").task is s_e.task
    # 带抖动的任务：重建时用构造前的 rng 状态副本 → 抖动与原任务相同，只是整体平移
    s_l = make_system(SystemConfig(condition="M3", object_mode="born"), "letter", dist_jitter=LAMBDA / 8,
                      power_jitter_db=3.0, rng=np.random.default_rng(33))
    d_l = s_l.derive(roi_offset=(0.6, 0.0, 0.0))
    assert [m.meta["dist_shift"] for m in d_l.task.samples] == [m.meta["dist_shift"] for m in s_l.task.samples]
    assert np.array_equal(d_l.power_scale, s_l.power_scale)
    for a, b in zip(d_l.task.samples, s_l.task.samples):
        assert np.allclose(a.objects.pos - b.objects.pos, [0.1, 0.0, 0.0])


# ============================================================ 审查 INT-SEM-4：零陷后的源功率标定
def test_cal_ref_nulled_and_object_reference_definitions(m3_born):
    s_n = m3_born.derive(cal_ref="nulled")
    assert s_n._rs is m3_born._rs and s_n.task is m3_born.task
    p1 = s_n.p1db_in_w
    pn = np.r_[s_n.cal_bits_rx[0], np.zeros(N_TX, dtype=np.int64)]
    ph = s_n.phases_from_params(pn)
    m = s_n.forward(ph, noisy=False)
    assert np.median(np.abs(m.u_in[:, 0]) ** 2) == pytest.approx(p1, rel=1e-9)        # 参考码本下中位开环驱动 = P1dB
    assert abs(s_n.cal_null_depth_db - s_n.cfg.null_target_db) < 3.0                 # 参考零陷深度 ≈ 训练目标
    assert s_n.null_depth_db(ph) == pytest.approx(s_n.cal_null_depth_db, abs=1e-9)
    assert s_n.p_feed_w > 30.0 * m3_born.p_feed_w                                     # 聚焦标定低估了零陷后的驱动
    assert s_n.u0_focus_abs2 == pytest.approx(s_n.p_feed_w * m3_born.p_ref_per_w, rel=1e-12)
    # object：物体散射聚焦码本下最大物体驱动 = P1dB·10^{bo/10}
    s_o = m3_born.derive(cal_ref="object", condition="M4")
    po = np.r_[s_o.cal_bits_rx[0], np.zeros(N_TX, dtype=np.int64)]
    ev = s_o.evaluate(po, np.random.default_rng(0), n_noise=0)
    assert ev["cal_ref"] == "object" and ev["pa_in_delta_rel_p1db_db"]["max"] == pytest.approx(10.0, abs=1e-6)
    mo = s_o.forward(s_o.phases_from_params(po), noisy=False)
    w, _ = weights(mo.phases["phase_rx"], mo.phases["phase_tx"], s_o.params)
    du = mo.u_in[:, 0] - s_o.s_b * np.sum(w[0] * s_o.H0[:25, 0])
    assert np.max(np.abs(du) ** 2) == pytest.approx(10.0 * p1, rel=1e-9)
    # 回退量照常作用；M0 与同 cal_ref 的 M3 同功率（M0 另解中继版本场景）
    assert m3_born.derive(cal_ref="nulled", condition="M4").p_feed_w == pytest.approx(10 * s_n.p_feed_w, rel=1e-12)
    assert m3_born.derive(cal_ref="nulled", condition="M0").p_feed_w == pytest.approx(s_n.p_feed_w, rel=1e-9)
    with pytest.raises(ValueError):
        PhysicalSystem(SystemConfig(cal_ref="object"), FocusTask(1))                  # 无物体：参考功率为 0
    with pytest.raises(ValueError):
        SystemConfig(cal_ref="peak")


# ============================================================ 审查 INT-SEM-5：logdet 的无噪声评估
def test_logdet_clean_metrics_use_detector_offset_and_clip(m3_full):
    s = m3_full.derive(detector="logdet")
    p = s.random_params(np.random.default_rng(34))
    m = s.forward(s.phases_from_params(p), noisy=False)
    y_det = s.detector.read(m.p_w, None)
    ev = s.evaluate(p, np.random.default_rng(0), n_noise=0)
    assert ev["acc_clean"] == pytest.approx(s._accuracy(y_det, m.flags, s.labels))
    assert ev["margin_db"] == pytest.approx(s.task.margin_db(y_det))
    lo, hi = s.detector.range_dbm
    raw = w2dbm(m.p_w) + s.detector.offset_db
    assert ev["det_clipped_frac"] == pytest.approx(np.mean((raw < lo) | (raw > hi)))
    assert ev["loss_meas_clean"] == pytest.approx(
        s.loss_from(dataclasses.replace(m, readings_dbm=y_det), model_penalties=False), rel=1e-12)
    assert ev["loss_clean"] == pytest.approx(s.loss(p, noisy=False), rel=1e-12)     # 模型目标仍用理想 dBm
    # 把读数整体压到量程下沿：全部钳在 −40 dBm → 两探头读数相等，裕量 0，评估反映真实检波器
    s.detector.offset_db = lo - float(np.max(w2dbm(m.p_w))) - 1.0
    ev2 = s.evaluate(p, np.random.default_rng(0), n_noise=0)
    assert ev2["det_clipped_frac"] == 1.0 and ev2["margin_db"] == 0.0 and not ev2["all_correct"]
    assert ev2["acc_clean"] == 0.0                     # 全部钳位平局 = 全部未决（审查 R2-INT-2；旧规则得 0.5）
    assert ev2["acc_clean_ideal"] == pytest.approx(s._accuracy(m.readings_dbm, m.flags, s.labels))
    assert s.clean_metrics(p)["acc_clean"] == pytest.approx(ev2["acc_clean"])
    assert m3_full.evaluate(p, np.random.default_rng(0), n_noise=0)["det_clipped_frac"] is None


# ============================================================ 审查 INT-NUM-3 / INT-SEM-6：被标记样本与自激 run
def test_flagged_samples_excluded_from_margin_and_operating_point():
    s = _sys("M3", "full", n=1, isolation_db=-22.0, kappa0_phase_deg=0.0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        p = s.random_params(rng)
        m = s.forward(s.phases_from_params(p), noisy=False)
        if 0 < m.flags.sum() < s.B:
            break
    else:
        pytest.fail("没有找到部分样本被标记的码本")
    ok = ~m.flags
    ev = s.evaluate(p, np.random.default_rng(1), n_noise=3)
    assert ev["oscillating_run"] and ev["margin_db"] is None                         # 标记 = 错 ⇒ 无正裕量
    assert ev["margin_db_unflagged"] == pytest.approx(s.task.margin_db(m.readings_dbm[ok], s.labels[ok]))
    assert ev["acc_clean_valid"] is None and ev["all_correct_valid"] is None and ev["acc_noisy_valid"] is None
    assert ev["probe_dbm"]["max"] == pytest.approx(np.max(m.readings_dbm[ok]))
    rel = 10 * np.log10(np.abs(m.u[ok]) ** 2 / s.p1db_in_w)
    assert ev["pa_in_rel_p1db_db"]["min"] == pytest.approx(np.min(rel))
    assert ev["pa_in_rel_p1db_db_max"] == pytest.approx(np.max(rel))
    rel_open = 10 * np.log10(np.abs(m.u_in) ** 2 / s.p1db_in_w)
    assert ev["pa_in_open_rel_p1db_db_max"] == pytest.approx(np.max(rel_open))     # 开环量与环路无关：全部样本
    # 无标记的 run：*_valid 等于原值
    s_ok = _sys("M3", "full", n=1)
    ev_ok = s_ok.evaluate(s_ok.random_params(np.random.default_rng(2)), np.random.default_rng(3), n_noise=3)
    assert not ev_ok["oscillating_run"]
    assert ev_ok["acc_clean_valid"] == ev_ok["acc_clean"] and ev_ok["all_correct_valid"] == ev_ok["all_correct"]
    assert ev_ok["acc_noisy_valid"] == ev_ok["acc_noisy"] and ev_ok["margin_db"] == ev_ok["margin_db_unflagged"]


# ============================================================ 审查 R2-INT-1：born 物体极化率与体素收敛
def _rx_object_field_db(s):
    """单物体（n=1 奇偶，样本 1）在接收单元处的散射场 / 入射场 (dB)。"""
    v = s._H[:, :N_TX, 0]
    return 20 * np.log10(np.linalg.norm(v[1] - v[0]) / np.linalg.norm(v[0]))


def test_born_objects_use_volume_rr_and_converge_with_voxel_size():
    """born + 水：make_system 自动用 volume_rr，λ/20、自动 λ/28、λ/40 的单物体场在 0.5 dB 内（旧 scalar_rr：
    +5.9 / +9.6 / +3.6 dB 非单调）；金属 born 同样收敛；evaluate / describe 记录实际模型。"""
    lv = {}
    for vd in (LAMBDA / 20, None, LAMBDA / 40):
        s = make_system(SystemConfig(condition="M3", object_mode="born", voxel_d=vd), "parity", n=1)
        assert s.task.model == "volume_rr" and s.object_model == "volume_rr"
        lv[vd] = _rx_object_field_db(s)
    assert np.ptp(list(lv.values())) < 0.5 and abs(lv[None]) < 1.0, lv
    ev = s.evaluate(s.focus_params(0), np.random.default_rng(0), n_noise=0)
    assert ev["object_model"] == "volume_rr" and ev["effective_eps"] == "78-10j"
    assert s.describe()["object_model"] == "volume_rr" and s.describe()["task"]["model"] == "volume_rr"
    # 反例（机制）：显式 scalar_rr 的 born 在 λ/20 与 λ/28 之间差 > 3 dB
    with pytest.warns(RuntimeWarning):
        old = [_rx_object_field_db(PhysicalSystem(SystemConfig(condition="M3", object_mode="born"),
                                                  make_task(SystemConfig(voxel_d=vd), "parity", n=1)))
               for vd in (LAMBDA / 20, None)]
    assert abs(old[1] - old[0]) > 3.0 and old[1] > 5.0
    # 金属：born 用 α0 = 3V（ε_eff = 4），λ/10 与 λ/20 一致；full / intra 仍用 scalar_rr（软球）
    lm = [_rx_object_field_db(make_system(SystemConfig(condition="M3", object_mode="born", material="metal",
                                                       voxel_d=vd), "parity", n=1)) for vd in (LAMBDA / 10, LAMBDA / 20)]
    assert abs(lm[1] - lm[0]) < 0.1
    sf = make_system(SystemConfig(condition="M3", object_mode="intra", material="metal", voxel_d=LAMBDA / 10),
                     "parity", n=1)
    assert sf.task.model == "scalar_rr" and sf.evaluate(sf.focus_params(0), None, n_noise=0)["effective_eps"] == "metal"


def test_intra_matches_full_for_single_object_and_is_affine_under_m0():
    """RQ2 的强度匹配对照（审查 R2-INT-1）：单物体时 intra 与 full 逐位相同（中继条件、无环境），
    而 M0 + intra 的探头功率差对 x 仿射（Walsh W≥3 为浮点噪声级），与 born 同样满足门 G1 的前提。"""
    si = _sys("M3", "intra", n=1)
    sf = _sys("M3", "full", n=1)
    assert np.allclose(si._H, sf._H, rtol=1e-9, atol=0.0)
    s0 = _sys("M0", "intra", n=3)
    rng = np.random.default_rng(36)
    for _ in range(3):
        m = s0.forward(s0.phases_from_params(s0.random_params(rng)), noisy=False)
        assert walsh_weight_ge(m.p_w[:, 0] - m.p_w[:, 1], 3, 3) < 1e-20


def test_derive_object_mode_switches_object_model(m3_born):
    ph = m3_born.phases_from_params(m3_born.random_params(np.random.default_rng(37)))
    for mode, model in (("full", "scalar_rr"), ("intra", "scalar_rr")):
        d = m3_born.derive(object_mode=mode)
        fresh = make_system(dataclasses.replace(m3_born.cfg, object_mode=mode), "parity", n=3)
        assert d.task is not m3_born.task and d.task.model == model == fresh.task.model
        assert np.array_equal(d.forward(ph, noisy=False).readings_dbm, fresh.forward(ph, noisy=False).readings_dbm)
        assert d.derive(object_mode="born").task.model == "volume_rr"
    # 显式 model 不随物体模式变
    s_x = make_system(SystemConfig(condition="M3", object_mode="full", voxel_d=VOX), "parity", n=2, model="cm_rr")
    assert s_x.derive(object_mode="born").task is s_x.task


# ============================================================ 审查 R2-INT-2：logdet 钳位平局
def test_logdet_clip_ties_count_as_wrong(m3_full):
    s = m3_full.derive(detector="logdet")
    p = s.random_params(np.random.default_rng(35))
    m = s.forward(s.phases_from_params(p), noisy=False)
    assert not m.flags.any()
    lo, _ = s.detector.range_dbm
    # 样本 0（x = 0，偶）的两个探头都钳在量程下沿 → y1 == y2；旧规则 y1 > y2 判"偶"并计对
    s.detector.offset_db = lo - float(np.max(m.readings_dbm[0])) - 0.5
    y = s.clean_readings(m)
    tie = y[:, 0] == y[:, 1]
    assert tie[0] and s.labels[0] == 0
    strict = (y[:, 0] > y[:, 1]).astype(int)
    ev = s.evaluate(p, np.random.default_rng(0), n_noise=0)
    assert ev["acc_clean"] == pytest.approx(np.mean((strict == s.labels) & ~tie))
    assert ev["acc_clean"] < np.mean(strict == s.labels)
    assert ev["acc_clean"] <= ev["acc_clean_ideal"] and not ev["all_correct"]
    assert s.clean_metrics(p)["acc_clean"] == pytest.approx(ev["acc_clean"])
    # 钳位单调 ⇒ 平局未决后检波器读数的准确率不超过理想读数（门 G1 不会被钳位"补"出来）：M0 + Born 扫 offset
    s0 = _sys("M0", "born", n=3, detector="logdet")
    rng = np.random.default_rng(38)
    for _ in range(4):
        p0 = s0.random_params(rng)
        y_id = s0.forward(s0.phases_from_params(p0), noisy=False).readings_dbm
        for off in np.linspace(lo - y_id.max() - 1.0, -10.0 - y_id.min() + 1.0, 9):
            s0.detector.offset_db = float(off)
            e0 = s0.evaluate(p0, None, n_noise=0)
            assert e0["acc_clean"] <= e0["acc_clean_ideal"] + 1e-12 and not e0["all_correct"]


# ============================================================ 审查 R2-INT-3：loc3bit 部分标记
def test_loc3bit_partial_flags_evaluate_does_not_raise():
    """部分样本被标记、未标记样本里某一位只剩一类时，evaluate 不再因在子集上重算自动阈值而 ValueError；
    margin_db_unflagged = 未标记样本上 min sgn·(y − t)，t 由全部样本求出（与 predict 一致）。"""
    s = make_system(SystemConfig(condition="M1", object_mode="born", preset="flat0", isolation_db=-22.0,
                                 kappa0_phase_deg=0.0, voxel_d=VOX), "loc3bit")
    rng = np.random.default_rng(0)
    for _ in range(400):
        p = s.random_params(rng)
        m = s.forward(s.phases_from_params(p), noisy=False)
        ok = ~m.flags
        if ok.any() and m.flags.any():
            bits = (s.labels[ok][:, None] >> np.arange(3)) & 1
            if np.any(bits.all(0) | (bits == 0).all(0)):
                break
    else:
        pytest.fail("没有找到使某一位只剩一类的部分标记码本")
    with pytest.raises(ValueError):
        s.task.margin_db(m.readings_dbm[ok], s.labels[ok])            # 旧做法
    ev = s.evaluate(p, np.random.default_rng(1), n_noise=2)
    y = m.readings_dbm
    t = s.task.thresholds(y)
    sgn = 2.0 * ((s.labels[:, None] >> np.arange(3)) & 1) - 1.0
    assert ev["margin_db_unflagged"] == pytest.approx(np.min((sgn * (y - t))[ok]))
    assert ev["margin_db"] is None and ev["oscillating_run"] and ev["all_correct"] is False
    assert len(ev["bit_acc_clean"]) == 3


# ============================================================ 审查 S0-0 / S2-1 / S2-2 / S2-3 增补
def test_bare_loop_gain_helpers_and_spec_pitfall():
    """S2-3：默认 30 dB PA、2.5+2.5 dB 损耗下 κ0 = −30 dB → −5 dB；INTEGRATION G 实验 D 的 −25 dB 恰好 0 dB，
    无物体样本 K = κ_eff 处在临界上（线性中继必被标记）。"""
    from ris_nlsim.system import bare_loop_gain_db, isolation_for_loop_gain_db

    cfg = SystemConfig()
    assert bare_loop_gain_db(cfg) == pytest.approx(-5.0)
    assert bare_loop_gain_db(dataclasses.replace(cfg, isolation_db=-25.0)) == pytest.approx(0.0)
    for g in (-15.0, -3.0, 0.0, 10.0):
        assert bare_loop_gain_db(dataclasses.replace(cfg, isolation_db=isolation_for_loop_gain_db(cfg, g))) == \
            pytest.approx(g)
    s = make_system(SystemConfig(condition="M1", object_mode="born", voxel_d=VOX, isolation_db=-25.0), "parity", n=2)
    m = s.forward(s.phases_from_params(s.random_params(np.random.default_rng(0))), noisy=False)
    assert abs(abs(s.small_signal_gain() * s.params.kappa0) - 1.0) < 1e-12
    assert m.flags[0] and np.isclose(m.loop_gain_ss[0], 1.0, atol=1e-12)          # x = 000：K = κ_eff


def test_osc_criterion_flags_measurement_fields_and_torch_consistency():
    """S0-0：cfg.osc_criterion 选择标记判据；Measurement / evaluate 总报三种判据下的量；TorchSystem 同一标记、同一损失。
    |G_v κ_eff| = +6 dB、φ = 90°、聚焦码本：M4（+10 dB）CW 稳定但相位稳健判据不稳定；backoff +20 dB 时只有小信号判据不稳定。"""
    from ris_nlsim.system import isolation_for_loop_gain_db

    cfg = SystemConfig(condition="M4", object_mode="full", voxel_d=VOX)
    cfg = dataclasses.replace(cfg, isolation_db=isolation_for_loop_gain_db(cfg, 6.0))
    base = make_system(cfg, "parity", n=2)
    p = base.focus_params(0)
    expect = {("cw", 10.0): False, ("robust", 10.0): True, ("small_signal", 10.0): True,
              ("cw", 20.0): False, ("robust", 20.0): False, ("small_signal", 20.0): True}
    for (crit, bo), flagged in expect.items():
        s = base.derive(osc_criterion=crit, backoff_db=bo)
        assert s._rs is base._rs
        m = s.forward(s.phases_from_params(p), noisy=False)
        assert m.ss_unstable.all() and m.robust_unstable.all() == (bo == 10.0)
        assert np.all(m.flags == flagged), (crit, bo, m.flags)
        assert np.allclose(m.d, s.direct_field()) and m.C.shape == (s.B, 2, 1)
        ev = s.evaluate(p, np.random.default_rng(0), n_noise=2)
        assert ev["osc_criterion"] == crit and ev["oscillating_run"] == flagged
        assert ev["n_ss_unstable"] == s.B and ev["ss_unstable_run"] is True
        assert ev["robust_unstable_run"] == (bo == 10.0) and ev["loop_gain_robust_max"] >= ev["loop_gain_op_max"]
        assert (ev["acc_clean_valid"] is None) == flagged
        ts = s.to_torch()
        assert ts.osc_criterion == crit
        with torch.no_grad():
            out = ts.forward(ts.as_phases(m.phases))
            lt = float(ts.loss(ts.as_phases(m.phases)))
        assert np.array_equal(np.asarray(out["flags"]), m.flags)
        assert lt == pytest.approx(s.loss_from(m), rel=1e-9)
    with pytest.raises(ValueError):
        SystemConfig(osc_criterion="strict")
    # M0：环路量为 0、无标记、d/C 为 None；evaluate 的环路键为 None
    m0 = base.derive(condition="M0")
    mm = m0.forward(m0.phases_from_params(m0.random_params(np.random.default_rng(1))), noisy=False)
    assert not mm.ss_unstable.any() and not mm.robust_unstable.any() and mm.d is None and mm.C is None
    ev0 = m0.evaluate(m0.random_params(np.random.default_rng(1)), np.random.default_rng(0), n_noise=0)
    assert ev0["n_ss_unstable"] is None and ev0["loop_gain_robust_max"] is None


@pytest.mark.parametrize("mode", ["full", "born", "intra"])
def test_direct_field_is_probe_field_without_relay_or_surface(mode):
    """S2-2：direct_field = 方案 §2 的 d_k。中继：= Measurement.d，z − d = C·g(u)；
    M0：= 去掉无源面（α_P → 0）的场景直接求解的探头场，z − d 即无源面的全部贡献。"""
    from ris_nlsim.scene import NodeSet
    from ris_nlsim.patterns import element_pattern

    s = make_system(SystemConfig(condition="M3", object_mode=mode, voxel_d=VOX), "parity", n=2)
    ph = s.phases_from_params(s.random_params(np.random.default_rng(0)))
    m = s.forward(ph, noisy=False)
    d = s.direct_field()
    assert d.shape == (s.B, 2) and np.allclose(d, m.d, rtol=1e-13)
    assert np.allclose(m.z - d, np.einsum("bkr,br->bk", m.C, m.g), rtol=1e-12, atol=1e-30)
    assert np.allclose(s.direct_field([3, 1]), d[[3, 1]])
    m0 = s.derive(condition="M0")
    sc_np = dataclasses.replace(m0.scene, passive=NodeSet.empty(element_pattern()))
    ref = np.stack([solver.solve_scene(sc_np, smp.objects, None, mode)[:, 0] for smp in m0.task.samples])
    assert np.allclose(m0.direct_field(), m0.s_b[:, None] * ref, rtol=1e-9)


def test_loc3bit_fixed_thresholds_on_held_out_set():
    """S2-1：留出集评估用训练集阈值（calibrate_thresholds → evaluate(thresholds_db=...)），不再从测试标签重算；
    评估后恢复任务原设置；非 loc3bit / 长度不符报错。"""
    cfg = SystemConfig(condition="M3", object_mode="full", voxel_d=VOX)
    s = make_system(cfg, "loc3bit")
    p = s.random_params(np.random.default_rng(0))
    t_train = s.calibrate_thresholds(p)
    assert t_train.shape == (3,)
    ev_auto = s.evaluate(p, np.random.default_rng(1), n_noise=2)
    assert ev_auto["thresholds_source"] == "auto" and np.allclose(ev_auto["thresholds_db"], t_train)
    # "测试集"：同一几何、带距离/功率抖动；阈值固定为训练集阈值
    test = make_system(cfg, "loc3bit", dist_jitter=LAMBDA / 8, power_jitter_db=3.0, rng=np.random.default_rng(5))
    ev = test.evaluate(p, np.random.default_rng(1), n_noise=3, thresholds_db=t_train)
    assert ev["thresholds_source"] == "given" and np.allclose(ev["thresholds_db"], t_train)
    assert test.task.thresholds_db == "auto"
    y = test.clean_readings(test.forward(test.phases_from_params(p), noisy=False))
    pred = ((y > t_train[None, :]).astype(int) << np.arange(3)[None, :]).sum(axis=1)
    assert ev["acc_clean"] == pytest.approx(np.mean(pred == test.labels))
    # 固定阈值下判决与标签无关（auto 阈值才依赖被评估样本的标签）
    test.task.thresholds_db = t_train
    assert np.array_equal(test.task.predict(y, test.labels[::-1]), pred)
    test.task.thresholds_db = "auto"
    t_test_auto = test.calibrate_thresholds(p)
    assert not np.allclose(t_test_auto, t_train)
    with pytest.raises(ValueError):
        test.evaluate(p, np.random.default_rng(1), n_noise=0, thresholds_db=[0.0, 1.0])
    with pytest.raises(ValueError):
        make_system(cfg, "parity", n=2).evaluate(np.zeros(50, int), np.random.default_rng(1), n_noise=0,
                                                   thresholds_db=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        make_system(cfg, "parity", n=2).calibrate_thresholds(np.zeros(50, int))


def test_localization_jitter_scales_feed_and_rebuilds_on_derive():
    """S2-0：LocalizationTask 的 dist_jitter / power_jitter_db 与 LetterTask 同一约定；系统按 meta 缩放馈源功率，
    derive 改 ROI 时用构造前 rng 状态重建出同一组抖动。"""
    cfg = SystemConfig(condition="M3", object_mode="full", voxel_d=VOX)
    s = make_system(cfg, "loc3bit", dist_jitter=LAMBDA / 4, power_jitter_db=6.0, rng=np.random.default_rng(3))
    pdb = s.task.power_db()
    assert np.all(np.abs(pdb) <= 6.0) and np.ptp(pdb) > 1.0
    assert np.allclose(s.s_b, s.s_feed * 10 ** (pdb / 20))
    d2 = s.derive(roi_offset=(0.6, 0.0, 0.0))
    assert d2.task is not s.task and np.allclose(d2.task.power_db(), pdb)
    assert np.allclose([x.meta["dist_shift"] for x in d2.task.samples], [x.meta["dist_shift"] for x in s.task.samples])
