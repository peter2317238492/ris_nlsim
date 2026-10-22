"""实验 C（exp_c_invariance）冒烟测试：抖动构造、held-out 测试集、定位阈值固定、端到端微型配置（单进程，< 30 s）。"""
import json
import os

import numpy as np
import pytest

from ris_nlsim.constants import LAMBDA
from ris_nlsim.experiments import common
from ris_nlsim.experiments import exp_c_invariance as C
from ris_nlsim.system import PhysicalSystem, SystemConfig


def test_parse_condition_labels():
    assert C.parse_condition("M1@M3") == {"condition": "M1", "backoff_db": 0.0}
    assert C.parse_condition("M1@M4") == {"condition": "M1", "backoff_db": 10.0}
    assert C.parse_condition("M4") == {"condition": "M4"}
    assert C.parse_condition("Rapp@+20") == {"condition": "M4", "backoff_db": 20.0}      # 审查 R-C3
    assert "Rapp@+20" in C.ALL_CONDITIONS and "Rapp@+20" in C.SAT_CONDS
    with pytest.raises(ValueError):
        C.parse_condition("M3@M4")


def test_oracle_trainer_uses_accuracy_objective():
    """审查 R-C1：主 oracle 是准确率目标（最好可达准确率），softplus 损失 oracle 只作为标注的次要训练器。
    准确率目标的坐标下降收敛后，任何单比特翻转都不能增加训练集判对数（判对数 + ½·mean tanh 的字典序局部最优）。"""
    assert C.ORACLE_OBJECTIVE == {"oracle": "accuracy", "oracle_loss": "loss"}
    assert C.build_config(C.parse_args(["--quick"]))["trainers"] == ["oracle", "dspsa"]
    cfg = SystemConfig(condition="M1", voxel_d=LAMBDA / 10)
    s = PhysicalSystem(cfg, C.make_c_task(cfg, "loc3bit", 1, "none", None))
    job = {"oracle_restarts": 1, "oracle_sweeps": 30}
    tr = C._train(s, "oracle", np.random.default_rng(3), job)
    assert tr["info"]["objective"] == "accuracy"
    acc = tr["train_eval"]["acc_clean"]
    p = tr["params"]
    for i in range(p.size):
        q = p.copy()
        q[i] = 1 - q[i]
        s.task.thresholds_db = "auto"
        assert s.evaluate(q, np.random.default_rng(0), n_noise=0)["acc_clean"] <= acc + 1e-12
    tl = C._train(s, "oracle_loss", np.random.default_rng(3), job)
    assert tl["info"]["objective"] == "model"


def test_saturation_fraction_reports_true_operating_point():
    """审查 R-C3：sat_frac = 闭环 PA 输入 ≥ P1dB 的样本比例；聚焦码本下 Rapp@+20 全饱和、M2（−10 dB）全不饱和。"""
    out = {}
    for cond in ("Rapp@+20", "M2"):
        cfg = SystemConfig(**C.parse_condition(cond))
        s = PhysicalSystem(cfg, C.make_c_task(cfg, "letter", 1, "none", None))
        p = s.focus_params()
        m = s.forward(s.phases_from_params(p), noisy=False)
        ok = ~m.flags
        assert C.saturation_fraction(s, p) == pytest.approx(np.mean(np.abs(m.u[ok]) ** 2 >= s.p1db_in_w))
        out[cond] = C.saturation_fraction(s, p)
    assert out["Rapp@+20"] == 1.0 and out["M2"] == 0.0


def _ptab(task, cond, jitter, metric, vals, trainer="oracle"):
    return [{"task": task, "trainer": trainer, "cond": cond, "jitter": jitter, "seed": i, metric: v}
            for i, v in enumerate(vals)]


def test_verdict_tests_localization_on_none_and_dist_and_flags_unsaturated():
    """审查 R-C2：定位的"饱和更差"只在 none / dist 检验，power / both 标为 AGC 检验；未饱和的条件注明不检验 AGC。"""
    rows = (_ptab("loc3bit", "M4", "none", "d_test_acc", [-0.25, -0.375, -0.25])
            + _ptab("loc3bit", "M4", "both", "d_test_acc", [0.0, 0.05, -0.05])
            + _ptab("letter", "M4", "both", "d_inv_loss", [-0.1, 0.1, 0.0]))
    pt = common.aggregate(rows, ["task", "trainer", "cond", "jitter"], ["d_test_acc", "d_inv_loss", "d_zs_inv_loss"])
    table = [{"task": "loc3bit", "trainer": "oracle", "cond": "M4", "jitter": "none", "test_acc_clean_mean": 0.4,
              "test_sat_frac_mean": 0.1},
             {"task": "loc3bit", "trainer": "oracle", "cond": "M1@M3", "jitter": "none", "test_acc_clean_mean": 0.7}]
    txt = C.verdict_text(pt, table)
    line_none = next(l for l in txt.splitlines() if "loc3bit / oracle / M4 / none" in l)
    assert "d_test_acc < 0" in line_none and "支持预测" in line_none and "未饱和" in line_none
    line_both = next(l for l in txt.splitlines() if "loc3bit / oracle / M4 / both" in l)
    assert "AGC" in line_both and "d_test_acc > 0" in line_both and "不显著" in line_both
    assert "letter / oracle / M4 / both" in txt


def test_apply_jitter_ranges_shift_along_normal_and_common_random_numbers():
    cfg = SystemConfig(condition="M1")
    base = C.make_c_task(cfg, "letter", 4, "none", None)
    both = C.make_c_task(cfg, "letter", 4, "both", np.random.default_rng(7))
    power = C.make_c_task(cfg, "letter", 4, "power", np.random.default_rng(7))
    dist = C.make_c_task(cfg, "letter", 4, "dist", np.random.default_rng(7))
    pb, pp = both.power_db(), power.power_db()
    assert np.all(np.abs(pb) <= 6.0) and np.ptp(pb) > 0
    assert np.allclose(pb, pp)                                  # 公共随机数：power 模式 = both 模式的功率抖动
    assert np.allclose(dist.power_db(), 0.0)
    for s0, sb, sp, sd in zip(base.samples, both.samples, power.samples, dist.samples):
        shift = sb.meta["dist_shift"]
        assert abs(shift) <= LAMBDA / 4 + 1e-15
        assert sd.meta["dist_shift"] == pytest.approx(shift)
        d = sb.objects.pos - s0.objects.pos                     # 只沿 +x（字母平面法向）平移
        assert np.allclose(d[:, 0], shift) and np.allclose(d[:, 1:], 0.0)
        assert np.allclose(sp.objects.pos, s0.objects.pos)      # power 模式不平移
        assert np.allclose(sb.objects.alpha, s0.objects.alpha)
    assert both.make_spec is None


def test_heldout_test_set_uses_fresh_draws():
    cfg = SystemConfig(condition="M1")
    tr = C.make_c_task(cfg, "letter", 3, "both", common.seed_rng(C.EXP, "letter", 0, "train"))
    te = C.make_c_task(cfg, "letter", 3, "both", common.seed_rng(C.EXP, "letter", 0, "test"))
    assert not np.any(np.isclose(tr.power_db()[:, None], te.power_db()[None, :]))
    # 同一随机流可复现
    tr2 = C.make_c_task(cfg, "letter", 3, "both", common.seed_rng(C.EXP, "letter", 0, "train"))
    assert np.allclose(tr.power_db(), tr2.power_db())


def test_loc_thresholds_fixed_on_test_and_reset_for_training():
    cfg = SystemConfig(condition="M1", voxel_d=LAMBDA / 10)
    s = PhysicalSystem(cfg, C.make_c_task(cfg, "loc3bit", 1, "none", None))
    p = s.random_params(np.random.default_rng(0))
    thr = C.train_thresholds(s, p)
    assert thr.shape == (3,) and np.all(np.isfinite(thr))
    ev = C.evaluate_on(s, p, np.full(3, 1e9), np.random.default_rng(1), n_noise=0)   # 阈值极大 → 全判 0 号格
    assert ev["acc_clean"] == pytest.approx(1 / 8)
    assert isinstance(s.task.thresholds_db, np.ndarray)
    ev_auto = C.evaluate_on(s, p, thr, np.random.default_rng(1), n_noise=0)          # 训练集阈值 = 自动阈值
    s.task.thresholds_db = "auto"
    assert ev_auto["acc_clean"] == pytest.approx(s.evaluate(p, np.random.default_rng(1), n_noise=0)["acc_clean"])
    s.task.thresholds_db = np.zeros(3)
    tr = C._train(s, "oracle", np.random.default_rng(2), {"oracle_restarts": 1, "oracle_sweeps": 1})
    assert s.task.thresholds_db == "auto" and tr["params"].shape == (50,)


def test_tiny_end_to_end(tmp_path):
    args = C.parse_args(["--quick", "--seeds", "1", "--conditions", "M1@M3", "Rapp@+20", "--jitters", "both",
                         "--n-iter", "20", "--oracle-restarts", "1"])
    cfg = C.build_config(args)
    cfg.update(oracle_sweeps=1, n_noise=2, n_rep_train={"letter": 1, "loc3bit": 1},
               n_rep_test={"letter": 2, "loc3bit": 1})
    res = C.run(cfg, workers=1, out=str(tmp_path))
    rows = res["rows"]
    assert not [r for r in rows if "error" in r]
    assert len(rows) == 2 * 2 * 2 * 2                            # 任务 × 条件 × (none, both) × 训练器
    for name in ("runs.jsonl", "summary.csv", "summary.md", "config.json", "paired.csv", "accuracy_by_jitter.png",
                 "invariance_loss.png"):
        assert os.path.exists(os.path.join(tmp_path, name)), name
    for r in rows:
        assert r["p_feed_dbm"] is not None and r["voxel_d"] > 0 and r["test_pa_in_rel_p1db_db_mean"] is not None
        assert 0.0 <= r["test_sat_frac"] <= 1.0
        assert r["objective"] == {"oracle": "accuracy", "dspsa": "measurable"}[r["trainer"]]
        if r["cond"] == "M1@M3":
            assert r["test_sat_frac"] == 0.0 or r["test_pa_in_rel_p1db_db_max"] >= 0.0
        if r["jitter"] == "none":
            assert not r["heldout"] and r["n_train"] == r["n_test"] and r["inv_loss"] == 0.0
        else:
            assert r["heldout"]
            assert r["inv_loss"] == pytest.approx(r["acc_none"] - r["test_acc_clean"])
            assert r["zs_inv_loss"] == pytest.approx(r["acc_none"] - r["zs_test_acc_clean"])
            assert max(abs(v) for v in r["test_power_db_range"]) <= 6.0
        if r["task"] == "loc3bit":
            assert len(r["thresholds_db"]) == 3
    md = open(os.path.join(tmp_path, "summary.md"), encoding="utf-8").read()
    assert "held-out" in md and "M1@M3" in md and "Rapp@+20" in md and "sat" in md and "AGC" in md
    with open(os.path.join(tmp_path, "config.json"), encoding="utf-8") as f:
        assert json.load(f)["exp"] == C.EXP
