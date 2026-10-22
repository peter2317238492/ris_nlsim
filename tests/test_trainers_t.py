"""INTEGRATION T：新默认训练器 sign_acc（run_dspsa）与 ILS（run_ils）——与 _scratch 探索插件逐位一致、预算不超、
默认训练器能解 parity2_born、旧变体不变；判决单元与可测判对优先目标；bench 参照池化全部缓存种子。"""
import os
import shutil

import numpy as np
import pytest

from ris_nlsim import train
from ris_nlsim.experiments import bench_trainers as BT
from ris_nlsim.ils import ils_bits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILY = os.path.join(ROOT, "_scratch", "trainers", "dspsa_family.py")
ALT = os.path.join(ROOT, "_scratch", "trainers", "alt_optimizers.py")
SCRATCH = os.path.join(ROOT, "_scratch", "pytest_tmp", "test_trainers_t")


def _plugin(path):
    if not os.path.isfile(path):
        pytest.skip(f"探索插件不存在：{path}")
    return BT.load_plugin(path)


# ------------------------------------------------------------ 与探索插件逐位一致
@pytest.mark.parametrize("task", ["parity2_born", "letters", "focus"])
def test_sign_acc_bitwise_equals_dspsa_family_df_u_nores(task):
    fam = _plugin(FAMILY)
    for seed in (0, 1):
        a = BT.run_trainer(BT.TRAINERS["dspsa_sign_acc"], task, seed, 120, name="new", curve_points=10, n_noise_eval=0)
        b = BT.run_trainer(fam.TRAINERS["df_u_nores"], task, seed, 120, name="plugin", curve_points=10,
                           n_noise_eval=0)
        assert a["params"] == b["params"] and a["curve"] == b["curve"] and a["n_meas_used"] == b["n_meas_used"] == 120
    # train.run_dspsa（算法 / 噪声随机流分开）与插件经 MeasurementBudget 同一码本
    s = BT.build_system(task, 0)
    r = train.run_dspsa(s, 60, np.random.default_rng(5), noise_rng=np.random.default_rng(6), n_noise_eval=0,
                        eval_every=60)
    meas = BT.MeasurementBudget(s, 120, np.random.default_rng(6))
    x = fam.TRAINERS["df_u_nores"](meas, np.random.default_rng(5))
    assert r["variant"] == "sign_acc" and np.array_equal(r["params"], x) and r["n_measurements"] == 120
    assert r["train_objective"] == ("loss" if task == "focus" else "accuracy_first")


@pytest.mark.parametrize("task", ["parity2_born", "letters"])
def test_ils_bitwise_equals_alt_optimizers(task):
    alt = _plugin(ALT)
    for name, ref in (("ils", "ao_ils_acc"), ("ils_extras", "ao_ils4_k0_drift")):
        a = BT.run_trainer(BT.TRAINERS[name], task, 0, 150, name=name, curve_points=10, n_noise_eval=0)
        b = BT.run_trainer(alt.TRAINERS[ref], task, 0, 150, name=ref, curve_points=10, n_noise_eval=0)
        assert a["params"] == b["params"] and a["curve"] == b["curve"], name
    s = BT.build_system(task, 0)
    for extras, ref in ((False, "ao_ils_acc"), (True, "ao_ils4_k0_drift")):
        r = train.run_ils(s, 150, np.random.default_rng(5), noise_rng=np.random.default_rng(6), n_noise_eval=0,
                          extras=extras)
        meas = BT.MeasurementBudget(s, 150, np.random.default_rng(6))
        x = alt.TRAINERS[ref](meas, np.random.default_rng(5))
        assert np.array_equal(r["params"], x) and r["n_measurements"] == 150


def test_old_subset_variant_unchanged_vs_independent_plugin_baseline():
    """旧默认 subset（a="auto"）与 pocket 版本：经 bench 包装的库实现与插件里独立重写的基线 df_base / df_base_pk 同一码本。"""
    fam = _plugin(FAMILY)
    for task in ("parity2_born", "letters"):
        for mine, ref in (("dspsa_subset", "df_base"), ("dspsa_subset_pocket", "df_base_pk")):
            a = BT.run_trainer(BT.TRAINERS[mine], task, 0, 120, name=mine, curve_points=4, n_noise_eval=0)
            b = BT.run_trainer(fam.TRAINERS[ref], task, 0, 120, name=ref, curve_points=4, n_noise_eval=0)
            assert a["params"] == b["params"], (task, mine)


# ------------------------------------------------------------ 预算与默认行为
def test_budget_never_exceeded():
    for name in ("dspsa_sign_acc", "ils", "ils_extras"):
        for budget in (7, 101):
            r = BT.run_trainer(BT.TRAINERS[name], "parity2_born", 1, budget, name=name, curve_points=4, n_noise_eval=0)
            assert not r["overran"] and not r["fallback_to_report"] and r["n_meas_used"] <= budget
            if name == "ils" or (name == "ils_extras" and budget > 12):    # extras 预留 4×3 次终选赛测量
                assert r["n_meas_used"] == budget
    s = BT.build_system("parity2_born", 0)
    r = train.run_dspsa(s, 13, np.random.default_rng(0), n_noise_eval=0, eval_every=5)
    assert r["n_measurements"] == 26 and len(r["history"]["loss_plus"]) == 13
    assert [c["iter"] for c in r["curve"]] == [0, 5, 10, 13]
    ri = train.run_ils(s, 37, np.random.default_rng(0), n_noise_eval=0, eval_every=10)
    assert ri["n_measurements"] == 37 and ri["curve"][-1]["n_meas"] == 37 and ri["variant"] == "ils"
    assert ri["curve"][-1]["acc_clean"] == pytest.approx(ri["eval"]["acc_clean"])


def test_default_trainer_and_ils_solve_parity2_born_within_1000_measurements():
    """M3 + Born（水，L90）2 位奇偶：新默认 run_dspsa（500 迭代 = 1000 次测量）与 run_ils（1000 次测量）全对。"""
    from ris_nlsim.system import SystemConfig, make_system

    s = make_system(SystemConfig(preset="L90", condition="M3", object_mode="born", material="water"), "parity", n=2)
    for seed in (0, 2):
        r = train.run_dspsa(s, 500, np.random.default_rng(seed), n_noise_eval=0, eval_every=250)
        assert r["variant"] == "sign_acc" and r["n_measurements"] == 1000
        assert r["eval"]["all_correct"] is True and r["eval"]["margin_db"] > 0.006, seed
    for seed in (0, 1):
        r = train.run_ils(s, 1000, np.random.default_rng(seed), n_noise_eval=0)
        assert r["n_measurements"] == 1000 and r["eval"]["all_correct"] is True and r["eval"]["margin_db"] > 0.5


def test_sign_acc_pocket_and_decision_outputs():
    s = BT.build_system("parity2_born", 0)
    r = train.run_dspsa(s, 30, np.random.default_rng(3), n_noise_eval=0, eval_every=10, pocket=True)
    pk = r["pocket"]
    assert np.array_equal(pk["params"], r["params"]) and pk["eval"] is r["eval"]
    assert 1 <= pk["at_measurement"] <= 60 and pk["loss_measured"] == r["pocket_key"]
    assert r["decision_params"].shape == (50,) and len(r["tau_final"]) == 1
    assert r["curve"][-1]["acc_clean"] == pytest.approx(r["eval"]["acc_clean"])
    assert all("pocket_acc_clean" in c and "n_meas" in c for c in r["curve"])
    with pytest.raises(ValueError):
        train.run_dspsa(s.derive(condition="M0", passive_levels=4), 2, np.random.default_rng(0), variant="sign_acc")


# ------------------------------------------------------------ 判决单元与可测目标
def test_decision_units_min_equals_sample_margins():
    from ris_nlsim.constants import LAMBDA
    from ris_nlsim.tasks import LetterTask, LocalizationTask, ParityTask

    rng = np.random.default_rng(0)
    roi = np.array([0.5, 0.0, 0.0])
    tasks = [ParityTask(1, roi, voxel_d=LAMBDA / 6), LetterTask(roi), LocalizationTask(roi, voxel_d=LAMBDA / 6)]
    for t in tasks:
        y = rng.normal(-40.0, 3.0, size=(t.n_samples, t.n_probes))
        M = t.decision_units(y)
        J = {"binary": 1, "multiclass": t.n_probes - 1, "loc3bit": t.n_probes}[t.kind]
        assert M.shape == (t.n_samples, J)
        assert np.array_equal(np.min(M, axis=1), t.sample_margins_db(y))       # 逐位相等（单调舍入）
        ok = np.all(M > 0, axis=1)
        assert np.array_equal(ok, t.decide(y) == t.labels)


def test_accuracy_objective_equals_oracle_objective_on_clean_readings():
    s = BT.build_system("letters", 0)
    f_or = train.oracle_objective(s, "accuracy")
    f_me = train.measurable_objective(s, "accuracy")
    for seed in range(3):
        x = s.random_params(np.random.default_rng(seed))
        m = s.forward(s.phases_from_params(x), noisy=False)
        assert f_me(s.clean_readings(m), m.flags) == f_or(x)
    obj = train.AccuracyObjective.from_task(s.task, s.labels, tie="unit", adaptive=True)
    assert obj.tau_now() == 1.0
    m = s.forward(s.phases_from_params(s.random_params(np.random.default_rng(9))), np.random.default_rng(1))
    obj.observe(m.readings_dbm, m.flags)
    tau = obj.tau_now()
    assert tau.shape == (s.n_probes - 1,) and np.all(tau >= 1e-3)
    assert np.floor(-obj(m.readings_dbm, m.flags) + 0.5) == np.sum(s.task.decide(m.readings_dbm) == s.labels)


def test_ils_bits_respects_budget_with_pure_callbacks():
    """纯算法层：可分离目标上 ILS 找到全局最优，测量数恰为 n_meas。"""
    p, target = 12, np.random.default_rng(0).integers(0, 2, size=12)
    n = {"k": 0}

    def measure(x):
        n["k"] += 1
        return np.array([[float(np.sum(x != target))]]), np.zeros(1, dtype=bool)

    res = ils_bits(measure, lambda y, fl: float(y[0, 0]), p, 200, np.random.default_rng(1))
    assert n["k"] == res["n_measurements"] == 200 and np.array_equal(res["params"], target) and res["J"] == 0.0
    n["k"] = 0
    res = ils_bits(measure, lambda y, fl: float(y[0, 0]), p, 200, np.random.default_rng(1), extras=True)
    assert n["k"] == 200 and np.array_equal(res["params"], target)


# ------------------------------------------------------------ bench：参照池化全部缓存种子
def test_build_references_pools_all_cached_seeds():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    cache = os.path.join(SCRATCH, "cache")
    docs = [BT.oracle_refs("parity2_born", s, cache_dir=cache, n_restarts=1, max_sweeps=1) for s in (0, 1)]
    assert docs[0]["instance_hash"] == docs[1]["instance_hash"]
    only = BT.build_references(docs[:1])
    pooled = BT.build_references(docs[:1], cache)
    assert only[("parity2_born", 0)]["n_pooled_seeds"] == 1
    assert pooled[("parity2_born", 0)]["n_pooled_seeds"] == 2 and pooled[("parity2_born", 0)]["n_pooled_docs"] == 2
    acc = [d["oracle"]["accuracy"]["acc"] for d in docs] + [d["oracle"]["loss"]["acc"] for d in docs]
    assert pooled[("parity2_born", 0)]["ref_acc"] >= max(acc) - 1e-12
    assert BT.build_references(docs, cache)[("parity2_born", 1)]["n_pooled_seeds"] == 2    # 不重复计数
    assert {"dspsa_sign_acc", "ils", "ils_extras"} <= set(BT.BUILTIN_TRAINERS)
