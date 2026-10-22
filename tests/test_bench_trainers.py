"""bench_trainers 冒烟测试：测量预算包装的计数 / 一致性、内置 DSPSA 经包装与 train.run_dspsa 逐位一致、插件加载、
超预算回退、oracle 缓存复用与端到端输出（全部写在 E: 的 _scratch 下，不用 C: 的 tmp_path）。"""
import json
import os
import shutil

import numpy as np
import pytest

from ris_nlsim import train
from ris_nlsim.experiments import bench_trainers as BT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(ROOT, "_scratch", "pytest_tmp", "test_bench_trainers")
TASK = "parity2_born"


@pytest.fixture(scope="module")
def scratch():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)
    return SCRATCH


@pytest.fixture(scope="module")
def system():
    return BT.build_system(TASK, 0)


def test_measurement_budget_counts_and_exposes_only_hardware_info(system):
    meas = BT.MeasurementBudget(system, 3, np.random.default_rng(1), task_name=TASK)
    assert meas.n_params == system.n_params == 50 and meas.kind == "binary" and meas.n_probes == 2
    assert np.array_equal(meas.labels, system.labels) and np.all(meas.levels == 2)
    x = meas.random_params(np.random.default_rng(2))
    meas.report(x)                                   # 不计数
    y, fl = meas.measure(x)
    assert y.shape == (system.B, 2) and fl.shape == (system.B,) and fl.dtype == bool
    f = meas.measure_loss(x)
    assert np.isfinite(f) and meas.n_used == 2 and meas.remaining == 1
    meas.measure(meas.focus_params())                # focus_params 不计数，measure 计数
    with pytest.raises(BT.BudgetExhausted):
        meas.measure_loss(x)
    assert meas.n_used == 3
    with pytest.raises(ValueError):
        meas.report(np.full(50, 2))                  # 越界电平
    assert not any(k in vars(meas) for k in ("system", "rng", "_system"))


def test_measure_loss_equals_loss_from_readings_on_noiseless_system(system):
    quiet = system.derive(detector="ideal", pa_noise=False)
    meas = BT.MeasurementBudget(quiet, 10, np.random.default_rng(0))
    x = meas.random_params(np.random.default_rng(3))
    y, fl = meas.measure(x)
    f = meas.measure_loss(x)
    assert f == pytest.approx(meas.loss_from_readings(y, fl), abs=1e-12)
    assert f == pytest.approx(quiet.loss(x, None, noisy=False, model_penalties=False), abs=1e-12)
    # 换 τ 只改损失、不改判决
    assert meas.loss_from_readings(y, fl, tau=0.1) != pytest.approx(f)
    assert meas.accuracy_from_readings(y, fl) == pytest.approx(BT.clean_eval(quiet, x)["acc"])


def test_dspsa_subset_through_wrapper_reproduces_run_dspsa(system):
    budget = 60
    n_iter = (budget - BT.DSPSA_AUTO_GAIN_MEAS) // 2
    rng = np.random.default_rng(11)
    meas = BT.MeasurementBudget(system, budget, rng)  # 噪声流 = 训练器流（与 run_dspsa 同一个 Generator）
    x = BT.dspsa_subset(meas, rng)
    ref = train.run_dspsa(system, n_iter, np.random.default_rng(11), n_noise_eval=0, variant="subset")
    assert np.array_equal(x, ref["params"]) and meas.n_used == ref["n_measurements"] == budget
    assert len(meas.reports) == n_iter + 1


def test_run_trainer_overrun_falls_back_to_last_report_and_curve():
    def greedy_overrun(meas, rng):
        x = meas.random_params(rng)
        while True:                                  # 故意不看预算
            meas.measure_loss(x)
            meas.report(x)

    r = BT.run_trainer(greedy_overrun, TASK, 0, 12, name="overrun", curve_points=4, n_noise_eval=0)
    assert r["overran"] and r["fallback_to_report"] and r["n_meas_used"] == 12
    assert r["curve"]["n"] == [0, 3, 6, 9, 12]
    assert r["curve"]["acc"][0] is None and all(a is not None for a in r["curve"]["acc"][1:])
    r2 = BT.run_trainer(BT.focus_init, TASK, 0, 12, name="focus_init", curve_points=4, n_noise_eval=0)
    assert r2["n_meas_used"] == 0 and len(set(r2["curve"]["acc"])) == 1
    with pytest.raises(RuntimeError):
        BT.run_trainer(lambda m, g: None, TASK, 0, 5)


def test_metrics_reached_and_measurements_to_criterion():
    ref = {(TASK, 0): {"ref_acc": 1.0, "ref_power": None, "ref_loss": 0.0, "chance_acc": 0.5}}
    row = {"task": TASK, "seed": 0, "kind": "binary", "acc_final": 0.75, "power_final": None,
           "curve": {"n": [0, 10, 20], "acc": [None, 0.95, 0.75], "power": [None, None, None]}}
    BT.add_metrics([row], ref)
    assert row["gap_acc"] == pytest.approx(0.25) and row["norm_score"] == pytest.approx(0.5)
    assert row["reached"] is False and row["meas_to_crit"] == 10
    foc = {("focus", 0): {"ref_acc": None, "ref_power": -6.0, "chance_acc": None}}
    rf = {"task": "focus", "seed": 0, "kind": "focus", "acc_final": None, "power_final": -6.8,
          "curve": {"n": [0, 5], "acc": [None, None], "power": [-9.0, -6.8]}}
    BT.add_metrics([rf], foc)
    assert rf["reached"] and rf["meas_to_crit"] == 5 and rf["power_gap_db"] == pytest.approx(-0.8)


def test_plugin_cache_and_end_to_end(scratch):
    plugin = os.path.join(scratch, "plugin_demo.py")
    with open(plugin, "w", encoding="utf-8") as f:
        f.write("import numpy as np\n\n"
                "def all_zero(meas, rng):\n"
                "    x = np.zeros(meas.n_params, dtype=np.int64)\n"
                "    meas.measure_loss(x)\n"
                "    meas.report(x)\n"
                "    return x\n\n"
                "TRAINERS = {'all_zero': all_zero}\n")
    trainers, _ = BT.registry([plugin])
    assert "all_zero" in trainers and "dspsa_subset" in trainers
    cache = os.path.join(scratch, "cache")
    out = os.path.join(scratch, "out")
    argv = ["--trainers", "random_search", "focus_init", "all_zero", "--plugin", plugin, "--tasks", TASK,
            "--seeds", "1", "--budget", "24", "--workers", "1", "--out", out, "--cache-dir", cache,
            "--oracle-restarts", "1", "--oracle-sweeps", "2", "--curve-points", "6", "--n-noise-eval", "0"]
    assert BT.main(argv) == 0
    for name in ("runs.jsonl", "summary.md", "summary.csv", "curves.png", "references.json", "config.json"):
        assert os.path.exists(os.path.join(out, name)), name
    rows = [json.loads(line) for line in open(os.path.join(out, "runs.jsonl"), encoding="utf-8")]
    assert {r["trainer"] for r in rows} == {"random_search", "focus_init", "all_zero"}
    assert all("error" not in r and r["ref_acc"] is not None and "reached" in r for r in rows)
    rs = next(r for r in rows if r["trainer"] == "random_search")
    assert rs["n_meas_used"] == 24 and rs["budget"] == 24
    files = os.listdir(cache)
    assert len(files) == 1 and files[0].startswith(f"{TASK}_s0_") and files[0].endswith("_r1w2.json")
    doc = BT.oracle_refs(TASK, 0, cache_dir=cache, n_restarts=1, max_sweeps=2)
    assert doc["cached"] and doc["oracle"]["accuracy"]["acc"] is not None
    md = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    assert "all_zero" in md and "oracle 参照" in md
    # 合并旧结果：只跑 focus_init，旧的 random_search / all_zero 行进入汇总
    out2 = os.path.join(scratch, "out2")
    res = BT.run_benchmark(["focus_init"], [TASK], seeds=1, budget=24, workers=1, out=out2, cache_dir=cache,
                           oracle_restarts=1, oracle_sweeps=2, curve_points=6, n_noise_eval=0, include_runs=[out])
    assert res["config"]["n_included_rows"] == 2
    assert {r["trainer"] for r in res["table"]} == {"focus_init", "random_search", "all_zero"}
    # 只重新汇总（不跑新 run），按 --trainers 过滤
    out3 = os.path.join(scratch, "out3")
    assert BT.main(["--no-run", "--include-runs", out, "--trainers", "random_search", "--tasks", TASK, "--seeds", "1",
                    "--budget", "24", "--workers", "1", "--out", out3, "--cache-dir", cache, "--oracle-restarts", "1",
                    "--oracle-sweeps", "2"]) == 0
    rows3 = [json.loads(line) for line in open(os.path.join(out3, "runs.jsonl"), encoding="utf-8")]
    assert [r["trainer"] for r in rows3] == ["random_search"] and "meas_to_settle" in rows3[0]
    assert "random_search" in open(os.path.join(out3, "summary.md"), encoding="utf-8").read()
