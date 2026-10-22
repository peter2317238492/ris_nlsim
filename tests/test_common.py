"""experiments/common.py（INTEGRATION F）：多进程、异常行、汇总统计与输出格式。"""
import json
import math
import os

import numpy as np
import pytest

from ris_nlsim.experiments import common


def test_run_jobs_two_processes_and_error_row():
    jobs = [{"x": 1}, {"x": 2}, {"x": 3, "fail": True}, {"x": 4}]
    rows = common.run_jobs(common.demo_job, jobs, workers=2)
    assert len(rows) == 4
    assert [r.get("y") for r in rows] == [1, 4, None, 16]           # 按 jobs 顺序
    assert "error" in rows[2] and "intentional failure" in rows[2]["error"] and rows[2]["job"] == jobs[2]
    assert all("error" not in r for i, r in enumerate(rows) if i != 2)
    pids = {r["pid"] for r in rows if "pid" in r}
    assert os.getpid() not in pids                                   # 在子进程中运行
    assert all(r["elapsed_s"] >= 0 for r in rows)


def test_run_jobs_serial_path():
    rows = common.run_jobs(common.demo_job, [{"x": 5}, {"x": 0, "fail": True}], workers=1)
    assert rows[0]["y"] == 25 and rows[0]["pid"] == os.getpid() and "error" in rows[1]


def test_aggregate_mean_std_ci():
    rows = [{"cond": "M3", "n": 3, "acc": a, "ok": a == 1.0} for a in (1.0, 0.75, 0.875, 1.0)]
    rows += [{"cond": "M1", "n": 3, "acc": 0.5, "ok": False}, {"cond": "M1", "n": 3, "acc": None},
             {"cond": "M1", "n": 3, "error": "boom"}]
    table = common.aggregate(rows, ["cond", "n"], ["acc", "ok"])
    assert [t["cond"] for t in table] == ["M3", "M1"]
    t3 = table[0]
    vals = np.array([1.0, 0.75, 0.875, 1.0])
    assert t3["n_runs"] == 4 and t3["acc_n"] == 4
    assert t3["acc_mean"] == pytest.approx(vals.mean())
    assert t3["acc_std"] == pytest.approx(vals.std(ddof=1))
    half = 3.182446305284263 * vals.std(ddof=1) / 2.0              # t_{0.975, 3}
    assert t3["acc_ci95"] == pytest.approx(half, rel=1e-9)
    # 比例类指标（acc*、布尔）的区间截到 [0, 1]（审查 INT-NUM-5）；半宽本身不截断
    assert t3["acc_lo"] == pytest.approx(vals.mean() - half) and t3["acc_hi"] == 1.0
    assert t3["ok_mean"] == pytest.approx(0.5)                     # 布尔 → 比例
    assert t3["ok_lo"] >= 0.0 and t3["ok_hi"] <= 1.0
    raw = common.aggregate(rows, ["cond", "n"], ["acc"], bounds=None)[0]
    assert raw["acc_hi"] == pytest.approx(vals.mean() + half)       # bounds=None：不截断
    t1 = table[1]
    assert t1["n_runs"] == 2 and t1["acc_n"] == 1 and t1["acc_mean"] == 0.5 and math.isnan(t1["acc_ci95"])
    assert t1["n_errors"] == 1 and t1["n_total"] == 3               # error 行归入分组、计入 n_errors
    assert math.isnan(t1["acc_std"])                                # n = 1：ddof=1 无定义 → nan（不是 0）


def test_aggregate_nested_keys_and_outputs(tmp_path):
    rows = [{"cfg": {"cond": c}, "eval": {"acc": a}, "arr": np.arange(3), "z": 1 + 2j, "bad": float("nan")}
            for c, a in (("M3", 1.0), ("M3", 0.5), ("M4", 0.25))]
    table = common.aggregate(rows, ["cfg.cond"], ["eval.acc"])
    assert table[0]["cfg.cond"] == "M3" and table[0]["eval.acc_mean"] == pytest.approx(0.75)
    out = common.make_outdir("unit", out=str(tmp_path / "res"))
    paths = common.write_outputs(out, rows, table, config={"seed": np.int64(3)}, title="T")
    back = common.load_jsonl(paths["runs"])
    assert back[0]["arr"] == [0, 1, 2] and back[0]["z"] == {"re": 1.0, "im": 2.0} and back[0]["bad"] is None
    md = open(paths["md"], encoding="utf-8").read()
    assert md.startswith("# T") and "| cfg.cond |" in md and "M4" in md
    assert json.load(open(paths["config"], encoding="utf-8")) == {"seed": 3}
    assert open(paths["csv"], encoding="utf-8").readline().startswith("cfg.cond,")


def test_markdown_savefig_and_seed_rng(tmp_path):
    md = common.to_markdown([{"a": 1, "b": 0.123456789, "c": None}, {"a": 2, "b": float("nan"), "c": "x|y"}])
    lines = md.strip().splitlines()
    assert lines[0] == "| a | b | c |" and lines[2] == "| 1 | 0.1235 |  |" and "x\\|y" in lines[3]
    assert common.to_markdown([]).startswith("_(empty)_")
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    p = common.savefig(fig, str(tmp_path / "f" / "fig.png"))
    assert os.path.getsize(p) > 0
    a = common.seed_rng(1, "M3", 2).integers(0, 2 ** 31, 4)
    b = common.seed_rng(1, "M3", 2).integers(0, 2 ** 31, 4)
    c = common.seed_rng(1, "M4", 2).integers(0, 2 ** 31, 4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    args = common.base_argparser("x").parse_args(["--quick", "--workers", "3"])
    assert args.quick and args.workers == 3 and args.seeds == 5 and args.out is None


def test_run_jobs_isolates_worker_crash_and_system_exit():
    """审查 INT-NUM-1：一个 job 硬崩溃（os._exit）只让它自己成为 crashed 行；sys.exit 被捕获为 error 行。"""
    jobs = [{"x": i, "sleep": 0.3} for i in range(6)]
    jobs[1] = {"x": 1, "crash": True}
    jobs[4] = {"x": 4, "exit": True}
    rows = common.run_jobs(common.demo_job, jobs, workers=3)
    assert len(rows) == 6 and [r["job_index"] for r in rows] == list(range(6))
    assert rows[1].get("crashed") is True and "BrokenProcessPool" in rows[1]["error"] and rows[1]["job"] == jobs[1]
    assert "error" in rows[4] and "SystemExit" in rows[4]["error"] and not rows[4].get("crashed")
    for i in (0, 2, 3, 5):                                          # 被连带的 job 在第二轮重跑成功
        assert "error" not in rows[i] and rows[i]["y"] == i * i
    serial = common.run_jobs(common.demo_job, [{"x": 2}, {"x": 3, "exit": True}], workers=1)
    assert serial[0]["y"] == 4 and "SystemExit" in serial[1]["error"]


def test_max_workers_cap():
    assert common._max_workers(100, 500) == (61 if os.name == "nt" else 100)
    assert common._max_workers(8, 3) == 3 and common._max_workers(0, 3) == 1


def test_aggregate_tuple_array_keys_all_error_group_and_exclusion():
    """审查 INT-NUM-4 / INT-NUM-5 / INT-SEM-6：tuple / ndarray 键可分组；全错格点保留；exclude_if 剔除自激 run。"""
    rows = [{"job": {"roi": (0.8, 0.0, 0.0), "c": "M4"}, "eval": {"acc": 1.0, "osc": False}},
            {"job": {"roi": (0.8, 0.0, 0.0), "c": "M4"}, "eval": {"acc": 0.0, "osc": True}},
            {"job": {"roi": np.array([0.8, 0.0, 0.0]), "c": "M4"}, "eval": {"acc": 0.5, "osc": False}},
            {"job": {"roi": (0.4, 0.0, 0.0), "c": "M1"}, "error": "boom"},
            {"job": {"roi": (0.4, 0.0, 0.0), "c": "M1"}, "error": "boom"}]
    table = common.aggregate(rows, ["job.roi", "job.c"], ["eval.acc"])
    assert len(table) == 2
    g4, g1 = table
    assert g4["job.roi"] == "[0.8, 0.0, 0.0]" and g4["n_runs"] == 3 and g4["eval.acc_mean"] == pytest.approx(0.5)
    assert g1["n_runs"] == 0 and g1["n_errors"] == 2 and g1["eval.acc_n"] == 0 and math.isnan(g1["eval.acc_mean"])
    ex = common.aggregate(rows, ["job.c"], ["eval.acc"], exclude_if="eval.osc")
    assert ex[0]["n_runs"] == 2 and ex[0]["n_excluded"] == 1 and ex[0]["eval.acc_mean"] == pytest.approx(0.75)
    # evaluate 的 *_valid 字段：自激 run 为 None → 自动不计入
    rows_v = [{"c": "M1", "acc_clean_valid": None}, {"c": "M1", "acc_clean_valid": 0.75}]
    assert common.aggregate(rows_v, ["c"], ["acc_clean_valid"])[0]["acc_clean_valid_n"] == 1


def test_exp_a_object_modes_include_rq2_reference_and_g1():
    """审查 R2-INT-1：实验 A 的物体模式含 intra（RQ2 强度匹配对照）与 born（门 G1），且都是系统接受的模式。"""
    from ris_nlsim.system import OBJECT_MODES

    assert set(common.EXP_A_OBJECT_MODES) == set(OBJECT_MODES)
    assert common.RQ2_REFERENCE_MODE == "intra" and common.G1_OBJECT_MODE == "born"
    assert common.RQ2_REFERENCE_MODE in common.EXP_A_OBJECT_MODES and common.G1_OBJECT_MODE in common.EXP_A_OBJECT_MODES


def test_cjk_font_fallback_renders_chinese_without_missing_glyphs(tmp_path):
    """审查 S0-1：导入 common 即设好中文字体回退（DejaVu Sans 优先、缺字回退中文字体），中文标题不再画成方框。"""
    import warnings

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    if not any(c in installed for c in common.CJK_FONT_CANDIDATES):
        pytest.skip("本机没有中文字体")
    assert common.setup_cjk_fonts() == common.CJK_FONT and common.CJK_FONT in installed
    assert matplotlib.rcParams["font.family"][:2] == ["DejaVu Sans", common.CJK_FONT]
    assert matplotlib.rcParams["font.sans-serif"][0] == common.CJK_FONT
    assert matplotlib.rcParams["axes.unicode_minus"] is False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig, ax = plt.subplots()
        ax.set_title("奇偶阶梯：全对种子数 κ0 = −25 dB |Gκ|≥1 W≥3 2ⁿ ψ=90°")
        ax.set_xlabel("接收90 15 发90 30")
        ax.plot([-1, 1], [-2, 2])
        common.savefig(fig, str(tmp_path / "cjk.png"))
    assert not [x for x in w if "missing from" in str(x.message)], [str(x.message) for x in w]
