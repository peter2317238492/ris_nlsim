"""实验 F（experiments/exp_f_hardware.py）：实测表解析、列名 → 几何、PO 金属板 RCS、线性闭式 vs relay 管线、
单 job 与端到端小规模运行（方案 §6.2 F、§8 门 G1；INTEGRATION G "F"）。"""
import json
import os
import pathlib
import shutil

import numpy as np
import pytest

from ris_nlsim.constants import F0, wavelength
from ris_nlsim.pa import LinearPA, RappPA
from ris_nlsim.patterns import horn_pattern
from ris_nlsim.relay import probe_fields, relay_linear, solve_fixed_point, weights
from ris_nlsim.solver import solve_scene
from ris_nlsim.experiments import exp_f_hardware as ef

HAS_XLSX = os.path.exists(ef.DEFAULT_XLSX)
needs_xlsx = pytest.mark.skipif(not HAS_XLSX, reason="测试数据.xlsx 不在默认位置")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def out_dir(request) -> pathlib.Path:
    """审查 P-3：pytest 默认 tmp_path 在 %TEMP%（C:/Users/…/pytest-of-…），而项目规定测试输出留在仓库内 →
    实验输出写到项目 _scratch/pytest_tmp/test_exp_f/<测试名>/（固定目录，每次运行前清空，测试后保留便于检查，约 1 MB）。"""
    d = PROJECT_ROOT / "_scratch" / "pytest_tmp" / "test_exp_f" / request.node.name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def _fc(delta=30.0, rx=(0.0, 0.0), tx=(0.0, 0.0), reading="phi_theta"):
    ar, er = ef.pointing_angles(rx, reading)
    at, et = ef.pointing_angles(tx, reading)
    return ef.FConfig("t", "t", reading, delta, 1.8, 1.8, ar, er, at, et)


# ------------------------------------------------------------ 解析
@needs_xlsx
def test_load_measurements_blocks_and_consistency():
    blocks = ef.load_measurements()
    assert [b.name for b in blocks] == ["az30", "az50"]
    b30, b50 = blocks
    assert (b30.feed_az_deg, b30.feed_dist_m, b50.feed_az_deg, b50.feed_dist_m) == (30.0, 1.8, 50.0, 1.5)
    assert len(b30.labels) == 11 and len(b50.labels) == 10
    assert b30.impr.shape == (11, 11) and np.allclose(b30.freqs_ghz, np.arange(3.45, 3.5501, 0.01))
    for b in blocks:
        assert sum(b.coplanar_note.values()) == 1                     # 只有最后一列标"无夹角"
        assert b.consistency_db() < 1e-9                              # 提升 = 绝对增益 − 金属板增益
        assert np.all(np.isfinite(b.impr))
    st = b30.stats("收00-发00")
    assert st["meas_min"] <= st["meas_mean"] <= st["meas_max"] and st["meas_n_freq"] == 11


def test_parse_label_and_fold_rule():
    assert ef.parse_label("收00-发00") == {"rx": (0.0, 0.0), "tx": (0.0, 0.0)}
    assert ef.parse_label("收00-发90 15") == {"rx": (0.0, 0.0), "tx": (90.0, 15.0)}
    assert ef.parse_label("接收15 15发00") == {"rx": (15.0, 15.0), "tx": (0.0, 0.0)}
    assert ef.parse_label("接收90 30 发00\n（收发阵列之间无夹角仅靠波束控制）") == {"rx": (90.0, 30.0), "tx": (0.0, 0.0)}
    assert ef.ascii_label("接收90 15 发90 30") == "rx90,15/tx90,30"
    # ψ = Δ − α_r + α_t：共面列 ψ = 0，"收00-发00" ψ = Δ，发射偏转加大折叠
    assert _fc(30, (90, 30)).fold_deg == 0.0
    assert _fc(50, (90, 50)).fold_deg == 0.0
    assert _fc(30).fold_deg == 30.0
    assert _fc(30, (0, 0), (90, 50)).fold_deg == pytest.approx(80.0)
    assert _fc(50, (90, 30), (90, 15)).fold_deg == pytest.approx(35.0)
    # 歧义列两种读法
    assert ef.is_ambiguous((15.0, 0.0)) and ef.is_ambiguous((15.0, 15.0)) and not ef.is_ambiguous((90.0, 15.0))
    assert ef.pointing_angles((15.0, 0.0), "phi_theta") == (0.0, 0.0)
    assert ef.pointing_angles((15.0, 0.0), "az_el") == (15.0, 0.0)
    a, e = ef.pointing_angles((15.0, 15.0), "phi_theta")
    assert a == pytest.approx(3.97, abs=0.01) and e == pytest.approx(14.48, abs=0.01)


@needs_xlsx
def test_configs_only_annotated_column_is_coplanar():
    for b in ef.load_measurements():
        cfgs = ef.configs_for_block(b)
        copl = [c for c in cfgs if abs(c.fold_deg) < 1e-9]
        assert len(copl) == 1 and copl[0].coplanar_note                # 折叠规则与表注一致（模块 docstring）
        assert all(0.0 <= c.fold_deg <= 180.0 for c in cfgs)


# ------------------------------------------------------------ 几何与物理
def test_geometry_beams_point_at_feed_and_probe():
    fc = _fc(30, (90, 15), (90, 30))
    g = ef.make_geometry(fc)
    r = g["relay"]
    d_feed = ef._unit(g["feed"] - r.rx_center)
    d_probe = ef._unit(g["probe"] - r.tx_center)
    # 接收波束 = 法向朝馈源一侧转 α_r；发射同理（同向安装：h_tx = h_rx 绕 z 转 ψ）
    a = np.deg2rad(15.0)
    assert np.allclose(d_feed, np.cos(a) * r.rx_normal + np.sin(a) * g["h_rx"])
    a = np.deg2rad(30.0)
    assert np.allclose(d_probe, np.cos(a) * r.tx_normal + np.sin(a) * g["h_tx"])
    assert np.degrees(np.arccos(d_feed @ d_probe)) == pytest.approx(30.0)


def test_plate_po_far_field_rcs():
    """镜面取向 PO 板在远场的单站 RCS = 4πA²/λ²（DESIGN §1.3：k²αD = 4πA/λ）。"""
    fc = _fc(30)
    g = ef.make_geometry(fc)
    r = g["relay"]
    pc = 0.5 * (r.rx_center + r.tx_center)
    R = 40.0
    n = ef._unit(np.array([1.0, 0.25, 0.0]))
    t = np.cross([0, 0, 1.0], n)
    g["feed"], g["probe"] = pc + R * n + 0.02 * t, pc + R * n - 0.02 * t
    g["relay"].rx_center[:] = pc                                       # 喇叭对准板心（只影响方向图取值）
    g["relay"].tx_center[:] = pc
    H = ef.plate_response(g, F0, "one", block_direct=True)
    lam = wavelength(F0)
    D = horn_pattern().peak_directivity
    sigma = abs(H) ** 2 * (4 * np.pi) ** 3 * R**4 / (lam**2 * D * D)
    A = r.width ** 2
    assert sigma == pytest.approx(4 * np.pi * A**2 / lam**2, rel=0.02)
    H2 = ef.plate_response(g, F0, "both", block_direct=True)
    assert abs(H2) ** 2 / abs(H) ** 2 == pytest.approx(4.0, rel=0.03)    # 面积加倍 → RCS ×4


def test_fast_probe_field_matches_relay_pipeline():
    fc = _fc(30, (90, 15), (0, 0))
    g = ef.make_geometry(fc)
    H = solve_scene(ef.relay_scene(g, F0), None, None, "full")
    rng = np.random.default_rng(3)
    prm = ef.relay_params(0.1 * rng.standard_normal((1, 25)), 0.1 * rng.standard_normal((1, 25)))
    for _ in range(3):
        bits = rng.integers(0, 2, 50)
        pa = LinearPA(28.0)
        w, ct = weights(np.pi * bits[None, :25], np.pi * bits[None, 25:], prm)
        lin = relay_linear(H, w, ct, 0.7, prm)
        fp = solve_fixed_point(lin.u_in, lin.K, pa)
        z = probe_fields(lin.d, lin.C, fp.g)[0]
        assert ef.fast_probe_field(H, bits, pa.gain_v, prm, 0.7) == pytest.approx(z, rel=1e-12)


def test_codebooks_and_oracle_not_worse_than_conj():
    fc = _fc(30, (90, 15), (90, 15))
    g = ef.make_geometry(fc)
    H = solve_scene(ef.relay_scene(g, F0), None, None, "full")
    prm = ef.relay_params()
    conj = ef.conj_bits(H)
    steer = ef.steer_bits(g, fc, F0)
    assert conj.shape == steer.shape == (50,)
    orc = ef.oracle_codebook(H, prm, 30.0, np.random.default_rng(0), n_restarts=1, start=conj)
    gv = 10 ** 1.5
    p = {k: abs(ef.fast_probe_field(H, b, gv, prm, 1.0)) for k, b in (("conj", conj), ("oracle", orc))}
    assert p["oracle"] >= p["conj"] * (1 - 1e-12)
    # 量化共轭的接收合成至少达到理想相干和的 (2/π)²（1-bit 量化损失上限 3.9 dB）
    v = H[:25, 0]
    u = abs(np.sum(np.exp(1j * np.pi * conj[:25]) * v)) / np.sum(np.abs(v))
    assert u ** 2 >= (2 / np.pi) ** 2 * 0.95


def test_run_job_rows_and_physics_window():
    fc = _fc(30)
    res = ef.run_job({"kind": "config", "config": fc.__dict__, "seed": 0, "freqs": [F0], "gains": [25.0, 30.0, 40.0],
                      "steer": True, "phase_err_deg": 10.0, "oracle_restarts": 1})
    rows = {(r["gain_db"], r["codebook"]): r for r in res["rows"]}
    assert len(rows) == 9
    r30 = rows[(30.0, "conj")]
    assert 20.0 < r30["improvement_db"] < 29.0                         # 实测 "收00-发00" 21–30 dB
    assert r30["pa_in_rel_p1db_db"] < -15 and not r30["flagged"]       # 深回退工作点
    assert rows[(30.0, "oracle")]["improvement_db"] >= r30["improvement_db"] - 0.01
    assert rows[(25.0, "conj")]["improvement_db"] < r30["improvement_db"]
    assert rows[(40.0, "conj")]["flagged"] and rows[(40.0, "conj")]["improvement_db"] is None   # |Gκ| = +5 dB
    assert r30["iso_geo_rms_db"] < -40                                 # 板间自由空间耦合远低于 κ0 = −30 dB


# ------------------------------------------------------------ 审查 S0-0：三种自激判据
def test_flag_criteria_robust_vs_small_signal():
    """flagged = CW 或相位稳健环路增益 ≥ 1（物理）；flagged_ss = CW 或小信号 |Gκ| ≥ 1（保守）。
    ∠κ0 = 0°、|Gκ| = +1 dB：强注入把 PA 压缩成注入锁定的稳定饱和态 → 只有保守列标记；
    ∠κ0 = 90°、|Gκ| = +5 dB：深回退下 PA 线性，三种判据都标记。"""
    fc = _fc(30)
    g = ef.make_geometry(fc)
    H = solve_scene(ef.relay_scene(g, F0), None, None, "full")
    Hp = ef.plate_response(g, F0)
    b = ef.conj_bits(H)
    ev = ef._eval_point(H, Hp, b, b, 36.0, ef.relay_params(isolation_db=-30.0, kappa0_phase_deg=0.0), -20.0, 20.0)
    assert ev["ss_unstable"] and not ev["robust_unstable"] and not ev["possibly_oscillating"]
    assert ev["loop_gain_ss"] > 1.0 > ev["loop_gain_robust"] >= ev["loop_gain_op"]
    assert ev["flagged_ss"] and not ev["flagged"]
    assert ev["improvement_ss_db"] is None and ev["improvement_db"] == ev["improvement_raw_db"]
    ev = ef._eval_point(H, Hp, b, b, 40.0, ef.relay_params(), -20.0, 20.0)
    assert ev["flagged"] and ev["flagged_ss"] and ev["robust_unstable"] and ev["improvement_db"] is None


# ------------------------------------------------------------ 审查 P-1：未测的 ∠κ0 系统误差
def test_kappa_sys_grid_and_phase_arcs():
    grid = ef.kappa_sys_grid()
    assert grid[0] == (ef.OPEN_LOOP_ISO_DB, 0.0) and ef.DEFAULT_KAPPA not in grid     # 默认点在 default 行里
    assert sorted(p for i, p in grid[1:]) == [p for p in ef.KAPPA_SYS_PHASES_DEG if p != ef.DEFAULT_KAPPA[1]]
    assert ef.kappa_case(*grid[0]) == "open" and ef.kappa_case(-30.0, 90.0) == "-30dB/90°"
    ph = list(ef.KAPPA_SYS_PHASES_DEG)
    assert ef._phase_arcs([90, 135, 180, 225, 270], ph) == "[90°, 270°]"
    assert ef._phase_arcs([315, 0, 45], ph) == "[315°, 45°]"                         # 跨 0° 回绕
    assert ef._phase_arcs([0, 180], ph) == "0° ∪ 180°"
    assert ef._phase_arcs(ph, ph) == "全部" and ef._phase_arcs([], ph) == "无"


def test_run_job_kappa_sys_rows_follow_loop_factor():
    """sys 行只在 f0、conj 上算；空场景 K = κ_eff 与码本无关 ⇒ 改善量(∠κ0) − 改善量(开环) ≈ 环路因子
    −20log10|1 − G_v κ_eff|（深回退近线性，只差视距项干涉）：G = 30 dB、κ0 = −30 dB 时 0° 为 +7.2 dB、180° 为 −3.9 dB。"""
    fc = _fc(30)
    res = ef.run_job({"kind": "config", "config": fc.__dict__, "seed": 0, "freqs": [F0], "gains": [30.0],
                      "steer": False, "oracle": False, "phase_err_deg": 0.0, "kappa_sys": ef.kappa_sys_grid()})
    rows = res["rows"]
    sys_rows = [r for r in rows if r["kappa_set"] == "sys"]
    assert len(sys_rows) == len(ef.kappa_sys_grid()) and all(r["codebook"] == "conj" for r in sys_rows)
    by = {r["kappa_case"]: r for r in rows if r["codebook"] == "conj"}
    assert by["-30dB/90°"]["kappa_set"] == "default" and len(by) == 1 + len(ef.KAPPA_SYS_PHASES_DEG)
    op = by["open"]
    assert abs(op["loop_factor_db"]) < 1e-6 and op["loop_gain_ss_db"] < -150
    lf_ref = {0.0: 7.18, 90.0: -1.19, 180.0: -3.88}
    for ph, lf in lf_ref.items():
        r = by[f"-30dB/{ph:g}°"]
        assert r["loop_factor_db"] == pytest.approx(lf, abs=0.02)
        assert r["improvement_db"] - op["improvement_db"] == pytest.approx(r["loop_factor_db"], abs=0.3)
    # 默认 90° 不是最坏也不是最好：门 G1 判定会随 ∠κ0 翻转（审查 P-1 的复现）
    assert by["-30dB/0°"]["improvement_db"] > by["-30dB/90°"]["improvement_db"] + 7.0
    assert by["-30dB/180°"]["improvement_db"] < by["-30dB/90°"]["improvement_db"] - 2.0


@needs_xlsx
def test_end_to_end_small_run(out_dir):
    """端到端（单进程、裁剪后的计划）：输出文件齐全、门 G1 字段（含 ∠κ0 条件结论）、主表对齐实测；输出不落 系统临时目录。"""
    assert not str(out_dir).lower().startswith("c:")
    args = ef.parse_args(["--quick", "--workers", "1", "--seeds", "1", "--out", str(out_dir)])

    def keep(jobs):
        out = []
        for j in jobs:
            c = j["config"]
            if j["kind"] == "config" and (c["label"].startswith("收00-发00") or c["coplanar_note"]):
                out.append({**j, "gains": [25.0, 30.0, 32.5], "oracle_restarts": 1})
            elif j["kind"] == "fold" and c["label"] in ("fold0", "fold40"):
                out.append(j)
            elif j["kind"] == "iso" and ef.FConfig(**c).fold_deg == 0.0:
                out.append({**j, "kappa_grid": [(-30.0, 90.0), (-30.0, 180.0), (-20.0, 180.0)]})
        return out

    res = ef.run(args, job_filter=keep)
    for f in ("runs.jsonl", "summary.csv", "summary.md", "config.json", "gain_fit.csv", "gain_fit_kappa.csv",
              "gate_kappa.csv", "kappa_sweep.csv", "fold_sweep.csv", "isolation_sweep.csv", "measured.csv",
              "coplanar_analysis.csv", "fig_gain_sweep.png", "fig_pred_vs_meas.png", "fig_fold_sweep.png",
              "fig_isolation_sweep.png", "fig_kappa_phase.png"):
        assert (out_dir / f).exists(), f
    assert not res["errors"]
    summ = res["summary"]
    main = {(r["block"], r["config"]): r for r in summ["main"]}
    r = main[("az30", "收00-发00")]
    assert r["gate_conj"] is True and abs(r["dev_conj_db"]) < 3.0
    assert r["p_feed_dbm"] is not None and r["pa_in_rel_p1db_db"] < -15
    # 审查 P-1：主表给开环与 ∠κ0 扫描范围；默认 90° 落在范围内
    assert r["pred_conj_kappa_min"] < r["pred_conj"] < r["pred_conj_kappa_max"]
    assert r["pred_conj_kappa_max"] - r["pred_conj_kappa_min"] > 9.0             # +7.2 … −3.9 dB 的环路因子
    assert r["pred_conj"] < r["pred_conj_open"]                                   # 90°：环路因子 −1.2 dB
    g = summ["gate"]
    kt = {t["kappa_case"]: t for t in g["kappa"]}
    assert len(kt) == 1 + len(ef.KAPPA_SYS_PHASES_DEG) and kt["-30dB/90°"]["is_default"]
    assert kt["-30dB/90°"]["n_pass_non_coplanar"] == g["non_coplanar"][0]       # 默认情形与原门 G1 计数一致
    assert kt["-30dB/0°"]["n_pass_non_coplanar"] < kt["-30dB/90°"]["n_pass_non_coplanar"]
    assert kt["-30dB/0°"]["mean_dev_db"] > kt["-30dB/90°"]["mean_dev_db"] + 7.0
    assert g["pass_non_coplanar"] and g["verdict_code"] == "CONDITIONAL" and 0.0 not in g["kappa_phases_pass"]
    assert g["open_loop"] == (2, 2) and g["pass_open_loop"]
    assert all((t["min_gate_margin_db"] >= 0.0) == t["pass_non_coplanar"] for t in g["kappa"])
    fit = {(f["block"], f["codebook"]): f for f in summ["fit"]}
    f30 = fit[("az30", "conj")]
    assert f30["G_match_00_open_db"] is not None and f30["G_match_00_kappa_min_db"] < f30["G_match_00_db"]
    assert fit[("az30", "steer")].get("G_match_00_open_db") is None             # 系统误差只对 conj 算
    assert {t["kappa_case"] for t in summ["fit_kappa"] if t["block"] == "az30"} == set(kt)
    cop = summ["coplanar"]
    assert {c["block"] for c in cop} == {"az30", "az50"}
    assert all(c["unexplained_db"] > 3.0 for c in cop)                 # 几何不含共面退化机理（报告给出）
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["voxel_d"] is None and "gate" in cfg and cfg["gate"]["verdict_code"] == "CONDITIONAL"
    assert cfg["kappa_sys_phases_deg"] == list(ef.KAPPA_SYS_PHASES_DEG)
    md = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "门 G1" in md and "有条件 PASS" in md and "开环（κ0 → 0）" in md and "假设值" in md
