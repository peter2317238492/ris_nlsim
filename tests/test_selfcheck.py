"""selfcheck.py（INTEGRATION H，DESIGN §10，门 G1 仿真侧）：小规模全流程 + 各检查不是空检查。"""
import numpy as np
import pytest  # noqa: F401  (保留：便于本文件追加 slow 标记)

from ris_nlsim import selfcheck as sc
from ris_nlsim.constants import LAMBDA


def test_run_selfcheck_tiny_all_pass():
    """tiny（λ/10 体素、少量 oracle/DSPSA 预算）+ fast：全部 PASS（体素收敛 SKIP、XOR 为 INFO）。"""
    res = sc.run_selfcheck(fast=True, tiny=True, verbose=False)
    st = {r.name: r.status for r in res}
    assert st == {"reciprocity": "PASS", "optical_theorem": "PASS", "born_vs_full": "PASS",
                  "voxel_convergence": "SKIP", "dwba_affinity": "PASS", "relay_dwba_affinity": "PASS",
                  "m1_born_parity_info": "INFO", "m0_born_parity3_unsolvable": "PASS",
                  "m0_born_presence_solvable": "PASS", "m0_born_xor_info": "INFO", "torch_numpy": "PASS",
                  "calibration": "PASS"}, st
    par = next(r for r in res if r.name == "m0_born_parity3_unsolvable")
    assert par.values["best_margin_db"] <= 0.0 and par.values["best_acc"] < 1.0
    assert all(r.line().startswith(f"[{r.status}]") for r in res)


def test_dwba_check_detects_non_affine_system():
    """把 M0 + full（物体多次散射，y 含三阶项）当作"Born 系统"喂给仿射检查 → 必须 FAIL（检查不是空的）。"""
    s_full = sc._parity_system("M0", "full", 3, tiny=True)
    r = sc.check_dwba_affinity(np.random.default_rng(0), s_full, tiny=True, n_codebooks=2)
    assert not r.passed and r.values["max_W3_ydiff"] > 1e-12


def test_relay_dwba_check_and_m1_born_info():
    """审查 P-2：M1/M3 + Born 下中继线性量 u_in、d、C、K 各自仿射（PASS）；但物体在中继发射波束中，
    M1 + Born 的 y1−y2 含 3 阶 Walsh 项（INFO），只有 objects_see_relay_tx=False 时才恢复方案 §6.1 的"二次"。"""
    r1, r2 = sc.check_relay_dwba_affinity(np.random.default_rng(3), tiny=True, n_codebooks=2)
    assert r1.passed and r1.status == "PASS" and r1.values["worst"] < 1e-20
    terms = r1.values["per_preset"]["flat0"]["max_W2_relay_terms"]
    assert set(terms) == {f"{c}/{k}" for c in ("M1", "M3") for k in ("u_in", "d", "C", "K")}
    assert r2.info and r2.status == "INFO"
    for p in sc.RELAY_DWBA_PRESETS:
        v = r2.values[p]
        assert v["W3_ydiff_objects_not_in_tx_beam"] < 1e-20                    # 前提成立时二次（不可解）
        assert v["W3_ydiff_default"] > 1e-8 and v["W3_ydiff_no_loop"] > 1e-8   # 再照射 + 环路 → 高阶项
    assert r2.values["flat0"]["W3_ydiff_default"] > 1e-3                        # flat0：物体在放大环路里


def test_relay_dwba_check_detects_non_affine_relay_terms():
    """把 M1 + full（物体多次散射 → u_in、C、K 非仿射）喂给中继 DWBA 检查 → 必须 FAIL（检查不是空的）。"""
    s_full = sc._parity_system("M1", "full", 3, tiny=True, preset="flat0")
    r1, _ = sc.check_relay_dwba_affinity(np.random.default_rng(0), tiny=True, n_codebooks=1, presets=("flat0",),
                                         systems={"flat0": s_full})
    assert not r1.passed and r1.values["worst"] > 1e-12


def test_born_full_error_scales_with_contrast():
    e1 = sc._born_full_err(1.01, LAMBDA / 8)
    e2 = sc._born_full_err(1.1, LAMBDA / 8)
    assert e1 < 0.02 and 5 < e2 / e1 < 20


def test_check_result_status_and_main_exit_code(monkeypatch, capsys):
    ok = sc.CheckResult("a", True, "x")
    bad = sc.CheckResult("b", False, "y")
    assert (ok.status, bad.status) == ("PASS", "FAIL")
    assert sc.CheckResult("c", True, "z", skipped=True).status == "SKIP"
    assert sc.CheckResult("d", True, "w", info=True).status == "INFO"
    monkeypatch.setattr(sc, "run_selfcheck", lambda **kw: [ok, sc.CheckResult("d", True, "w", info=True)])
    assert sc.main([]) == 0
    monkeypatch.setattr(sc, "run_selfcheck", lambda **kw: [ok, bad])
    assert sc.main([]) == 1
    assert "1 FAIL" in capsys.readouterr().out


def test_selfcheck_main_fast_exit_zero():
    """真实配置（水自动体素 λ/28）下 --fast 全部通过（约 5 s）。"""
    assert sc.main(["--fast"]) == 0
