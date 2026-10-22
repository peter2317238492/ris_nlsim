"""实验 B（experiments/exp_b_snr.py）：SNR 定义、所需 SNR 汇总、端到端小网格（进程内，< 30 s）。"""
import json
import math
import os

import numpy as np
import pytest

from ris_nlsim.experiments import exp_b_snr as B


def _job(**kw):
    j = {"exp": "B", "preset": "L90", "task": "parity2", "cond": "M1@M3", "mode": "born", "material": "water",
         "seed": 0, "n_noise_eval": 3, "kind": "dspsa", "snr_target_db": 20.0, "dspsa_iters": 10, "eval_every": 5}
    j.update(kw)
    return j


def test_dspsa_job_snr_definition():
    r = B.run_job(_job(dspsa_variant="subset"))
    assert r["noise_dbm"] == pytest.approx(r["p_ref_dbm"] - 20.0)
    assert r["snr_db"] == pytest.approx(r["signal_dbm"] - r["noise_dbm"])
    s = B.base_system(_job())
    assert r["p_ref_dbm"] == pytest.approx(B.reference_signal_dbm(s))          # 与种子无关、确定性
    ph = s.phases_from_params(r["eval"]["params"])
    assert r["signal_dbm"] == pytest.approx(B.signal_dbm(s, ph))
    assert 0.0 <= r["acc_noisy_valid"] <= 1.0 and r["n_measurements"] == 2 * 10 + 20


def test_oracle_job_noise_curve_monotone_trend():
    job = _job(kind="oracle", oracle_restarts=1, oracle_sweeps=2, cont=True, cont_restarts=1, cont_steps=3,
               snr_dense=[-20.0, 60.0], n_noise_eval=20)
    r = B.run_job(job)
    s = B.base_system(job)
    for tr in ("oracle", "cont"):
        c = r[tr]["curve"]
        assert [p["snr_target_db"] for p in c] == [-20.0, 60.0]
        assert c[1]["snr_db"] - c[0]["snr_db"] == pytest.approx(80.0)
        # 60 dB 名义 SNR 下探头噪声可忽略，只剩 VNA σ = 0.002 dB：与无噪声准确率的差不超过"贴着判决边界"
        # （|裕量| ≤ 0.05 dB = 25σ）的样本比例；−20 dB 下不高于它
        ph = (s.phases_from_params(r[tr]["eval"]["params"]) if tr == "oracle"
              else {k: np.asarray(v) for k, v in r[tr]["eval"]["phases"].items()})
        sm = s.task.sample_margins_db(s.clean_readings(s.forward(ph, noisy=False)), s.labels)
        fragile = float(np.mean(np.abs(sm) <= 0.05))
        assert abs(c[1]["acc_noisy_valid"] - r[tr]["acc_clean_valid"]) <= fragile + 1e-12
        assert c[0]["acc_noisy_valid"] <= c[1]["acc_noisy_valid"] + 1e-12
    # 1-bit oracle = accuracy 目标 + polish（审查 Q-1）；loss-oracle 只记可测目标（实验 E 的 L*）
    assert r["oracle"]["eval"]["acc_clean"] >= r["oracle"]["stage1_acc_clean"] - 1e-12
    assert r["oracle_loss"]["loss_meas_clean"] is not None


def test_required_snr_table_synthetic():
    pts = []
    for seed in range(3):
        for snr, acc in ((0.0, 0.5), (10.0, 0.8), (20.0, 1.0)):
            pts.append({"task": "t", "mode": "born", "cond": "M3", "seed": seed, "trainer": "cont",
                        "snr_target_db": snr, "noise_dbm": -snr, "snr_db": snr + seed, "signal_dbm": 0.0,
                        "acc": acc, "acc_clean": 1.0, "osc": False, "p_ref_dbm": 0.0})
    ct = B.curve_table(pts)
    rt = B.required_snr_table(pts, ct)
    assert len(rt) == 1
    r = rt[0]
    assert r["req_snr_db_meancurve"] == pytest.approx(16.0)                  # 平均 SNR 11 → 21，插值 0.9
    assert r["n_seeds_reaching"] == 3 and r["req_snr_db_seed_mean"] == pytest.approx(16.0)


def test_main_end_to_end_tiny_grid(tmp_path):
    out = str(tmp_path / "expB")
    B.main(["--dspsa-variant", "subset",
            "--tasks", "parity2", "--conditions", "M1@M3", "M3", "--modes", "born", "--seeds", "2", "--snr", "0", "40",
            "--snr-dense", "-10", "40", "--dspsa-iters", "10", "--oracle-restarts", "1", "--oracle-sweeps", "1",
            "--cont-restarts", "1", "--cont-steps", "3", "--n-noise", "3", "--workers", "1", "--out", out])
    for f in ("runs.jsonl", "summary.csv", "summary.md", "config.json", "required_snr.csv", "acc_vs_snr.png",
              "acc_vs_noise_dbm.png"):
        assert os.path.exists(os.path.join(out, f)), f
    rows = [json.loads(line) for line in open(os.path.join(out, "runs.jsonl"), encoding="utf-8")]
    assert len(rows) == 2 * 2 * (1 + 2) and not any("error" in r for r in rows)
    md = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    for s in ("所需 SNR", "RQ3", "工作点与信号电平", "M3 − M1@M3", "M3 − M1b@M3", "DSPSA 训练诊断", "DSPSA pocket"):
        assert s in md, s
    B.main(["--resummarize", out])


# ------------------------------------------------------------ 审查 Q-4：M1b（Bussgang）对照
def test_m1b_conditions_and_rq3_columns():
    assert {"M1b@M3", "M1b@M4"} <= set(B.ALL_CONDITIONS) and set(B.parse_args([]).conditions) == set(B.ALL_CONDITIONS)
    pts = []
    for cond, acc in (("M3", 0.9), ("M1@M3", 0.6), ("M1b@M3", 0.7), ("M4", 0.8), ("M1@M4", 0.5), ("M1b@M4", 0.4),
                      ("M0", 0.55)):
        for seed in range(2):
            pts.append({"task": "parity3", "mode": "born", "cond": cond, "seed": seed, "trainer": "oracle",
                        "snr_target_db": 20.0, "noise_dbm": -60.0, "snr_db": 20.0, "signal_dbm": -40.0, "acc": acc,
                        "acc_clean": acc, "osc": False, "p_ref_dbm": -40.0})
    md = B.rq3_md(B.curve_table(pts))
    assert "M3 − M1b@M3" in md and "M4 − M1b@M4" in md
    row = [ln for ln in md.splitlines() if ln.startswith("| parity3")][0]
    assert [c.strip() for c in row.strip("|").split("|")][4:] == ["+0.30", "+0.20", "+0.30", "+0.40", "+0.05"]


# ------------------------------------------------------------ 审查 Q-5：字母任务的物体模式与材料记录
def test_letters_skip_born_and_record_actual_material():
    args = B.parse_args(["--tasks", "letters", "parity3", "--modes", "born", "intra", "--material", "water",
                         "--seeds", "1", "--snr", "10", "--conditions", "M1@M3"])
    jobs = B.build_jobs(args)
    assert not any(j["task"] == "letters" and j["mode"] == "born" for j in jobs)
    assert any(j["task"] == "letters" and j["mode"] == "intra" for j in jobs)
    assert any(j["task"] == "parity3" and j["mode"] == "born" for j in jobs)
    assert args.skipped_cells == [{"task": "letters", "mode": "born",
                                   "reason": args.skipped_cells[0]["reason"]}]
    lj = [j for j in jobs if j["task"] == "letters"]
    assert all(j["material"] == "metal" for j in lj)                 # --letter-material（默认 metal），不是 --material
    dj = [j for j in lj if j["kind"] == "dspsa"][0]
    r = B.run_job(dict(dj, dspsa_iters=4, eval_every=2, n_noise_eval=2))
    assert r["material"] == "metal" and r["object_model"] == "scalar_rr" and r["effective_eps"] == "metal"
    rp = B.run_job(_job(task="parity3", dspsa_iters=4, eval_every=2, n_noise_eval=2))
    assert rp["material"] == "water" and rp["object_model"] == "volume_rr"


# ------------------------------------------------------------ 审查 Q-2：pocket、平台诊断、训练到位判定
def test_dspsa_job_reports_pocket_and_quick_budget_matches_exp_a():
    r = B.run_job(_job(dspsa_iters=30, dspsa_variant="subset"))      # 旧变体才单列 pocket
    pk = r["pocket"]
    assert pk is not None and pk["eval"]["acc_noisy_valid"] is not None
    assert pk["snr_db"] == pytest.approx(pk["signal_dbm"] - r["noise_dbm"])
    assert r["L0"] == pytest.approx(r["curve"][0]["loss_meas_clean"])
    assert r["chance_loss"] == pytest.approx(math.log(2.0))
    assert B.parse_args(["--quick", "--scenario", "baseline"]).dspsa_iters == 1500   # 与实验 A 同预算（审查 Q-2）
    assert B.parse_args(["--quick"]).dspsa_iters == 1000                # informative 场景 quick（INTEGRATION S）


def test_trained_table_and_rq3_withholding():
    def drow(cond, seed, snr, L, Lpk):
        return {"kind": "dspsa", "task": "parity3", "mode": "born", "cond": cond, "seed": seed, "snr_target_db": snr,
                "chance_loss": math.log(2.0), "L0": 1.2, "loss_meas_clean": L, "pocket": {"loss_meas_clean": Lpk}}

    def orow(cond, seed, Ls):
        return {"kind": "oracle", "task": "parity3", "mode": "born", "cond": cond, "seed": seed,
                "oracle": {"loss_meas_clean": Ls + 0.1}, "oracle_loss": {"loss_meas_clean": Ls}}

    rows = []
    for seed in range(5):
        rows += [orow("M3", seed, 0.2), orow("M1@M3", seed, 0.2)]
        rows += [drow("M3", seed, 40.0, 0.69, 0.25), drow("M1@M3", seed, 40.0, 0.69, 0.25 if seed < 4 else 0.69),
                 drow("M3", seed, 0.0, 0.7, 0.7)]
    B.annotate_dspsa(rows)
    d = [r for r in rows if r["kind"] == "dspsa" and r["cond"] == "M3" and r["snr_target_db"] == 40.0][0]
    assert d["Lstar"] == pytest.approx(0.2)
    assert d["plateau"] is True and d["trained"] is False                  # 停在 τ·ln2 平台
    assert d["pocket"]["plateau"] is False and d["pocket"]["reach90"] is True and d["pocket"]["trained"] is True
    tt = B.trained_table(rows)
    assert tt[("parity3", "born", "M3", "dspsa")]["trained"] is False
    assert tt[("parity3", "born", "M3", "dspsa_pocket")]["trained"] is True
    assert tt[("parity3", "born", "M1@M3", "dspsa_pocket")]["k"] == 4      # 4/5 = 80% → 训练到位
    assert tt[("parity3", "born", "M1@M3", "dspsa_pocket")]["trained"] is True
    assert tt[("parity3", "born", "M3", "dspsa")]["plateau_k"] == 10
    ct = []
    for tr in ("dspsa", "dspsa_pocket"):
        for cond in ("M3", "M1@M3"):
            ct.append({"task": "parity3", "mode": "born", "cond": cond, "trainer": tr, "snr_target_db": 40.0,
                       "acc_n": 5, "acc_mean": 0.9 if cond == "M3" else 0.6})
    md = B.rq3_md(ct, trained=tt)
    lines = {ln.split("|")[3].strip(): ln for ln in md.splitlines() if ln.startswith("| parity3")}
    assert "未训练好" in lines[B.TRAINER_NAMES["dspsa"]]
    assert "+0.30" in lines[B.TRAINER_NAMES["dspsa_pocket"]] and "未训练好" not in lines[B.TRAINER_NAMES["dspsa_pocket"]]
