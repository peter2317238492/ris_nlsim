"""实验 B：增益 vs 非线性——准确率–SNR 曲线（方案 §3 RQ3、§6.2；INTEGRATION G "B"）。

任务：3 位奇偶（主任务；可选 2 位奇偶 = XOR 作为"线性 + 平方律可解"的对照，可选字母 A/B/C）。
条件：M0、M1@M3、M1b@M3、M3、M4、M1@M4、M1b@M4——M1@Mk / M1b@Mk 为小信号增益 / Bussgang 等效复增益的线性中继、
  源功率与 Mk 相同（INTEGRATION B："实验 A/B 的非线性收益同时报相对 M1 与 M1b 的值"，审查 Q-4）。
噪声：探头加性复高斯噪声 probe_noise_dbm（DESIGN §4，加在 |·|² 之前），另有 VNA 0.002 dB 读数噪声与 PA 输入热噪声。
  噪声网格按"名义 SNR" = P_ref − P_n 给出，P_ref = 该条件 16 个随机码本（固定 rng）× 全部样本 × 全部探头的无噪声
  探头功率中位数；每个条件各自跨过其信号电平（quick 5 点）。
SNR 轴（INTEGRATION G "B"）：训练后码本的无噪声探头功率中位数（全部样本 × 探头）/ P_n。
场景（--scenario，system.SCENARIOS，INTEGRATION S）：quick 默认 "informative"（L90 + ROI 0.35 m + 遮挡馈源→探头视距，
  full 耦合，条件 M0 / M1 / M1b@+25 / Rapp@+25 / M3）；"baseline" = 原默认场景（born，偏置主导）；full 模式默认 baseline。
每个格点（任务 × 条件 × 模式 × 种子）：
  1. DSPSA 在该噪声下训练（每个噪声点一次训练；可测目标，方案 §4.7），评估同噪声下的有噪声准确率（n_noise 次独立读数）：
     "dspsa" = train.run_dspsa 默认变体的输出（INTEGRATION T：1-bit sign_acc，判对优先目标 + sign-SPSA，输出即 pocket 码本）；
     旧变体（--dspsa-variant subset）另有 "dspsa_pocket" = 训练中测到的可测损失最小的码本（独立噪声复评；审查 Q-2 / S1-0）；
  2. 无噪声训练的 1-bit oracle（accuracy 目标 + 最小裕量 polish，审查 Q-1；与实验 A 同一实现）与连续相位 oracle 码本，
     在稠密噪声网格上评估有噪声准确率（"这个结构最好的码本对噪声有多鲁棒"，区分"噪声下训不动"与"噪声下判不对"）；
     另跑默认 softplus 损失 oracle 只为实验 E 判据的 L*（不画曲线）。
DSPSA 训练到位（审查 Q-2，与实验 A 同一规则）：旧变体 = 实验 E 判据 (L0 − L) ≥ 0.9 (L0 − L*) 且 loss_meas_clean 离开
  τ·ln2 平台（L0 = 初始码本，L* = 同种子 1-bit oracle 的无噪声可测目标）；sign_acc（判对优先，不优化 softplus）= 无噪声
  acc ≥ 0.9·acc*（acc* = 同种子 1-bit accuracy oracle，INTEGRATION T）。某条件在最高名义 SNR（噪声可忽略）上训练到位的种子
  < 80% 时，其 DSPSA 曲线、所需 SNR 与 RQ3 差值只反映训练失败，汇总里标"未训练好"并不给出 RQ3 差值。
所需 SNR（方案 §7）= metrics.required_snr_db(SNR, acc, 0.9)：由种子平均曲线求一次，也逐种子求（报达标种子数）。
RQ3 的两种看法：横轴 = 各自 SNR（非线性能否在给定 SNR 下更好地利用信号）；横轴 = 绝对噪声功率 dBm（同一检测噪声地板下
增益的作用：M0 的探头信号弱，同一 P_n 下 SNR 更低）。
字母任务（审查 Q-5）：只跑 intra / full——born（volume_rr）下 "metal" 是 εr_eff = 4 介质（INTEGRATION B3）；材料由
  --letter-material 显式传给字母任务（默认 metal，方案 §6.2），每行记录任务的实际材料与 effective_eps。

用法：python -m ris_nlsim.experiments.exp_b_snr [--quick] [--workers N] [--seeds S] [--out DIR] [选项]
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import math
import os
import sys
import time

import numpy as np

from . import common
from .exp_a_parity import (POLISH_SWEEPS, TRAINED_RUN_FRAC, _setup_fonts, chance_loss_of, dspsa_variant_text,
                           e_reach, gain_pairs, is_rapp, parse_cfg_overrides, parse_condition, run_accuracy_oracle,
                           run_cont_plus, scenario_overrides, system_info, CONT_PLUS_INIT_RESTARTS, PLATEAU_FRAC,
                           REACH_FRAC)

EXP = "exp_b"
ALL_CONDITIONS = ("M0", "M1@M3", "M1b@M3", "M3", "M4", "M1@M4", "M1b@M4")
INFORMATIVE_CONDITIONS = ("M0", "M1", "M1b@+25", "Rapp@+25", "M3")      # informative 场景 quick 默认（INTEGRATION S）
TASKS = ("parity3", "parity2", "letters")
LETTER_MODES = ("intra", "full")              # 审查 Q-5：born 下金属退化为 εr_eff = 4 介质
SNR_QUICK = (-10.0, 0.0, 10.0, 25.0, 40.0)
SNR_FULL = (-15.0, -7.5, 0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5)
SNR_DENSE = tuple(float(v) for v in np.arange(-20.0, 60.0 + 1e-9, 5.0))     # oracle 码本评估用（只评估，便宜）
N_REF_CODEBOOKS = 16
DSPSA_TRAINERS = ("dspsa", "dspsa_pocket")
TRAINER_NAMES = {"dspsa": "DSPSA（噪声下训练）", "dspsa_pocket": "DSPSA pocket（噪声下训练）",
                 "oracle": "1-bit oracle 码本", "cont": "连续 oracle 码本"}
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_INK, _INK2, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ============================================================ 系统
_SYS_CACHE: "collections.OrderedDict[tuple, object]" = collections.OrderedDict()


def _task_kw(task: str, material: str | None = None) -> tuple[str, dict]:
    if task == "parity3":
        return "parity", {"n": 3}
    if task == "parity2":
        return "parity", {"n": 2}
    if task == "letters":
        return "letter", ({} if material is None else {"material": material})   # 审查 Q-5：材料显式传给字母任务
    raise ValueError(f"未知任务 {task!r}")


def base_system(job: dict):
    """无探头噪声的基础系统（每 worker 缓存）；噪声系统由 derive(probe_noise_dbm=…) 得到（复用样本响应）。
    job["material"]：奇偶任务经 cfg.material，字母任务显式传给 LetterTask（审查 Q-5）。"""
    from ..system import SystemConfig, make_system

    key = (job["preset"], job["task"], job["cond"], job["mode"], job["material"],
           tuple(sorted((job.get("cfg_over") or {}).items())))
    if key in _SYS_CACHE:
        _SYS_CACHE.move_to_end(key)
        return _SYS_CACHE[key]
    kind, kw = _task_kw(job["task"], job["material"] if job["task"] == "letters" else None)
    cfg = SystemConfig(preset=job["preset"], object_mode=job["mode"], material=job["material"],
                       **parse_condition(job["cond"]), **(job.get("cfg_over") or {}))
    s = make_system(cfg, kind, **kw)
    _SYS_CACHE[key] = s
    while len(_SYS_CACHE) > 6:
        _SYS_CACHE.popitem(last=False)
    return s


def reference_signal_dbm(s) -> float:
    """P_ref：16 个随机码本（固定 Generator，与种子无关）× 全部样本 × 全部探头的无噪声探头功率中位数 (dBm)。"""
    rng = np.random.default_rng(20261019)
    p = [s.forward(s.phases_from_params(s.random_params(rng)), noisy=False).p_w for _ in range(N_REF_CODEBOOKS)]
    return float(10.0 * np.log10(np.median(np.concatenate([x.ravel() for x in p]))) + 30.0)


def signal_dbm(s, phases: dict) -> float:
    """训练后码本的无噪声探头功率中位数 (dBm)（全部样本 × 探头；SNR 轴的分子）。"""
    m = s.forward(phases, noisy=False)
    return float(10.0 * np.log10(np.median(m.p_w)) + 30.0)


def _noisy_system(s, noise_dbm: float):
    return s.derive(probe_noise_dbm=float(noise_dbm))


def _eval_lite(ev: dict) -> dict:
    keys = ("acc_clean", "acc_clean_valid", "all_correct", "all_correct_valid", "margin_db", "margin_db_unflagged",
            "acc_noisy", "acc_noisy_std", "acc_noisy_valid", "oscillating_run", "n_flagged", "noisy_flagged_frac",
            "loss_clean", "loss_meas_clean", "pa_in_rel_p1db_db_mean", "pa_in_rel_p1db_db_min", "pa_in_rel_p1db_db_max",
            "pa_in_open_rel_p1db_db_mean", "pa_in_delta_rel_p1db_db_max", "null_depth_db", "loop_gain_op_max",
            "probe_dbm", "params", "p_feed_dbm", "voxel_d", "material", "object_model", "effective_eps")
    return {k: ev.get(k) for k in keys if k in ev}


def task_material_info(s) -> dict:
    """任务的实际材料 / 极化率模型 / effective_eps（审查 Q-5：不再用 job 的 material 字段）。"""
    d = s.task.describe()
    return {"material": None if getattr(s.task, "material", None) is None else str(s.task.material),
            "object_model": d.get("model"), "effective_eps": d.get("effective_eps")}


# ============================================================ job（模块顶层，可 pickle）
def run_job(job: dict) -> dict:
    """job["kind"] = "dspsa"：在 job["snr_target_db"] 对应的噪声下训练 DSPSA（最终迭代 + pocket）并评估；
    job["kind"] = "oracle"：无噪声训练 1-bit accuracy oracle（+ polish）/ 连续 oracle，再在稠密噪声网格上评估；
    另记 softplus loss-oracle 的可测目标作实验 E 判据的 L*。"""
    from .. import train

    t0 = time.perf_counter()
    s = base_system(job)
    p_ref = reference_signal_dbm(s)
    mat = task_material_info(s)
    row = {"exp": "B", "scenario": job.get("scenario", "baseline"), "kind": job["kind"], "task": job["task"],
           "cond": job["cond"], "mode": job["mode"],
           "material": mat["material"], "effective_eps": mat["effective_eps"], "object_model": mat["object_model"],
           "seed": int(job["seed"]), "p_ref_dbm": p_ref, "sys": system_info_any(s), "chance_loss": chance_loss_of(s)}
    cell = (job["preset"], job["task"], job["mode"], job["material"], int(job["seed"]))
    n_noise = int(job.get("n_noise_eval", 50))
    if job["kind"] == "dspsa":
        snr_t = float(job["snr_target_db"])
        noise = p_ref - snr_t
        sn = _noisy_system(s, noise)
        rng = common.seed_rng("expB-dspsa", *cell)        # 公共随机数：与条件、噪声点无关
        d = train.run_dspsa(sn, int(job["dspsa_iters"]), rng, eval_every=int(job.get("eval_every", 250)),
                            n_noise_eval=n_noise, pocket=True, variant=job.get("dspsa_variant", "auto"))
        ph = sn.phases_from_params(d["params"])
        sig = signal_dbm(sn, ph)
        ev = _eval_lite(d["eval"])
        curve = d.get("curve") or []
        row.update(snr_target_db=snr_t, noise_dbm=noise, signal_dbm=sig, snr_db=sig - noise, eval=ev,
                   acc_noisy_valid=ev.get("acc_noisy_valid"), acc_clean_valid=ev.get("acc_clean_valid"),
                   oscillating_run=ev.get("oscillating_run"), loss_meas_clean=ev.get("loss_meas_clean"),
                   L0=curve[0].get("loss_meas_clean") if curve else None,
                   train_flagged_frac=float(d["n_flagged_measurements"]) / max(1, int(d["n_measurements"])),
                   curve=curve, n_measurements=int(d["n_measurements"]), dspsa_time_s=d["time_s"],
                   variant=d["variant"], train_objective=d.get("train_objective"))
        pk = d.get("pocket") or {}
        if d["variant"] != "sign_acc" and pk.get("params") is not None and pk.get("eval") is not None:
            pev = _eval_lite(pk["eval"])                  # 独立噪声复评（训练噪声下；审查 Q-2，赢家诅咒）
            psig = signal_dbm(sn, sn.phases_from_params(pk["params"]))
            row["pocket"] = {"eval": pev, "signal_dbm": psig, "snr_db": psig - noise,
                             "acc_noisy_valid": pev.get("acc_noisy_valid"), "acc_clean_valid": pev.get("acc_clean_valid"),
                             "oscillating_run": pev.get("oscillating_run"), "loss_meas_clean": pev.get("loss_meas_clean"),
                             "loss_measured": pk.get("loss_measured"), "at_measurement": pk.get("at_measurement")}
        else:
            row["pocket"] = None
    else:
        rng_e = common.seed_rng("expB-evalnoise", *cell)
        res = {}
        o = run_accuracy_oracle(s, common.seed_rng("expB-oracle", *cell), common.seed_rng("expB-polish", *cell),
                                int(job["oracle_restarts"]), int(job["oracle_sweeps"]),
                                int(job.get("polish_sweeps", POLISH_SWEEPS)), 0, common.seed_rng("expB-oracle-eval", *cell))
        res["oracle"] = (s.phases_from_params(o["params"]), o)
        if job.get("cont"):
            cseed = int(common.seed_rng("expB-cont", *cell).integers(0, 2 ** 31 - 1))
            if job.get("cont_plus"):          # 设计初值 + soft-min 精修（INTEGRATION S，与实验 A 同一实现；非奇偶任务自动退化）
                c = run_cont_plus(s, cseed, int(job["cont_restarts"]), int(job["cont_steps"]), 0,
                                  common.seed_rng("expB-cont-eval", *cell),
                                  init_restarts=int(job.get("cont_init_restarts", CONT_PLUS_INIT_RESTARTS)),
                                  refine_steps=int(job.get("cont_refine_steps", 300)))
            else:
                c = train.run_oracle_continuous(s, n_restarts=int(job["cont_restarts"]), steps=int(job["cont_steps"]),
                                                seed=cseed, n_noise_eval=0)
            res["cont"] = (c["phases"], c)
        if job.get("oracle_loss", True):
            # 实验 E 判据的 L*（softplus 损失 oracle；同一随机重启起点）
            ol = train.run_oracle_bits(s, common.seed_rng("expB-oracle", *cell), n_restarts=int(job["oracle_restarts"]),
                                       max_sweeps=int(job["oracle_sweeps"]), n_noise_eval=0, objective="loss")
            row["oracle_loss"] = {"eval": _eval_lite(ol["eval"]), "time_s": ol["time_s"],
                                  "loss_meas_clean": ol["eval"].get("loss_meas_clean")}
        else:
            row["oracle_loss"] = None
        sig = {tr: signal_dbm(s, ph) for tr, (ph, _) in res.items()}
        pts = {tr: [] for tr in res}
        for snr_t in job["snr_dense"]:
            noise = p_ref - float(snr_t)
            sn = _noisy_system(s, noise)                     # 每个噪声点 derive 一次（复用样本响应）
            for tr, (ph, _) in res.items():
                ev = sn.evaluate_phases(ph, rng_e, n_noise=n_noise)
                pts[tr].append({"snr_target_db": float(snr_t), "noise_dbm": noise, "snr_db": sig[tr] - noise,
                                "acc_noisy": ev.get("acc_noisy"), "acc_noisy_valid": ev.get("acc_noisy_valid"),
                                "acc_noisy_std": ev.get("acc_noisy_std")})
        for tr, (ph, r) in res.items():
            ev0 = _eval_lite(r["eval"])
            if tr == "cont":
                ev0["phases"] = {k: np.asarray(v).tolist() for k, v in ph.items()}
            row[tr] = {"eval": ev0, "signal_dbm": sig[tr], "curve": pts[tr], "time_s": r["time_s"],
                       "acc_clean_valid": ev0.get("acc_clean_valid"), "oscillating_run": ev0.get("oscillating_run"),
                       "loss_meas_clean": ev0.get("loss_meas_clean"), "margin_db": ev0.get("margin_db_unflagged"),
                       "all_correct": ev0.get("all_correct")}
            if tr == "oracle":
                row[tr].update(stage1_acc_clean=r.get("stage1_acc_clean"), stage1_margin_db=r.get("stage1_margin_db"),
                               polish_sweeps=r.get("polish_sweeps"))
            if tr == "cont" and r.get("source") is not None:
                row[tr].update(cont_source=r["source"], cont_candidates=r.get("candidates"))
    row["run_time_s"] = time.perf_counter() - t0
    return row


def system_info_any(s) -> dict:
    """system_info 的任务无关版本（字母任务没有 n_vox_per_object / spacing）。"""
    try:
        return system_info(s)
    except AttributeError:
        d = {"label": s.label, "p_feed_dbm": float(10.0 * np.log10(s.p_feed_w) + 30.0), "backoff_db": s.backoff_db,
             "voxel_d": float(s.task.voxel_d), "lambda_over_voxel_d": float(s.lam / s.task.voxel_d),
             "object_model": s.object_model, "effective_eps": s.task.describe().get("effective_eps")}
        if s.bussgang_gain is not None:
            d["bussgang_gain_db"] = float(20.0 * np.log10(abs(s.bussgang_gain)))
        return d


# ============================================================ 网格
def task_modes(task: str, modes) -> list[str]:
    """任务实际运行的物体模式：字母任务去掉 born（审查 Q-5：volume_rr 下 metal = εr_eff 4 介质）。"""
    return [m for m in modes if not (task == "letters" and m not in LETTER_MODES)]


def build_jobs(args) -> list[dict]:
    jobs = []
    skipped = []
    for task in args.tasks:
        modes = task_modes(task, args.modes)
        skipped += [{"task": task, "mode": m, "reason": "born 下 metal 是 εr_eff = 4 介质（INTEGRATION B3）"}
                    for m in args.modes if m not in modes]
        mat = getattr(args, "letter_material", "metal") if task == "letters" else args.material
        for mode in modes:
            for cond in args.conditions:
                for seed in range(args.seeds):
                    base = {"exp": "B", "preset": args.preset, "task": task, "cond": cond, "mode": mode,
                            "material": mat, "seed": int(seed), "n_noise_eval": int(args.n_noise),
                            "cfg_over": dict(args.cfg_over), "scenario": getattr(args, "scenario", "baseline")}
                    if args.oracle_curves:
                        jobs.append(dict(base, kind="oracle", oracle_restarts=int(args.oracle_restarts),
                                         oracle_sweeps=int(args.oracle_sweeps), cont=bool(args.oracle_cont),
                                         cont_restarts=int(args.cont_restarts), cont_steps=int(args.cont_steps),
                                         polish_sweeps=int(getattr(args, "polish_sweeps", POLISH_SWEEPS)),
                                         oracle_loss=True, snr_dense=[float(v) for v in args.snr_dense],
                                         cont_plus=bool(getattr(args, "cont_plus", False)),
                                         cont_init_restarts=int(getattr(args, "cont_init_restarts",
                                                                        CONT_PLUS_INIT_RESTARTS)),
                                         cont_refine_steps=int(getattr(args, "cont_refine_steps", 300))))
                    for snr in args.snr:
                        jobs.append(dict(base, kind="dspsa", snr_target_db=float(snr),
                                         dspsa_iters=int(args.dspsa_iters), eval_every=250,
                                         dspsa_variant=getattr(args, "dspsa_variant", "auto")))
    args.skipped_cells = skipped
    jobs.sort(key=lambda j: (-(3 if is_rapp(j["cond"]) else 1) - (2 if j["kind"] == "oracle" else 0), j["task"],
                             j["mode"]))
    return jobs


# ============================================================ DSPSA 训练诊断（审查 Q-2）
def annotate_dspsa(rows: list[dict]) -> None:
    """给 DSPSA 行补上（原地）：plateau（loss_meas_clean ≥ 0.9·chance）、reach90（实验 E 判据，L* = 同 task/mode/cond/seed
    的 1-bit oracle 与 loss-oracle 的无噪声可测目标较小者）、trained = reach90 且不在平台；最终迭代与 pocket 各一份。"""
    lstar, astar = {}, {}
    for r in rows:
        if "error" in r or r.get("kind") != "oracle":
            continue
        v = [x["loss_meas_clean"] for x in (r.get("oracle"), r.get("oracle_loss"))
             if x and x.get("loss_meas_clean") is not None]
        if v:
            lstar[(r["task"], r["mode"], r["cond"], r["seed"])] = float(min(v))
        a = (r.get("oracle") or {}).get("acc_clean_valid")
        if a is not None:
            astar[(r["task"], r["mode"], r["cond"], r["seed"])] = float(a)
    for r in rows:
        if "error" in r or r.get("kind") != "dspsa":
            continue
        Ls = lstar.get((r["task"], r["mode"], r["cond"], r["seed"]))
        r["Lstar"] = Ls
        ch = r.get("chance_loss")
        As = astar.get((r["task"], r["mode"], r["cond"], r["seed"]))
        for tgt in (r, r.get("pocket")):
            if not tgt:
                continue
            L = tgt.get("loss_meas_clean")
            tgt["plateau"] = None if (L is None or ch is None) else bool(float(L) >= PLATEAU_FRAC * float(ch))
            if r.get("train_objective") == "accuracy_first":
                # INTEGRATION T：判对优先训练器按准确率判据（无噪声 acc ≥ 0.9·acc*），平台只作诊断
                a = tgt.get("acc_clean_valid")
                tgt["reach90"] = None if (a is None or As is None) else bool(float(a) >= REACH_FRAC * As - 1e-12)
                tgt["trained"] = tgt["reach90"]
                continue
            tgt["reach90"] = e_reach(r.get("L0"), L, Ls)
            tgt["trained"] = (None if tgt["reach90"] is None or tgt["plateau"] is None
                              else bool(tgt["reach90"] and not tgt["plateau"]))


def trained_table(rows: list[dict]) -> dict:
    """{(task, mode, cond, trainer): {"trained": bool, "k": 训练到位种子数, "n": 种子数, "snr": 最高名义 SNR,
    "plateau_k", "plateau_n"（全部 SNR 点）}}：在最高名义 SNR（噪声可忽略）上训练到位的种子比例 ≥ TRAINED_RUN_FRAC 才算
    该条件的 DSPSA 训练到位（审查 Q-2）。"""
    out = {}
    groups = collections.defaultdict(list)
    for r in rows:
        if "error" in r or r.get("kind") != "dspsa":
            continue
        groups[(r["task"], r["mode"], r["cond"])].append(r)
    for (task, mode, cond), rs in groups.items():
        smax = max(r["snr_target_db"] for r in rs)
        top = [r for r in rs if abs(r["snr_target_db"] - smax) < 1e-9]
        for tr in DSPSA_TRAINERS:
            def get(r):
                return r if tr == "dspsa" else r.get("pocket")
            if tr == "dspsa_pocket" and all(get(r) is None for r in rs):
                continue                                # sign_acc：输出即 pocket，不单列（INTEGRATION T）
            tv = [get(r).get("trained") for r in top if get(r) is not None]
            known = [bool(v) for v in tv if v is not None]
            pl = [bool(get(r).get("plateau")) for r in rs if get(r) is not None and get(r).get("plateau") is not None]
            k, n = sum(known), len(top)
            out[(task, mode, cond, tr)] = {"trained": bool(n and len(known) == n and k >= TRAINED_RUN_FRAC * n - 1e-12),
                                           "k": k, "n": n, "snr": smax, "plateau_k": sum(pl), "plateau_n": len(pl)}
    return out


# ============================================================ 汇总
def _curve_points(rows: list[dict]) -> list[dict]:
    """把两类行展开成统一的曲线点：{task, cond, mode, seed, trainer, snr_target_db, noise_dbm, snr_db, acc}。"""
    pts = []
    for r in rows:
        if "error" in r:
            continue
        base = {k: r[k] for k in ("task", "cond", "mode", "seed")}
        base["p_ref_dbm"] = r["p_ref_dbm"]
        if r["kind"] == "dspsa":
            pts.append(dict(base, trainer="dspsa", snr_target_db=r["snr_target_db"], noise_dbm=r["noise_dbm"],
                            snr_db=r["snr_db"], signal_dbm=r["signal_dbm"], acc=r["acc_noisy_valid"],
                            acc_clean=r["acc_clean_valid"], osc=bool(r.get("oscillating_run"))))
            pk = r.get("pocket")
            if pk:
                pts.append(dict(base, trainer="dspsa_pocket", snr_target_db=r["snr_target_db"], noise_dbm=r["noise_dbm"],
                                snr_db=pk["snr_db"], signal_dbm=pk["signal_dbm"], acc=pk["acc_noisy_valid"],
                                acc_clean=pk["acc_clean_valid"], osc=bool(pk.get("oscillating_run"))))
        else:
            for tr in ("oracle", "cont"):
                if not r.get(tr):
                    continue
                for p in r[tr]["curve"]:
                    pts.append(dict(base, trainer=tr, snr_target_db=p["snr_target_db"], noise_dbm=p["noise_dbm"],
                                    snr_db=p["snr_db"], signal_dbm=r[tr]["signal_dbm"], acc=p["acc_noisy_valid"],
                                    acc_clean=r[tr]["acc_clean_valid"], osc=bool(r[tr].get("oscillating_run"))))
    return pts


def _cond_idx(c: str) -> int:
    return ALL_CONDITIONS.index(c) if c in ALL_CONDITIONS else 99


def curve_table(pts: list[dict]) -> list[dict]:
    keys = ["task", "mode", "cond", "trainer", "snr_target_db"]
    t = common.aggregate(pts, keys, ["acc", "snr_db", "noise_dbm", "signal_dbm", "acc_clean", "osc"])
    t.sort(key=lambda r: (r["task"], r["mode"], _cond_idx(r["cond"]), r["cond"], r["trainer"], r["snr_target_db"]))
    return t


def required_snr_table(pts: list[dict], ct: list[dict], target: float = 0.9, trained: dict | None = None) -> list[dict]:
    """所需 SNR：(a) 种子平均曲线（各名义 SNR 点的平均 SNR 与平均准确率）上求一次；(b) 逐种子求，报均值/CI 与达标种子数。
    trained（trained_table）给出时，DSPSA 行另记 trained / trained_k / trained_n（审查 Q-2）。"""
    from ..metrics import required_snr_db

    out = []
    groups = collections.OrderedDict()
    for r in ct:
        groups.setdefault((r["task"], r["mode"], r["cond"], r["trainer"]), []).append(r)
    by_seed = collections.defaultdict(list)
    for p in pts:
        by_seed[(p["task"], p["mode"], p["cond"], p["trainer"], p["seed"])].append(p)
    for key, rs in groups.items():
        rs = [r for r in rs if r["acc_n"] and math.isfinite(r["snr_db_mean"])]
        if not rs:
            continue
        snr = np.array([r["snr_db_mean"] for r in rs])
        acc = np.array([r["acc_mean"] for r in rs])
        req_mean = required_snr_db(snr, acc, target)
        per = []
        n_seed = 0
        for (t, m, c, tr, sd), ps in by_seed.items():
            if (t, m, c, tr) != key:
                continue
            ps = [p for p in ps if p["acc"] is not None]
            if len(ps) < 2:
                continue
            n_seed += 1
            v = required_snr_db([p["snr_db"] for p in ps], [p["acc"] for p in ps], target)
            per.append(v)
        fin = [v for v in per if math.isfinite(v)]
        row = {"task": key[0], "mode": key[1], "cond": key[2], "trainer": key[3],
               "req_snr_db_meancurve": req_mean, "n_seeds": n_seed, "n_seeds_reaching": len(fin),
               "req_snr_db_seed_mean": float(np.mean(fin)) if fin else float("nan"),
               "req_snr_db_seed_ci95": (float(common._t975(len(fin) - 1) * np.std(fin, ddof=1) / np.sqrt(len(fin)))
                                        if len(fin) > 1 else float("nan")),
               "acc_at_max_snr": float(acc[np.argmax(snr)]),
               # 无噪声 acc 取最高 SNR 点：DSPSA 每个 SNR 各训一套码本，rs[0]（最低 SNR）的码本不代表"训练好"的结果
               "acc_clean_mean": rs[int(np.argmax(snr))].get("acc_clean_mean"),
               "p_ref_dbm": None}
        if trained is not None and key[3] in DSPSA_TRAINERS:
            tt = trained.get(key, {})
            row.update(trained=bool(tt.get("trained", False)), trained_k=tt.get("k"), trained_n=tt.get("n"))
        out.append(row)
    return out


def _fmt(v, f=".2f"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return format(float(v), f)


def curve_md(ct: list[dict], snr_targets: list[float], trainer: str) -> str:
    rows = [r for r in ct if r["trainer"] == trainer]
    if not rows:
        return "_(无)_\n"
    tg = sorted({r["snr_target_db"] for r in rows if any(abs(r["snr_target_db"] - s) < 1e-9 for s in snr_targets)})
    cols = ["任务", "模式", "条件"] + [f"名义 {s:g} dB：acc [CI] / SNR" for s in tg]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault((r["task"], r["mode"], r["cond"]), {})[r["snr_target_db"]] = r
    for (task, mode, cond), d in groups.items():
        vals = [task, mode, cond]
        for s in tg:
            r = d.get(s)
            if r is None or not r["acc_n"]:
                vals.append("–")
                continue
            ci = "" if not (r["acc_n"] > 1 and math.isfinite(r["acc_lo"])) else f" [{r['acc_lo']:.2f},{r['acc_hi']:.2f}]"
            vals.append(f"{r['acc_mean']:.2f}{ci} / {_fmt(r['snr_db_mean'], '.1f')}")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def req_md(rt: list[dict]) -> str:
    cols = ["任务", "模式", "条件", "训练器", "所需 SNR dB（平均曲线）", "所需 SNR dB（逐种子均值 ± CI）", "达标种子",
            "最高 SNR 处 acc", "无噪声 acc", "训练到位（最高名义 SNR）"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rt:
        ci = "" if not math.isfinite(r["req_snr_db_seed_ci95"]) else f" ± {r['req_snr_db_seed_ci95']:.1f}"
        tr_cell = "—"
        warn = ""
        if r["trainer"] in DSPSA_TRAINERS and "trained" in r:
            tr_cell = f"{r.get('trained_k')}/{r.get('trained_n')}" + ("" if r["trained"] else "（未训练好）")
            warn = "" if r["trained"] else "（未训练好）"
        lines.append("| " + " | ".join([r["task"], r["mode"], r["cond"], TRAINER_NAMES[r["trainer"]],
                                         _fmt(r["req_snr_db_meancurve"], ".1f") + warn,
                                         _fmt(r["req_snr_db_seed_mean"], ".1f") + ci,
                                         f"{r['n_seeds_reaching']}/{r['n_seeds']}", _fmt(r["acc_at_max_snr"]),
                                         _fmt(r["acc_clean_mean"]), tr_cell]) + " |")
    return "\n".join(lines) + "\n"


def oppoint_md(rows: list[dict]) -> str:
    ok = [r for r in rows if "error" not in r]
    groups = collections.OrderedDict()
    for r in sorted(ok, key=lambda r: (r["task"], r["mode"], _cond_idx(r["cond"]), r["cond"])):
        groups.setdefault((r["task"], r["mode"], r["cond"]), []).append(r)
    cols = ["任务", "模式", "条件", "材料（实际，εr_eff）", "P_feed dBm", "backoff dB", "voxel_d mm (λ/d)",
            "P_ref dBm（随机码本探头中位数）", "DSPSA 码本信号中位数 dBm", "oracle / 连续 oracle 码本信号 dBm",
            "闭环 PA 输入 rel P1dB：DSPSA / oracle / 连续 oracle", "DSPSA 训练测量被标记比例",
            "自激 run（DSPSA / oracle / 连续）"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for (task, mode, cond), rs in groups.items():
        ds = [r for r in rs if r["kind"] == "dspsa"]
        os_ = [r for r in rs if r["kind"] == "oracle"]
        sysi = rs[0]["sys"]

        def pa_mean(evs):
            v = [e.get("pa_in_rel_p1db_db_mean") for e in evs if e and e.get("pa_in_rel_p1db_db_mean") is not None]
            return _fmt(np.mean(v) if v else None, ".1f")
        sig_o = [r["oracle"]["signal_dbm"] for r in os_ if r.get("oracle")]
        sig_c = [r["cont"]["signal_dbm"] for r in os_ if r.get("cont")]
        osc = (f"{sum(bool(r.get('oscillating_run')) for r in ds)}/{len(ds)} / "
               f"{sum(bool(r['oracle'].get('oscillating_run')) for r in os_ if r.get('oracle'))}/{len(os_)} / "
               f"{sum(bool(r['cont'].get('oscillating_run')) for r in os_ if r.get('cont'))}/{len(sig_c)}")
        vd = sysi.get("voxel_d")
        mat = f"{rs[0].get('material')} ({rs[0].get('effective_eps')})"
        lines.append("| " + " | ".join([
            task, mode, cond, mat, _fmt(sysi.get("p_feed_dbm"), ".2f"), _fmt(sysi.get("backoff_db"), "+.0f"),
            "–" if vd is None else f"{vd * 1e3:.2f} ({_fmt(sysi.get('lambda_over_voxel_d'), '.1f')})",
            _fmt(rs[0]["p_ref_dbm"], ".1f"), _fmt(np.mean([r["signal_dbm"] for r in ds]) if ds else None, ".1f"),
            f"{_fmt(np.mean(sig_o) if sig_o else None, '.1f')} / {_fmt(np.mean(sig_c) if sig_c else None, '.1f')}",
            f"{pa_mean([r['eval'] for r in ds])} / {pa_mean([r['oracle']['eval'] for r in os_ if r.get('oracle')])} / "
            f"{pa_mean([r['cont']['eval'] for r in os_ if r.get('cont')])}",
            _fmt(np.mean([r["train_flagged_frac"] for r in ds]) if ds else None, ".3f"), osc]) + " |")
    return "\n".join(lines) + "\n"


RQ3_PAIRS = (("M3", "M1@M3"), ("M3", "M1b@M3"), ("M4", "M1@M4"), ("M4", "M1b@M4"), ("M1@M3", "M0"))


def rq3_pairs(conds, any_power_m1: bool = False) -> list[tuple[str, str]]:
    """RQ3 的条件对：exp_a.gain_pairs（Rapp 条件 − 同回退 M1 / M1b；无同回退 M1 时与任一 M1 配对，线性中继无噪声判决与
    源功率无关）+ (第一个 M1 类条件 − M0)。默认条件集下与 RQ3_PAIRS 相同。"""
    conds = list(dict.fromkeys(conds))
    out = gain_pairs(conds, any_power_m1)
    lin = [c for c in conds if c.startswith("M1") and not c.startswith("M1b")]
    if lin and "M0" in conds:
        out.append((lin[0], "M0"))
    return out


def rq3_md(ct: list[dict], snr_points: dict | None = None, trained: dict | None = None, pairs=None) -> str:
    """RQ3：同一名义 SNR 点上 Mx − M1@Mx、Mx − M1b@Mx（审查 Q-4）与 M1 − M0 的准确率差（种子平均）。
    snr_points：{训练器: 要列出的名义 SNR 点}（None → 全部）。trained（trained_table）给出时，DSPSA / pocket 的差值只在
    两个条件都训练到位时给出，否则写"未训练好"（审查 Q-2）。"""
    pairs = list(RQ3_PAIRS if pairs is None else pairs)
    if not pairs:
        return "_(无可比较的条件对)_\n"
    head = " | ".join(f"{a} − {b}" for a, b in pairs)
    lines = [f"| 任务 | 模式 | 训练器 | 名义 SNR dB | {head} |", "|" + "---|" * (4 + len(pairs))]
    idx = {(r["task"], r["mode"], r["cond"], r["trainer"], r["snr_target_db"]): r for r in ct}
    keys = sorted({(r["task"], r["mode"], r["trainer"], r["snr_target_db"]) for r in ct})
    if snr_points is not None:
        keys = [k for k in keys if any(abs(k[3] - v) < 1e-9 for v in snr_points.get(k[2], []))]
    any_row = False
    for (task, mode, tr, s) in keys:
        def acc(c):
            r = idx.get((task, mode, c, tr, s))
            return None if r is None or not r["acc_n"] else r["acc_mean"]

        def ok(c):
            if trained is None or tr not in DSPSA_TRAINERS:
                return True
            return bool(trained.get((task, mode, c, tr), {}).get("trained", False))

        def diff(a, b):
            va, vb = acc(a), acc(b)
            if va is None or vb is None:
                return "–"
            if not (ok(a) and ok(b)):
                return "未训练好"
            return f"{va - vb:+.2f}"
        cells = [diff(a, b) for a, b in pairs]
        if all(c == "–" for c in cells):
            continue
        any_row = True
        lines.append(f"| {task} | {mode} | {TRAINER_NAMES[tr]} | {s:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n" if any_row else "_(无可比较的条件对)_\n"


def plot_curves(ct: list[dict], outdir: str, xkey: str, fname: str, xlabel: str,
                trained: dict | None = None) -> str | None:
    """准确率曲线：行 = (任务, 模式)，列 = 训练器；每条线一个条件（固定顺序的类别色），带 95% CI 带；0.9 参考线。
    trained 给出时，未训练到位的 DSPSA / pocket 曲线画成浅色点线（审查 Q-2）。"""
    import matplotlib.pyplot as plt

    if not ct:
        return None
    panels = sorted({(r["task"], r["mode"]) for r in ct})
    trs = [t for t in ("dspsa", "dspsa_pocket", "oracle", "cont") if any(r["trainer"] == t for r in ct)]
    conds = [c for c in ALL_CONDITIONS if any(r["cond"] == c for r in ct)]
    conds += sorted({r["cond"] for r in ct} - set(conds))
    fig, axes = plt.subplots(len(panels), len(trs), figsize=(4.2 * len(trs), 3.0 * len(panels)), squeeze=False,
                             sharey=True)
    for i, (task, mode) in enumerate(panels):
        for j, tr in enumerate(trs):
            ax = axes[i][j]
            for c_i, cond in enumerate(conds):
                rs = sorted([r for r in ct if r["task"] == task and r["mode"] == mode and r["trainer"] == tr
                             and r["cond"] == cond and r["acc_n"]], key=lambda r: r[f"{xkey}_mean"])
                if not rs:
                    continue
                x = np.array([r[f"{xkey}_mean"] for r in rs])
                y = np.array([r["acc_mean"] for r in rs])
                lo = np.array([r["acc_lo"] if math.isfinite(r["acc_lo"]) else r["acc_mean"] for r in rs])
                hi = np.array([r["acc_hi"] if math.isfinite(r["acc_hi"]) else r["acc_mean"] for r in rs])
                col = _SERIES[c_i % len(_SERIES)]
                faded = (trained is not None and tr in DSPSA_TRAINERS
                         and not trained.get((task, mode, cond, tr), {}).get("trained", False))
                ax.fill_between(x, lo, hi, color=col, alpha=0.05 if faded else 0.12, lw=0)
                ls = ":" if faded else ("--" if "@M4" in cond else "-")
                ax.plot(x, y, ls, color=col, lw=1.2 if faded else 2, alpha=0.45 if faded else 1.0,
                        label=cond if (i == 0 and j == 0) else None,
                        marker="o" if tr in DSPSA_TRAINERS else None, ms=4 if faded else 5)
            ax.axhline(0.9, color=_INK2, lw=0.8, ls="--")
            ax.set_ylim(0.0, 1.05)
            ax.grid(color=_GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=_INK2, labelsize=8)
            ax.set_title(f"{task} · {mode} · {TRAINER_NAMES[tr]}", fontsize=9, color=_INK)
            if i == len(panels) - 1:
                ax.set_xlabel(xlabel, fontsize=8, color=_INK2)
            if j == 0:
                ax.set_ylabel("有噪声准确率", fontsize=8, color=_INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(conds), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    if trained is not None and any(t in DSPSA_TRAINERS for t in trs):
        fig.text(0.5, -0.01, "DSPSA / pocket：浅色点线 = 该条件在最高名义 SNR 上未训练到位（实验 E 判据 + 离开 τ·ln2 平台，"
                 "≥ 80% 种子），曲线只反映训练失败", ha="center", fontsize=8, color=_INK2)
    fig.tight_layout()
    return common.savefig(fig, os.path.join(outdir, fname))


def plateau_md(tt: dict) -> str:
    """DSPSA 训练诊断表（审查 Q-2）：平台比例（全部 SNR 点）与最高名义 SNR 上训练到位的种子数。"""
    if not tt:
        return "_(无 DSPSA 行)_\n"
    lines = ["| 任务 | 模式 | 条件 | 训练器 | 平台（loss_meas_clean ≥ 0.9·chance，全部 SNR 点） | 训练到位 @最高名义 SNR | 判定 |",
             "|---|---|---|---|---|---|---|"]
    for (task, mode, cond, tr), v in sorted(tt.items(), key=lambda kv: (kv[0][0], kv[0][1], _cond_idx(kv[0][2]),
                                                                        kv[0][2], kv[0][3])):
        lines.append(f"| {task} | {mode} | {cond} | {TRAINER_NAMES[tr]} | {v['plateau_k']}/{v['plateau_n']} | "
                     f"{v['k']}/{v['n']} @{v['snr']:g} dB | {'训练到位' if v['trained'] else '未训练好'} |")
    return "\n".join(lines) + "\n"


def summarize(rows: list[dict], outdir: str, config: dict) -> dict:
    _setup_fonts()
    annotate_dspsa(rows)
    tt = trained_table(rows)
    pts = _curve_points(rows)
    ct = curve_table(pts)
    rt = required_snr_table(pts, ct, trained=tt)
    paths = common.write_outputs(outdir, rows, ct, config=config, title="实验 B：准确率–SNR")
    paths["req_csv"] = common.save_csv(rt, os.path.join(outdir, "required_snr.csv"))
    figs = [plot_curves(ct, outdir, "snr_db", "acc_vs_snr.png", "SNR = 训练后码本探头信号中位数 / P_n (dB)", trained=tt),
            plot_curves(ct, outdir, "noise_dbm", "acc_vs_noise_dbm.png", "探头噪声功率 P_n (dBm)", trained=tt)]
    paths["figs"] = [f for f in figs if f]
    n_err = sum(1 for r in rows if "error" in r)
    ok = [r for r in rows if "error" not in r]
    ds = [r for r in ok if r.get("kind") == "dspsa"]
    n_pl = sum(1 for r in ds if r.get("plateau"))
    n_tr = sum(1 for r in ds if r.get("trained"))
    n_tr_pk = sum(1 for r in ds if r.get("pocket") and r["pocket"].get("trained"))
    mats = sorted({f"{r['task']}: {r.get('material')} (εr_eff {r.get('effective_eps')})" for r in ok})
    skipped = config.get("skipped_cells") or []
    skip_txt = "；".join(f"{d['task']} × {d['mode']}（{d['reason']}）" for d in skipped)
    md = ["# 实验 B：增益 vs 非线性——准确率–SNR（方案 RQ3、§6.2）\n",
          f"- 运行：{len(rows)} 个 job（{n_err} 个出错）；模式 {'quick' if config.get('quick') else 'full'}；种子数 "
          f"{config.get('seeds')}；任务 {'/'.join(config['tasks'])}；物体模式 {'/'.join(config['modes'])}；"
          f"条件 {'/'.join(config.get('conditions') or ALL_CONDITIONS)}。",
          f"- 实际材料（任务记录，审查 Q-5）：{'；'.join(mats) if mats else '–'}。"
          + (f"跳过：{skip_txt}。" if skipped else ""),
          f"- **场景 {config.get('scenario', 'baseline')}**（system.SCENARIOS，INTEGRATION S）："
          f"{scenario_overrides(config.get('scenario', 'baseline')) or '默认 L90 奇偶场景'}。",
          f"- DSPSA：每个名义 SNR 点在该噪声下独立训练 {config['dspsa_iters']} 迭代"
          f"（{dspsa_variant_text(config.get('dspsa_variant', 'auto'))}），"
          f"评估 {config['n_noise']} 次独立有噪声读数"
          f"{'；另报 pocket 码本（训练中测到的最好码本，独立噪声复评）' if config.get('dspsa_variant') == 'subset' else ''}；"
          f"名义 SNR（相对 P_ref）= {', '.join(f'{v:g}' for v in config['snr'])} dB。",
          f"- oracle 码本曲线：{'开' if config.get('oracle_curves') else '关'}（1-bit oracle = accuracy 目标 "
          f"{config['oracle_restarts']} 重启 + 最小裕量 polish，审查 Q-1；"
          f"连续 oracle {('%d 重启 × %d 步' % (config['cont_restarts'], config['cont_steps']) + ('，cont-plus 设计初值 %d 次 + soft-min 精修' % config.get('cont_init_restarts', CONT_PLUS_INIT_RESTARTS) if config.get('cont_plus') else '')) if config.get('oracle_cont') else '关'}"
          f"（softplus 训练，acc 不是准确率上界）；无噪声训练，在 {len(config['snr_dense'])} 个名义 SNR 点 "
          f"{config['snr_dense'][0]:g}…{config['snr_dense'][-1]:g} dB 上评估）。",
          f"- 总墙钟 {config.get('wall_s', float('nan')):.0f} s（workers = {config.get('workers')}）；"
          f"单 job 中位 {np.median([r['run_time_s'] for r in ok]) if ok else float('nan'):.1f} s。",
          "- SNR = 训练后码本的无噪声探头功率中位数（全部样本 × 探头）/ P_n；准确率 = 有噪声读数下的平均准确率"
          "（自激 run 剔除，方案 §4.5）；区间 = 种子间 t 分布 95% CI。\n"]
    if config.get("quick"):
        md.append(f"> quick 模式：{config.get('quick_note', '')}\n")
    md.append("## DSPSA 训练诊断（审查 Q-2 / S1-0）\n")
    if config.get("dspsa_variant", "auto") in ("auto", "sign_acc"):
        md.append(f"- 全部 DSPSA run（sign_acc，INTEGRATION T）：训练到位（无噪声 acc ≥ {REACH_FRAC:g}·acc*，acc* = 同种子 1-bit "
                  f"accuracy oracle）{n_tr}/{len(ds)}（含低 SNR 点，按最高名义 SNR 的判定见下表）；（诊断）softplus 可测损失 ≥ "
                  f"{PLATEAU_FRAC:g}·chance 的 run {n_pl}/{len(ds)}——判决量 ≪ τ = 1 dB 时即使全对也落在这里，不作失败判据。")
    else:
        md.append(f"- 全部 DSPSA run：最终迭代停在平台（loss_meas_clean ≥ {PLATEAU_FRAC:g}·chance）{n_pl}/{len(ds)}；"
                  f"训练到位（实验 E 判据 (L0−L) ≥ {REACH_FRAC:g}(L0−L*) 且离开平台，L* = 同种子 1-bit oracle）："
                  f"最终迭代 {n_tr}/{len(ds)}，pocket {n_tr_pk}/{len(ds)}。")
    md.append(f"- 规则：某条件在最高名义 SNR（噪声可忽略）上训练到位的种子 < {TRAINED_RUN_FRAC:.0%} 时，该条件的 DSPSA 曲线 / 所需 SNR"
              " 只反映训练失败，RQ3 差值写\"未训练好\"，图中画成浅色点线；RQ3 结论请看 oracle 码本。"
              "sign_acc（判对优先，INTEGRATION T）的\"训练到位\" = 无噪声 acc ≥ 0.9·acc*（同种子 1-bit accuracy oracle），"
              "旧变体按上面的损失判据。\n")
    md.append(plateau_md(tt))
    md.append("\n## 所需 SNR（准确率 ≥ 0.9；metrics.required_snr_db）\n")
    md.append(req_md(rt))
    md.append("\n## RQ3：同一名义 SNR 下的准确率差（种子平均）\n")
    md.append("Mx − M1@Mx、Mx − M1b@Mx = 非线性的贡献（同源功率；M1 = 小信号增益、M1b = Bussgang 等效复增益，INTEGRATION B；"
              "没有同回退 M1 的 Rapp 条件与 M1 配对：线性中继的无噪声判决与源功率无关，但同一名义 SNR 下噪声相对信号相同）；"
              "M1 − M0 = 增益的贡献（名义 SNR 各自相对本条件 P_ref，绝对噪声不同，见 acc_vs_noise_dbm.png 的同一噪声地板视图）。"
              "DSPSA / pocket 行只在两个条件都训练到位时给出差值。\n")
    dense_sub = list(config["snr_dense"][::2])
    md.append(rq3_md(ct, {"dspsa": list(config["snr"]), "dspsa_pocket": list(config["snr"]), "oracle": dense_sub,
                          "cont": dense_sub}, trained=tt,
                     pairs=(None if config.get("scenario", "baseline") == "baseline"     # baseline：固定 RQ3_PAIRS（原表）
                            else rq3_pairs(config.get("conditions") or ALL_CONDITIONS, any_power_m1=True))))
    for tr in ("dspsa", "dspsa_pocket", "oracle", "cont"):
        if any(r["trainer"] == tr for r in ct):
            md.append(f"\n## 曲线：{TRAINER_NAMES[tr]}（acc [95% CI] / 实际 SNR dB）\n")
            md.append(curve_md(ct, config["snr"] if tr in DSPSA_TRAINERS else config["snr_dense"][::2], tr))
    md.append("\n## 工作点与信号电平\n")
    md.append(oppoint_md(rows))
    md.append("\n## 图\n")
    for p in paths["figs"]:
        md.append(f"- `{os.path.basename(p)}`")
    md.append("")
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return {"curve": ct, "required": rt, "trained": tt, "paths": paths}


# ============================================================ CLI
def parse_args(argv=None):
    p = common.base_argparser("实验 B：准确率–SNR（RQ3）")
    p.add_argument("--tasks", nargs="+", default=None, choices=list(TASKS),
                   help="parity3（主）/ parity2（XOR 对照）/ letters（字母 A/B/C，只跑 intra/full）；quick 默认 parity3 parity2")
    p.add_argument("--conditions", nargs="+", default=None, help="默认 " + " ".join(ALL_CONDITIONS))
    p.add_argument("--modes", nargs="+", default=None, choices=["born", "intra", "full"],
                   help="物体模式（默认 born；full 模式下 born full；字母任务自动去掉 born）")
    p.add_argument("--material", default="water", help="奇偶任务的材料")
    p.add_argument("--letter-material", default="metal",
                   help="字母任务的材料（显式传给 LetterTask，默认 metal = 方案 §6.2；审查 Q-5）")
    p.add_argument("--preset", default="L90")
    p.add_argument("--scenario", default=None, choices=["baseline", "informative", "informative_probe"],
                   help="命名场景（system.SCENARIOS，INTEGRATION S）；默认 quick → informative，full → baseline")
    p.add_argument("--dspsa-variant", default="auto", choices=["auto", "sign_acc", "subset", "faithful"],
                   help="DSPSA 变体（默认 auto = 1-bit sign_acc，INTEGRATION T；subset = 旧默认）")
    p.add_argument("--cont-plus", dest="cont_plus", action="store_true", default=None,
                   help="连续 oracle 加设计初值 + soft-min 精修（informative 场景默认开，INTEGRATION S）")
    p.add_argument("--no-cont-plus", dest="cont_plus", action="store_false")
    p.add_argument("--cont-init-restarts", type=int, default=CONT_PLUS_INIT_RESTARTS)
    p.add_argument("--cont-refine-steps", type=int, default=300)
    p.add_argument("--snr", type=float, nargs="+", default=None, help="DSPSA 的名义 SNR 点 (dB，相对 P_ref)")
    p.add_argument("--snr-dense", type=float, nargs="+", default=None, help="oracle 码本评估的名义 SNR 点")
    p.add_argument("--dspsa-iters", type=int, default=None,
                   help="DSPSA 迭代数（quick 1500（informative 场景 1000），与实验 A 相同 / full 4000）。旧 subset 变体在奇偶"
                        "任务上常停在 y1≈y2 的 τ·ln2 平台（审查 S1-0）；汇总按训练到位判据标出")
    p.add_argument("--n-noise", type=int, default=50, help="有噪声评估的独立读数次数")
    p.add_argument("--no-oracle-curves", dest="oracle_curves", action="store_false", default=True)
    p.add_argument("--oracle-restarts", type=int, default=8)
    p.add_argument("--oracle-sweeps", type=int, default=30)
    p.add_argument("--polish-sweeps", type=int, default=POLISH_SWEEPS,
                   help="1-bit accuracy oracle 之后的最小裕量 polish 轮数（0 = 关；审查 Q-1）")
    p.add_argument("--no-oracle-cont", dest="oracle_cont", action="store_false", default=True)
    p.add_argument("--cont-restarts", type=int, default=None)
    p.add_argument("--cont-steps", type=int, default=None)
    p.add_argument("--cfg", nargs="+", default=[], metavar="KEY=VALUE", help="SystemConfig 字段覆盖（全部 job 共用）")
    p.add_argument("--resummarize", type=str, default=None)
    a = p.parse_args(argv)
    q = a.quick
    a.scenario = a.scenario or ("informative" if q else "baseline")
    info = a.scenario != "baseline"
    a.cfg_over = {**scenario_overrides(a.scenario), **parse_cfg_overrides(a.cfg)}
    if "probe_noise_dbm" in a.cfg_over:
        raise ValueError("probe_noise_dbm 由 SNR 网格决定，不能用 --cfg 覆盖")
    a.tasks = a.tasks or (["parity3", "parity2"] if q else ["parity3", "parity2", "letters"])
    a.conditions = a.conditions or list(INFORMATIVE_CONDITIONS if (q and info) else ALL_CONDITIONS)
    for c in a.conditions:
        parse_condition(c)
    a.modes = a.modes or (["full"] if (q and info) else (["born"] if q else ["born", "full"]))
    if a.cont_plus is None:
        a.cont_plus = info
    a.snr = a.snr or list(SNR_QUICK if q else SNR_FULL)
    a.snr_dense = a.snr_dense or list(SNR_DENSE)
    a.dspsa_iters = a.dspsa_iters or ((1000 if info else 1500) if q else 4000)
    if a.cont_plus:
        a.cont_restarts = a.cont_restarts or 1
        a.cont_steps = a.cont_steps or 600
    a.cont_restarts = a.cont_restarts or (2 if q else 4)
    a.cont_steps = a.cont_steps or (300 if q else 800)
    return a


def main(argv=None) -> str:
    args = parse_args(argv)
    from ris_nlsim.experiments.exp_b_snr import run_job as job_fn    # 规范模块名（spawn 可反序列化）

    if args.resummarize:
        rows = common.load_jsonl(os.path.join(args.resummarize, "runs.jsonl"))
        with open(os.path.join(args.resummarize, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        summarize(rows, args.resummarize, config)
        print(f"重新汇总完成：{args.resummarize}")
        return args.resummarize
    outdir = common.make_outdir(EXP, args.out)
    jobs = build_jobs(args)
    if args.skipped_cells:
        print("跳过：" + "；".join(f"{d['task']} × {d['mode']}（{d['reason']}）" for d in args.skipped_cells), flush=True)
    config = {k: v for k, v in vars(args).items() if k != "resummarize"}
    config.update(exp="B", n_jobs=len(jobs), started=_dt.datetime.now().isoformat(timespec="seconds"),
                  python=sys.version.split()[0],
                  quick_note=(f"场景 {args.scenario}；任务 {'/'.join(args.tasks)}（字母关），物体模式 {'/'.join(args.modes)}，"
                              f"{args.seeds} 个种子，DSPSA {args.dspsa_iters} 迭代"
                              f"{'（informative 场景 full 耦合 + Rapp 条件更贵，quick 从 1500 减到 1000）' if args.scenario != 'baseline' else ''}"
                              f"，{len(args.snr)} 个名义 SNR 点；连续 oracle {args.cont_restarts} 重启 × {args.cont_steps} 步"
                              f"{'（cont-plus：设计初值 %d 次 + soft-min 精修）' % args.cont_init_restarts if args.cont_plus else '（少于 full 的 4 × 800）'}。"))
    print(f"实验 B：{len(jobs)} 个 job → {outdir}（workers={args.workers}）", flush=True)
    t0 = time.perf_counter()
    rows = common.run_jobs(job_fn, jobs, workers=args.workers, progress=True)
    config["wall_s"] = time.perf_counter() - t0
    summarize(rows, outdir, config)
    print(f"完成：{config['wall_s']:.0f} s；输出 {outdir}", flush=True)
    return outdir


if __name__ == "__main__":
    main()
