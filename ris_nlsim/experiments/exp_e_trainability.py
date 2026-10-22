"""实验 E：可训练性（RQ6；方案 §3 RQ6、§4.7、§6.2 表"E 可训练性"、§7"收敛迭代数"；INTEGRATION G "E"、D、G2、B3）。

假设（方案 RQ6）：深度饱和时损失地形变平（类似梯度消失），DSPSA 收敛迭代数上升；预测：达到 oracle 90% 所需迭代数
随饱和加深（M2 → M3 → M4）上升。

任务（L90 预设，full 耦合）：
  parity3  3 位奇偶（INTEGRATION G 的主任务；水球 λ/3，自动体素）。本几何下 1-bit oracle 的损失也只到 chance 附近
           （L* ≈ τ·log2），DSPSA 学不会 → 按下面的判据通常"不可检验"，如实报告；
  letter   字母 A/B/C（3 探头 argmax）：oracle 显著低于 chance，DSPSA 偶尔低于 chance；
  focus    单探头聚焦（无物体，最大化探头 0 功率，损失 = −y0 dBm）：DSPSA 能学会的任务（审查 R-E2），也是"饱和使地形变平"
           最直接的检验——PA 饱和后接收比特对输出功率几乎没有影响。
条件：M2 / M3 / M4 / Rapp@+20（Rapp PA、源功率回退 +20 dB，比 M4 再深 10 dB）/ M1@M3（线性 PA、小信号增益、与 M3 同源功率）。

每个 run（条件 × 任务 × 种子）：
  1. DSPSA（train.run_dspsa；--variant 默认 auto = 1-bit 新默认 sign_acc（INTEGRATION T：判对优先可测目标 + sign-SPSA，
     首步 0.1、α = 0.602、A = 400，无需标定——符号梯度只比较两次测量，各条件的步长天然相同，相当于"shared"；曲线记录
     当时的输出码本 = pocket）；--variant subset = 旧默认（A = 400，α = 0.501，下面的 shared / auto 增益规则））学习曲线：
     每 eval_every = 10 次迭代记录无噪声可测目标 loss_meas_clean 与准确率（callback，不计入测量次数，INTEGRATION D）。
     **增益 a（审查 R-E1，INTEGRATION B3）**：默认 "shared"——每个 (任务, 种子) 在参考条件（gain_ref，默认 M3）上用
     train.calibrate_dspsa_gain（n_starts = 3，随机流 seed_rng(EXP, "cal", 任务, 种子)）标定一次，同一个数值传给该任务
     该种子的所有条件（含 faithful 变体）。a = "auto"（每条件按自己的 mean|ĝ| 归一化）会把 RQ6 要检验的"整体变平"抵消掉，
     只作为 --gain auto 的标注对照。每行记录实际所用的 a 与 a_mode（sign_acc 时 a_mode = "sign"，不跑标定 job）。
  2. 1-bit 模型 oracle：train.run_oracle_bits 的损失目标（"loss"）与准确率目标（"accuracy"，分类任务；B3 / S0-2：损失 oracle 的
     acc 不是最好可达准确率）。L* = 同任务同条件**全部种子、两种目标**的 oracle 码本 loss_meas_clean 的最小值（pooled；系统与
     种子无关，pooled 是可达损失的最好估计）；acc* = 同样 pooled 的最高准确率。本 run 自己的 oracle 另记为 *_run。
  3. 收敛判据（审查 R-E2，替代 INTEGRATION G "E" 的 L0 参照）：
       分类任务：参照点 = chance（零信息损失，奇偶 τ·log2、字母 log3）；首个 k 使 (chance − L_k) ≥ 0.9·(chance − L*)。
         原 L0 参照 (L0 − L_k) ≥ 0.9(L0 − L*) 在奇偶上退化：L0 ≈ 3 是 y1 − y2 约 6 dB 的常数偏置，L* ≈ chance，
         "达到 90%" 只量"把偏置拉平"的速度（每个 run 都在第一个曲线点达到）。
         只有 oracle 明显优于 chance（chance − L* > DELTA_CHANCE）时才**可检验**；否则整行记"不可检验"。
       focus：无 chance，参照点 = L0（随机初始码本的 −y0），L0 − L* > DELTA_FOCUS_DB 才可检验。
       准确率判据（分类任务）：首个 k 使 acc_k − acc_chance ≥ 0.9·(acc* − acc_chance)，acc* − acc_chance ≥ DELTA_ACC 才可检验。
     未达到 → 删失（记 > n_iter）；另报"达到并保持"（settled）。判读：oracle 不优于 chance、或被比较条件的全部 DSPSA run
     都删失（从未低于阈值 / chance）时报"不可检验"，不给 Spearman。测量次数 = 2k（shared a 时标定测量在参考条件上，
     不计入各 run；auto 时 + 20）。
  4. 地形平坦度（训练无关）：随机码本上翻转 m = max(2, p//8) 个随机比特（DSPSA subset 的扰动）的无噪声 |ΔL| 中位数，
     与同码本有噪声测量的标准差之比（"梯度信噪比"）；DSPSA 历史里前 100 次迭代的 mean|ĝ|（shared a 下各条件可比）。
  5. faithful vs subset（1-bit 退化，DESIGN §6.1、INTEGRATION G2）：中继全部 50 比特取反 ⇒ u → −u、C → −C，
     z = d + C g(u) 不变（Rapp + AM-PM 有 g(−u) = −g(u)），即 L(x) = L(1 − x) 精确成立；faithful DSPSA 在 1-bit 下 x± 恒为
     一对互补码本 ⇒ 无噪声 ĝ ≡ 0，有噪声时 ĝ 只是测量噪声。本脚本对每个条件数值验证 max|L(x) − L(1−x)|，并对
     --faithful-conds（默认 M3）用同一种子、同一个 a 跑 faithful 变体。
实际工作点：每行报最终码本的 PA 输入（闭环 |u|²）相对 P1dB 的 min/mean/max、P_feed、体素尺寸。

场景：--scenario（system.SCENARIOS，INTEGRATION S；默认 baseline = 原 L90 场景）。

用法：python -m ris_nlsim.experiments.exp_e_trainability [--quick] [--workers N] [--seeds S] [--out DIR]
      [--conditions M2 M3 M4 Rapp@+20 M1@M3] [--tasks parity3 letter focus] [--n-iter N] [--faithful-conds M3]
      [--variant auto|sign_acc|subset] [--gain shared|auto] [--gain-ref M3] [--scenario baseline|informative|…]
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from . import common

EXP = "exp_e_trainability"
ALL_CONDITIONS = ("M2", "M3", "M4", "Rapp@+20", "M1@M3")
RAPP_PREFIX = "Rapp@"                # "Rapp@+20"：Rapp PA、源功率回退 +20 dB（比 M4 再深 10 dB 的饱和，相对 INTEGRATION G 的增补）
TASKS = ("parity3", "letter", "focus")
FRACTION = 0.9
EVAL_EVERY = 10                      # 审查 R-E2：原 50 的分辨率下奇偶在第一个曲线点就"达到"
N_CAL_MEAS = 20                      # dspsa a="auto" 的标定测量次数
GAIN_MODES = ("shared", "auto")
GAIN_REF = "M3"                      # shared a 的参考条件（INTEGRATION B3）
N_CAL_STARTS = 3
DELTA_CHANCE = 0.05                  # 分类：chance − L* 须 > 它才可检验（损失单位）
DELTA_FOCUS_DB = 1.0                 # focus：L0 − L* 须 > 它才可检验（dB）
DELTA_ACC = 0.1                      # 准确率判据：acc* − acc_chance 须 ≥ 它才可检验


def parse_condition(label: str) -> dict:
    """条件标签 → SystemConfig 字段（"M1@M3" → M1 + M3 的回退 0 dB；其余按条件表）。"""
    from ..system import CONDITION_BACKOFF_DB, CONDITIONS

    lab = str(label).strip()
    if lab.startswith(RAPP_PREFIX):
        return {"condition": "M4", "backoff_db": float(lab[len(RAPP_PREFIX):])}     # M2/M3/M4 都是 RappPA
    if "@" in lab:
        base, ref = lab.split("@", 1)
        if base not in ("M1", "M1b") or ref not in CONDITION_BACKOFF_DB:
            raise ValueError(f"无法解析条件 {label!r}")
        return {"condition": base, "backoff_db": float(CONDITION_BACKOFF_DB[ref])}
    if lab not in CONDITIONS:
        raise ValueError(f"未知条件 {label!r}")
    return {"condition": lab}


def cond_backoff_db(label: str) -> float:
    """条件标签的源功率回退（dB，INTEGRATION B 的标定定义）。"""
    from ..system import CONDITION_BACKOFF_DB

    f = parse_condition(label)
    return float(f["backoff_db"]) if "backoff_db" in f else float(CONDITION_BACKOFF_DB[f["condition"]])


def make_e_system(task: str, cond: str, preset: str = "L90", object_mode: str = "full", scenario: str = "baseline"):
    from ..system import SCENARIOS, SystemConfig, make_system

    if scenario not in SCENARIOS:
        raise ValueError(f"未知场景 {scenario!r}，可选 {list(SCENARIOS)}")
    over = {k: v for k, v in SCENARIOS[scenario].items() if k != "preset"}
    cfg = SystemConfig(preset=preset, object_mode=object_mode, **parse_condition(cond), **over)
    if task == "parity3":
        return make_system(cfg, "parity", n=3)
    if task == "letter":
        return make_system(cfg, "letter")
    if task == "focus":
        return make_system(cfg, "focus")
    raise ValueError(f"未知任务 {task!r}")


def chance_level(system) -> float | None:
    """零信息损失（losses.chance_loss）：奇偶 τ·log2，字母 log3；focus 无判决边界 → None。"""
    from ..losses import chance_loss

    ch = chance_loss(system.task.kind, system.n_probes, system.cfg.tau_db, **system.task.task_kwargs)
    return None if ch is None else float(ch)


def chance_accuracy(system) -> float | None:
    """随机猜测的准确率：二分类 1/2、K 类 1/K；focus None。"""
    kind = system.task.kind
    if kind == "binary":
        return 0.5
    if kind == "multiclass":
        return 1.0 / float(system.n_probes)
    if kind == "loc3bit":
        return 1.0 / float(2 ** system.n_probes)
    return None


# ============================================================ 判据
def iters_to_fraction(iters, losses, ref: float, Lstar: float, frac: float = FRACTION):
    """首个满足 (ref − L_k) ≥ frac·(ref − L*) 的迭代数 k；(k, censored, gap)。ref = chance（分类，审查 R-E2）或 L0（focus）。
    gap = ref − L* ≤ 0（参照点已不差于 oracle）时返回 (0, False, gap)；未达到返回 (None, True, gap)。"""
    gap = float(ref) - float(Lstar)
    if not np.isfinite(gap):
        return None, True, gap
    if gap <= 0:
        return 0, False, gap
    for k, L in zip(iters, losses):
        if L is not None and np.isfinite(L) and float(ref) - float(L) >= frac * gap:
            return int(k), False, gap
    return None, True, gap


def iters_to_settle(iters, losses, ref: float, Lstar: float, frac: float = FRACTION):
    """首个 k 使其后（含 k）所有曲线点都满足 (ref − L_j) ≥ frac·(ref − L*)（"达到并保持"）；(k, censored)。
    iters_to_fraction 只要求某一点达到，随机游走或带噪下探偶然碰到的低损失码本也算；本判据排除这种情况。"""
    gap = float(ref) - float(Lstar)
    if not np.isfinite(gap):
        return None, True
    if gap <= 0:
        return 0, False
    ok = [L is not None and np.isfinite(L) and float(ref) - float(L) >= frac * gap for L in losses]
    k_set = None
    for k, good in zip(reversed(list(iters)), reversed(ok)):
        if not good:
            break
        k_set = int(k)
    return (k_set, False) if k_set is not None else (None, True)


def progress_criterion(iters, losses, L0: float, Lstar: float, chance: float | None, n_iter: int,
                       frac: float = FRACTION) -> dict:
    """审查 R-E2 的收敛判据：分类（chance 非 None）以 chance 为参照、chance − L* > DELTA_CHANCE 才可检验；
    focus 以 L0 为参照、L0 − L* > DELTA_FOCUS_DB 才可检验。不可检验时 k 类字段为 None（aggregate 不计入）。
    返回 {ref_kind, ref, gap, testable, iters_to_90, censored, reached_90, iters_to_90_lb, iters_to_90_settled,
          settled_90, iters_to_90_settled_lb, below_chance, progress_final}。"""
    ref_kind = "chance" if chance is not None else "L0"
    ref = float(chance) if chance is not None else float(L0)
    delta = DELTA_CHANCE if chance is not None else DELTA_FOCUS_DB
    gap = ref - float(Lstar) if Lstar is not None and np.isfinite(Lstar) else float("nan")
    Lv = [float(v) for v in losses if v is not None and np.isfinite(v)]
    out = {"ref_kind": ref_kind, "ref": ref, "gap": gap, "testable": bool(np.isfinite(gap) and gap > delta),
           "below_chance": (bool(min(Lv) < chance) if (chance is not None and Lv) else None),
           "iters_to_90": None, "censored": None, "reached_90": None, "iters_to_90_lb": None,
           "iters_to_90_settled": None, "settled_90": None, "iters_to_90_settled_lb": None, "progress_final": None}
    if not out["testable"]:
        return out
    k, cens, _ = iters_to_fraction(iters, losses, ref, Lstar, frac)
    ks, cs = iters_to_settle(iters, losses, ref, Lstar, frac)
    out.update(iters_to_90=k, censored=cens, reached_90=not cens, iters_to_90_lb=(int(n_iter) if cens else k),
               iters_to_90_settled=ks, settled_90=not cs, iters_to_90_settled_lb=(int(n_iter) if cs else ks),
               progress_final=(ref - float(losses[-1])) / gap if losses and losses[-1] is not None else None)
    return out


def accuracy_criterion(iters, accs, acc_star, acc_chance, n_iter: int, frac: float = FRACTION) -> dict:
    """准确率判据（审查 R-E2）：首个 k 使 acc_k − acc_chance ≥ frac·(acc* − acc_chance)；acc* − acc_chance ≥ DELTA_ACC 才可检验。"""
    out = {"acc_testable": False, "acc_iters_to_90": None, "acc_reached_90": None, "acc_iters_to_90_lb": None}
    if acc_star is None or acc_chance is None or not (acc_star - acc_chance >= DELTA_ACC):
        return out
    need = acc_chance + frac * (acc_star - acc_chance)
    k = next((int(i) for i, a in zip(iters, accs) if a is not None and np.isfinite(a) and a >= need - 1e-12), None)
    out.update(acc_testable=True, acc_iters_to_90=k, acc_reached_90=k is not None,
               acc_iters_to_90_lb=(int(n_iter) if k is None else k))
    return out


def flip_symmetry(system, rng: np.random.Generator, n: int = 8) -> float:
    """max_x |L(x) − L(1 − x)|（无噪声可测目标）：中继全部比特取反时 z 不变 ⇒ 1-bit faithful DSPSA 的 x± 损失相等。"""
    out = 0.0
    for _ in range(int(n)):
        x = system.random_params(rng)
        a = system.loss_meas_clean(system.forward(system.phases_from_params(x), noisy=False))
        b = system.loss_meas_clean(system.forward(system.phases_from_params(1 - x), noisy=False))
        out = max(out, abs(a - b))
    return float(out)


def landscape_probe(system, rng: np.random.Generator, n_pairs: int = 16, n_noise: int = 10) -> dict:
    """地形平坦度：随机码本 x 与翻转 m 个随机比特的 x'（m = DSPSA subset 默认 max(2, p//8)）的无噪声 |ΔL|；
    以及同一码本 n_noise 次有噪声测量的损失标准差（测量噪声对 ĝ 的贡献）。"""
    p = system.n_params
    m = max(2, p // 8)
    dl = []
    for _ in range(int(n_pairs)):
        x = system.random_params(rng)
        y = x.copy()
        idx = rng.choice(p, size=m, replace=False)
        y[idx] = 1 - y[idx]
        a = system.loss_meas_clean(system.forward(system.phases_from_params(x), noisy=False))
        b = system.loss_meas_clean(system.forward(system.phases_from_params(y), noisy=False))
        dl.append(abs(a - b))
    x = system.random_params(rng)
    noisy = [system.loss_from(system.forward(system.phases_from_params(x), rng, noisy=True), model_penalties=False)
             for _ in range(int(n_noise))]
    sd = float(np.std(noisy, ddof=1)) if len(noisy) > 1 else float("nan")
    med = float(np.median(dl))
    return {"land_dL_med": med, "land_dL_mean": float(np.mean(dl)), "land_noise_sd": sd,
            "land_snr": (med / sd) if sd > 0 else None, "land_m": m}


def _curve(res: dict) -> list[dict]:
    return [{k: c.get(k) for k in ("iter", "loss_meas_clean", "loss_clean", "acc_clean", "n_flagged")}
            for c in res["curve"]]


def _dspsa_summary(res: dict, prefix: str) -> dict:
    h = res["history"]
    lp, lm = np.asarray(h["loss_plus"], float), np.asarray(h["loss_minus"], float)
    flips = np.asarray(h["n_flips"])
    g = np.asarray(h["grad_abs_mean"], float)
    ev = res["eval"]
    return {f"{prefix}curve": _curve(res), f"{prefix}a": float(res["a"]),
            f"{prefix}dL_pm_med": float(np.nanmedian(np.abs(lp - lm))),
            f"{prefix}g_abs_early": float(np.nanmean(g[:100])) if g.size else None,
            f"{prefix}flips_total": int(flips.sum()), f"{prefix}flips_last_half": int(flips[flips.size // 2:].sum()),
            f"{prefix}final_loss_meas": ev.get("loss_meas_clean"), f"{prefix}final_acc": ev.get("acc_clean"),
            f"{prefix}final_acc_noisy": ev.get("acc_noisy"), f"{prefix}final_osc": ev.get("oscillating_run"),
            f"{prefix}flagged_meas_frac": res["n_flagged_measurements"] / max(1, res["n_measurements"]),
            f"{prefix}n_measurements": int(res["n_measurements"]), f"{prefix}time_s": float(res["time_s"]),
            f"{prefix}a_auto_degenerate": bool(np.asarray(h.get("a_auto_degenerate", [False]))[0])}


# ============================================================ job
def run_e_cal_job(job: dict) -> dict:
    """shared 增益的标定（审查 R-E1）：在参考条件上 train.calibrate_dspsa_gain，随机流只依赖 (任务, 种子)。"""
    from ..train import calibrate_dspsa_gain

    t0 = time.perf_counter()
    task, seed, ref = job["task"], int(job["seed"]), job["gain_ref"]
    s = make_e_system(task, ref, job.get("preset", "L90"), job.get("object_mode", "full"),
                      job.get("scenario", "baseline"))
    cal = calibrate_dspsa_gain(s, common.seed_rng(EXP, "cal", task, seed), n_starts=int(job.get("n_cal_starts",
                                                                                              N_CAL_STARTS)),
                               variant="subset")
    return {"kind": "cal", "task": task, "seed": seed, "gain_ref": ref, "a": cal["a"], "mean_abs_g": cal["mean_abs_g"],
            "n_measurements": cal["n_measurements"], "job_time_s": time.perf_counter() - t0}


def run_e_job(job: dict) -> dict:
    from ..train import run_dspsa, run_oracle_bits

    t0 = time.perf_counter()
    task, cond, seed = job["task"], job["cond"], int(job["seed"])
    from ..train import _variant

    a = job.get("a", "auto")
    s = make_e_system(task, cond, job.get("preset", "L90"), job.get("object_mode", "full"),
                      job.get("scenario", "baseline"))
    var = _variant(s, job.get("variant", "auto"))
    sign = var == "sign_acc"
    if a is None and not sign:
        raise RuntimeError(f"shared 增益标定失败（任务 {task}，种子 {seed}）")
    n_iter = int(job["n_iter"])
    eval_every = int(job.get("eval_every", EVAL_EVERY))
    a_mode = "sign" if sign else ("auto" if a == "auto" else "shared")
    row = {"task": task, "cond": cond, "seed": seed, "condition_internal": s.label, "n_iter": n_iter,
           "scenario": job.get("scenario", "baseline"), "variant": var,
           "chance": chance_level(s), "acc_chance": chance_accuracy(s), "a_mode": a_mode,
           "a_ref_cond": job.get("gain_ref") if a_mode == "shared" else None, "build_time_s": time.perf_counter() - t0}
    # 1. DSPSA（公共随机数：同种子 → 同初始码本与扰动序列，与条件无关；sign_acc 无标定，subset 的 shared a 与条件无关）
    rng_d = common.seed_rng(EXP, "dspsa", task, seed)
    res = run_dspsa(s, n_iter, rng_d, variant=var, a="auto" if sign else a, eval_every=eval_every,
                    n_noise_eval=int(job["n_noise"]))
    row.update(_dspsa_summary(res, ""))
    ev = res["eval"]
    for k in ("pa_in_rel_p1db_db_min", "pa_in_rel_p1db_db_mean", "pa_in_rel_p1db_db_max", "pa_in_open_rel_p1db_db_mean",
              "null_depth_db", "p_feed_dbm", "backoff_db", "voxel_d", "effective_eps", "object_model",
              "loop_gain_ss_max", "loop_gain_op_max", "margin_db"):
        row[k] = ev.get(k)
    # 2. oracle：损失目标 +（分类任务）准确率目标（审查 R-E2 / B3）
    t1 = time.perf_counter()
    orc = run_oracle_bits(s, common.seed_rng(EXP, "oracle", task, cond, seed), n_restarts=int(job["oracle_restarts"]),
                          max_sweeps=int(job["oracle_sweeps"]), n_noise_eval=0)
    eo = orc["eval"]
    row.update(oracle_loss_meas=eo.get("loss_meas_clean"), oracle_loss_model=float(orc["loss"]),
               oracle_acc=eo.get("acc_clean"), oracle_margin_db=eo.get("margin_db"), oracle_n_evals=int(orc["n_evals"]),
               oracle_pa_in_rel_p1db_db_mean=eo.get("pa_in_rel_p1db_db_mean"))
    if row["chance"] is not None:
        oa = run_oracle_bits(s, common.seed_rng(EXP, "oracle_acc", task, cond, seed),
                             n_restarts=int(job["oracle_restarts"]), max_sweeps=int(job["oracle_sweeps"]),
                             n_noise_eval=0, objective="accuracy")
        ea = oa["eval"]
        row.update(oracle_acc_obj_loss_meas=ea.get("loss_meas_clean"), oracle_acc_obj_acc=ea.get("acc_clean"),
                   oracle_acc_obj_n_evals=int(oa["n_evals"]))
    else:
        row.update(oracle_acc_obj_loss_meas=None, oracle_acc_obj_acc=None, oracle_acc_obj_n_evals=None)
    row["oracle_time_s"] = time.perf_counter() - t1
    Ls = [c["loss_meas_clean"] for c in row["curve"]]
    row.update(L0=Ls[0], L_final=Ls[-1], L_min=float(np.nanmin(Ls)),
               meas_offset=(N_CAL_MEAS if (a == "auto" and not sign) else 0))
    # 本 run 自己的 oracle 作 L*（次要；主判据在 finalize_rows 里用 pooled L*）
    own = [v for v in (row["oracle_loss_meas"], row["oracle_acc_obj_loss_meas"]) if v is not None]
    pc = progress_criterion([c["iter"] for c in row["curve"]], Ls, Ls[0], min(own) if own else None, row["chance"],
                            n_iter)
    row.update({f"run_{k}": v for k, v in pc.items() if k in ("testable", "iters_to_90", "censored", "reached_90",
                                                              "iters_to_90_lb", "gap")})
    # 4. 地形平坦度 + 1-bit 取反对称
    row.update(landscape_probe(s, common.seed_rng(EXP, "land", task, seed), int(job["n_land"])))
    row["flip_sym_max_abs_dL"] = flip_symmetry(s, common.seed_rng(EXP, "sym", task, seed))
    # 5. faithful 变体（同一种子、同一个 a；sign_acc 模式下没有 shared a → faithful 用 a = auto）
    if cond in job.get("faithful_conds", ()):
        rf = run_dspsa(s, n_iter, common.seed_rng(EXP, "dspsa", task, seed), variant="faithful",
                       a="auto" if sign else a, eval_every=eval_every, n_noise_eval=int(job["n_noise"]))
        row.update(_dspsa_summary(rf, "faithful_"))
        Lf = [c["loss_meas_clean"] for c in row["faithful_curve"]]
        row.update(faithful_L0=Lf[0], faithful_L_final=Lf[-1], faithful_L_min=float(np.nanmin(Lf)))
    row["job_time_s"] = time.perf_counter() - t0
    return row


def run_any_job(job: dict) -> dict:
    """run_jobs 的顶层入口：job["kind"] == "cal" → shared 增益标定，否则训练 run。"""
    return run_e_cal_job(job) if job.get("kind") == "cal" else run_e_job(job)


# ============================================================ 网格
def build_config(args) -> dict:
    quick = bool(args.quick)
    seeds = args.seeds if args.seeds is not None else 5          # 单次 run 很便宜：quick 也用 5 个种子
    cfg = {"exp": EXP, "quick": quick, "seeds": int(seeds), "scenario": getattr(args, "scenario", None) or "baseline",
           "variant": getattr(args, "variant", None) or "auto",
           "conditions": list(args.conditions or ALL_CONDITIONS), "tasks": list(args.tasks or TASKS),
           "n_iter": int(args.n_iter or (1500 if quick else 4000)),
           "oracle_restarts": int(args.oracle_restarts or (4 if quick else 8)), "oracle_sweeps": 30,
           "faithful_conds": list(args.faithful_conds or ["M3"]), "n_land": 16, "n_noise": 10 if quick else 20,
           "eval_every": EVAL_EVERY, "fraction": FRACTION, "preset": "L90", "object_mode": "full",
           "gain": args.gain or "shared", "gain_ref": args.gain_ref or GAIN_REF, "n_cal_starts": N_CAL_STARTS,
           "delta_chance": DELTA_CHANCE, "delta_focus_db": DELTA_FOCUS_DB, "delta_acc": DELTA_ACC}
    if cfg["gain"] not in GAIN_MODES:
        raise ValueError(f"--gain 只能是 {GAIN_MODES}")
    for c in cfg["conditions"] + cfg["faithful_conds"] + [cfg["gain_ref"]]:
        parse_condition(c)
    cfg["condition_backoff_db"] = {c: cond_backoff_db(c) for c in cfg["conditions"]}
    return cfg


def _sign_mode(cfg: dict) -> bool:
    """1-bit 中继上 variant auto / sign_acc → 符号梯度（无需增益标定）。实验 E 的系统全是中继（1-bit）。"""
    return cfg.get("variant", "auto") in ("auto", "sign_acc")


def make_cal_jobs(cfg: dict) -> list[dict]:
    if cfg.get("gain", "shared") != "shared" or _sign_mode(cfg):
        return []
    return [{"kind": "cal", "task": t, "seed": s, "gain_ref": cfg["gain_ref"], "n_cal_starts": cfg["n_cal_starts"],
             "preset": cfg["preset"], "object_mode": cfg["object_mode"], "scenario": cfg.get("scenario", "baseline")}
            for t in cfg["tasks"] for s in range(cfg["seeds"])]


def make_jobs(cfg: dict, gains: dict | None = None) -> list[dict]:
    """训练 job；gains = {(任务, 种子): a}（shared 模式的标定结果；缺失 → None → job 报错）。auto 模式 a = "auto"。"""
    shared = cfg.get("gain", "shared") == "shared" and not _sign_mode(cfg)
    jobs = [{"task": t, "cond": c, "seed": s, "n_iter": cfg["n_iter"], "oracle_restarts": cfg["oracle_restarts"],
             "variant": cfg.get("variant", "auto"), "scenario": cfg.get("scenario", "baseline"),
             "oracle_sweeps": cfg["oracle_sweeps"], "faithful_conds": cfg["faithful_conds"], "n_land": cfg["n_land"],
             "n_noise": cfg["n_noise"], "preset": cfg["preset"], "object_mode": cfg["object_mode"],
             "eval_every": cfg.get("eval_every", EVAL_EVERY),
             "a": ((gains or {}).get((t, s)) if shared else "auto"), "gain_ref": cfg.get("gain_ref", GAIN_REF)}
            for t in cfg["tasks"] for c in cfg["conditions"] for s in range(cfg["seeds"])]
    jobs.sort(key=lambda j: (j["cond"] not in j["faithful_conds"], j["cond"].startswith("M1"), j["task"] != "parity3"))
    return jobs


# ============================================================ 汇总
METRICS = ["testable", "reached_90", "iters_to_90", "iters_to_90_lb", "meas_to_90", "settled_90",
           "iters_to_90_settled_lb", "below_chance", "progress_final", "acc_testable", "acc_reached_90",
           "acc_iters_to_90_lb", "run_testable", "run_reached_90", "faithful_reached_90", "faithful_settled_90",
           "L0", "L_final", "L_min", "Lstar", "oracle_loss_meas", "oracle_acc_obj_loss_meas", "oracle_acc",
           "oracle_acc_obj_acc", "acc_star", "final_acc", "final_acc_noisy", "land_dL_med", "land_noise_sd", "land_snr",
           "g_abs_early", "a", "dL_pm_med", "flips_total", "flip_sym_max_abs_dL", "final_osc", "flagged_meas_frac",
           "pa_in_rel_p1db_db_min", "pa_in_rel_p1db_db_mean", "pa_in_rel_p1db_db_max", "null_depth_db", "p_feed_dbm",
           "voxel_d", "chance", "faithful_final_loss_meas", "faithful_final_acc", "faithful_dL_pm_med",
           "faithful_flips_total", "faithful_L0", "faithful_L_final", "faithful_a", "time_s", "oracle_time_s"]


def finalize_rows(rows: list[dict]) -> None:
    """pooled 参照（原地加字段，审查 R-E2）：L* = 同任务同条件全部种子、两种 oracle 目标的最小 loss_meas_clean；
    acc* = 同样 pooled 的最高准确率；按 progress_criterion / accuracy_criterion 计算主判据（subset 与 faithful）。"""
    best_L, best_acc = {}, {}
    for r in rows:
        if "error" in r or r.get("kind") == "cal" or "curve" not in r:
            continue
        key = (r["task"], r["cond"])
        for v in (r.get("oracle_loss_meas"), r.get("oracle_acc_obj_loss_meas")):
            if v is not None and np.isfinite(v):
                best_L[key] = min(best_L.get(key, np.inf), float(v))
        for v in (r.get("oracle_acc"), r.get("oracle_acc_obj_acc")):
            if v is not None and np.isfinite(v):
                best_acc[key] = max(best_acc.get(key, -np.inf), float(v))
    for r in rows:
        if "error" in r or r.get("kind") == "cal" or "curve" not in r:
            continue
        key = (r["task"], r["cond"])
        Ls = best_L.get(key)
        it = [c["iter"] for c in r["curve"]]
        L = [c["loss_meas_clean"] for c in r["curve"]]
        pc = progress_criterion(it, L, L[0], Ls, r.get("chance"), r["n_iter"])
        r.update(Lstar=Ls, **pc)
        r["meas_to_90"] = None if not pc["reached_90"] else 2 * pc["iters_to_90"] + int(r.get("meas_offset", 0))
        acc_star = best_acc.get(key)
        r["acc_star"] = None if acc_star is None or not np.isfinite(acc_star) else acc_star
        r.update(accuracy_criterion(it, [c.get("acc_clean") for c in r["curve"]], r["acc_star"], r.get("acc_chance"),
                                    r["n_iter"]))
        if "faithful_curve" in r:
            Lf = [c["loss_meas_clean"] for c in r["faithful_curve"]]
            pf = progress_criterion([c["iter"] for c in r["faithful_curve"]], Lf, Lf[0], Ls, r.get("chance"),
                                    r["n_iter"])
            r.update(faithful_testable=pf["testable"], faithful_reached_90=pf["reached_90"],
                     faithful_settled_90=pf["settled_90"], faithful_iters_to_90=pf["iters_to_90"])


def _ci(t: dict, m: str, f: str = ".3g") -> str:
    mean, half, n = t.get(f"{m}_mean"), t.get(f"{m}_ci95"), t.get(f"{m}_n", 0)
    if not n or mean is None or not math.isfinite(mean):
        return "—"
    if half is None or not math.isfinite(half):
        return format(mean, f)
    return f"{format(mean, f)} ± {format(half, f)}"


def _m(t: dict, m: str, f: str = ".2f") -> str:
    v = t.get(f"{m}_mean")
    return "—" if v is None or not math.isfinite(v) else format(v, f)


def median_iters(rows: list[dict], task: str, cond: str, key: str = "iters_to_90", cens_key: str = "censored") -> str:
    """删失数据的中位数：删失记为 +∞；中位数落在删失上时写 "> n_iter"；不可检验的 run 不计入（全部不可检验 → "n/t"）。"""
    rr = [r for r in _train_rows(rows) if r["task"] == task and r["cond"] == cond]
    if not rr:
        return "—"
    rr = [r for r in rr if r.get("testable")]
    if not rr:
        return "n/t"
    v = np.array([np.inf if r.get(cens_key, r.get(key) is None) or r.get(key) is None else float(r[key]) for r in rr])
    med = float(np.median(v))
    return f"> {rr[0]['n_iter']}" if not np.isfinite(med) else f"{med:.0f}"


def _train_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if "error" not in r and r.get("kind") != "cal"]


def summary_markdown(table: list[dict], rows: list[dict], cfg: dict) -> str:
    seeds = cfg["seeds"]
    gain = "sign" if _sign_mode(cfg) else cfg.get("gain", "shared")
    gain_txt = ("sign_acc 变体（INTEGRATION T：判对优先可测目标 + sign-SPSA，首步 0.1、α = 0.602、A = 400，无需增益标定，"
                "各条件步长天然相同；曲线 = 当时的输出码本（pocket））" if gain == "sign" else
                (f"subset 变体，a = shared（每个 (任务, 种子) 在 {cfg.get('gain_ref', GAIN_REF)} 上用 calibrate_dspsa_gain 标定一次、"
                 f"{cfg.get('n_cal_starts', N_CAL_STARTS)} 个起点，同一数值用于全部条件，审查 R-E1 / INTEGRATION B3；A = 400，α = 0.501）"
                 if gain == "shared" else "subset 变体，a = auto（每条件各自标定：对照臂，会把整体变平归一化掉；A = 400，α = 0.501）"))
    lines = [f"# 实验 E：可训练性（RQ6）{'— quick' if cfg['quick'] else ''}", "",
             f"- L90 预设、场景 {cfg.get('scenario', 'baseline')}（INTEGRATION S），full 耦合；DSPSA {cfg['n_iter']} 次迭代"
             f"（{gain_txt}），每 "
             f"{cfg['eval_every']} 次迭代记录无噪声可测目标（不计入测量次数）；1-bit oracle {cfg['oracle_restarts']} 次重启"
             "（损失目标 + 分类任务的准确率目标）。",
             f"- 每格 {seeds} 个种子" + ("（quick 模式少于 5 个种子）" if seeds < 5 else "") +
             "；同种子在所有条件下用同一初始码本、扰动序列与增益 a（公共随机数）。",
             f"- **判据（审查 R-E2）**：L* = 同任务同条件全部种子、两种 oracle 目标的最低 loss_meas_clean（pooled）。分类任务以 "
             f"chance 为参照：首个 k 使 (chance − L_k) ≥ {cfg['fraction']:g}·(chance − L*)，只有 chance − L* > "
             f"{cfg.get('delta_chance', DELTA_CHANCE):g} 才可检验（testable）；focus 以 L0 为参照，L0 − L* > "
             f"{cfg.get('delta_focus_db', DELTA_FOCUS_DB):g} dB 才可检验。未达到删失为 > n_iter；median 把删失当 +∞，"
             "mean_lb 把删失记为 n_iter（下界）；不可检验的 run 不计入（n/t）。settled = 达到并保持。"
             "acc 判据：acc_k − acc_chance ≥ 0.9·(acc* − acc_chance)，acc* = pooled oracle 最高准确率。",
             "- 原 L0 参照判据 (L0 − L_k) ≥ 0.9(L0 − L*) 在奇偶上退化：L0 ≈ 3 是 y1 − y2 约 6 dB 的常数偏置、L* ≈ chance，"
             "它只量\"把偏置拉平\"的速度（本节不再使用；runs.jsonl 的 run_* 是以本 run 自己的 oracle 为 L* 的同一 chance 判据）。",
             "- 地形：dL = 随机码本翻转 m 比特的无噪声 |ΔL| 中位数；noise_sd = 同码本有噪声测量的损失标准差；"
             "g_early = DSPSA 前 100 次迭代 mean|ĝ|（shared a 下各条件可直接比较；sign_acc 的符号梯度下恒为 1，无信息）。PA in = 最终码本闭环 PA 输入相对 P1dB（dB）。",
             ""]
    tasks = [t for t in TASKS if any(r.get("task") == t for r in table)]
    for task in tasks:
        lines += [f"## {task}", "",
                  "| cond | backoff (dB) | n | a | testable | reached 90% | iters median | iters mean_lb | settled 90% | "
                  "below chance | L0 | L_final | L_min | L* (pooled) | chance | acc DSPSA | acc* oracle | acc reached | "
                  "acc iters mean_lb | dL | noise_sd | g_early | PA in min/mean/max | null (dB) | P_feed (dBm) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in cfg["conditions"]:
            t = next((x for x in table if x["task"] == task and x["cond"] == c), None)
            if t is None:
                continue
            pa = "/".join(_m(t, f"pa_in_rel_p1db_db_{k}", ".1f") for k in ("min", "mean", "max"))
            lines.append(
                f"| {c} | {cond_backoff_db(c):+g} | {t['n_runs']}" + (f" (err {t['n_errors']})" if t["n_errors"] else "") +
                f" | {_ci(t, 'a', '.2f')} | {_m(t, 'testable')} | {_m(t, 'reached_90')} | {median_iters(rows, task, c)} | "
                f"{_ci(t, 'iters_to_90_lb', '.0f')} | {_m(t, 'settled_90')} | {_m(t, 'below_chance')} | "
                f"{_ci(t, 'L0')} | {_ci(t, 'L_final')} | {_ci(t, 'L_min')} | {_m(t, 'Lstar', '.3f')} | "
                f"{_m(t, 'chance', '.3f')} | {_ci(t, 'final_acc', '.2f')} | {_m(t, 'acc_star')} | "
                f"{_m(t, 'acc_reached_90')} | {_ci(t, 'acc_iters_to_90_lb', '.0f')} | "
                f"{_ci(t, 'land_dL_med')} | {_ci(t, 'land_noise_sd')} | {_ci(t, 'g_abs_early')} | {pa} | "
                f"{_m(t, 'null_depth_db', '.1f')} | {_m(t, 'p_feed_dbm', '.1f')} |")
        lines.append("")
    # faithful vs subset
    fr = [r for r in _train_rows(rows) if "faithful_curve" in r]
    mv = "sign_acc" if gain == "sign" else "subset"
    if fr:
        lines += [("## faithful vs sign_acc（1-bit 中继，同种子；faithful 用 a = auto）" if gain == "sign" else
                   "## faithful vs subset（1-bit 中继，同种子、同一个 a）"), "",
                  f"| task | cond | n | max|L(x)−L(1−x)| | {mv}: L_final | faithful: L_final | faithful: L0 | "
                  f"median |L(x+)−L(x−)| {mv} | … faithful | flips {mv} | flips faithful | faithful reached 90% | "
                  f"faithful settled 90% | {mv} settled 90% |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        ft = common.aggregate(fr, ["task", "cond"], ["flip_sym_max_abs_dL", "L_final", "faithful_L_final", "faithful_L0",
                                                     "dL_pm_med", "faithful_dL_pm_med", "flips_total",
                                                     "faithful_flips_total", "faithful_reached_90", "faithful_settled_90",
                                                     "settled_90"])
        for t in ft:
            lines.append(f"| {t['task']} | {t['cond']} | {t['n_runs']} | {_m(t, 'flip_sym_max_abs_dL', '.1e')} | "
                         f"{_ci(t, 'L_final')} | {_ci(t, 'faithful_L_final')} | {_ci(t, 'faithful_L0')} | "
                         f"{_m(t, 'dL_pm_med', '.2e')} | {_m(t, 'faithful_dL_pm_med', '.2e')} | {_m(t, 'flips_total', '.0f')} | "
                         f"{_m(t, 'faithful_flips_total', '.0f')} | {_m(t, 'faithful_reached_90')} | "
                         f"{_m(t, 'faithful_settled_90')} | {_m(t, 'settled_90')} |")
        sym = [r["flip_sym_max_abs_dL"] for r in _train_rows(rows) if r.get("flip_sym_max_abs_dL") is not None]
        walk = ("shared a 由 subset 的 |ĝ|（信号级）标定，faithful 的 ĝ 只有测量噪声级 → θ 几乎不动，判决码本停在初始码本附近"
                if gain == "shared" else ("（sign_acc 模式下 faithful 臂用 a = auto）" if gain == "sign" else "") +
                "a = auto 按 faithful 自己的噪声级 |ĝ| 标定 → 每步约 0.3 的随机游走，判决码本随机翻转；reached 是随机游走"
                "偶然碰到低损失码本，settled（达到并保持）才表示收敛")
        lines += ["", f"全部条件的 max|L(x) − L(1−x)| = {max(sym):.1e}（取反对称精确成立 → faithful 的 ĝ 只含测量噪声）。"
                  if sym else "", f"faithful 的 θ 只被测量噪声驱动：{walk}。", ""]
    lines += [verdict_text(rows, cfg), ""]
    n_err = sum(1 for r in rows if "error" in r)
    if n_err:
        lines.append(f"{n_err} 个 run 出错（见 runs.jsonl 的 error 字段）。")
    return "\n".join(lines) + "\n"


def verdict_text(rows: list[dict], cfg: dict) -> str:
    """预测检验（审查 R-E2）：Rapp 条件（M2 → M3 → M4 → Rapp@+20）的回退 dB 与收敛迭代数（删失下界）的种子级 Spearman；
    oracle 不优于 chance（不可检验）或被比较条件的 DSPSA 全部删失时报"不可检验"，不给相关系数。
    另报训练无关的地形指标（dL、g_early，shared a 下可比）与回退的相关。"""
    from scipy.stats import spearmanr

    out = ["## 预测检验（RQ6）", "",
           "预测：达到 oracle 90% 的迭代数随饱和加深（回退增大）上升。种子级 Spearman ρ(回退 dB, ·)；"
           "iters 用删失下界（未达到记 n_iter）。", ""]
    bo = {c: cond_backoff_db(c) for c in cfg["conditions"] if parse_condition(c)["condition"] in ("M2", "M3", "M4")}
    names = ", ".join(sorted(bo, key=bo.get))

    def rho(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            return None
        return float(spearmanr(x[ok], y[ok]).statistic)

    def fmt(v):
        return "—" if v is None else f"{v:+.2f}"

    for task in TASKS:
        rr = [r for r in _train_rows(rows) if r.get("task") == task and r.get("cond") in bo]
        if len(rr) < 4:
            continue
        x = [bo[r["cond"]] for r in rr]
        land = (f"地形（训练无关）：ρ(回退, dL) = {fmt(rho(x, [r.get('land_dL_med') for r in rr]))}，"
                f"ρ(回退, g_early) = {fmt(rho(x, [r.get('g_abs_early') for r in rr]))}")
        testable = [r for r in rr if r.get("testable")]
        Lst = [r["Lstar"] for r in rr if r.get("Lstar") is not None]
        ref = rr[0].get("chance")
        if len(testable) < len(rr) / 2:
            why = (f"oracle L* 平均 {np.mean(Lst):.3f}，chance {ref:.3f}：oracle 不比零信息损失好 "
                   f"{cfg.get('delta_chance', DELTA_CHANCE):g} 以上" if ref is not None and Lst
                   else "oracle 相对初始损失的改进不足")
            out.append(f"- {task}：**不可检验**（{why}）。{land}（n = {len(rr)}，条件 {names}）。")
            continue
        if not any(r.get("reached_90") for r in testable):
            bc = np.mean([bool(r.get("below_chance")) for r in testable]) if ref is not None else None
            extra = f"；DSPSA 曾低于 chance 的 run 比例 {bc:.2f}" if bc is not None else ""
            out.append(f"- {task}：**不可检验**（全部 {len(testable)} 个可检验 run 都删失：DSPSA 从未达到 90% 阈值"
                       f"{extra}，无法比较收敛速度）。{land}（条件 {names}）。")
            continue
        xt = [bo[r["cond"]] for r in testable]
        r1 = rho(xt, [r["iters_to_90_lb"] for r in testable])
        r2 = rho(xt, [r["iters_to_90_settled_lb"] for r in testable])
        ra = [r for r in testable if r.get("acc_testable")]
        r3 = rho([bo[r["cond"]] for r in ra], [r["acc_iters_to_90_lb"] for r in ra]) if ra else None
        reach = {c: np.mean([bool(r.get("reached_90")) for r in testable if r["cond"] == c])
                 for c in sorted(bo, key=bo.get) if any(r["cond"] == c for r in testable)}
        out.append(f"- {task}：ρ(回退, 首达) = {fmt(r1)}，ρ(回退, 达到并保持) = {fmt(r2)}，ρ(回退, acc 首达) = {fmt(r3)}"
                   f"（n = {len(testable)}；达到比例 " + "、".join(f"{c} {v:.2f}" for c, v in reach.items())
                   + f"；L* 平均 {np.mean(Lst):.3f}" + (f" vs chance {ref:.3f}" if ref is not None else "")
                   + f"）。{land}。")
    return "\n".join(out)


def _progress(r: dict, key: str = "curve"):
    """曲线的判据进度 (ref − L_k)/(ref − L*)（ref = chance 或 L0）；不可检验时 None。"""
    if not r.get("testable") or r.get("Lstar") is None:
        return None
    L = np.array([c["loss_meas_clean"] for c in r[key]], dtype=float)
    ref = r["ref"] if r.get("ref_kind") == "chance" else L[0]
    gap = ref - r["Lstar"]
    return (ref - L) / gap if gap > 0 else None


def plot_curves(rows: list[dict], cfg: dict, outdir: str) -> list[str]:
    import matplotlib.pyplot as plt

    paths = []
    tr = _train_rows(rows)
    tasks = [t for t in TASKS if any(r.get("task") == t for r in tr)]
    conds = cfg["conditions"]
    fig, axes = plt.subplots(2, len(tasks), figsize=(7.0 * len(tasks), 8.4), squeeze=False, constrained_layout=True)
    for j, task in enumerate(tasks):
        for k, c in enumerate(conds):
            rr = [r for r in tr if r["task"] == task and r["cond"] == c]
            if not rr:
                continue
            it = np.array([p["iter"] for p in rr[0]["curve"]])
            L = np.array([[p["loss_meas_clean"] for p in r["curve"]] for r in rr], dtype=float)
            mu, sd = np.nanmean(L, axis=0), np.nanstd(L, axis=0)
            ax = axes[0][j]
            ax.plot(it, mu, color=f"C{k}", label=c)
            ax.fill_between(it, mu - sd, mu + sd, color=f"C{k}", alpha=0.15)
            if rr[0].get("Lstar") is not None:
                ax.axhline(rr[0]["Lstar"], color=f"C{k}", ls="--", lw=1)
            prog = [p for p in (_progress(r) for r in rr) if p is not None]
            if prog:
                P = np.array(prog)
                ax2 = axes[1][j]
                ax2.plot(it, np.nanmean(P, axis=0), color=f"C{k}", label=c)
                ax2.fill_between(it, np.nanmean(P, axis=0) - np.nanstd(P, axis=0),
                                 np.nanmean(P, axis=0) + np.nanstd(P, axis=0), color=f"C{k}", alpha=0.12)
        ch = next((r["chance"] for r in tr if r["task"] == task), None)
        if ch is not None:
            axes[0][j].axhline(ch, color="k", ls=":", lw=1, label="chance")
        axes[0][j].set_title(f"{task}：DSPSA loss_meas_clean（均值 ± sd）\n虚线 = 各条件 pooled oracle L*", fontsize=10)
        axes[0][j].set_xlabel("迭代")
        axes[0][j].set_ylabel("loss_meas_clean")
        axes[0][j].legend(fontsize=8)
        ref_txt = "(chance − L_k)/(chance − L*)" if ch is not None else "(L0 − L_k)/(L0 − L*)"
        axes[1][j].axhline(cfg["fraction"], color="k", ls=":", lw=1)
        axes[1][j].set_ylim(-0.5, 1.3)
        axes[1][j].set_title(f"{task}：进度 {ref_txt}（仅可检验的 run）", fontsize=10)
        axes[1][j].set_xlabel("迭代")
        if axes[1][j].get_legend_handles_labels()[0]:
            axes[1][j].legend(fontsize=8)
        else:
            axes[1][j].text(0.5, 0.5, "不可检验（oracle 不优于 chance）", ha="center", va="center",
                            transform=axes[1][j].transAxes, fontsize=9)
    paths.append(common.savefig(fig, os.path.join(outdir, "learning_curves.png")))
    # faithful vs subset
    fr = [r for r in tr if "faithful_curve" in r]
    if fr:
        keys = list(dict.fromkeys((r["task"], r["cond"]) for r in fr))
        fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 3.6), squeeze=False)
        for j, (task, c) in enumerate(keys):
            ax = axes[0][j]
            rr = [r for r in fr if r["task"] == task and r["cond"] == c]
            it = np.array([p["iter"] for p in rr[0]["curve"]])
            for name, key, col in (("subset", "curve", "C0"), ("faithful", "faithful_curve", "C3")):
                L = np.array([[p["loss_meas_clean"] for p in r[key]] for r in rr], dtype=float)
                for q in range(L.shape[0]):
                    ax.plot(it, L[q], color=col, alpha=0.35, lw=0.8)
                ax.plot(it, L.mean(axis=0), color=col, lw=2, label=name)
            if rr[0].get("Lstar") is not None:
                ax.axhline(rr[0]["Lstar"], color="k", ls="--", lw=1, label="pooled oracle L*")
            ax.set_title(f"{task} / {c}：1-bit DSPSA 两种变体")
            ax.set_xlabel("迭代")
            ax.set_ylabel("loss_meas_clean")
            ax.legend(fontsize=8)
        paths.append(common.savefig(fig, os.path.join(outdir, "faithful_vs_subset.png")))
    return paths


def summarize(rows: list[dict], outdir: str, cfg: dict) -> dict:
    finalize_rows(rows)
    tr = [r for r in rows if r.get("kind") != "cal"]
    table = common.aggregate(tr, ["task", "cond"], METRICS)
    paths = common.write_outputs(outdir, rows, table, config=cfg)
    md = summary_markdown(table, tr, cfg)
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(md)
    try:
        paths["figs"] = plot_curves(tr, cfg, outdir)
    except Exception as e:          # noqa: BLE001 —— 作图失败不影响数据输出
        paths["fig_error"] = repr(e)
    return {"table": table, "paths": paths, "markdown": md}


def run(cfg: dict, workers: int = 1, out: str | None = None, progress: bool = False) -> dict:
    """两阶段：(1) shared 模式下每个 (任务, 种子) 在参考条件上标定 a（审查 R-E1）；(2) 训练 job。"""
    outdir = common.make_outdir(EXP, out)
    t0 = time.perf_counter()
    cal_jobs = make_cal_jobs(cfg)
    cal_rows = common.run_jobs(run_any_job, cal_jobs, workers=workers, progress=progress) if cal_jobs else []
    gains = {(r["task"], r["seed"]): r["a"] for r in cal_rows if "error" not in r and r.get("a") is not None}
    for r in cal_rows:
        if "error" in r:
            j = r.get("job", {})
            r.update(kind="cal", task=j.get("task"), seed=j.get("seed"))
    jobs = make_jobs(cfg, gains)
    rows = common.run_jobs(run_any_job, jobs, workers=workers, progress=progress)
    for r in rows:
        if "error" in r:
            j = r.get("job", {})
            r.update(task=j.get("task"), cond=j.get("cond"), seed=j.get("seed"))
    cfg = dict(cfg, n_jobs=len(jobs) + len(cal_jobs), wall_time_s=time.perf_counter() - t0, workers=int(workers),
               shared_gains={f"{t}/{s}": a for (t, s), a in gains.items()})
    res = summarize(cal_rows + rows, outdir, cfg)
    res.update(outdir=outdir, rows=rows, cal_rows=cal_rows)
    return res


def parse_args(argv=None):
    p = common.base_argparser("实验 E：可训练性（RQ6，方案 §6.2）")
    p.set_defaults(seeds=None)
    p.add_argument("--conditions", nargs="+", default=None,
                   help="条件（默认 M2 M3 M4 Rapp@+20 M1@M3；Rapp@<dB> = Rapp PA + 指定源功率回退）")
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=None, help="任务（默认 parity3 letter focus）")
    p.add_argument("--n-iter", type=int, default=None, help="DSPSA 迭代数（quick 1500 / full 4000）")
    p.add_argument("--oracle-restarts", type=int, default=None, help="1-bit oracle 重启数（quick 4 / full 8）")
    p.add_argument("--faithful-conds", nargs="+", default=None, help="额外跑 faithful 变体的条件（默认 M3）")
    p.add_argument("--variant", choices=["auto", "sign_acc", "subset"], default=None,
                   help="DSPSA 变体（默认 auto = 1-bit sign_acc，INTEGRATION T；subset = 旧默认，配合 --gain）")
    p.add_argument("--scenario", default=None, choices=["baseline", "informative", "informative_probe"],
                   help="命名场景（system.SCENARIOS，INTEGRATION S；默认 baseline）")
    p.add_argument("--gain", choices=list(GAIN_MODES), default=None,
                   help="subset 变体的增益：shared（默认，参考条件标定一次、各条件共用，INTEGRATION B3）| auto（对照臂）")
    p.add_argument("--gain-ref", default=None, help="shared 增益的参考条件（默认 M3）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    t0 = time.perf_counter()
    res = run(cfg, workers=args.workers, out=args.out, progress=True)
    print(res["markdown"])
    print(f"输出目录：{res['outdir']}（{time.perf_counter() - t0:.0f} s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
