"""实验 C：饱和归一化的利弊（RQ4；方案 §3 RQ4、§6.2 表"C 归一化的利弊"、§7"不变性损失"；INTEGRATION G "C"、B3）。

假设（方案 RQ4）：PA 饱和像 AGC，提高对入射功率 / 物体距离变化的不变性（形状识别受益），但损失幅度信息
（阈值定位受损）。预测：M4 在字母上更稳，在定位上变差。

任务（INTEGRATION E）：
  letter   字母 A/B/C，金属薄板（厚度 1 体素，d = λ/10，INTEGRATION B2），像素 λ/2，3 探头 argmax；
  loc3bit  Song 2026 式 8 格定位：2×4 格（格宽 λ/2，水球 λ/3，自动体素），**一套码本、3 个探头各自阈值 → 3-bit 格号**。
两个任务的物体平面都是 plane_u = +y、plane_v = +z，法向 n̂ = û×v̂ = +x（正对 L90 接收板）。

抖动（方案 §6.2，逐样本独立）：入射功率 U(−6, +6) dB（meta["power_db"]，系统层按它缩放馈源功率，INTEGRATION C）、
物体沿 n̂ 平移 U(−λ/4, +λ/4)（"物体距离"）。抖动模式 none / power / dist / both；同一个种子下各模式共用同一组
均匀随机数（公共随机数：power 模式的功率抖动值与 both 模式相同）。

协议：
  - 训练集 / 测试集：none 模式两者相同（样本确定，无可 hold-out 的东西）；有抖动时测试集是**独立重新抽取**的抖动样本
    （held-out：抖动值来自与训练集不同的随机流，与训练集不共享任何样本）。
  - 训练器（审查 R-C1，INTEGRATION B3）：
      oracle       = 1-bit 模型 oracle，**准确率目标**（train.run_oracle_bits(objective="accuracy")：最大化训练集判对样本数，
                     裕量只打破平局）——"这个结构在训练集上最多判对几个"，主结果；
      oracle_loss  = 同一 oracle 但用默认 softplus 损失目标（可选，标注为次要）：不可解 / 难任务上它最大化平均带符号裕量
                     而不是判对数（≈ τlog2 − mean(s)/2），acc 可远低于可达值（定位实测 0.29–0.38 vs 准确率目标 0.58），
                     **不是**"最好可达准确率"；
      dspsa        = DSPSA（train.run_dspsa，可测目标、有噪声测量，"实际训练"）。
  - loc3bit 的阈值：用最终码本在**训练集**上的无噪声读数求出（两类中位数中点，losses.loc3bit_thresholds），固定后
    用于测试集（不使用测试集标签）。none 模式下训练集即测试集，与 "auto" 阈值相同。
  - 不变性损失（方案 §7）= acc(none，同条件同种子同训练器) − acc(有抖动，held-out 测试集)；
    另报"零样本"不变性损失 = acc(none) − acc(none 训练的码本直接用于抖动测试集，阈值同样来自无抖动训练集)：
    它直接量"训练好的物理映射本身"对抖动的不变性（AGC 效应），不含"在抖动数据上重新训练"的补偿。
  - 准确率：被标记可能自激的样本算错（system.evaluate）；acc_clean 为无噪声读数，acc_noisy 为 VNA 级读数噪声。
条件（INTEGRATION B，"同功率"配对）：M1@M3（线性 PA、小信号增益，源功率 = M3）、M3、M4、Rapp@+20（Rapp PA、源功率
回退 +20 dB，审查 R-C3）、M1@M4（线性，源功率 = M4）。
饱和是否真的发生（审查 R-C3）：训练会把码本推向对入射波零陷（方案推论 3），名义 M4（+10 dB）训练后的闭环 PA 输入常在
P1dB 以下 3–14 dB（Rapp p=2 在 −5 dB 处压缩 < 0.1 dB，近线性）——此时"M4 vs M1@M3"并不检验 AGC 假设。所以加
Rapp@+20（比 M4 再深 10 dB，零陷后仍多在 P1dB 以上），并对每行报测试集上闭环 PA 输入 ≥ P1dB 的样本比例 sat_frac；
判读只把 sat_frac ≥ SAT_MIN 的饱和条件当作 AGC 检验。
定位判读（审查 R-C2）：±6 dB 功率抖动（标准差 3.5 dB）对固定 dBm 阈值的线性中继按构造是毁灭性的（多数探头在 8 格间的
读数对比度 < 1 dB），power / both 下"M4 定位更差"无法被检验——它们是 AGC 检验（饱和可能帮助）。"M4 定位更差"的检验
用 none 与 dist 两行（d_test_acc、d_inv_loss）。
实际工作点：每个 run 报测试集上每样本 PA 输入（闭环 |u|²，相对 P1dB）的 min/mean/max、sat_frac、开环值、零陷深度、
P_feed、体素尺寸。
场景（--scenario，system.SCENARIOS / apply_scenario，INTEGRATION S）：默认 baseline（原 L90 场景，ROI 0.5 m、馈源→探头视距保留）；
informative = ROI 拉近到 0.35 m + 遮挡馈源→探头视距；informative_probe 另把探头方位改为 45° / 75°（第三探头 60°）。
场景对 quick / full 都不改默认（与实验 E 相同）。

用法：python -m ris_nlsim.experiments.exp_c_invariance [--quick] [--workers N] [--seeds S] [--out DIR]
      [--tasks letter loc3bit] [--conditions M1@M3 M3 M4 Rapp@+20 M1@M4] [--jitters none power dist both]
      [--trainers oracle oracle_loss dspsa] [--n-iter N] [--scenario baseline|informative|informative_probe]
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time

import numpy as np

from . import common

EXP = "exp_c_invariance"
TASKS = ("letter", "loc3bit")
JITTERS = ("none", "power", "dist", "both")
TRAINERS = ("oracle", "oracle_loss", "dspsa")
DEFAULT_TRAINERS = ("oracle", "dspsa")
ORACLE_OBJECTIVE = {"oracle": "accuracy", "oracle_loss": "loss"}      # 审查 R-C1：主 oracle 用准确率目标
TRAINER_LABEL = {"oracle": "oracle（准确率目标）", "oracle_loss": "oracle（softplus 损失目标）", "dspsa": "DSPSA"}
ALL_CONDITIONS = ("M1@M3", "M3", "M4", "Rapp@+20", "M1@M4")
RAPP_PREFIX = "Rapp@"                  # "Rapp@+20"：Rapp PA、源功率回退 +20 dB（审查 R-C3，同实验 E）
PLANE_U = (0.0, 1.0, 0.0)
PLANE_V = (0.0, 0.0, 1.0)
POWER_JITTER_DB = 6.0                  # 方案 §6.2：±6 dB
DIST_JITTER_LAM = 0.25                 # 方案 §6.2：±λ/4
REF_COND = "M1@M3"                     # 配对差分的线性参考
SAT_CONDS = ("M4", "Rapp@+20")         # 名义饱和条件（判读的 RQ4 检验对象）
SAT_MIN = 0.5                          # 测试集 PA 输入 ≥ P1dB 的样本比例 ≥ 它才算"饱和"（审查 R-C3）


# ============================================================ 条件
def parse_condition(label: str) -> dict:
    """条件标签 → SystemConfig 字段："M1@M4" → M1 + M4 的回退（+10 dB）；"Rapp@+20" → Rapp PA（M2–M4 同一 PA）+
    回退 +20 dB；"M3" → 条件表（INTEGRATION B）。"""
    from ..system import CONDITION_BACKOFF_DB, CONDITIONS

    lab = str(label).strip()
    if lab.startswith(RAPP_PREFIX):
        return {"condition": "M4", "backoff_db": float(lab[len(RAPP_PREFIX):])}
    if "@" in lab:
        base, ref = lab.split("@", 1)
        if base not in ("M1", "M1b") or ref not in CONDITION_BACKOFF_DB:
            raise ValueError(f"无法解析条件 {label!r}")
        return {"condition": base, "backoff_db": float(CONDITION_BACKOFF_DB[ref])}
    if lab not in CONDITIONS:
        raise ValueError(f"未知条件 {label!r}")
    return {"condition": lab}


def jitter_amplitudes(mode: str, lam: float) -> tuple[float, float]:
    """抖动模式 → (距离半幅 m, 功率半幅 dB)。"""
    if mode not in JITTERS:
        raise ValueError(f"未知抖动模式 {mode!r}")
    dist = DIST_JITTER_LAM * lam if mode in ("dist", "both") else 0.0
    pdb = POWER_JITTER_DB if mode in ("power", "both") else 0.0
    return dist, pdb


# ============================================================ 任务与抖动
def apply_jitter(task, rng: np.random.Generator, dist: float, power_db: float, normal=None):
    """逐样本抖动（原地修改 task，返回 task）：物体沿 n̂ 平移 dist·(2U₁−1)，meta["power_db"] = power_db·(2U₂−1)。

    每个样本总是抽两个均匀数（与幅度无关）→ 同一 rng 状态下 power / dist / both 模式共用同一组抖动值（公共随机数）。
    物体极化率与位置无关，只平移体素坐标。修改后清除 make_spec（PhysicalSystem.derive 不得按原规格无抖动重建）。
    """
    n = np.cross(PLANE_U, PLANE_V) if normal is None else np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    for s in task.samples:
        u1, u2 = rng.uniform(0.0, 1.0, size=2)
        shift = float(dist) * (2.0 * u1 - 1.0)
        pdb = float(power_db) * (2.0 * u2 - 1.0)
        if shift != 0.0 and len(s.objects):
            s.objects = dataclasses.replace(s.objects, pos=s.objects.pos + shift * n[None, :])
        s.meta = dict(s.meta, dist_shift=shift, power_db=pdb)
    task.make_spec = None
    return task


def make_c_task(cfg, kind: str, n_rep: int, jitter: str, rng: np.random.Generator | None, loc_cell: float | None = None):
    """实验 C 的任务：letter（n_per_class = n_rep）或 loc3bit（n_per_cell = n_rep），再按抖动模式施加抖动。"""
    from ..constants import wavelength
    from ..system import make_task

    if kind == "letter":
        task = make_task(cfg, "letter", n_per_class=int(n_rep), plane_u=PLANE_U, plane_v=PLANE_V)
    elif kind == "loc3bit":
        kw = {} if loc_cell is None else {"cell": float(loc_cell)}
        task = make_task(cfg, "loc3bit", n_per_cell=int(n_rep), plane_u=PLANE_U, plane_v=PLANE_V, **kw)
    else:
        raise ValueError(f"未知任务 {kind!r}")
    dist, pdb = jitter_amplitudes(jitter, wavelength(cfg.freq_hz))
    if dist > 0 or pdb > 0:
        if rng is None:
            raise ValueError("有抖动时必须给 rng")
        apply_jitter(task, rng, dist, pdb)
    return task


def train_thresholds(system, params) -> np.ndarray | None:
    """loc3bit：最终码本在训练集上的无噪声检波器读数 → 自动阈值（两类中位数中点，losses.loc3bit_thresholds，
    与任务当前的 thresholds_db 设置无关）；其他任务 None。"""
    from ..losses import loc3bit_thresholds

    if system.task.kind != "loc3bit":
        return None
    m = system.forward(system.phases_from_params(params), noisy=False)
    return np.asarray(loc3bit_thresholds(system.clean_readings(m), system.labels), dtype=float)


def evaluate_on(system, params, thresholds, rng, n_noise: int) -> dict:
    """在 system（测试集）上评估码本；loc3bit 时先把阈值固定为训练集阈值（不用测试标签）。"""
    if system.task.kind == "loc3bit":
        system.task.thresholds_db = "auto" if thresholds is None else np.asarray(thresholds, dtype=float)
    return system.evaluate(params, rng, n_noise=n_noise)


def saturation_fraction(system, params) -> float | None:
    """审查 R-C3：码本下（无噪声）未标记样本 × 中继的闭环 PA 输入 ≥ P1dB_in 的比例（"真正工作在饱和区"的样本比例）。
    M0 / 无未标记样本时 None。"""
    if system.is_passive:
        return None
    m = system.forward(system.phases_from_params(params), noisy=False)
    ok = ~np.asarray(m.flags, dtype=bool)
    if not ok.any():
        return None
    return float(np.mean(np.abs(m.u[ok]) ** 2 >= system.p1db_in_w))


# ============================================================ job
_EVAL_KEYS = ("acc_clean", "acc_noisy", "all_correct", "margin_db", "margin_db_unflagged", "loss_clean",
              "loss_meas_clean", "oscillating_run", "n_flagged", "pa_in_rel_p1db_db_min", "pa_in_rel_p1db_db_mean",
              "pa_in_rel_p1db_db_max", "pa_in_open_rel_p1db_db_min", "pa_in_open_rel_p1db_db_mean",
              "pa_in_open_rel_p1db_db_max", "pa_in_delta_rel_p1db_db_mean", "null_depth_db", "loop_gain_ss_max",
              "loop_gain_op_max", "noisy_flagged_frac", "bit_acc_clean")


def _pick(ev: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": ev.get(k) for k in _EVAL_KEYS}


def _train(system, trainer: str, rng, job: dict) -> dict:
    from ..train import run_dspsa, run_oracle_bits

    t0 = time.perf_counter()
    if system.task.kind == "loc3bit":
        system.task.thresholds_db = "auto"      # 训练损失用训练集自身的自动阈值（none 模式下训练集即测试集，可能被改过）
    if trainer in ORACLE_OBJECTIVE:
        obj = ORACLE_OBJECTIVE[trainer]          # 审查 R-C1："oracle" = 准确率目标（最好可达准确率），"oracle_loss" = 损失目标
        res = run_oracle_bits(system, rng, n_restarts=int(job["oracle_restarts"]), max_sweeps=int(job["oracle_sweeps"]),
                              n_noise_eval=0, objective=obj)
        info = {"n_evals": int(res["n_evals"]), "n_measurements": None, "flagged_meas_frac": None,
                "objective": res["objective"]}
    elif trainer == "dspsa":
        n_iter = int(job["n_iter"])
        res = run_dspsa(system, n_iter, rng, eval_every=max(50, n_iter // 4), n_noise_eval=0)
        info = {"n_evals": None, "n_measurements": int(res["n_measurements"]),
                "flagged_meas_frac": res["n_flagged_measurements"] / max(1, res["n_measurements"]),
                "objective": res["objective"]}
    else:
        raise ValueError(f"未知训练器 {trainer!r}")
    info["train_time_s"] = time.perf_counter() - t0
    return {"params": np.asarray(res["params"], dtype=np.int64), "train_eval": res["eval"], "info": info}


def make_c_cfg(job: dict):
    """job → SystemConfig：预设 / 物体模式 / 条件，再叠加命名场景（system.apply_scenario；baseline = 原配置不变）。"""
    from ..system import SystemConfig, apply_scenario

    cfg = SystemConfig(preset=job.get("preset", "L90"), object_mode=job.get("object_mode", "full"),
                       **parse_condition(job["cond"]))
    return apply_scenario(cfg, job.get("scenario", "baseline"))


def run_c_job(job: dict) -> dict:
    """一个 job = (任务, 条件, 种子)：对每个训练器先训练 none，再训练每种抖动模式并在 held-out 测试集评估。

    返回 {"rows": [每个 (抖动模式, 训练器) 一行]}。公共随机数：训练/测试抖动与训练器随机流只依赖 (任务, 种子[, 训练器])，
    与条件无关 → 同一种子下各条件的数据集与初始码本相同（配对比较）。
    """
    from ..constants import wavelength
    from ..system import PhysicalSystem

    kind, cond, seed = job["task"], job["cond"], int(job["seed"])
    scenario = job.get("scenario", "baseline")
    cfg = make_c_cfg(job)
    lam = wavelength(cfg.freq_hz)
    loc_cell = None if job.get("loc_cell_lam") is None else float(job["loc_cell_lam"]) * lam
    n_noise = int(job["n_noise"])
    jitters = list(job["jitters"])
    if "none" not in jitters:
        jitters = ["none"] + jitters
    t_job = time.perf_counter()
    # 数据集：none（训练 = 测试）与各抖动模式的训练集 / held-out 测试集（随机流与条件无关）
    systems = {}
    for jm in jitters:
        if jm == "none":
            tr = make_c_task(cfg, kind, 1, "none", None, loc_cell)
            st = PhysicalSystem(cfg, tr)
            systems[jm] = (st, st)
        else:
            n_tr, n_te = (job["n_rep_train"], job["n_rep_test"])
            tr = make_c_task(cfg, kind, n_tr, jm, common.seed_rng(EXP, kind, seed, "train"), loc_cell)
            te = make_c_task(cfg, kind, n_te, jm, common.seed_rng(EXP, kind, seed, "test"), loc_cell)
            systems[jm] = (PhysicalSystem(cfg, tr), PhysicalSystem(cfg, te))
    s_none = systems["none"][0]
    sysinfo = {"p_feed_dbm": float(10 * np.log10(s_none.p_feed_w) + 30), "backoff_db": s_none.backoff_db,
               "voxel_d": float(s_none.task.voxel_d), "object_model": s_none.object_model,
               "material": str(s_none.task.material), "condition_internal": s_none.label,
               "loc_cell": None if kind != "loc3bit" else float(s_none.task.cell)}
    rows = []
    for trainer in job["trainers"]:
        base = None
        for jm in jitters:
            s_tr, s_te = systems[jm]
            rng_train = common.seed_rng(EXP, "train", kind, seed, trainer, jm)
            rng_eval = common.seed_rng(EXP, "eval", kind, seed, trainer, jm)
            tr = _train(s_tr, trainer, rng_train, job)
            params = tr["params"]
            thr = train_thresholds(s_tr, params)
            ev = evaluate_on(s_te, params, thr, rng_eval, n_noise)
            row = {"task": kind, "cond": cond, "seed": seed, "jitter": jm, "trainer": trainer, "scenario": scenario,
                   "n_train": s_tr.B, "n_test": s_te.B, "heldout": jm != "none", **sysinfo,
                   **_pick(tr["train_eval"], "train_"), **_pick(ev, "test_"), **tr["info"],
                   "thresholds_db": None if thr is None else [float(v) for v in thr],
                   "test_sat_frac": saturation_fraction(s_te, params),      # 审查 R-C3：真正饱和的测试样本比例
                   "params": [int(v) for v in params],
                   "test_power_db_range": [float(v) for v in (s_te.task.power_db().min(), s_te.task.power_db().max())],
                   "test_dist_shift_range": [float(min(s.meta.get("dist_shift", 0.0) for s in s_te.task.samples)),
                                             float(max(s.meta.get("dist_shift", 0.0) for s in s_te.task.samples))]}
            if jm == "none":
                base = {"params": params, "thr": thr, "acc": ev.get("acc_clean"), "acc_noisy": ev.get("acc_noisy")}
                row.update(acc_none=base["acc"], inv_loss=0.0, inv_loss_noisy=0.0, zs_test_acc_clean=None,
                           zs_test_acc_noisy=None, zs_inv_loss=None)
            else:
                zs = evaluate_on(s_te, base["params"], base["thr"], common.seed_rng(EXP, "zs", kind, seed, trainer, jm),
                                 n_noise)
                row.update(acc_none=base["acc"],
                           inv_loss=_diff(base["acc"], ev.get("acc_clean")),
                           inv_loss_noisy=_diff(base["acc_noisy"], ev.get("acc_noisy")),
                           zs_test_acc_clean=zs.get("acc_clean"), zs_test_acc_noisy=zs.get("acc_noisy"),
                           zs_inv_loss=_diff(base["acc"], zs.get("acc_clean")),
                           zs_test_oscillating_run=zs.get("oscillating_run"),
                           zs_test_pa_in_rel_p1db_db_mean=zs.get("pa_in_rel_p1db_db_mean"),
                           zs_test_sat_frac=saturation_fraction(s_te, base["params"]))
            rows.append(row)
    return {"rows": rows, "job_time_s": time.perf_counter() - t_job}


def _diff(a, b):
    return None if a is None or b is None else float(a) - float(b)


# ============================================================ 网格
def build_config(args) -> dict:
    quick = bool(args.quick)
    seeds = args.seeds if args.seeds is not None else (3 if quick else 5)
    cfg = {
        "exp": EXP, "quick": quick, "seeds": int(seeds),
        "tasks": list(args.tasks or TASKS),
        "conditions": list(args.conditions or ALL_CONDITIONS),
        "jitters": list(args.jitters or JITTERS),
        "trainers": list(args.trainers or DEFAULT_TRAINERS),
        "oracle_objectives": dict(ORACLE_OBJECTIVE),
        "n_iter": int(args.n_iter or (1000 if quick else 3000)),
        "oracle_restarts": int(args.oracle_restarts or (4 if quick else 8)),
        "oracle_sweeps": 30,
        "n_rep_train": {"letter": 4 if quick else 6, "loc3bit": 2 if quick else 4},
        "n_rep_test": {"letter": 8 if quick else 16, "loc3bit": 4 if quick else 8},
        "n_noise": 10 if quick else 20,
        "preset": "L90", "object_mode": "full", "loc_cell_lam": args.loc_cell_lam,
        "power_jitter_db": POWER_JITTER_DB, "dist_jitter_lam": DIST_JITTER_LAM,
        "scenario": getattr(args, "scenario", None) or "baseline",
    }
    from ..system import SCENARIOS

    if cfg["scenario"] not in SCENARIOS:
        raise ValueError(f"未知场景 {cfg['scenario']!r}，可选 {list(SCENARIOS)}")
    for c in cfg["conditions"]:
        parse_condition(c)
    return cfg


def make_jobs(cfg: dict) -> list[dict]:
    jobs = []
    for kind in cfg["tasks"]:
        for cond in cfg["conditions"]:
            for seed in range(cfg["seeds"]):
                jobs.append({"task": kind, "cond": cond, "seed": seed, "jitters": cfg["jitters"],
                             "trainers": cfg["trainers"], "n_iter": cfg["n_iter"],
                             "oracle_restarts": cfg["oracle_restarts"], "oracle_sweeps": cfg["oracle_sweeps"],
                             "n_rep_train": cfg["n_rep_train"][kind], "n_rep_test": cfg["n_rep_test"][kind],
                             "n_noise": cfg["n_noise"], "preset": cfg["preset"], "object_mode": cfg["object_mode"],
                             "loc_cell_lam": cfg["loc_cell_lam"], "scenario": cfg.get("scenario", "baseline")})
    # 慢的 job（Rapp 中继、定位样本多）先提交，改善负载均衡
    jobs.sort(key=lambda j: (j["cond"].startswith("M1"), j["task"] != "loc3bit"))
    return jobs


def flatten(job_rows: list[dict]) -> list[dict]:
    """job 结果 → 每 (抖动, 训练器) 一行；出错的 job 展开成对应格点的 error 行（aggregate 计入 n_errors）。"""
    out = []
    for r in job_rows:
        if "error" in r:
            job = r.get("job", {})
            jit = list(job.get("jitters", JITTERS))
            if "none" not in jit:
                jit = ["none"] + jit
            for tr in job.get("trainers", DEFAULT_TRAINERS):
                for jm in jit:
                    out.append({"task": job.get("task"), "cond": job.get("cond"), "seed": job.get("seed"),
                                "jitter": jm, "trainer": tr, "error": r["error"]})
            continue
        for row in r["rows"]:
            out.append(dict(row, job_time_s=r.get("job_time_s"), elapsed_s=r.get("elapsed_s")))
    return out


# ============================================================ 汇总
METRICS = ["test_acc_clean", "test_acc_noisy", "inv_loss", "inv_loss_noisy", "zs_test_acc_clean", "zs_inv_loss",
           "train_acc_clean", "test_margin_db_unflagged", "test_oscillating_run", "test_sat_frac", "zs_test_sat_frac",
           "test_pa_in_rel_p1db_db_min",
           "test_pa_in_rel_p1db_db_mean", "test_pa_in_rel_p1db_db_max", "test_pa_in_open_rel_p1db_db_mean",
           "test_null_depth_db", "test_loop_gain_ss_max", "flagged_meas_frac", "p_feed_dbm", "voxel_d", "train_time_s"]
KEYS = ["task", "trainer", "cond", "jitter"]


def paired_rows(rows: list[dict]) -> list[dict]:
    """按种子配对（同任务/训练器/抖动/种子）：Δ = 条件 − M1@M3 的 test_acc_clean、inv_loss、zs_inv_loss。"""
    idx = {}
    for r in rows:
        if "error" not in r:
            idx[(r["task"], r["trainer"], r["jitter"], r["seed"], r["cond"])] = r
    out = []
    for (task, tr, jm, seed, cond), r in idx.items():
        if cond == REF_COND:
            continue
        ref = idx.get((task, tr, jm, seed, REF_COND))
        if ref is None:
            continue
        out.append({"task": task, "trainer": tr, "jitter": jm, "cond": cond, "seed": seed,
                    "d_test_acc": _diff(r.get("test_acc_clean"), ref.get("test_acc_clean")),
                    "d_inv_loss": _diff(r.get("inv_loss"), ref.get("inv_loss")),
                    "d_zs_inv_loss": _diff(r.get("zs_inv_loss"), ref.get("zs_inv_loss"))})
    return out


def _ci(t: dict, m: str, f: str = ".2f") -> str:
    mean, half, n = t.get(f"{m}_mean"), t.get(f"{m}_ci95"), t.get(f"{m}_n", 0)
    if not n or mean is None or not math.isfinite(mean):
        return "—"
    if half is None or not math.isfinite(half):
        return format(mean, f)
    return f"{format(mean, f)} ± {format(half, f)}"


def _order(vals, ref):
    return sorted(set(vals), key=lambda v: (ref.index(v) if v in ref else len(ref), str(v)))


def scenario_text(name: str) -> str:
    """命名场景的一句话说明（summary 头部）。"""
    from ..system import SCENARIOS

    if name == "baseline":
        return "默认 L90 场景，ROI 0.5 m、馈源→探头视距保留"
    over = {k: v for k, v in SCENARIOS.get(name, {}).items() if k != "preset"}
    return f"L90 + {over}"


def summary_markdown(table: list[dict], ptab: list[dict], cfg: dict, rows: list[dict]) -> str:
    seeds = cfg["seeds"]
    lines = [f"# 实验 C：饱和归一化的利弊（RQ4）{'— quick' if cfg['quick'] else ''}", ""]
    lines += [f"- 预设 {cfg['preset']}、场景 {cfg.get('scenario', 'baseline')}（system.SCENARIOS，INTEGRATION S："
              f"{scenario_text(cfg.get('scenario', 'baseline'))}），物体模式 {cfg['object_mode']}；条件 {', '.join(cfg['conditions'])}；"
              f"训练器 {', '.join(cfg['trainers'])}（oracle = 1-bit 模型 oracle、**准确率目标**（最大化训练集判对数，"
              f"审查 R-C1），{cfg['oracle_restarts']} 次重启；oracle_loss = 同一 oracle 的 softplus 损失目标（次要：不可解 / "
              f"难任务上它不是最好可达准确率）；dspsa = {cfg['n_iter']} 次迭代、可测目标）。",
              f"- 每格 {seeds} 个种子" + ("（quick 模式少于 5 个种子）" if seeds < 5 else "") +
              f"；抖动：功率 ±{POWER_JITTER_DB:g} dB（meta power_db）、距离 ±{DIST_JITTER_LAM:g}λ（沿物体平面法向 +x）。",
              f"- 训练集样本数：letter {cfg['n_rep_train']['letter']}×3、loc3bit {cfg['n_rep_train']['loc3bit']}×8；"
              f"**held-out 测试集是独立重新抽取的抖动样本**（letter {cfg['n_rep_test']['letter']}×3、"
              f"loc3bit {cfg['n_rep_test']['loc3bit']}×8）；none 模式训练集 = 测试集（无抖动，样本确定）。",
              "- 不变性损失 inv = acc(none) − acc(抖动 held-out)；零样本 zs_inv = acc(none) − acc(none 训练的码本在抖动测试集)。"
              "定位阈值由训练集读数确定后固定（不用测试标签）。准确率为无噪声读数（被标记自激的样本算错）。",
              "- 工作点列：测试集每样本闭环 PA 输入相对 P1dB（dB）的 min/mean/max（种子平均）；**sat = 测试集闭环 PA 输入 ≥ P1dB "
              f"的样本比例**（审查 R-C3：训练把码本推向零陷，名义 M4 训练后常在 P1dB 以下；sat < {SAT_MIN:g} 的行不检验 AGC 假设）；"
              "null = 空场景零陷深度；P_feed、体素尺寸见右侧列。", ""]
    tasks = _order([t["task"] for t in table], list(TASKS))
    trs = _order([t["trainer"] for t in table], list(TRAINERS))
    conds = _order([t["cond"] for t in table], list(ALL_CONDITIONS))
    jits = _order([t["jitter"] for t in table], list(JITTERS))
    idx = {(t["task"], t["trainer"], t["cond"], t["jitter"]): t for t in table}
    for task in tasks:
        for tr in trs:
            lines += [f"## {task} / {tr}", "",
                      "| cond | jitter | n | acc_test | acc_noisy | inv_loss | zs_acc | zs_inv_loss | acc_train | "
                      "osc | PA in min/mean/max (dB re P1dB) | sat | null (dB) | P_feed (dBm) | voxel_d (mm) |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in conds:
                for jm in jits:
                    t = idx.get((task, tr, c, jm))
                    if t is None:
                        continue
                    pa = "/".join(_ci(t, f"test_pa_in_rel_p1db_db_{k}", ".1f").split(" ±")[0]
                                  for k in ("min", "mean", "max"))
                    vd = t.get("voxel_d_mean")
                    lines.append(
                        f"| {c} | {jm} | {t['n_runs']}" + (f" (err {t['n_errors']})" if t["n_errors"] else "") +
                        f" | {_ci(t, 'test_acc_clean')} | {_ci(t, 'test_acc_noisy')} | {_ci(t, 'inv_loss')} | "
                        f"{_ci(t, 'zs_test_acc_clean')} | {_ci(t, 'zs_inv_loss')} | {_ci(t, 'train_acc_clean')} | "
                        f"{_ci(t, 'test_oscillating_run')} | {pa} | {_ci(t, 'test_sat_frac').split(' ±')[0]} | "
                        f"{_ci(t, 'test_null_depth_db', '.1f')} | {_ci(t, 'p_feed_dbm', '.1f')} | "
                        f"{'—' if vd is None or not math.isfinite(vd) else format(1e3 * vd, '.2f')} |")
            lines.append("")
    # 配对差分（预测检验）
    pt = common.aggregate(ptab, ["task", "trainer", "cond", "jitter"], ["d_test_acc", "d_inv_loss", "d_zs_inv_loss"])
    lines += ["## 配对差分（相对 M1@M3，同种子）与预测检验", "",
              "预测（方案 §6.2）：字母上饱和中继更稳 → d_inv_loss < 0；定位上饱和中继变差 → none / dist 下 d_test_acc < 0"
              "（power / both 是 AGC 检验，见判读）。", "",
              "| task | trainer | cond | jitter | n | d_test_acc | d_inv_loss | d_zs_inv_loss |", "|---|---|---|---|---|---|---|---|"]
    for t in pt:
        if t["jitter"] == "none" and t["cond"] not in SAT_CONDS:
            continue
        lines.append(f"| {t['task']} | {t['trainer']} | {t['cond']} | {t['jitter']} | {t['n_runs']} | "
                     f"{_ci(t, 'd_test_acc')} | {_ci(t, 'd_inv_loss')} | {_ci(t, 'd_zs_inv_loss')} |")
    lines += ["", verdict_text(pt, table), "", sanity_text(rows, ptab), ""]
    n_err = sum(1 for r in rows if "error" in r)
    if n_err:
        lines.append(f"{n_err} 行出错（见 runs.jsonl 的 error 字段）。")
    return "\n".join(lines) + "\n"


def sanity_text(rows: list[dict], ptab: list[dict]) -> str:
    """线性对照的自检：(1) M1@M4 与 M1@M3 只差整体功率 → dB 判决（argmax / 自动阈值 / softmax）逐 run 相同；
    (2) 线性中继 + 字母 argmax 对逐样本整体功率缩放不变 → power 抖动的零样本不变性损失应恰为 0。"""
    out = ["### 自检（线性对照）", ""]
    d = [abs(r["d_test_acc"]) for r in ptab if r["cond"] == "M1@M4" and r.get("d_test_acc") is not None]
    if d:
        out.append(f"- max |acc(M1@M4) − acc(M1@M3)| = {max(d):.3g}（{len(d)} 对；应为 0：线性系统只差整体功率）"
                   + (" → PASS" if max(d) == 0 else " → 偏离（检查噪声或标记）"))
    z = [abs(r["zs_inv_loss"]) for r in rows if "error" not in r and r["task"] == "letter" and r["jitter"] == "power"
         and r["cond"].startswith("M1") and r.get("zs_inv_loss") is not None]
    if z:
        out.append(f"- 字母 / 线性中继 / 仅功率抖动：max |zs_inv_loss| = {max(z):.3g}（{len(z)} 个 run；应为 0："
                   "argmax 对整体功率缩放不变）" + (" → PASS" if max(z) == 0 else " → 偏离"))
    return "\n".join(out)


CHANCE = {"letter": 1.0 / 3.0, "loc3bit": 1.0 / 8.0}
FLOOR_MARGIN = 0.15                     # 两条件的测试准确率都 < chance + 0.15 → 地板效应，预测无法检验
# 审查 R-C2：各任务的预测检验 (抖动, 差分量, 预测符号, 说明)
VERDICT_SPECS = {
    "letter": [("both", "d_inv_loss", -1, "饱和更不变（方案 §6.2 的主检验）"),
               ("power", "d_inv_loss", -1, "仅功率抖动：AGC 的直接检验"),
               ("dist", "d_inv_loss", -1, "仅距离抖动")],
    "loc3bit": [("none", "d_test_acc", -1, "无抖动：幅度信息被压掉 → 定位更差（主检验）"),
                ("dist", "d_test_acc", -1, "距离抖动：定位更差"),
                ("dist", "d_inv_loss", +1, "距离抖动：不变性损失更大"),
                ("power", "d_test_acc", +1, "AGC 检验：固定 dBm 阈值下线性中继按构造失败，饱和可能帮助"),
                ("both", "d_test_acc", +1, "AGC 检验（同上）")],
}


def _sig_text(t: dict, m: str) -> tuple[str, int]:
    """(观测描述, 符号)：符号 = +1 / −1（95% 区间整体在 0 的一侧）或 0（含 0 / 无区间）。"""
    mean, lo, hi = t.get(f"{m}_mean"), t.get(f"{m}_lo"), t.get(f"{m}_hi")
    if mean is None or not math.isfinite(mean):
        return "无数据", 0
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return f"均值 {mean:+.2f}（n < 2，无区间）", 0
    if lo <= 0 <= hi:
        return f"均值 {mean:+.2f}，95% 区间 [{lo:+.2f}, {hi:+.2f}] 含 0 → 不显著", 0
    sgn = 1 if lo > 0 else -1
    return f"均值 {mean:+.2f}，95% 区间 [{lo:+.2f}, {hi:+.2f}] → 显著{'> 0' if sgn > 0 else '< 0'}", sgn


def verdict_text(pt: list[dict], table: list[dict] | None = None) -> str:
    """预测检验（审查 R-C2 / R-C3）：对每个饱和条件（SAT_CONDS）与训练器，按 VERDICT_SPECS 的 (抖动, 差分量) 给出
    配对差分（相对 M1@M3）的判读；同时注明 (1) 地板效应（两条件准确率都 < chance + FLOOR_MARGIN），(2) 该条件测试集
    是否真的饱和（sat_frac < SAT_MIN → 近线性，不检验 AGC 假设）。"""
    out = ["### 判读（饱和条件 − M1@M3，同种子配对）", "",
           "定位：±6 dB 功率抖动对固定 dBm 阈值的线性中继按构造是毁灭性的（多数探头 8 格间对比度 < 1 dB，抖动跨 12 dB），"
           "所以\"饱和中继定位更差\"只用 none / dist 检验；power / both 是 AGC 检验（预测方向相反：饱和可能帮助）。"
           f"sat = 测试集闭环 PA 输入 ≥ P1dB 的样本比例；sat < {SAT_MIN:g} 的条件实际近线性，不检验 AGC 假设。", ""]
    tab = {(t["task"], t["trainer"], t["cond"], t["jitter"]): t for t in (table or [])}
    pidx = {(t["task"], t["trainer"], t["cond"], t["jitter"]): t for t in pt}
    tasks = _order([t["task"] for t in pt], list(TASKS))
    trs = _order([t["trainer"] for t in pt], list(TRAINERS))
    for task in tasks:
        for tr in trs:
            for cond in SAT_CONDS:
                for jm, metric, want_sign, what in VERDICT_SPECS.get(task, []):
                    t = pidx.get((task, tr, cond, jm))
                    if t is None:
                        continue
                    obs, sgn = _sig_text(t, metric)
                    verdict = {want_sign: "（支持预测）", -want_sign: "（与预测相反）"}.get(sgn, "")
                    a_s = (tab.get((task, tr, cond, jm)) or {}).get("test_acc_clean_mean")
                    a_r = (tab.get((task, tr, REF_COND, jm)) or {}).get("test_acc_clean_mean")
                    sat = (tab.get((task, tr, cond, jm)) or {}).get("test_sat_frac_mean")
                    notes = []
                    if a_s is not None and a_r is not None and math.isfinite(a_s) and math.isfinite(a_r):
                        ch = CHANCE.get(task, 0.0)
                        notes.append(f"acc {cond} {a_s:.2f} / {REF_COND} {a_r:.2f}，chance {ch:.2f}")
                        if max(a_s, a_r) < ch + FLOOR_MARGIN:
                            notes.append("两者都接近 chance → 地板效应，无法检验")
                    if sat is not None and math.isfinite(sat):
                        notes.append(f"sat {sat:.2f}" + ("（未饱和 → 不检验 AGC）" if sat < SAT_MIN else ""))
                    want = f"{metric} {'<' if want_sign < 0 else '>'} 0"
                    out.append(f"- {task} / {tr} / {cond} / {jm}：{what}，预测 {want}；观测 {obs}{verdict}，"
                               f"n = {t['n_runs']}" + (f"；{'；'.join(notes)}" if notes else "") + "。")
    return "\n".join(out)


def plot_accuracy(table: list[dict], outdir: str) -> list[str]:
    import matplotlib.pyplot as plt

    paths = []
    tasks = _order([t["task"] for t in table], list(TASKS))
    trs = _order([t["trainer"] for t in table], list(TRAINERS))
    conds = _order([t["cond"] for t in table], list(ALL_CONDITIONS))
    jits = _order([t["jitter"] for t in table], list(JITTERS))
    idx = {(t["task"], t["trainer"], t["cond"], t["jitter"]): t for t in table}
    fig, axes = plt.subplots(len(tasks), len(trs), figsize=(5.2 * len(trs), 3.6 * len(tasks)), squeeze=False)
    w = 0.8 / max(1, len(jits))
    for i, task in enumerate(tasks):
        for j, tr in enumerate(trs):
            ax = axes[i][j]
            for k, jm in enumerate(jits):
                ys, es = [], []
                for c in conds:
                    t = idx.get((task, tr, c, jm), {})
                    ys.append(t.get("test_acc_clean_mean", np.nan))
                    e = t.get("test_acc_clean_ci95", np.nan)
                    es.append(0.0 if e is None or not np.isfinite(e) else e)
                x = np.arange(len(conds)) + (k - (len(jits) - 1) / 2) * w
                ax.bar(x, ys, w, yerr=es, capsize=2, label=jm)
            chance = 1 / 3 if task == "letter" else 1 / 8
            ax.axhline(chance, color="k", ls=":", lw=1)
            ax.set_xticks(np.arange(len(conds)))
            ax.set_xticklabels(conds)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{task} / {TRAINER_LABEL.get(tr, tr)}：held-out 测试准确率", fontsize=9)
            ax.set_ylabel("accuracy (clean readings)")
            ax.legend(title="jitter", fontsize=8)
    paths.append(common.savefig(fig, os.path.join(outdir, "accuracy_by_jitter.png")))
    # 不变性损失
    fig, axes = plt.subplots(len(tasks), len(trs), figsize=(5.2 * len(trs), 3.6 * len(tasks)), squeeze=False)
    jj = [j for j in jits if j != "none"]
    w = 0.8 / max(1, 2 * len(jj))
    for i, task in enumerate(tasks):
        for j, tr in enumerate(trs):
            ax = axes[i][j]
            for k, jm in enumerate(jj):
                for q, (m, hatch) in enumerate((("inv_loss", None), ("zs_inv_loss", "//"))):
                    ys, es = [], []
                    for c in conds:
                        t = idx.get((task, tr, c, jm), {})
                        ys.append(t.get(f"{m}_mean", np.nan))
                        e = t.get(f"{m}_ci95", np.nan)
                        es.append(0.0 if e is None or not np.isfinite(e) else e)
                    x = np.arange(len(conds)) + (2 * k + q - (2 * len(jj) - 1) / 2) * w
                    ax.bar(x, ys, w, yerr=es, capsize=2, hatch=hatch, color=f"C{k}",
                           alpha=0.9 if q == 0 else 0.5, label=f"{jm} ({'retrained' if q == 0 else 'zero-shot'})")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(np.arange(len(conds)))
            ax.set_xticklabels(conds)
            ax.set_title(f"{task} / {TRAINER_LABEL.get(tr, tr)}：不变性损失", fontsize=9)
            ax.set_ylabel("acc(none) - acc(jitter)")
            ax.legend(fontsize=7)
    paths.append(common.savefig(fig, os.path.join(outdir, "invariance_loss.png")))
    return paths


def summarize(rows: list[dict], outdir: str, cfg: dict) -> dict:
    table = common.aggregate(rows, KEYS, METRICS)
    ptab = paired_rows(rows)
    paths = common.write_outputs(outdir, rows, table, config=cfg, title=None)
    common.save_csv(common.aggregate(ptab, ["task", "trainer", "cond", "jitter"],
                                     ["d_test_acc", "d_inv_loss", "d_zs_inv_loss"]),
                    os.path.join(outdir, "paired.csv"))
    md = summary_markdown(table, ptab, cfg, rows)
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(md)
    try:
        paths["figs"] = plot_accuracy(table, outdir)
    except Exception as e:          # noqa: BLE001 —— 作图失败不影响数据输出
        paths["fig_error"] = repr(e)
    return {"table": table, "paired": ptab, "paths": paths, "markdown": md}


# ============================================================ 入口
def run(cfg: dict, workers: int = 1, out: str | None = None, progress: bool = False) -> dict:
    outdir = common.make_outdir(EXP, out)
    t0 = time.perf_counter()
    jobs = make_jobs(cfg)
    job_rows = common.run_jobs(run_c_job, jobs, workers=workers, progress=progress)
    rows = flatten(job_rows)
    cfg = dict(cfg, n_jobs=len(jobs), wall_time_s=time.perf_counter() - t0, workers=int(workers))
    res = summarize(rows, outdir, cfg)
    res.update(outdir=outdir, rows=rows)
    return res


def parse_args(argv=None):
    p = common.base_argparser("实验 C：饱和归一化的利弊（RQ4，方案 §6.2）")
    p.set_defaults(seeds=None)
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=None)
    p.add_argument("--conditions", nargs="+", default=None, help="条件标签（默认 M1@M3 M3 M4 Rapp@+20 M1@M4；Rapp@<dB> = Rapp PA + 指定回退）")
    p.add_argument("--jitters", nargs="+", choices=list(JITTERS), default=None)
    p.add_argument("--trainers", nargs="+", choices=list(TRAINERS), default=None,
                   help="训练器（默认 oracle dspsa；oracle = 准确率目标，oracle_loss = softplus 损失目标）")
    p.add_argument("--n-iter", type=int, default=None, help="DSPSA 迭代数（quick 1000 / full 3000）")
    p.add_argument("--oracle-restarts", type=int, default=None, help="1-bit oracle 重启数（quick 4 / full 8）")
    p.add_argument("--loc-cell-lam", type=float, default=None, help="定位格宽（单位 λ；默认任务默认 λ/2）")
    p.add_argument("--scenario", default=None, choices=["baseline", "informative", "informative_probe"],
                   help="命名场景（system.SCENARIOS / apply_scenario，INTEGRATION S）；默认 baseline")
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
