"""实验 A：奇偶校验阶梯（方案 §6.1 主实验，RQ1 + RQ2；INTEGRATION G "A"）。

网格：n × 条件 × object_mode × material × N_r × spacing × seed。每个 run（一个格点的一个种子）：
  1. DSPSA（train.run_dspsa 默认变体，有噪声测量；INTEGRATION T：1-bit 默认 sign_acc = 判对优先可测目标 + sign-SPSA，
     输出即 pocket 码本）→ "dspsa"；旧变体（--dspsa-variant subset）另有 "dspsa_pocket"（训练中测到的可测损失最小的码本，
     审查 S1-0 / Q-2：subset 的最终迭代常停在 y1≈y2 的 τ·ln2 平台）；
  2. 1-bit 模型 oracle "oracle"：run_oracle_bits(objective="accuracy")（INTEGRATION G-A / B3："最好可达准确率"）
     + polish_min_margin 第二阶段（判对数不降、最小带符号裕量最大化，审查 Q-1：accuracy 目标的平局项可给出 0.001 dB 的全对）；
     诊断 "oracle_loss"：默认 softplus 损失 oracle（审查 S0-2：不可解任务上 ≈ τlog2 − mean(s)/2，acc_clean 可低于随机，
     **不是**最好可达准确率；与 oracle 用同一随机重启）；门 G1 格点另跑 "oracle_margin"（最大化最小裕量，> 0 ⇔ 全对）；
  3. 可选连续相位 oracle "cont"（train.run_oracle_continuous，softplus 模型目标 → 其 acc_clean 同样不是准确率上界）；
     --cont-plus（informative 场景默认开，INTEGRATION S）：τ = 0.1 dB 的随机重启 + 设计初值（接收码本零陷馈源、相干接收物体，
     方案推论 3）重启 + soft-min 裕量精修（40 dB 动态范围受限读数），按 (判对数, 最小裕量) 取最好者；
  4. nonlinearity_profile：5 个随机码本的平均（方案 §7 "在随机（未训练）配置上量"）+ 各训练后码本。
"solved" = 全对且无噪声最小裕量 > 3σ_det（VNA 0.006 dB；审查 Q-1）；"robust" = 全对且最小裕量 ≥ 0.5 dB（INTEGRATION S：
≫ VNA 噪声、≈ 5σ 的 0.1 dB 检波器级噪声），非线性收益主表用 robust，solved 并列；全对数与裕量并列报告。
场景（--scenario，system.SCENARIOS，INTEGRATION S）：quick 默认 "informative"（L90 + ROI 0.35 m + 遮挡馈源→探头视距，
  条件 M0 / M1 / M1b@+25 / Rapp@+20 / +25 / +30 / M3 / M4，full 耦合 + 门 G1 的 born 格点）；"baseline" = 原默认场景
  （偏置主导，方案推论 3），full 模式默认仍为 baseline。
DSPSA 是否"训练到位"按实验 E 判据（INTEGRATION G-E）：(L0 − L) ≥ 0.9 (L0 − L*)，L0 = 初始码本、L* = 1-bit oracle 的
可测目标（loss_meas_clean，取 oracle / oracle_loss 较小者）；格点内 ≥ 80% 的 run 达标才报告基于 DSPSA 的非线性收益。
条件标签（INTEGRATION B "同功率"配对）：
  M0、M2、M3、M4      DESIGN §7.1（M0 与 M3 同源功率）；
  M1@Mk / M1b@Mk      小信号增益 / Bussgang 等效复增益的线性中继，源功率 = Mk 的源功率（backoff 同 Mk）；
  Rapp@<dB>           = M3@<dB>（Rapp PA、显式回退；与实验 C / E 同一写法，内部条件 M4）；
  M3@<dB>、M1@<dB>    显式回退（审查 Q-3 功率扫描，例如 M3@+20 / M1@+20 / M1b@+20）：标定后码本把背景零陷，intra/full 的
                      M3/M4 实际闭环工作点比名义回退低 2–16 dB、物体驱动低 20–37 dB（方案推论 3），名义 M4 并不压缩物体项；
                      --power-sweep 加一个 n ≥ 3 × intra/full 的回退扫描块，让 PA 真正压缩物体相关部分。
  后缀 ":noTx"        objects_see_relay_tx=False（理想化对照，审查 INT-PHYS-1）：物体不被中继发射场照射，
                      K、C 与 x 无关 → M1 + Born 下 y 对 x 二次，3 位奇偶必不可解（门 G1 自检的线性中继扩展）。
门 G1（方案 §6.1、§8）：M0 + Born 的探头场对 x 仿射 → y1−y2 二次 → 3 位奇偶不可解（Minsky–Papert）。
若任何训练器在 M0+Born（或 M1:noTx+Born）上 3 位全对，就是 bug，summary.md 的自检段落会标 FAIL。
注意（system.py 模块 docstring）：默认 objects_see_relay_tx=True 时 M1 + Born 的 u = (I − G K(x))⁻¹u_in(x) 已是 x 的
有理函数（方案推论 4），M1 + Born 解出 3 位奇偶不违背理论，只是"环路/再照射"非线性，不是 PA 非线性。

非线性度分解（方案 §7）：物体贡献 = W≥3(e)（e = 接收单元馈源列场，M0 为无源单元激励场）；PA 贡献 = W≥3(g) − W≥3(u)
（PA 映射本身加入的高阶权重；M1 恒为 0）；另报 pa_excess_y = W≥3(y) − W≥3(y_linref)（同码本、增益匹配线性 PA）。
非线性收益（方案 §7 主结论）= robust / solved / acc(Rapp 条件) − 同一指标(M1 同功率)，也对 M1b（同回退）；非 baseline 场景下
没有同回退的 M1 时与任一 M1 配对（线性中继的无噪声判决量与源功率无关）。按种子配对（同种子 = 同初始码本与扰动序列，公共随机数），t 分布 95% 区间。自激 run（最终码本任一样本被标记）不参与精度均值（方案 §4.5），单独报比例。

用法：python -m ris_nlsim.experiments.exp_a_parity [--quick] [--workers N] [--seeds S] [--out DIR] [网格选项]
"""
from __future__ import annotations

import collections
import datetime as _dt
import math
import os
import sys
import time

import numpy as np

from . import common

EXP = "exp_a"
TRAINERS = ("dspsa", "dspsa_pocket", "oracle", "oracle_loss", "oracle_margin", "cont")
TRAINER_NAMES = {"dspsa": "DSPSA", "dspsa_pocket": "DSPSA pocket", "oracle": "1-bit oracle",
                 "oracle_loss": "loss-oracle", "oracle_margin": "margin-oracle", "cont": "连续 oracle", "best": "best"}
DSPSA_TRAINERS = ("dspsa", "dspsa_pocket")
CAPABILITY_TRAINERS = ("oracle", "cont", "best")          # 主表："这个结构能做到多好"
LADDER_TRAINERS = ("dspsa", "dspsa_pocket", "oracle", "cont", "best")
SOLVED_SIGMA_MULT = 3.0              # solved：全对且最小裕量 > 3σ_det（审查 Q-1）
ROBUST_MARGIN_DB = 0.5               # robust：全对且最小裕量 ≥ 0.5 dB（INTEGRATION S；非线性收益主表）
RAPP_PREFIX = "Rapp@"                # "Rapp@+25" = Rapp PA + 显式回退（实验 C / E 同一写法）
INFORMATIVE_QUICK_CONDITIONS = ("M0", "M1", "M1b@+25", "Rapp@+20", "Rapp@+25", "Rapp@+30", "M3", "M4")
CONT_PLUS_TAU_DB = 0.1               # cont-plus：连续 oracle 的 softplus 温度（探索者 τ_cont）
CONT_PLUS_INIT_RESTARTS = 6          # cont-plus：设计初值重启次数（每次 cont_steps 步；按 margin_key 取最好）
DR_DB = 40.0                         # cont-plus 选择 / 精修用的读数动态范围（相对本组最强读数）
REACH_FRAC = 0.9                     # 实验 E 判据：(L0 − L) ≥ 0.9 (L0 − L*)（INTEGRATION G-E）
TRAINED_RUN_FRAC = 0.8               # 格点内达标 run 比例 ≥ 此值才报告基于 DSPSA 的收益（审查 Q-2）
PLATEAU_FRAC = 0.9                   # 平台诊断：loss_meas_clean ≥ 0.9 × chance（τ·ln2）
POLISH_SWEEPS = 10
ALL_CONDITIONS = ("M0", "M1@M2", "M1@M3", "M1@M4", "M1b@M3", "M1b@M4", "M2", "M3", "M4")
QUICK_CONDITIONS = ("M0", "M1@M3", "M1@M4", "M1b@M3", "M1b@M4", "M3", "M4")
RAPP_BASES = ("M2", "M3", "M4")
POWER_SWEEP_FULL = (20.0, 30.0)      # 审查 Q-3：功率扫描块的回退 dB（M3@+20 / M1@+20 …）
POWER_SWEEP_QUICK = (30.0,)
ALL_MODES = ("born", "intra", "full")
ALL_MATERIALS = ("eps1.5", "eps4", "eps10", "water")
NR_LEVELS = (1, 2, 4)
DEFAULT_SPACING_LAM = 0.5            # 方案 §6.1：槽间距 λ/2（多次散射在间距 < λ/2 时才可能显著）
DIAMETER_LAM = 1.0 / 3.0             # 方案 §6.1：散射体直径约 λ/3
NOTX = ":noTx"
W_FLOOR = 1e-8                       # W≥3 对数坐标的下限（更小的值画在下限上）


# ============================================================ 条件标签
def parse_condition(label: str) -> dict:
    """条件标签 → SystemConfig 字段。"M1@M4" → M1、backoff = M4 的回退（+10 dB）；"M1b@M3" → M1b、0 dB；
    "M1@-3" → 显式回退 −3 dB；"M3@+20"（M2/M3/M4 只接受数值，审查 Q-3）→ Rapp PA、显式回退 +20 dB
    （与 "M1@+20" / "M1b@+20" 配对）；后缀 ":noTx" → objects_see_relay_tx=False。"""
    from ..system import CONDITION_BACKOFF_DB, CONDITIONS

    lab = str(label).strip()
    out = {"objects_see_relay_tx": True}
    if lab.endswith(NOTX):
        out["objects_see_relay_tx"] = False
        lab = lab[: -len(NOTX)]
    if lab.startswith(RAPP_PREFIX):             # "Rapp@+25"：Rapp PA（M2–M4 同一 PA），显式回退（实验 C / E 同一约定）
        try:
            backoff = float(lab[len(RAPP_PREFIX):])
        except ValueError as e:
            raise ValueError(f"无法解析条件 {label!r}（Rapp@ 后只能是回退 dB 数值）") from e
        if not math.isfinite(backoff):
            raise ValueError(f"回退必须有限：{label!r}")
        out.update(condition="M4", backoff_db=backoff)
        return out
    if "@" in lab:
        base, ref = lab.split("@", 1)
        if base in ("M1", "M1b") and ref in CONDITION_BACKOFF_DB:
            backoff = CONDITION_BACKOFF_DB[ref]
        elif base in ("M1", "M1b") + RAPP_BASES:
            try:
                backoff = float(ref)          # Rapp 条件只接受数值（"M3@M4" 无意义 → ValueError）
            except ValueError as e:
                raise ValueError(f"无法解析条件 {label!r}（M2/M3/M4 的 '@' 后只能是回退 dB 数值）") from e
            if not math.isfinite(backoff):
                raise ValueError(f"回退必须有限：{label!r}")
        else:
            raise ValueError(f"只有 M1 / M1b / M2 / M3 / M4 可以写 '@'：{label!r}")
        out.update(condition=base, backoff_db=float(backoff))
        return out
    if lab not in CONDITIONS:
        raise ValueError(f"未知条件 {label!r}")
    out["condition"] = lab
    return out


def condition_backoff_db(label: str) -> float:
    """标签的有效回退 (dB)：显式 '@' 优先，否则条件表（M0 / M1 为 0，与 M3 同功率）。"""
    from ..system import CONDITION_BACKOFF_DB

    p = parse_condition(label)
    return float(p["backoff_db"]) if "backoff_db" in p else float(CONDITION_BACKOFF_DB[p["condition"]])


def is_rapp(label: str) -> bool:
    """Rapp PA 条件（M2/M3/M4 及其显式回退形式）。"""
    try:
        return parse_condition(label)["condition"] in RAPP_BASES
    except ValueError:
        return False


def parse_cfg_overrides(items) -> dict:
    """--cfg key=value … → SystemConfig 字段覆盖（例如 bg_cancel=null cal_ref=object isolation_db=-40）。
    值依次尝试 int / float / true|false / none，否则为字符串；未知字段报错。条件相关字段请用条件标签。"""
    import dataclasses

    from ..system import SystemConfig

    names = {f.name for f in dataclasses.fields(SystemConfig)}
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--cfg 需要 key=value，得到 {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in names:
            raise ValueError(f"SystemConfig 没有字段 {k!r}")
        if k in ("condition", "backoff_db", "m1_gain", "objects_see_relay_tx", "object_mode", "material", "n_relays",
                 "preset"):
            raise ValueError(f"字段 {k!r} 由网格/条件标签决定，不能用 --cfg 覆盖")
        vl = v.strip().lower()
        if vl in ("true", "false"):
            val = vl == "true"
        elif vl == "none":
            val = None
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v.strip()
        out[k] = val
    return out


def scenario_overrides(name: str) -> dict:
    """命名场景（system.SCENARIOS）除 preset 外的 SystemConfig 字段（进 job 的 cfg_over；值为可哈希的 tuple / 标量）。
    baseline → {}（与改动前的 job 完全相同）。"""
    from ..system import SCENARIOS

    if name not in SCENARIOS:
        raise ValueError(f"未知场景 {name!r}，可选 {list(SCENARIOS)}")
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in SCENARIOS[name].items() if k != "preset"}


def base_label(label: str) -> tuple[str, str]:
    """(去掉后缀的标签, 后缀)。"""
    return (label[: -len(NOTX)], NOTX) if label.endswith(NOTX) else (label, "")


def gain_pairs(labels, any_power_m1: bool = False) -> list[tuple[str, str]]:
    """非线性收益配对 (Mx, 参考)：Rapp 条件（M2/M3/M4、M3@+20 …）对**同一回退**（同源功率）、同后缀的 M1 与 M1b
    （"M3" ↔ "M1@M3"、"M3@+20" ↔ "M1@+20"；按解析出的回退值配对，不依赖标签写法）。顺序：标签顺序，M1 在 M1b 前。
    any_power_m1=True（非 baseline 场景，INTEGRATION S）：没有同回退 M1 时与回退最接近的 M1 配对（线性中继的无噪声
    y1 − y2（dB）与源功率无关）；M1b 始终只与同回退配对。"""
    present = list(dict.fromkeys(labels))
    parsed = {}
    for lab in present:
        try:
            parsed[lab] = (parse_condition(lab), condition_backoff_db(lab))
        except ValueError:
            continue
    out = []
    for lab in present:
        if lab not in parsed or parsed[lab][0]["condition"] not in RAPP_BASES:
            continue
        p, bo = parsed[lab]
        for kind in ("M1", "M1b"):
            same = [ref for ref in present if ref in parsed and parsed[ref][0]["condition"] == kind
                    and abs(parsed[ref][1] - bo) < 1e-9
                    and parsed[ref][0]["objects_see_relay_tx"] == p["objects_see_relay_tx"]]
            if not same and kind == "M1" and any_power_m1:
                # 没有同回退的 M1：线性中继（小信号增益）的无噪声 y1 − y2（dB）与源功率无关 → 与任一 M1 配对（回退最接近者）
                same = sorted([ref for ref in present if ref in parsed and parsed[ref][0]["condition"] == "M1"
                               and parsed[ref][0]["objects_see_relay_tx"] == p["objects_see_relay_tx"]],
                              key=lambda r: abs(parsed[r][1] - bo))[:1]
            out += [(lab, ref) for ref in same]
    return out


# ============================================================ 每个 worker 内的系统缓存
_SYS_CACHE: "collections.OrderedDict[tuple, object]" = collections.OrderedDict()
_SYS_CACHE_MAX = 6


def _system_key(job: dict) -> tuple:
    return (job["preset"], int(job["n"]), job["cond"], job["mode"], job["material"], int(job["n_relays"]),
            float(job["spacing_lam"]), tuple(sorted((job.get("cfg_over") or {}).items())))


def get_system(job: dict):
    """按 job 构造（或从本进程缓存取）PhysicalSystem。系统构造是确定性的（calib_rng=None → default_rng(0)），
    构造后不被训练/评估修改（随机性全部来自调用方 Generator），所以同一 worker 内不同种子可安全复用。"""
    from ..constants import LAMBDA
    from ..system import SystemConfig, make_system

    key = _system_key(job)
    if key in _SYS_CACHE:
        _SYS_CACHE.move_to_end(key)
        return _SYS_CACHE[key]
    cfg = SystemConfig(preset=job["preset"], n_relays=int(job["n_relays"]), object_mode=job["mode"],
                       material=job["material"], **parse_condition(job["cond"]), **(job.get("cfg_over") or {}))
    s = make_system(cfg, "parity", n=int(job["n"]), spacing=float(job["spacing_lam"]) * LAMBDA,
                    diameter=DIAMETER_LAM * LAMBDA)
    _SYS_CACHE[key] = s
    while len(_SYS_CACHE) > _SYS_CACHE_MAX:
        _SYS_CACHE.popitem(last=False)
    return s


def system_info(s) -> dict:
    """写入 runs.jsonl 的系统摘要：实际工作点相关量（p_feed、backoff、voxel_d、材料、Bussgang 增益……）。"""
    t = s.task
    d = {"label": s.label, "p_feed_dbm": float(10.0 * np.log10(s.p_feed_w) + 30.0), "backoff_db": s.backoff_db,
         "p1db_in_dbm": float(10.0 * np.log10(s.p1db_in_w) + 30.0), "voxel_d": float(t.voxel_d),
         "lambda_over_voxel_d": float(s.lam / t.voxel_d), "n_vox_per_object": int(t.n_vox_per_object),
         "object_model": s.object_model, "effective_eps": t.describe().get("effective_eps"),
         "spacing": float(t.spacing), "diameter": float(t.diameter), "n_params": int(s.n_params),
         "objects_see_relay_tx": bool(s.cfg.objects_see_relay_tx), "build_time_s": float(s.build_time_s),
         "kappa0_eff_db": float(20.0 * np.log10(abs(s.params.kappa0))),
         "loop_gain_ss_nominal_db": (None if s.pa is None
                                     else float(20.0 * np.log10(abs(s.small_signal_gain() * s.params.kappa0))))}
    if s.bussgang_gain is not None:
        d["bussgang_gain_db"] = float(20.0 * np.log10(abs(s.bussgang_gain)))
        d["bussgang_phase_deg"] = float(np.degrees(np.angle(s.bussgang_gain)))
    return d


# ============================================================ 非线性度
_NL_KEYS = ("e", "u_in", "u", "g", "z", "y_diff_lin", "y_diff_db", "K", "C", "u_linref", "g_linref", "z_linref",
            "y_diff_lin_linref", "y_diff_db_linref", "pa_excess_u", "pa_excess_g", "pa_excess_z",
            "pa_excess_y_diff_lin", "pa_excess_y_diff_db")


def compact_profile(prof: dict) -> dict:
    """nonlinearity_profile 的紧凑版：每个量只留 W2、W3（K、C 另留 rel_var）、profile。"""
    out = {}
    for k in _NL_KEYS:
        v = prof.get(k)
        if isinstance(v, dict):
            out[k] = {f: v[f] for f in ("W2", "W3", "rel_var", "profile") if f in v}
    for k in ("flags_any", "linref_flags_any"):
        if k in prof:
            out[k] = bool(prof[k])
    return out


def _w(prof: dict | None, key: str, f: str):
    if not prof or key not in prof or not isinstance(prof[key], dict):
        return None
    v = prof[key].get(f)
    return None if v is None or not math.isfinite(float(v)) else float(v)


def decompose(prof: dict | None) -> dict | None:
    """W≥d 分解（方案 §7）：obj = W(e)；pa = W(g) − W(u)（PA 映射本身，M0 为 None）；y = W(y1−y2 线性功率）；
    ydb = W(y1−y2 dB)；pa_excess_y = W(y) − W(y_linref)；K/C 的 rel_var（环路/发射路径对 x 的依赖强度）。"""
    if prof is None:
        return None
    out = {}
    for d in ("W2", "W3"):
        tag = d[1]
        out[f"obj_W{tag}"] = _w(prof, "e", d)
        g, u = _w(prof, "g", d), _w(prof, "u", d)
        out[f"pa_W{tag}"] = None if (g is None or u is None) else g - u
        out[f"uin_W{tag}"] = _w(prof, "u_in", d)
        out[f"u_W{tag}"] = u
        out[f"g_W{tag}"] = g
        out[f"z_W{tag}"] = _w(prof, "z", d)
        out[f"y_W{tag}"] = _w(prof, "y_diff_lin", d)
        out[f"ydb_W{tag}"] = _w(prof, "y_diff_db", d)
        out[f"pa_excess_y_W{tag}"] = _w(prof, "pa_excess_y_diff_lin", d)
    out["K_relvar"] = _w(prof, "K", "rel_var")
    out["C_relvar"] = _w(prof, "C", "rel_var")
    return out


# ============================================================ 训练器结果
def solved_threshold_db(s) -> float:
    """"solved" 的裕量门限：SOLVED_SIGMA_MULT × 检波器读数噪声 σ（VNA 0.002 dB → 0.006 dB；ideal → 0）（审查 Q-1）。"""
    if s.cfg.detector == "ideal":
        return 0.0
    return SOLVED_SIGMA_MULT * float(s.detector.sigma_db)


def is_robust(ev: dict, thr_db: float = ROBUST_MARGIN_DB) -> bool:
    """全对且无噪声最小带符号裕量 ≥ thr_db（默认 0.5 dB，INTEGRATION S：物理上可靠的判别）。"""
    m = ev.get("margin_db_unflagged")
    return bool(ev.get("all_correct")) and m is not None and math.isfinite(float(m)) and float(m) >= thr_db


def is_solved(ev: dict, thr_db: float) -> bool:
    """全对且无噪声最小带符号裕量 > thr_db（裕量 ≤ 3σ 的"全对"在一次 VNA 读数下就可能翻转，不算解出）。"""
    m = ev.get("margin_db_unflagged")
    return bool(ev.get("all_correct")) and m is not None and math.isfinite(float(m)) and float(m) > thr_db


def trainer_row(res: dict, s, kind: str) -> dict:
    """训练器结果 → 行：完整 evaluate() 字典 + 训练开销 + 便于汇总的扁平字段（含 solved / solved_valid，审查 Q-1）。"""
    ev = dict(res["eval"])
    B = int(s.B)
    row = {"eval": ev, "time_s": float(res.get("time_s", float("nan"))), "objective": res.get("objective")}
    for k in ("acc_clean", "all_correct", "acc_clean_valid", "all_correct_valid", "margin_db",
              "margin_db_unflagged", "acc_noisy", "acc_noisy_valid", "oscillating_run", "n_flagged",
              "loss_clean", "loss_meas_clean", "null_depth_db", "loop_gain_ss_max", "loop_gain_op_max",
              "pa_in_rel_p1db_db_mean", "pa_in_rel_p1db_db_min", "pa_in_rel_p1db_db_max",
              "pa_in_open_rel_p1db_db_mean", "pa_in_delta_rel_p1db_db_max"):
        row[k] = ev.get(k)
    thr = solved_threshold_db(s)
    row["solved_thr_db"] = thr
    row["solved"] = is_solved(ev, thr)
    row["solved_valid"] = None if ev.get("oscillating_run") else row["solved"]
    row["robust"] = is_robust(ev)
    row["robust_valid"] = None if ev.get("oscillating_run") else row["robust"]
    row["flagged_frac"] = float(ev.get("n_flagged", 0)) / B
    if kind == "dspsa":
        row.update(n_measurements=int(res["n_measurements"]), variant=res["variant"], a=res.get("a"),
                   train_objective=res.get("train_objective"),
                   train_flagged_frac=float(res["n_flagged_measurements"]) / max(1, int(res["n_measurements"])),
                   curve=res.get("curve"))
    elif kind == "dspsa_pocket":
        row.update(loss_measured=res.get("loss_measured"), at_measurement=res.get("at_measurement"))
    elif kind.startswith("oracle"):
        row.update(loss=float(res["loss"]), n_evals=int(res["n_evals"]))
        for k in ("stage1_acc_clean", "stage1_margin_db", "polish_sweeps", "polish_evals"):
            if k in res:
                row[k] = res[k]
    else:
        row.update(loss=float(res["loss"]), restart_best=[float(v) for v in res.get("restart_best", [])])
        if res.get("source") is not None:
            row["cont_source"] = res["source"]
            row["cont_candidates"] = res.get("candidates")
    return row


def best_of(trainers: dict) -> dict:
    """各训练器中（非自激）最好的结果：acc 取最大、all_correct / solved 取任一、margin 取最大（"这个结构能做到多好"）。"""
    valid = {k: v for k, v in trainers.items() if v is not None and not v.get("oscillating_run")}
    if not valid:
        return {"acc_clean_valid": None, "all_correct_valid": None, "solved_valid": None, "robust_valid": None,
                "margin_db": None, "oscillating_run": True, "source": None}
    key = max(valid, key=lambda k: (valid[k]["acc_clean_valid"] or 0.0,
                                    -math.inf if valid[k]["margin_db"] is None else valid[k]["margin_db"]))
    margins = [v["margin_db"] for v in valid.values() if v["margin_db"] is not None]
    return {"acc_clean_valid": valid[key]["acc_clean_valid"],
            "all_correct_valid": bool(any(bool(v["all_correct_valid"]) for v in valid.values())),
            "solved_valid": bool(any(bool(v.get("solved_valid")) for v in valid.values())),
            "robust_valid": bool(any(bool(v.get("robust_valid")) for v in valid.values())),
            "margin_db": max(margins) if margins else None, "acc_noisy_valid": valid[key]["acc_noisy_valid"],
            "oscillating_run": False, "source": key}


def polish_min_margin(s, params, rng: np.random.Generator, max_sweeps: int = POLISH_SWEEPS) -> dict:
    """accuracy-oracle 的第二阶段（审查 Q-1）：从 params 出发做字典序 (判对数, 最小带符号裕量 dB) 的坐标下降——
    逐分量尝试全部电平，只接受字典序严格上升的改动，所以判对数不降、在 accuracy 最优码本附近把最小裕量推开
    （accuracy 目标的 tanh 平局项可停在裕量 0.001 dB < VNA σ 的"全对"上）。判决与 train.oracle_objective 一致：
    被标记样本裕量 −FLAG_MARGIN_DB、判错；平局判错。无噪声模型前向（"这个离散结构最多能做到多好"）。
    返回 {"params", "n_ok", "min_margin_db", "n_evals", "sweeps"}。"""
    from ..train import FLAG_MARGIN_DB

    task, labels = s.task, np.asarray(s.labels)
    L = np.asarray(s.levels, dtype=np.int64)

    def key(x):
        m = s.forward(s.phases_from_params(x), noisy=False)
        y = s.clean_readings(m)
        flags = np.asarray(m.flags, dtype=bool)
        if not np.all(np.isfinite(y[~flags])):
            return (-1, -math.inf)
        sm = np.where(flags, -FLAG_MARGIN_DB, np.asarray(task.sample_margins_db(y, labels), dtype=float))
        pred = np.asarray(task.decide(y, labels)).copy()
        pred[flags] = -1
        return (int(np.sum(pred == labels)), float(np.min(sm)))

    x = np.asarray(params, dtype=np.int64).copy()
    k_cur = key(x)
    n_evals, sweeps = 1, 0
    for _ in range(int(max_sweeps)):
        sweeps += 1
        improved = False
        for i in rng.permutation(x.size):
            cur = int(x[i])
            best_l, best_k = cur, k_cur
            for lv in range(int(L[i])):
                if lv == cur:
                    continue
                x[i] = lv
                kk = key(x)
                n_evals += 1
                if kk > best_k:
                    best_l, best_k = lv, kk
            x[i] = best_l
            if best_l != cur:
                k_cur = best_k
                improved = True
        if not improved:
            break
    return {"params": x, "n_ok": int(k_cur[0]), "min_margin_db": float(k_cur[1]), "n_evals": n_evals, "sweeps": sweeps}


def run_accuracy_oracle(s, rng_search: np.random.Generator, rng_polish: np.random.Generator, n_restarts: int,
                        max_sweeps: int, polish_sweeps: int, n_noise_eval: int, rng_eval: np.random.Generator) -> dict:
    """1-bit "最好可达准确率" oracle（INTEGRATION G-A / B3，审查 Q-1）：run_oracle_bits(objective="accuracy")
    → polish_min_margin → evaluate。返回 run_oracle_bits 同形字典（"objective" = "accuracy+polish"）+ 第一阶段指标。"""
    from .. import train

    t0 = time.perf_counter()
    o = train.run_oracle_bits(s, rng_search, n_restarts=n_restarts, max_sweeps=max_sweeps, n_noise_eval=0,
                              objective="accuracy")
    ev1 = o["eval"]
    out = {"params": o["params"], "loss": o["loss"], "n_evals": int(o["n_evals"]), "objective": "accuracy+polish",
           "stage1_acc_clean": ev1.get("acc_clean"), "stage1_margin_db": ev1.get("margin_db_unflagged"),
           "polish_sweeps": 0, "polish_evals": 0}
    if polish_sweeps > 0:
        p = polish_min_margin(s, o["params"], rng_polish, polish_sweeps)
        out.update(params=p["params"], n_evals=out["n_evals"] + p["n_evals"], polish_sweeps=p["sweeps"],
                   polish_evals=p["n_evals"], loss=-(p["n_ok"] + 0.5 * math.tanh(p["min_margin_db"] / s.cfg.tau_db)))
    out["eval"] = s.evaluate(out["params"], rng_eval, n_noise=n_noise_eval)
    out["time_s"] = time.perf_counter() - t0
    return out


def chance_loss_of(s) -> float:
    """"零信息"逐样本损失（losses.chance_loss；奇偶 = τ·ln2），平台诊断的基准。"""
    from ..losses import chance_loss

    return float(chance_loss(s.task.kind, s.n_probes, s.cfg.tau_db, **s.task.task_kwargs))


def e_reach(L0, L, Lstar, frac: float = REACH_FRAC):
    """实验 E 判据（INTEGRATION G-E）：(L0 − L) ≥ frac·(L0 − L*)；L* ≥ L0（oracle 不比初始码本好）时记为达标。
    任一量缺失 → None。"""
    if L0 is None or L is None or Lstar is None:
        return None
    L0, L, Lstar = float(L0), float(L), float(Lstar)
    if not (math.isfinite(L0) and math.isfinite(L) and math.isfinite(Lstar)):
        return None
    gap = L0 - Lstar
    if gap <= 0.0:
        return True
    return bool((L0 - L) >= frac * gap - 1e-12)           # 恰在边界上（浮点舍入）算达标


def dspsa_diagnostics(trainers: dict, d: dict, chance: float) -> dict:
    """DSPSA 训练诊断（审查 Q-2 / S1-0），写回 trainers["dspsa"] / ["dspsa_pocket"]：
    plateau = loss_meas_clean ≥ PLATEAU_FRAC × chance（y1 ≈ y2 平台）；reach90 = 实验 E 判据，
    L0 = 初始码本（curve[0]）、L* = 1-bit oracle（oracle / oracle_loss）的 loss_meas_clean 最小者；
    trained = reach90 且不在平台上。返回 {"L0", "Lstar", "source", "chance", "Lstar_on_plateau"}。"""
    curve = d.get("curve") or []
    L0 = curve[0].get("loss_meas_clean") if curve else None
    refs = {k: trainers[k]["loss_meas_clean"] for k in ("oracle", "oracle_loss")
            if trainers.get(k) is not None and trainers[k].get("loss_meas_clean") is not None}
    src = min(refs, key=refs.get) if refs else None
    Lstar = refs[src] if src else None
    acc_star = (trainers.get("oracle") or {}).get("acc_clean_valid")
    for k in DSPSA_TRAINERS:
        t = trainers.get(k)
        if t is None:
            continue
        L = t.get("loss_meas_clean")
        t["plateau"] = None if L is None else bool(float(L) >= PLATEAU_FRAC * chance)
        if (trainers.get("dspsa") or {}).get("train_objective") == "accuracy_first":
            # INTEGRATION T：判对优先训练器不优化 softplus 损失 → 按准确率判据（同 bench_trainers）：
            # acc ≥ 0.9·acc*（acc* = 同种子 1-bit accuracy oracle）；平台只作诊断
            a = t.get("acc_clean_valid")
            t["reach90"] = (None if a is None or acc_star is None
                            else bool(float(a) >= REACH_FRAC * float(acc_star) - 1e-12))
            t["trained"] = t["reach90"]
            continue
        t["reach90"] = e_reach(L0, L, Lstar)
        # 训练到位 = E 判据达标且离开平台：难格点里 softplus 的 L* 本身就在 τ·ln2 平台上（L* ≈ 0.69），
        # 单看 E 判据会把停在平台上的 DSPSA 记为"达标"
        t["trained"] = None if t["reach90"] is None or t["plateau"] is None else bool(t["reach90"] and not t["plateau"])
    return {"L0": L0, "Lstar": Lstar, "source": src, "chance": chance,
            "Lstar_on_plateau": None if Lstar is None else bool(float(Lstar) >= PLATEAU_FRAC * chance)}


def is_g1_cell(cond: str, mode: str, n: int) -> bool:
    """门 G1 格点：M0 + Born（及 M1:noTx + Born 理想化线性中继），n ≥ 3。"""
    return (mode == "born" and int(n) >= 3
            and (cond == "M0" or (cond.endswith(NOTX) and base_label(cond)[0].startswith("M1"))))


# ============================================================ 连续 oracle 增强（cont-plus，INTEGRATION S）
def readings_floor_db(p_w: np.ndarray, dr_db: float | None, flags=None) -> np.ndarray:
    """动态范围受限读数 y = 10log10(P + P_floor) + 30，P_floor = max(未标记样本的 P)·10^(−dr/10)；dr_db=None → 理想 dBm。
    （深零陷上的 dB 裕量在实测中不可靠：选择 / 精修连续码本时不让裕量靠 > dr dB 的深零陷取得。）"""
    p = np.asarray(p_w, dtype=float)
    if dr_db is None:
        return 10.0 * np.log10(np.maximum(p, 1e-300)) + 30.0
    ok = np.ones(p.shape[0], bool) if flags is None else ~np.asarray(flags, bool)
    pmax = float(np.max(p[ok])) if ok.any() else float(np.max(p))
    return 10.0 * np.log10(p + pmax * 10.0 ** (-dr_db / 10.0)) + 30.0


def margin_key(s, phases: dict, dr_db: float | None = DR_DB) -> tuple:
    """无噪声 (判对数, 最小带符号裕量 dB)（字典序越大越好）；读数经 readings_floor_db。被标记样本判错、裕量 −1e3。"""
    from ..train import FLAG_MARGIN_DB

    m = s.forward(phases, noisy=False)
    fl = np.asarray(m.flags, dtype=bool)
    y = s.clean_readings(m) if dr_db is None else readings_floor_db(m.p_w, dr_db, fl)
    if not np.all(np.isfinite(y[~fl])):
        return (-1, -math.inf)
    sm = np.where(fl, -FLAG_MARGIN_DB, np.asarray(s.task.sample_margins_db(y, s.labels), dtype=float))
    pred = np.asarray(s.task.decide(y, s.labels)).copy()
    pred[fl] = -1
    return (int(np.sum(pred == s.labels)), float(np.min(sm)))


def design_rx_phases(s, iters: int = 400, lr: float = 0.05, weight: float = 300.0) -> np.ndarray:
    """接收码本设计初值（方案推论 3：入射波是偏置）：连续相位最大化 |Σ e^{jφ} δv_sum|² − weight·|Σ e^{jφ} v0|²（各自归一化），
    v0 = 空场景馈源列接收场，δv_sum = 单物体样本相对 v0 的接收场之和（奇偶任务）。投影初值 + 梯度下降。返回 (R, 25)。"""
    R, N = s.n_relays, s.n_elem
    v0 = s.H0[:R * N, 0].reshape(R, N)
    single = [i for i in range(s.B) if bin(i).count("1") == 1]
    dvs = (s._H[single, :R * N, 0].reshape(len(single), R, N) - v0[None]).sum(axis=0)
    out = np.zeros((R, N))
    for r in range(R):
        a = dvs[r] - (np.vdot(v0[r], dvs[r]) / np.vdot(v0[r], v0[r])) * v0[r]
        phi = -np.angle(a)
        n0 = np.sum(np.abs(v0[r])) ** 2
        nd = np.sum(np.abs(dvs[r])) ** 2
        for _ in range(int(iters)):
            e = np.exp(1j * phi)
            u0 = np.sum(e * v0[r])
            ud = np.sum(e * dvs[r])
            g0 = 2 * np.real(np.conj(u0) * 1j * e * v0[r]) / n0
            gd = 2 * np.real(np.conj(ud) * 1j * e * dvs[r]) / nd
            phi = phi - lr * (weight * g0 - gd)
        out[r] = phi
    return out


def softmin_refine(s, phases: dict, steps: int = 300, lr: float = 0.01, tau_m: float = 0.05,
                   dr_db: float | None = DR_DB) -> dict:
    """连续相位 soft-min 裕量精修：最小化 τ_m·logsumexp(−s_b/τ_m)（→ −min_b s_b）+ 模型惩罚（光滑自激 / 零陷），
    读数为动态范围受限形式；每 20 步按 margin_key(dr_db) 记录，返回全程最好的相位（mod 2π）。仅二分类（奇偶）任务。"""
    import torch

    ts = s.to_torch()
    ph = {k: torch.as_tensor(np.asarray(v, dtype=float).reshape(ts.param_shapes[k]), dtype=torch.float64)
          .clone().requires_grad_(True) for k, v in phases.items() if k in ts.param_shapes}
    sgn = torch.as_tensor(np.where(np.asarray(s.labels) != 0, 1.0, -1.0))
    opt = torch.optim.Adam(list(ph.values()), lr=lr)
    best = (margin_key(s, {k: v.detach().numpy() for k, v in ph.items()}, dr_db),
            {k: v.detach().clone() for k, v in ph.items()})
    for it in range(int(steps)):
        opt.zero_grad()
        out = ts.forward(ph)
        z = out["z"]
        p = z.real ** 2 + z.imag ** 2
        fl = torch.as_tensor(np.asarray(out["flags"], dtype=bool))
        if dr_db is not None:
            pm = torch.where(fl.unsqueeze(-1), torch.zeros_like(p), p).max().detach()
            p = p + pm * 10.0 ** (-dr_db / 10.0)
        y = 10.0 * torch.log10(torch.clamp(p, min=1e-300))
        sm = torch.where(fl, torch.full_like(p[:, 0], -10.0), sgn * (y[:, 0] - y[:, 1]))
        L = tau_m * torch.logsumexp(-sm / tau_m, dim=0)
        if ts.kind == "relay":
            if ts.null_ref is not None:
                L = L + ts._null_penalty(out["w"])
            L = L + ts._osc_penalty(out)
        if not torch.isfinite(L):
            break
        L.backward()
        for v in ph.values():
            if v.grad is not None:
                torch.nan_to_num_(v.grad, nan=0.0, posinf=0.0, neginf=0.0)
        opt.step()
        if it % 20 == 19 or it == steps - 1:
            kk = margin_key(s, {k: v.detach().numpy() for k, v in ph.items()}, dr_db)
            if kk > best[0]:
                best = (kk, {k: v.detach().clone() for k, v in ph.items()})
    return {k: np.mod(v.numpy().astype(float), 2 * np.pi) for k, v in best[1].items()}


def run_cont_plus(s, seed: int, n_restarts: int, steps: int, n_noise_eval: int, rng_eval: np.random.Generator,
                  tau_cont: float = CONT_PLUS_TAU_DB, init_restarts: int = 1, refine_steps: int = 300) -> dict:
    """cont-plus 连续 oracle（INTEGRATION S；= _scratch/scenarios/geometry_power 的 cont / cinit / crefine）：
    τ_cont softplus 模型目标的随机重启（n_restarts × steps）+ 设计初值重启（init_restarts，中继奇偶任务，按 margin_key 取最好）
    + 对两者中较好者做 soft-min 裕量精修（refine_steps）；按 margin_key（40 dB 动态范围受限读数）取最好者，再用
    s.evaluate_phases 复评（理想读数）。返回 run_oracle_continuous 同形字典 + "source"（cont / cinit / crefine）与 "candidates"。"""
    from .. import train
    from ..tasks import ParityTask

    t0 = time.perf_counter()
    sc = s.derive(tau_db=float(tau_cont)) if float(tau_cont) != float(s.cfg.tau_db) else s
    cands, losses_ = {}, {}
    c = train.run_oracle_continuous(sc, n_restarts=n_restarts, steps=steps, seed=int(seed), n_noise_eval=0)
    cands["cont"], losses_["cont"] = c["phases"], float(c["loss"])
    rb = list(c.get("restart_best", []))
    parity = isinstance(s.task, ParityTask)
    keys = {"cont": margin_key(s, cands["cont"])}
    if parity and not s.is_passive and int(init_restarts) > 0:
        # 设计初值重启：接收相位 = design_rx_phases，发射相位随机；按 margin_key（不是 softplus 损失）取最好者——
        # 损失最低的重启未必判对最多（_scratch 原实现按损失选，会丢掉已全对的候选）
        ph_rx = design_rx_phases(s)
        best = None
        for r in range(int(init_restarts)):
            ci = train.run_oracle_continuous(sc, n_restarts=1, steps=steps, seed=int(seed) * 100 + r, n_noise_eval=0,
                                             init={"phase_rx": ph_rx})
            kk = margin_key(s, ci["phases"])
            if best is None or kk > best[0]:
                best = (kk, ci)
        cands["cinit"], losses_["cinit"], keys["cinit"] = best[1]["phases"], float(best[1]["loss"]), best[0]
    if parity and int(refine_steps) > 0:
        kb = max(keys, key=keys.get)
        cands["crefine"] = softmin_refine(s, cands[kb], steps=int(refine_steps))
        keys["crefine"] = margin_key(s, cands["crefine"])
    src = max(keys, key=keys.get)
    ev = s.evaluate_phases(cands[src], rng_eval, n_noise=n_noise_eval)
    return {"phases": cands[src], "loss": losses_.get(src, float("nan")), "eval": ev, "history": c.get("history"),
            "restart_best": rb, "objective": "model+softmin", "source": src,
            "candidates": {k: {"n_ok": int(v[0]), "margin_dr_db": float(v[1])} for k, v in keys.items()},
            "time_s": time.perf_counter() - t0}


# ============================================================ job（模块顶层，可 pickle）
def run_job(job: dict) -> dict:
    """一个 run：DSPSA（最终迭代 + pocket）+ 1-bit accuracy oracle（+ polish）+ loss-oracle（诊断）
    （+ 门 G1 格点的 margin-oracle）(+ 连续 oracle) + 非线性度。随机数全部由 common.seed_rng 按
    (种子, 格点除条件外的全部因素) 派生 → 同一种子下各条件的初始码本与扰动序列相同（公共随机数，配对比较）；
    三个 1-bit oracle 用同一个 "expA-oracle" 派生（同一组随机重启起点）。"""
    from .. import train

    t0 = time.perf_counter()
    s = get_system(job)
    t_build = time.perf_counter() - t0
    cell = (job["preset"], int(job["seed"]), int(job["n"]), job["mode"], job["material"], int(job["n_relays"]),
            float(job["spacing_lam"]))
    n_noise = int(job.get("n_noise_eval", 20))
    rng_d = common.seed_rng("expA-dspsa", *cell)
    rng_p = common.seed_rng("expA-profile", *cell)
    row = {"exp": "A", "scenario": job.get("scenario", "baseline"), "n": int(job["n"]), "cond": job["cond"],
           "mode": job["mode"], "material": job["material"],
           "n_relays": int(job["n_relays"]), "spacing_lam": float(job["spacing_lam"]), "seed": int(job["seed"]),
           "preset": job["preset"], "cfg_over": job.get("cfg_over") or {}, "sys": system_info(s),
           "build_or_cache_s": t_build}
    trainers, nl = {k: None for k in TRAINERS}, {k: None for k in TRAINERS}

    def prof(params):
        return compact_profile(s.nonlinearity_profile(s.phases_from_params(params)))

    # 1. DSPSA：默认变体（INTEGRATION T：1-bit = sign_acc，输出即 pocket）；旧变体另报 pocket（审查 Q-2）
    var = job.get("dspsa_variant", "auto")
    d = train.run_dspsa(s, int(job["dspsa_iters"]), rng_d, eval_every=int(job.get("eval_every", 100)),
                        n_noise_eval=n_noise, pocket=True, variant=var)
    trainers["dspsa"] = trainer_row(d, s, "dspsa")
    nl["dspsa"] = prof(d["params"])
    pk = d.get("pocket") or {}
    if d["variant"] != "sign_acc" and pk.get("params") is not None and pk.get("eval") is not None:
        trainers["dspsa_pocket"] = trainer_row({"eval": pk["eval"], "time_s": d["time_s"],
                                                "objective": "measurable (pocket)",
                                                "loss_measured": pk.get("loss_measured"),
                                                "at_measurement": pk.get("at_measurement")}, s, "dspsa_pocket")
        nl["dspsa_pocket"] = prof(pk["params"])
    # 2. 1-bit accuracy oracle + polish（审查 Q-1）
    o = run_accuracy_oracle(s, common.seed_rng("expA-oracle", *cell), common.seed_rng("expA-polish", *cell),
                            int(job["oracle_restarts"]), int(job["oracle_sweeps"]), int(job.get("polish_sweeps",
                                                                                                POLISH_SWEEPS)),
                            n_noise, common.seed_rng("expA-oracle-eval", *cell))
    trainers["oracle"] = trainer_row(o, s, "oracle")
    nl["oracle"] = prof(o["params"])
    # 诊断：默认 softplus 损失 oracle（同一随机重启；审查 S0-2 / Q-1）
    if job.get("oracle_loss", True):
        ol = train.run_oracle_bits(s, common.seed_rng("expA-oracle", *cell), n_restarts=int(job["oracle_restarts"]),
                                   max_sweeps=int(job["oracle_sweeps"]), n_noise_eval=n_noise, objective="loss")
        trainers["oracle_loss"] = trainer_row(ol, s, "oracle_loss")
        nl["oracle_loss"] = prof(ol["params"])
    # 门 G1 格点：最大化最小裕量（> 0 ⇔ 全对；selfcheck 同判据）
    if is_g1_cell(job["cond"], job["mode"], job["n"]):
        om = train.run_oracle_bits(s, common.seed_rng("expA-oracle", *cell), n_restarts=int(job["oracle_restarts"]),
                                   max_sweeps=int(job["oracle_sweeps"]), n_noise_eval=n_noise, objective="margin")
        trainers["oracle_margin"] = trainer_row(om, s, "oracle_margin")
        nl["oracle_margin"] = prof(om["params"])
    # 3. 连续相位 oracle（softplus 模型目标；cont-plus 另加设计初值 + soft-min 精修，INTEGRATION S）
    if job.get("cont"):
        cseed = int(common.seed_rng("expA-cont", *cell).integers(0, 2 ** 31 - 1))
        if job.get("cont_plus"):
            c = run_cont_plus(s, cseed, int(job["cont_restarts"]), int(job["cont_steps"]), n_noise,
                              common.seed_rng("expA-cont-eval", *cell),
                              init_restarts=int(job.get("cont_init_restarts", 1)),
                              refine_steps=int(job.get("cont_refine_steps", 300)))
        else:
            c = train.run_oracle_continuous(s, n_restarts=int(job["cont_restarts"]), steps=int(job["cont_steps"]),
                                            seed=cseed, n_noise_eval=n_noise)
        trainers["cont"] = trainer_row(c, s, "cont")
        trainers["cont"]["phases"] = {k: np.asarray(v).tolist() for k, v in c["phases"].items()}
        nl["cont"] = compact_profile(s.nonlinearity_profile(c["phases"]))
    row["e_ref"] = dspsa_diagnostics(trainers, d, chance_loss_of(s))
    nl["random"] = s.nonlinearity_profile_random(rng_p, int(job.get("n_random", 5)))
    row.update(trainers)
    row["best"] = best_of(trainers)
    row["nl"] = nl
    row["w"] = {k: decompose(v) for k, v in nl.items()}
    row["run_time_s"] = time.perf_counter() - t0
    return row


# ============================================================ 网格
def build_jobs(args) -> list[dict]:
    """格点 × 种子 → job 列表（去重；同一系统的种子相邻，便于 worker 缓存）。"""
    cells = []
    if args.design == "blocks":
        # 方案 §5："全因子组合太大，每个实验只扫两三个因素，其余固定在默认值"
        main = dict(n=args.n, cond=args.conditions, mode=args.modes, material=["water"], n_relays=[1],
                    spacing_lam=[DEFAULT_SPACING_LAM])
        rq2 = dict(n=[v for v in args.n if v >= 3] or args.n, cond=[c for c in args.conditions if c in
                                                                      ("M0", "M1@M3", "M1@M4", "M3", "M4")],
                   mode=[m for m in args.modes if m in ("intra", "full")] or args.modes, material=args.materials,
                   n_relays=[1], spacing_lam=sorted(set(args.spacing) | {0.35}))
        nr = dict(n=[v for v in args.n if v >= 3] or args.n, cond=[c for c in args.conditions if c in
                                                                     ("M0", "M1@M3", "M1@M4", "M3", "M4")],
                  mode=[m for m in args.modes if m in ("born", "full")] or args.modes, material=["water"],
                  n_relays=args.n_relays, spacing_lam=[DEFAULT_SPACING_LAM])
        blocks = [main, rq2, nr]
    else:
        blocks = [dict(n=args.n, cond=args.conditions, mode=args.modes, material=args.materials,
                       n_relays=args.n_relays, spacing_lam=args.spacing)]
    sweep = list(getattr(args, "power_sweep", None) or [])
    if sweep and any(m in ("intra", "full") for m in args.modes):
        # 审查 Q-3：训练码本零陷背景后 intra/full 的 M3/M4 实际只在 P1dB 以下工作、物体驱动低 20–37 dB；
        # 回退扫描块（n ≥ 3 × intra/full，水，N_r = 1）把源功率再抬高，让 PA 压缩物体相关部分；每个回退配对 M1（+M1b）。
        conds = []
        for b in sweep:
            conds += [f"M3@{float(b):+g}", f"M1@{float(b):+g}"]
            if getattr(args, "power_sweep_m1b", False):
                conds.append(f"M1b@{float(b):+g}")
        blocks.append(dict(n=[v for v in args.n if v >= 3] or args.n, cond=conds,
                           mode=[m for m in args.modes if m in ("intra", "full")], material=["water"],
                           n_relays=[1], spacing_lam=[DEFAULT_SPACING_LAM]))
    seen = set()
    for b in blocks:
        for n in b["n"]:
            for mat in b["material"]:
                for nr in b["n_relays"]:
                    for sp in b["spacing_lam"]:
                        for mode in b["mode"]:
                            for cond in b["cond"]:
                                key = (n, cond, mode, mat, nr, sp)
                                if key not in seen:
                                    seen.add(key)
                                    cells.append(key)
    if args.g1_controls and getattr(args, "scenario", "baseline") != "baseline" and "born" not in args.modes:
        # informative 场景只跑 full：另加门 G1 的 M0 + Born 格点（n ≥ 3），让自检在本场景下也成立（INTEGRATION S）
        for n in args.n:
            if n >= 3:
                key = (n, "M0", "born", "water" if "water" in args.materials else args.materials[0], 1,
                       DEFAULT_SPACING_LAM if DEFAULT_SPACING_LAM in args.spacing else args.spacing[0])
                if key not in seen:
                    seen.add(key)
                    cells.append(key)
    if args.g1_controls:                     # 线性中继的门 G1 理想化对照（M1 + Born + 物体不受中继照射）
        for n in args.n:
            if n >= 3 and ("born" in args.modes or getattr(args, "scenario", "baseline") != "baseline"):
                for nr in sorted(set(args.n_relays) if args.design == "factorial" else {1}):
                    key = (n, "M1@M3" + NOTX, "born", "water" if "water" in args.materials else args.materials[0],
                           nr, DEFAULT_SPACING_LAM if DEFAULT_SPACING_LAM in args.spacing else args.spacing[0])
                    if key not in seen:
                        seen.add(key)
                        cells.append(key)
    # 开销大的先跑（Rapp 不动点迭代 ≈ 线性 PA 的 6–8 倍；n 大、N_r 大、full/intra），减少尾部空等
    cost = {"born": 1, "intra": 2, "full": 3}
    cells.sort(key=lambda c: (-(2 ** c[0]) * c[4] * cost[c[2]] * (8 if is_rapp(c[1]) else 1), c))
    jobs = []
    for (n, cond, mode, mat, nr, sp) in cells:
        if sp <= DIAMETER_LAM:
            raise ValueError(f"间距 {sp}λ 必须大于散射体直径 {DIAMETER_LAM:.3f}λ")
        for seed in range(args.seeds):
            jobs.append({"exp": "A", "preset": args.preset, "n": int(n), "cond": cond, "mode": mode, "material": mat,
                         "n_relays": int(nr), "spacing_lam": float(sp), "seed": int(seed),
                         "dspsa_iters": int(args.dspsa_iters), "eval_every": 100, "n_noise_eval": 20,
                         "oracle_restarts": int(args.oracle_restarts), "oracle_sweeps": int(args.oracle_sweeps),
                         "polish_sweeps": int(getattr(args, "polish_sweeps", POLISH_SWEEPS)),
                         "oracle_loss": bool(getattr(args, "oracle_loss", True)),
                         "cont": bool(args.oracle_cont), "cont_restarts": int(args.cont_restarts),
                         "cont_steps": int(args.cont_steps), "n_random": int(args.n_random),
                         "cont_plus": bool(getattr(args, "cont_plus", False)),
                         "cont_refine_steps": int(getattr(args, "cont_refine_steps", 300)),
                         "cont_init_restarts": int(getattr(args, "cont_init_restarts", 1)),
                         "dspsa_variant": getattr(args, "dspsa_variant", "auto"),
                         "scenario": getattr(args, "scenario", "baseline"), "cfg_over": dict(args.cfg_over)})
    return jobs


# ============================================================ 汇总
CELL_KEYS = ["job.n", "job.cond", "job.mode", "job.material", "job.n_relays", "job.spacing_lam"]
SHORT = {k: k.split(".", 1)[1] for k in CELL_KEYS}


def _trainers_present(rows) -> list[str]:
    """出现过的训练器（TRAINERS 顺序）+ "best"。"""
    ok = [r for r in rows if "error" not in r]
    return [tr for tr in TRAINERS if any(r.get(tr) for r in ok)] + ["best"]


def cell_metrics(trs: list[str]) -> list[str]:
    m = []
    for tr in trs:
        m += [f"{tr}.acc_clean_valid", f"{tr}.all_correct_valid", f"{tr}.solved_valid", f"{tr}.robust_valid",
              f"{tr}.margin_db", f"{tr}.acc_noisy_valid", f"{tr}.oscillating_run"]
        if tr != "best":
            m += [f"{tr}.flagged_frac", f"{tr}.pa_in_rel_p1db_db_mean", f"{tr}.pa_in_rel_p1db_db_min",
                  f"{tr}.pa_in_rel_p1db_db_max", f"{tr}.pa_in_open_rel_p1db_db_mean",
                  f"{tr}.pa_in_delta_rel_p1db_db_max", f"{tr}.time_s", f"{tr}.loss_meas_clean",
                  f"{tr}.margin_db_unflagged"]
        if tr in DSPSA_TRAINERS:
            m += [f"{tr}.plateau", f"{tr}.reach90", f"{tr}.trained"]
    m += ["dspsa.train_flagged_frac", "e_ref.L0", "e_ref.Lstar", "e_ref.Lstar_on_plateau", "sys.p_feed_dbm", "sys.backoff_db", "sys.voxel_d",
          "sys.lambda_over_voxel_d", "sys.bussgang_gain_db", "sys.bussgang_phase_deg", "run_time_s"]
    for src in ["random"] + [t for t in trs if t != "best"]:
        for f in ("obj_W2", "obj_W3", "pa_W2", "pa_W3", "uin_W3", "u_W3", "g_W3", "z_W3", "y_W2", "y_W3",
                  "ydb_W3", "pa_excess_y_W3", "K_relvar", "C_relvar"):
            m.append(f"w.{src}.{f}")
    return m


def _cell_index(table: list[dict]) -> dict:
    return {(t["n"], t["cond"], t["mode"], t["material"], t["n_relays"], t["spacing_lam"]): t for t in table}


def dspsa_trained(t: dict | None, tr: str) -> bool:
    """格点内 DSPSA（tr = dspsa / dspsa_pocket）训练到位（E 判据达标且离开 τ·ln2 平台）的 run 比例
    ≥ TRAINED_RUN_FRAC（审查 Q-2）。"""
    if not t or not t.get(f"{tr}.trained_n"):
        return False
    return bool(t[f"{tr}.trained_mean"] >= TRAINED_RUN_FRAC - 1e-12)


def gain_rows(rows: list[dict], trs: list[str], any_power_m1: bool = False) -> list[dict]:
    """按种子配对的非线性收益：Δ = (Mx) − (参考)，acc_clean_valid、all_correct_valid 与 solved_valid（自激 run → None）。"""
    ok = [r for r in rows if "error" not in r]
    labels = list(dict.fromkeys(r["cond"] for r in ok))
    idx = {(r["n"], r["cond"], r["mode"], r["material"], r["n_relays"], r["spacing_lam"], r["seed"]): r for r in ok}
    out = []
    for (lab, ref) in gain_pairs(labels, any_power_m1):
        for key, r in idx.items():
            if key[1] != lab:
                continue
            q = idx.get(key[:1] + (ref,) + key[2:])
            if q is None:
                continue
            g = {"n": key[0], "gain_of": lab, "vs": ref, "mode": key[2], "material": key[3], "n_relays": key[4],
                 "spacing_lam": key[5], "seed": key[6]}
            for tr in trs:
                a, b = r.get(tr), q.get(tr)
                g[tr] = {}
                for f, name in (("acc_clean_valid", "d_acc"), ("all_correct_valid", "d_allc"),
                                ("solved_valid", "d_solved"), ("robust_valid", "d_robust")):
                    va = None if not a else a.get(f)
                    vb = None if not b else b.get(f)
                    g[tr][name] = None if (va is None or vb is None) else float(va) - float(vb)
            out.append(g)
    return out


def _fmt(v, f=".2f"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return format(float(v), f)


def _ci(t: dict, m: str, f=".2f") -> str:
    mean, lo, hi, n = t.get(f"{m}_mean"), t.get(f"{m}_lo"), t.get(f"{m}_hi"), t.get(f"{m}_n", 0)
    if not n:
        return "–"
    if n > 1 and lo is not None and math.isfinite(lo):
        return f"{_fmt(mean, f)} [{_fmt(lo, f)},{_fmt(hi, f)}]"
    return _fmt(mean, f)


def _sci(v) -> str:
    if v is None or not math.isfinite(float(v)):
        return "–"
    v = float(v)
    if abs(v) < 1e-3 and v != 0.0:
        return f"{v:.1e}"
    return f"{v:.3f}"


def _frac(t: dict, m: str) -> str:
    """布尔比例 → "k/n"。"""
    n = t.get(f"{m}_n", 0)
    if not n:
        return "–"
    return f"{int(round(t[f'{m}_mean'] * n))}/{n}"


def _cond_rank(label: str) -> tuple:
    """条件排序：ALL_CONDITIONS 顺序在前；其余（功率扫描 M3@+20、M1@+20 …）按 (回退, M1 < M1b < Rapp, 标签)。"""
    order = {c: i for i, c in enumerate(ALL_CONDITIONS)}
    b, suf = base_label(label)
    if b in order:
        return (0, float(order[b]), 0, suf, label)
    try:
        p = parse_condition(label)
        bo, kind = condition_backoff_db(label), {"M1": 0, "M1b": 1}.get(p["condition"], 2)
    except ValueError:
        bo, kind = math.inf, 9
    return (1, bo, kind, suf, label)


def _cond_order(labels) -> list[str]:
    return sorted(dict.fromkeys(labels), key=_cond_rank)


def g1_check(rows: list[dict], trs: list[str]) -> dict:
    """门 G1 自检：M0 + Born（及 M1:noTx + Born）n ≥ 3 不得有任何训练器全对（含 accuracy / margin oracle，审查 Q-1；
    未标记 run 的 margin_db_unflagged > 0 ⇔ 全对，也计违例）；W≥3(y_lin) 必须 ≈ 0。"""
    out = {"checked": 0, "violations": [], "max_W3_y": 0.0, "groups": []}
    ok = [r for r in rows if "error" not in r and is_g1_cell(r["cond"], r["mode"], r["n"])]
    for r in ok:
        out["checked"] += 1
        for tr in trs:
            t = r.get(tr)
            if not t:
                continue
            mu = t.get("margin_db_unflagged")
            pos_margin = (not t.get("n_flagged")) and mu is not None and math.isfinite(float(mu)) and float(mu) > 0.0
            if t.get("all_correct") or pos_margin:
                out["violations"].append({"cond": r["cond"], "n": r["n"], "seed": r["seed"], "trainer": tr,
                                          "acc": t.get("acc_clean"), "margin_db": mu})
        for src, w in (r.get("w") or {}).items():
            if w and w.get("y_W3") is not None:
                out["max_W3_y"] = max(out["max_W3_y"], float(w["y_W3"]))
    out["pass"] = bool(out["checked"] > 0 and not out["violations"] and out["max_W3_y"] < 1e-20)
    best_acc = collections.defaultdict(float)
    for r in ok:
        for tr in trs:
            t = r.get(tr)
            if t and t.get("acc_clean") is not None:
                k = (r["cond"], r["n"])
                best_acc[k] = max(best_acc[k], float(t["acc_clean"]))
    out["best_acc"] = {f"{c} n={n}": a for (c, n), a in sorted(best_acc.items())}
    return out


PREDICTED = [
    ("M0 / M1 + Born 物体", "可解", "—（理论不可解）", "—（理论不可解）"),
    ("M0 / M1 + 完全耦合、高 εr、近距", "可解", "可能，裕量小", "不太可能"),
    ("M3 / M4 + Born 物体，N_r = 1", "可解", "可解，对入射功率敏感", "不可解"),
    ("M3 / M4，N_r = 2–4", "可解", "可解", "可能"),
    ("M3 / M4 + 完全耦合", "可解", "待定（RQ2）", "待定"),
]


def _drive(t: dict | None, tr: str, key: str = "pa_in_rel_p1db_db_mean", f: str = ".1f") -> str:
    """格点内训练器码本的工作点量（种子平均，dB re P1dB_in）。"""
    if not t or not t.get(f"{tr}.{key}_n"):
        return "–"
    return _fmt(t[f"{tr}.{key}_mean"], f)


def _rng3(t: dict, tr: str) -> str:
    if not t.get(f"{tr}.pa_in_rel_p1db_db_mean_n"):
        return "–"
    return (f"{_fmt(t[f'{tr}.pa_in_rel_p1db_db_mean_mean'], '.1f')} [{_fmt(t[f'{tr}.pa_in_rel_p1db_db_min_mean'], '.1f')},"
            f"{_fmt(t[f'{tr}.pa_in_rel_p1db_db_max_mean'], '.1f')}]")


def predicted_vs_observed(table: list[dict], ns: list[int]) -> str:
    """方案 §6.1 预测表 + 观察值：best 的 solved 种子数（全对且裕量 > 3σ），括号内 DSPSA pocket 的 solved 数，
    ⟨⟩ 内为 1-bit oracle 码本的实测闭环 PA 输入均值 dB re P1dB（审查 Q-3：名义条件 ≠ 实际工作点）。"""
    def obs(pred):
        cells = {}
        for t in table:
            if pred(t):
                cells.setdefault(t["n"], []).append(t)
        return cells

    def fmt_cells(ts):
        parts = []
        for t in sorted(ts, key=lambda t: (_cond_rank(t["cond"]), t["mode"], t["material"], t["n_relays"],
                                           t["spacing_lam"])):
            tag = t["cond"]
            extra = []
            if t["material"] != "water":
                extra.append(t["material"])
            if t["n_relays"] != 1:
                extra.append(f"Nr{t['n_relays']}")
            if t["spacing_lam"] != DEFAULT_SPACING_LAM:
                extra.append(f"{t['spacing_lam']}λ")
            if extra:
                tag += "(" + ",".join(extra) + ")"
            pk = _frac(t, "dspsa_pocket.solved_valid") if t.get("dspsa_pocket.solved_valid_n") else \
                _frac(t, "dspsa.solved_valid")
            drv = _drive(t, "oracle")
            parts.append(f"{tag} {_frac(t, 'best.solved_valid')} ({pk})" + ("" if drv == "–" else f" ⟨{drv}⟩"))
        return "; ".join(parts) if parts else "未运行"

    def linear(t):
        return t["cond"] == "M0" or t["cond"].startswith("M1")

    def rapp(t):
        return is_rapp(t["cond"])

    rows_def = [
        lambda t: t["mode"] == "born" and t["n_relays"] == 1 and linear(t),
        lambda t: t["mode"] == "full" and t["n_relays"] == 1 and linear(t),
        lambda t: t["mode"] == "born" and t["n_relays"] == 1 and rapp(t),
        lambda t: t["n_relays"] >= 2 and rapp(t),
        lambda t: t["mode"] == "full" and t["n_relays"] == 1 and rapp(t),
    ]
    lines = ["| 条件 | " + " | ".join(f"n = {n}（预测）" for n in ns) + " | "
             + " | ".join(f"n = {n}（观察：solved 种子数 best (DSPSA pocket) ⟨oracle 码本闭环驱动 dB re P1dB⟩）"
                          for n in ns) + " |",
             "|" + "---|" * (1 + 2 * len(ns))]
    pred_col = {2: 1, 3: 2, 4: 3}
    for (name, *pred), fn in zip(PREDICTED, rows_def):
        cells = obs(fn)
        lines.append(f"| {name} | " + " | ".join(pred[pred_col[n] - 1] if n in pred_col else "?" for n in ns) + " | "
                     + " | ".join(fmt_cells(cells.get(n, [])) for n in ns) + " |")
    intra = obs(lambda t: t["mode"] == "intra")
    if intra:
        lines.append("| （补充）intra：物体内多次散射、物体间不耦合 | " + " | ".join("—" for _ in ns) + " | "
                     + " | ".join(fmt_cells(intra.get(n, [])) for n in ns) + " |")
    return "\n".join(lines) + "\n"


def operating_point_caveat(table: list[dict]) -> str:
    """审查 Q-3：各 Rapp 条件 × 模式的实测闭环驱动（oracle / 连续 oracle 码本）与物体驱动，相对名义回退的差。"""
    groups = collections.OrderedDict()
    for t in sorted(table, key=lambda t: (ALL_MODES.index(t["mode"]) if t["mode"] in ALL_MODES else 9,
                                          _cond_rank(t["cond"]), t["n"])):
        if not is_rapp(t["cond"]) or t["n_relays"] != 1 or t["material"] != "water" \
                or t["spacing_lam"] != DEFAULT_SPACING_LAM:
            continue
        groups.setdefault((t["mode"], t["cond"]), []).append(t)
    if not groups:
        return ""
    lines = ["| 模式 | 条件 | 名义回退 dB | 闭环 |u|² rel P1dB：1-bit oracle / 连续 oracle（n 平均） | "
             "物体驱动 max|u_in−u0|²：oracle / 连续 oracle | 闭环 − 名义（oracle） |", "|---|---|---|---|---|---|"]
    for (mode, cond), ts in groups.items():
        def mean_of(tr, key):
            v = [t[f"{tr}.{key}_mean"] for t in ts if t.get(f"{tr}.{key}_n")]
            return float(np.mean(v)) if v else None
        bo = condition_backoff_db(cond)
        do, dc = mean_of("oracle", "pa_in_rel_p1db_db_mean"), mean_of("cont", "pa_in_rel_p1db_db_mean")
        oo, oc = mean_of("oracle", "pa_in_delta_rel_p1db_db_max"), mean_of("cont", "pa_in_delta_rel_p1db_db_max")
        lines.append(f"| {mode} | {cond} | {bo:+g} | {_fmt(do, '.1f')} / {_fmt(dc, '.1f')} | {_fmt(oo, '.1f')} / "
                     f"{_fmt(oc, '.1f')} | {_fmt(None if do is None else do - bo, '+.1f')} |")
    return "\n".join(lines) + "\n"


def main_table_md(table: list[dict], trs: list[str]) -> str:
    """主表（结构能力）：1-bit accuracy oracle、连续 oracle、best。solved = 全对且裕量 > 3σ_det；全对与裕量并列（审查 Q-1）。"""
    show = [tr for tr in CAPABILITY_TRAINERS if tr in trs]
    cols = ["n", "条件", "模式", "材料", "N_r", "间距/λ", "runs", "自激 run oracle / cont", "P_feed dBm",
            "闭环 PA 输入 rel P1dB：oracle / cont 均值", "λ/voxel_d"]
    for tr in show:
        cols += [f"{TRAINER_NAMES[tr]} robust", f"{TRAINER_NAMES[tr]} solved", f"{TRAINER_NAMES[tr]} 全对",
                 f"{TRAINER_NAMES[tr]} acc", f"{TRAINER_NAMES[tr]} 裕量 dB"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in table:
        osc = f"{_frac(t, 'oracle.oscillating_run')} / {_frac(t, 'cont.oscillating_run')}"
        vals = [str(t["n"]), t["cond"], t["mode"], t["material"], str(t["n_relays"]), f"{t['spacing_lam']:g}",
                str(t["n_runs"]) + (f" (err {t['n_errors']})" if t["n_errors"] else ""), osc,
                _fmt(t.get("sys.p_feed_dbm_mean"), ".1f"), f"{_drive(t, 'oracle')} / {_drive(t, 'cont')}",
                _fmt(t.get("sys.lambda_over_voxel_d_mean"), ".1f")]
        for tr in show:
            vals += [_frac(t, f"{tr}.robust_valid"), _frac(t, f"{tr}.solved_valid"),
                     _frac(t, f"{tr}.all_correct_valid"), _ci(t, f"{tr}.acc_clean_valid"),
                     _ci(t, f"{tr}.margin_db", ".3g")]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def training_table_md(table: list[dict], trs: list[str]) -> str:
    """训练器表（审查 Q-2）：DSPSA 最终迭代 / pocket（平台比例、实验 E 判据达标数）与 loss-oracle / margin-oracle（诊断）。"""
    cols = ["n", "条件", "模式", "材料", "N_r", "间距/λ", "L0 → L*（可测目标）", "L* 在平台"]
    if "dspsa" in trs:
        cols += ["DSPSA solved", "DSPSA acc", "DSPSA loss_meas_clean", "DSPSA 平台", "DSPSA 达标 (E / sign_acc: acc)",
                 "DSPSA 训练到位"]
    if "dspsa_pocket" in trs:
        cols += ["pocket solved", "pocket acc", "pocket loss_meas_clean", "pocket 达标 (E)", "pocket 训练到位"]
    if "oracle_loss" in trs:
        cols += ["loss-oracle solved", "loss-oracle acc", "loss-oracle 裕量 dB"]
    if "oracle_margin" in trs:
        cols += ["margin-oracle acc", "margin-oracle 裕量 dB"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in table:
        vals = [str(t["n"]), t["cond"], t["mode"], t["material"], str(t["n_relays"]), f"{t['spacing_lam']:g}",
                f"{_fmt(t.get('e_ref.L0_mean'), '.3f')} → {_fmt(t.get('e_ref.Lstar_mean'), '.3f')}",
                _frac(t, "e_ref.Lstar_on_plateau")]
        if "dspsa" in trs:
            vals += [_frac(t, "dspsa.solved_valid"), _ci(t, "dspsa.acc_clean_valid"),
                     _fmt(t.get("dspsa.loss_meas_clean_mean"), ".3f"), _frac(t, "dspsa.plateau"),
                     _frac(t, "dspsa.reach90"), _frac(t, "dspsa.trained")]
        if "dspsa_pocket" in trs:
            vals += [_frac(t, "dspsa_pocket.solved_valid"), _ci(t, "dspsa_pocket.acc_clean_valid"),
                     _fmt(t.get("dspsa_pocket.loss_meas_clean_mean"), ".3f"), _frac(t, "dspsa_pocket.reach90"),
                     _frac(t, "dspsa_pocket.trained")]
        if "oracle_loss" in trs:
            vals += [_frac(t, "oracle_loss.solved_valid"), _ci(t, "oracle_loss.acc_clean_valid"),
                     _ci(t, "oracle_loss.margin_db", ".3g")]
        if "oracle_margin" in trs:
            vals += [_ci(t, "oracle_margin.acc_clean_valid"), _ci(t, "oracle_margin.margin_db_unflagged", ".3g")]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def dspsa_plateau_summary(rows: list[dict]) -> dict:
    """平台诊断（审查 Q-2）：DSPSA 最终迭代 loss_meas_clean ≥ 0.9·chance 的 run 数；E 判据达标数与 solved 数（最终 / pocket）。"""
    ok = [r for r in rows if "error" not in r and r.get("dspsa")]
    out = {"n": len(ok)}
    for tr in DSPSA_TRAINERS:
        pl = [bool(r[tr]["plateau"]) for r in ok if r.get(tr) and r[tr].get("plateau") is not None]
        re_ = [bool(r[tr]["reach90"]) for r in ok if r.get(tr) and r[tr].get("reach90") is not None]
        tn = [bool(r[tr]["trained"]) for r in ok if r.get(tr) and r[tr].get("trained") is not None]
        sv = [bool(r[tr]["solved"]) for r in ok if r.get(tr)]
        out[tr] = {"plateau": (sum(pl), len(pl)), "reach90": (sum(re_), len(re_)), "trained": (sum(tn), len(tn)),
                   "solved": (sum(sv), len(sv))}
    lp = [bool(r["e_ref"]["Lstar_on_plateau"]) for r in ok if (r.get("e_ref") or {}).get("Lstar_on_plateau") is not None]
    out["Lstar_on_plateau"] = (sum(lp), len(lp))
    ora = [bool(r["oracle"]["solved"]) for r in ok if r.get("oracle")]
    out["oracle_solved"] = (sum(ora), len(ora))
    return out


def w_table_md(table: list[dict]) -> str:
    """W≥3 分解：[随机] = 随机码本；[oracle] = 1-bit accuracy oracle 码本（DSPSA 最终迭代多停在平台上，审查 Q-2）。"""
    cols = ["n", "条件", "模式", "材料", "N_r", "间距/λ", "W≥3(e) 物体 [随机]", "W≥3(g)−W≥3(u) PA [随机]",
            "W≥3(y) 探头 [随机]", "W≥3(e) [oracle]", "PA [oracle]", "W≥3(u_in) [oracle]", "W≥3(y) [oracle]",
            "W≥3(y)−W≥3(y_linref) [oracle]", "rel_var K / C [随机]", "W≥2(y) [随机]"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in table:
        g = lambda k: t.get(f"w.{k}_mean")  # noqa: E731
        kc = "–" if g("random.K_relvar") is None or not math.isfinite(g("random.K_relvar")) else \
            f"{_sci(g('random.K_relvar'))} / {_sci(g('random.C_relvar'))}"
        lines.append("| " + " | ".join([str(t["n"]), t["cond"], t["mode"], t["material"], str(t["n_relays"]),
                                         f"{t['spacing_lam']:g}", _sci(g("random.obj_W3")), _sci(g("random.pa_W3")),
                                         _sci(g("random.y_W3")), _sci(g("oracle.obj_W3")), _sci(g("oracle.pa_W3")),
                                         _sci(g("oracle.uin_W3")), _sci(g("oracle.y_W3")),
                                         _sci(g("oracle.pa_excess_y_W3")), kc, _sci(g("random.y_W2"))]) + " |")
    return "\n".join(lines) + "\n"


def annotate_gain_table(gt: list[dict], table: list[dict]) -> None:
    """给非线性收益表每行补上（审查 Q-2 / Q-3）：DSPSA / pocket 在两个格点是否都训练到位（E 判据 ≥ 80% run），
    Mx 格点的实测闭环驱动（oracle / 连续 oracle 码本）与物体驱动（oracle 码本），dB re P1dB_in。原地修改。"""
    idx = _cell_index(table)
    for g in gt:
        a = idx.get((g["n"], g["gain_of"], g["mode"], g["material"], g["n_relays"], g["spacing_lam"]))
        b = idx.get((g["n"], g["vs"], g["mode"], g["material"], g["n_relays"], g["spacing_lam"]))
        for tr in DSPSA_TRAINERS:
            g[f"{tr}.trained"] = bool(dspsa_trained(a, tr) and dspsa_trained(b, tr))
            g[f"{tr}.reach"] = f"{_frac(a, f'{tr}.trained') if a else '–'}·{_frac(b, f'{tr}.trained') if b else '–'}"
        for tr in ("oracle", "cont"):
            ok = a is not None and a.get(f"{tr}.pa_in_rel_p1db_db_mean_n")
            g[f"drive_{tr}_db"] = a[f"{tr}.pa_in_rel_p1db_db_mean_mean"] if ok else None
        ok = a is not None and a.get("oracle.pa_in_delta_rel_p1db_db_max_n")
        g["obj_drive_oracle_db"] = a["oracle.pa_in_delta_rel_p1db_db_max_mean"] if ok else None
        g["nominal_backoff_db"] = condition_backoff_db(g["gain_of"])


def gain_table_md(gt: list[dict], trs: list[str]) -> str:
    """非线性收益主表（INTEGRATION S：Δrobust = 全对且裕量 ≥ 0.5 dB 的指示差为主判据，Δsolved（> 3σ_det = 0.006 dB）与
    Δacc 并列；均值 [95% CI]）。DSPSA / pocket 列只在两个格点都训练到位（≥ 80% run）时给出，否则写
    "未训练好 (达标 a·b)"（审查 Q-2）；右侧为 Mx 的实测工作点（审查 Q-3）。"""
    show = [tr for tr in ("dspsa", "dspsa_pocket", "oracle", "oracle_loss", "cont", "best") if tr in trs]
    cols = ["n", "Mx", "参考", "模式", "材料", "N_r", "间距/λ"]
    for tr in show:
        cols += [f"Δrobust {TRAINER_NAMES[tr]}", f"Δsolved {TRAINER_NAMES[tr]}", f"Δacc {TRAINER_NAMES[tr]}"]
    cols += ["Mx 名义回退 dB", "Mx 闭环驱动 oracle / cont dB re P1dB", "Mx 物体驱动 max (oracle) dB re P1dB"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in gt:
        vals = [str(t["n"]), t["gain_of"], t["vs"], t["mode"], t["material"], str(t["n_relays"]),
                f"{t['spacing_lam']:g}"]
        for tr in show:
            if tr in DSPSA_TRAINERS and not t.get(f"{tr}.trained", False):
                vals += [f"未训练好 ({t.get(f'{tr}.reach', '–')})", "–", "–"]
            else:
                vals += [_ci(t, f"{tr}.d_robust", "+.2f"), _ci(t, f"{tr}.d_solved", "+.2f"), _ci(t, f"{tr}.d_acc", "+.2f")]
        vals += [_fmt(t.get("nominal_backoff_db"), "+g"),
                 f"{_fmt(t.get('drive_oracle_db'), '.1f')} / {_fmt(t.get('drive_cont_db'), '.1f')}",
                 _fmt(t.get("obj_drive_oracle_db"), ".1f")]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def oppoint_table_md(table: list[dict]) -> str:
    """实际工作点（审查 Q-3：含 1-bit oracle 与连续 oracle 码本的闭环 / 开环 / 物体驱动）。"""
    cols = ["n", "条件", "模式", "材料", "N_r", "backoff dB", "P_feed dBm", "voxel_d mm (λ/d)", "Bussgang |G| dB / ∠°",
            "闭环 PA 输入 rel P1dB：oracle 均值 [min,max]", "连续 oracle", "DSPSA",
            "开环 |u_in|² rel P1dB：oracle / cont / DSPSA", "物体驱动 max|u_in−u0|² rel P1dB：oracle / cont / DSPSA",
            "最终码本被标记样本比例 DSPSA / oracle / cont", "训练测量被标记比例", "自激 run DSPSA / oracle / cont"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in table:
        vd = t.get("sys.voxel_d_mean")
        bg = "–" if not t.get("sys.bussgang_gain_db_n") else \
            f"{_fmt(t['sys.bussgang_gain_db_mean'], '.2f')} / {_fmt(t['sys.bussgang_phase_deg_mean'], '.1f')}"
        lines.append("| " + " | ".join([
            str(t["n"]), t["cond"], t["mode"], t["material"], str(t["n_relays"]), _fmt(t.get("sys.backoff_db_mean"), "+.0f"),
            _fmt(t.get("sys.p_feed_dbm_mean"), ".2f"),
            "–" if vd is None else f"{vd * 1e3:.2f} ({_fmt(t.get('sys.lambda_over_voxel_d_mean'), '.1f')})", bg,
            _rng3(t, "oracle"), _rng3(t, "cont"), _rng3(t, "dspsa"),
            " / ".join(_drive(t, tr, "pa_in_open_rel_p1db_db_mean") for tr in ("oracle", "cont", "dspsa")),
            " / ".join(_drive(t, tr, "pa_in_delta_rel_p1db_db_max") for tr in ("oracle", "cont", "dspsa")),
            " / ".join(_fmt(t.get(f"{tr}.flagged_frac_mean"), ".2f") for tr in ("dspsa", "oracle", "cont")),
            _fmt(t.get("dspsa.train_flagged_frac_mean"), ".3f"),
            " / ".join(_frac(t, f"{tr}.oscillating_run") for tr in ("dspsa", "oracle", "cont"))]) + " |")
    return "\n".join(lines) + "\n"


# ============================================================ 作图
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_INK, _INK2, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(_INK2)
    ax.tick_params(colors=_INK2, labelsize=8)


def plot_ladder(table: list[dict], trs: list[str], outdir: str) -> list[str]:
    """奇偶阶梯热图：每个 (材料, N_r, 间距) 一张图；行 = 物体模式，列 = 训练器（LADDER_TRAINERS 中出现者）；
    格子 = 条件 × n，颜色 = 平均无噪声准确率（单色蓝，≤ 0.5 机会水平 → 最浅，1 → 最深），
    文字 = solved 种子数 k/n（全对且裕量 > 3σ_det，审查 Q-1）与平均准确率。"""
    trs = [tr for tr in LADDER_TRAINERS if tr in trs]
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_seq", _BLUE_RAMP)
    paths = []
    groups = collections.OrderedDict()
    for t in table:
        groups.setdefault((t["material"], t["n_relays"], t["spacing_lam"]), []).append(t)
    for (mat, nr, sp), ts in groups.items():
        modes = [m for m in ALL_MODES if any(t["mode"] == m for t in ts)]
        conds = _cond_order(t["cond"] for t in ts)
        ns = sorted({t["n"] for t in ts})
        fig, axes = plt.subplots(len(modes), len(trs), figsize=(2.3 + 1.0 * len(ns) * len(trs) + 0.6 * len(trs),
                                                                0.6 + 0.42 * len(conds) * len(modes) + 0.4 * len(modes)),
                                 squeeze=False)
        for i, mode in enumerate(modes):
            for j, tr in enumerate(trs):
                ax = axes[i][j]
                M = np.full((len(conds), len(ns)), np.nan)
                txt = [["" for _ in ns] for _ in conds]
                for t in ts:
                    if t["mode"] != mode:
                        continue
                    a, b = conds.index(t["cond"]), ns.index(t["n"])
                    m_acc = t.get(f"{tr}.acc_clean_valid_mean")
                    if t.get(f"{tr}.solved_valid_n"):
                        M[a, b] = m_acc
                        txt[a][b] = _frac(t, f"{tr}.solved_valid") + "\n" + f"{m_acc:.2f}"
                    else:
                        txt[a][b] = "自激/无"
                ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.5, vmax=1, aspect="auto")   # 0.5 = 机会水平
                for a in range(len(conds)):
                    for b in range(len(ns)):
                        v = M[a, b]
                        ax.text(b, a, txt[a][b] or "–", ha="center", va="center", fontsize=7, linespacing=1.1,
                                color="white" if (np.isfinite(v) and v > 0.78) else _INK)
                ax.set_xticks(range(len(ns)), [f"n={n}" for n in ns])
                ax.set_yticks(range(len(conds)), conds if j == 0 else [""] * len(conds))
                ax.tick_params(labelsize=8, colors=_INK2, length=0)
                for sp_ in ax.spines.values():
                    sp_.set_visible(False)
                ax.set_title(f"{mode} · {TRAINER_NAMES[tr]}", fontsize=9, color=_INK)
        fig.suptitle(f"奇偶阶梯：上 = solved 种子数（裕量 > 3σ），下 = 平均准确率（颜色 0.5→1）（{mat}, N_r={nr}, 间距 {sp:g}λ）", fontsize=10,
                     color=_INK)
        fig.tight_layout()
        p = os.path.join(outdir, f"ladder_{mat}_Nr{nr}_sp{sp:g}.png")
        paths.append(common.savefig(fig, p))
    return paths


def plot_w3(rows: list[dict], outdir: str) -> list[str]:
    """W≥d 分解柱状图：每个 (材料, N_r, 间距) 一张；子图 = n × 模式；每个条件三根柱：
    物体 W≥d(e)、PA W≥d(g)−W≥d(u)、探头 W≥d(y1−y2)；d = 3（n ≥ 3），n = 2 时 W≥3 ≡ 0，改画 d = 2。
    柱 = 随机码本（种子平均），点 = 1-bit accuracy oracle 码本（DSPSA 最终迭代多停在平台上，审查 Q-2）。对数坐标（≤ W_FLOOR 画在下限）。"""
    import matplotlib.pyplot as plt

    ok = [r for r in rows if "error" not in r]
    if not ok:
        return []
    paths = []
    groups = collections.OrderedDict()
    for r in ok:
        groups.setdefault((r["material"], r["n_relays"], r["spacing_lam"]), []).append(r)
    comps = [("obj", "物体 W≥d(e)"), ("pa", "PA W≥d(g)−W≥d(u)"), ("y", "探头 W≥d(y1−y2)")]
    for (mat, nr, sp), rs in groups.items():
        modes = [m for m in ALL_MODES if any(r["mode"] == m for r in rs)]
        ns = sorted({r["n"] for r in rs})
        conds = _cond_order(r["cond"] for r in rs)
        fig, axes = plt.subplots(len(ns), len(modes), figsize=(1.0 + 0.75 * len(conds) * len(modes), 2.6 * len(ns)),
                                 squeeze=False, sharey=True)
        width = 0.26
        for i, n in enumerate(ns):
            for j, mode in enumerate(modes):
                ax = axes[i][j]
                d = 3 if n >= 3 else 2
                for c_i, (comp, name) in enumerate(comps):
                    key = f"{comp}_W{d}"
                    xs, ys, ds = [], [], []
                    for k, cond in enumerate(conds):
                        sel = [r for r in rs if r["n"] == n and r["mode"] == mode and r["cond"] == cond]
                        vr = [r["w"]["random"][key] for r in sel if r["w"].get("random") and
                              r["w"]["random"].get(key) is not None]
                        vd = [r["w"]["oracle"][key] for r in sel if r["w"].get("oracle") and
                              r["w"]["oracle"].get(key) is not None]
                        x = k + (c_i - 1) * width
                        if vr:
                            xs.append(x)
                            ys.append(max(float(np.mean(vr)), W_FLOOR))
                        if vd:
                            ds.append((x, max(float(np.mean(vd)), W_FLOOR)))
                    ax.bar(xs, ys, width=width * 0.92, color=_SERIES[c_i], label=name if (i == 0 and j == 0) else None,
                           bottom=0)
                    if ds:
                        ax.plot([d[0] for d in ds], [d[1] for d in ds], "o", ms=5, color=_INK, mfc="white", mew=1.2,
                                label="oracle 码本" if (i == 0 and j == 0 and c_i == 0) else None, linestyle="none")
                ax.set_yscale("log")
                ax.set_ylim(W_FLOOR * 0.7, 1.5)
                ax.set_xticks(range(len(conds)), conds, rotation=40, ha="right", fontsize=7)
                ax.grid(axis="y", color=_GRID, lw=0.6)
                ax.set_axisbelow(True)
                _style_axes(ax)
                ax.set_title(f"n={n} · {mode}（W≥{d}）", fontsize=9, color=_INK)
                if j == 0:
                    ax.set_ylabel(f"W≥{d}（≤{W_FLOOR:g} 画在下限）", fontsize=8, color=_INK2)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"W≥d 分解（{mat}, N_r={nr}, 间距 {sp:g}λ；柱 = 随机码本 × 种子平均，点 = 1-bit oracle 码本）", fontsize=10,
                     color=_INK, y=1.07)
        fig.tight_layout()
        p = os.path.join(outdir, f"w3_decomposition_{mat}_Nr{nr}_sp{sp:g}.png")
        paths.append(common.savefig(fig, p))
    return paths


def _setup_fonts():
    """中文字体：委托 experiments.common.setup_cjk_fonts（审查 S0-1，导入 common 时已设置；保留本函数供旧调用方）。"""
    common.setup_cjk_fonts()


def dspsa_variant_text(variant: str) -> str:
    """summary 头里的 DSPSA 变体说明（INTEGRATION T）。"""
    if variant in ("auto", "sign_acc"):
        return ("sign_acc 变体 = 1-bit 默认：判对优先可测目标（逐单元自适应 τ）+ sign-SPSA（首步 0.1、α 0.602、A 400）"
                "+ 子集 ±½ 扰动 + pocket，输出 pocket 码本；INTEGRATION T")
    if variant == "subset":
        return "旧 subset 变体：softplus τ = 1 dB、a = auto；另报 pocket 码本"
    return f"{variant} 变体"


def cont_text(config: dict) -> str:
    if not config.get("oracle_cont"):
        return "关"
    base = "开（%d 重启 × %d 步" % (config["cont_restarts"], config["cont_steps"])
    if config.get("cont_plus"):
        return (base + f"，cont-plus：τ = {CONT_PLUS_TAU_DB:g} dB + 设计初值 {config.get('cont_init_restarts', 1)} 次 + "
                f"soft-min 精修 {config.get('cont_refine_steps', 300)} 步，{DR_DB:g} dB 动态范围受限选择）")
    return base + "，softplus τ = 1 dB）"


def scenario_text(config: dict) -> str:
    from ..system import SCENARIOS

    name = config.get("scenario", "baseline")
    over = {k: v for k, v in SCENARIOS.get(name, {}).items() if k != "preset"}
    if name == "baseline":
        return ("默认 L90 奇偶场景（ROI 0.5 m、馈源→探头视距保留）；偏置主导（方案推论 3），1-bit 对 2 位 XOR 的裕量只有 "
                "~0.01–0.06 dB，3 位奇偶在 intra/full 下无判别力")
    return (f"L90 + {over}；馈源→探头视距被吸波屏遮挡、ROI 拉近：M1 XOR 1-bit 裕量中位数 ~ +3.9 dB；3 位奇偶只有连续相位"
            "（cont-plus）在 Rapp@+20…+30 下可解，任何 1-bit 码本都解不出（探索结论）")


def summarize(rows: list[dict], outdir: str, config: dict) -> dict:
    """runs → summary.csv / gain.csv / summary.md / 图。返回 {"table", "gain", "g1", "plateau", "paths"}。"""
    _setup_fonts()
    trs = _trainers_present(rows)
    table = common.aggregate(rows, CELL_KEYS, cell_metrics(trs))
    for t in table:
        for k, s in SHORT.items():
            t[s] = t.pop(k)
    table.sort(key=lambda t: (t["material"], t["n_relays"], t["spacing_lam"], t["n"],
                              ALL_MODES.index(t["mode"]) if t["mode"] in ALL_MODES else 9, _cond_rank(t["cond"])))
    gr = gain_rows(rows, trs, any_power_m1=config.get("scenario", "baseline") != "baseline")
    gkeys = ["n", "gain_of", "vs", "mode", "material", "n_relays", "spacing_lam"]
    dm = ("d_acc", "d_allc", "d_solved", "d_robust")
    gt = common.aggregate(gr, gkeys, [f"{tr}.{m}" for tr in trs for m in dm],
                          bounds={f"{tr}.{m}": (-1.0, 1.0) for tr in trs for m in dm})
    gt.sort(key=lambda t: (t["material"], t["n_relays"], t["spacing_lam"], t["n"],
                           ALL_MODES.index(t["mode"]) if t["mode"] in ALL_MODES else 9, _cond_rank(t["gain_of"]),
                           _cond_rank(t["vs"])))
    annotate_gain_table(gt, table)
    g1 = g1_check(rows, [t for t in trs if t != "best"])
    pl = dspsa_plateau_summary(rows)
    paths = common.write_outputs(outdir, rows, table, config=config, title="实验 A：奇偶校验阶梯")
    paths["gain_csv"] = common.save_csv(gt, os.path.join(outdir, "gain.csv"))
    paths["figs"] = plot_ladder(table, trs, outdir) + plot_w3(rows, outdir)
    ns = sorted({t["n"] for t in table})
    n_err = sum(1 for r in rows if "error" in r)
    ok = [r for r in rows if "error" not in r]
    thr = next((r["oracle"]["solved_thr_db"] for r in ok if r.get("oracle")), None)
    sweep = config.get("power_sweep") or []
    md = [f"# 实验 A：奇偶校验阶梯（方案 §6.1，RQ1 + RQ2）\n",
          f"- 运行：{len(rows)} 个 run（{n_err} 个出错），{len(table)} 个格点；模式 "
          f"{'quick' if config.get('quick') else 'full'}；种子数 {config.get('seeds')}；"
          f"DSPSA {config.get('dspsa_iters')} 迭代（{dspsa_variant_text(config.get('dspsa_variant', 'auto'))}，"
          f"每迭代 2 次有噪声测量，VNA 0.002 dB）；"
          f"1-bit oracle = accuracy 目标 {config.get('oracle_restarts')} 次重启 × ≤{config.get('oracle_sweeps')} 轮"
          f" + 最小裕量 polish ≤{config.get('polish_sweeps', POLISH_SWEEPS)} 轮；"
          f"loss-oracle（诊断）{'开' if config.get('oracle_loss', True) else '关'}；"
          f"连续 oracle {cont_text(config)}；"
          f"非线性度随机码本 {config.get('n_random')} 个/种子；"
          f"功率扫描块 {('回退 ' + ', '.join(f'{float(b):+g}' for b in sweep) + ' dB') if sweep else '关'}。",
          f"- 总墙钟 {config.get('wall_s', float('nan')):.0f} s，workers = {config.get('workers')}；"
          f"单 run 中位耗时 {np.median([r['run_time_s'] for r in ok]) if ok else float('nan'):.1f} s。",
          f"- SystemConfig 覆盖（--cfg）：{config.get('cfg_over') or '无'}。",
          f"- **场景 {config.get('scenario', 'baseline')}**（system.SCENARIOS，INTEGRATION S）：{scenario_text(config)}。",
          "- 几何 L90 预设（DESIGN §1.5），κ0 = −30 dB 端口隔离（环路 κ_eff 再乘 5 dB 网络损耗，INTEGRATION B2），"
          "PA Rapp p=2 + AM-PM 12°，材料自动体素（INTEGRATION B2），Born 物体 = DWBA + volume_rr（审查 R2-INT-1）。",
          "- 准确率/全对/裕量为无噪声评估；自激 run（最终码本任一样本被标记）不参与均值（方案 §4.5），单列比例。"
          "区间 = 种子间 t 分布 95% CI；\"k/n\" = n 个有效种子中的个数。"
          f"**robust** = 全部 2ⁿ 个样本判对且最小带符号裕量 ≥ {ROBUST_MARGIN_DB:g} dB（INTEGRATION S，非线性收益主判据）；"
          f"**solved** = 全对且最小裕量 > 3σ_det = {_fmt(thr, '.3g')} dB（审查 Q-1：裕量 ≤ 3σ 的全对"
          "一次 VNA 读数就可能翻转）；全对数与裕量在主表并列。best = 每个种子各训练器（非自激）中最好者（结构能力的下界估计）。",
          "- **训练器口径（审查 Q-1 / S0-2）**：1-bit oracle 用 accuracy 目标（INTEGRATION G-A / B3 的\"最好可达准确率\"）"
          "+ 最小裕量 polish；loss-oracle 与连续 oracle 都是 softplus 损失训练的，在不可解任务上 softplus ≈ τlog2 − mean(s)/2，"
          "其 acc_clean **不是**最好可达准确率（可低于随机），只作诊断 / 连续相位参考。\n"]
    if config.get("quick"):
        md.append(f"> quick 模式：{config.get('quick_note', '')}\n")
    md.append("## 门 G1 自检（M0 + Born 3 位奇偶不可解；仿真器自检）\n")
    if g1["checked"]:
        md.append(f"- 检查 {g1['checked']} 个 run（M0 + Born，及 M1:noTx + Born 理想化线性中继，n ≥ 3；训练器含 accuracy oracle 与"
                  f" margin-oracle，后者直接最大化最小裕量，> 0 ⇔ 全对）："
                  f"**{'PASS' if g1['pass'] else 'FAIL'}**；全对 / 正裕量违例 {len(g1['violations'])} 个；"
                  f"max W≥3(y₁−y₂ 线性功率) = {g1['max_W3_y']:.1e}（随机 + 训练后码本，理论 0）。")
        md.append("- 各条件最好的无噪声准确率（任一训练器）：" + "，".join(f"{k}: {v:.3f}" for k, v in g1["best_acc"].items())
                  + "（3 位奇偶在二次阈值函数下至多 7/8 = 0.875）。")
        if g1["violations"]:
            md.append("- **违例（应作为 bug 排查）**：" + str(g1["violations"]))
        md.append("")
    else:
        md.append("- 本次网格不含 M0 + Born 且 n ≥ 3，未检查。\n")
    md.append("## DSPSA 训练诊断（审查 Q-2 / S1-0；INTEGRATION T）\n")
    if config.get("dspsa_variant", "auto") in ("auto", "sign_acc"):
        md.append(f"- 默认 sign_acc 变体优化判对优先目标而不是 softplus 损失：\"达标 / 训练到位\" 按准确率判据 "
                  f"acc ≥ {REACH_FRAC:g}·acc*（acc* = 同种子 1-bit accuracy oracle，同 bench_trainers），平台（loss ≥ "
                  f"{PLATEAU_FRAC:g}·τ·ln2）只作诊断；sign_acc 的输出就是 pocket 码本，不再单列 pocket。")
    sign = config.get("dspsa_variant", "auto") in ("auto", "sign_acc")
    if pl["n"] and sign:
        f = lambda kv: f"{kv[0]}/{kv[1]}"  # noqa: E731
        md.append(f"- sign_acc 输出码本达到准确率判据（acc ≥ {REACH_FRAC:g}·acc*）= 训练到位：{f(pl['dspsa']['trained'])}；"
                  f"solved：DSPSA {f(pl['dspsa']['solved'])}，1-bit oracle {f(pl['oracle_solved'])}；"
                  f"（诊断）softplus 可测损失 ≥ {PLATEAU_FRAC:g}·τ·ln2：{f(pl['dspsa']['plateau'])}——判决量 ≪ τ = 1 dB 时"
                  "即使全对也会落在这里，不再作训练失败的判据。")
        md.append(f"- 规则：只有两个配对格点都有 ≥ {TRAINED_RUN_FRAC:.0%} 的 run 训练到位时，才报告基于 DSPSA 的非线性收益；"
                  "否则收益表写\"未训练好\"。DSPSA 只能找 1-bit 码本：1-bit 结构本身解不出的格点（oracle 也解不出）上它同样解不出。\n")
    elif pl["n"]:
        f = lambda kv: f"{kv[0]}/{kv[1]}"  # noqa: E731
        md.append(f"- 最终迭代停在平台（loss_meas_clean ≥ {PLATEAU_FRAC:g}·τ·ln2）：{f(pl['dspsa']['plateau'])}；"
                  f"达到实验 E 判据（(L0−L) ≥ {REACH_FRAC:g}(L0−L*)，L* = 1-bit oracle 可测目标）：最终迭代 "
                  f"{f(pl['dspsa']['reach90'])}，pocket {f(pl['dspsa_pocket']['reach90'])}；"
                  f"训练到位（E 达标且离开平台）：最终迭代 {f(pl['dspsa']['trained'])}，"
                  f"pocket {f(pl['dspsa_pocket']['trained'])}；"
                  f"solved：最终迭代 {f(pl['dspsa']['solved'])}，pocket {f(pl['dspsa_pocket']['solved'])}，"
                  f"1-bit oracle {f(pl['oracle_solved'])}。")
        md.append(f"- softplus 的 L* 本身落在平台上的 run：{f(pl['Lstar_on_plateau'])}——这些格点里 E 判据只说明"
                  "\"和 loss-oracle 一样停在 τ·ln2\"，所以\"训练到位\"另要求离开平台。")
        md.append(f"- 规则：只有两个配对格点都有 ≥ {TRAINED_RUN_FRAC:.0%} 的 run 训练到位时，才报告基于 DSPSA（或 pocket）的非线性收益；"
                  "否则收益表写\"未训练好\"——DSPSA 列反映的是训练失败（τ·ln2 平台），不是结构能力，不能据此回答 RQ1/RQ2。\n")
    md.append(training_table_md(table, trs))
    md.append("\n## 方案 §6.1 预测表 vs 观察\n")
    md.append("观察值：各格点有效种子中 solved（全对且裕量 > 3σ）的个数，best（括号内 DSPSA pocket）；⟨⟩ = 1-bit oracle 码本的"
              "实测闭环 PA 输入均值（dB re P1dB_in）。"
              "注意本仿真里中继条件的物体处在中继发射波束中（objects_see_relay_tx=True，审查 INT-PHYS-1），"
              "M1 + Born 不再是严格二次（环路 K(x) 与发射路径 C(x) 依赖 x，方案推论 4）；严格的线性中继对照是 M1@M3:noTx。\n")
    md.append(predicted_vs_observed(table, ns))
    cav = operating_point_caveat(table)
    if cav:
        md.append("\n**工作点注意（审查 Q-3）**：源功率按空场景聚焦码本开环标定（INTEGRATION B），训练后的码本把背景偏置 u0 零陷，"
                  "所以 intra/full 下名义 M3（≈ P1dB）/ M4（+10 dB）的实际闭环驱动低于名义值，物体引起的驱动 |u_in − u0|² 更是"
                  "比 P1dB 低 20–37 dB（方案推论 3）：名义\"深饱和\"并不压缩物体相关部分。下表为各条件的实测值"
                  "（n 平均；功率扫描块 M3@+20 / +30 等把源功率再抬高，见非线性收益表右侧列）。\n")
        md.append(cav)
    md.append("\n## 主表（结构能力：1-bit accuracy oracle / 连续 oracle / best）\n")
    md.append(main_table_md(table, trs))
    md.append("\n## 非线性收益（方案 §7 主结论：Rapp 条件 − 同功率 M1 / M1b，按种子配对；主判据 Δrobust）\n")
    md.append(f"**Δrobust** = robust 指示差（全对且最小裕量 ≥ {ROBUST_MARGIN_DB:g} dB，INTEGRATION S），Δsolved = solved（> 3σ_det）"
              "指示差，Δacc = 无噪声准确率差；均值 [95% CI]（种子间 t 分布；配对 = 同种子同初始码本）。"
              + ("没有同回退 M1 的 Rapp 条件与 M1（0 dB）配对：线性中继的无噪声判决量与源功率无关；M1b 只与同回退配对。"
                 if config.get("scenario", "baseline") != "baseline" else "") +
              "CI 不含 0 才能说 PA 非线性本身有用。DSPSA / pocket 列只在两个格点都训练到位时给出（审查 Q-2），"
              "\"未训练好 (a·b)\" 为 Mx·参考 格点中训练到位（E 判据达标且离开平台）的 run 数。右侧 = Mx 码本的实测工作点（审查 Q-3）。"
              "全对指示差 d_allc 见 gain.csv。\n")
    md.append(gain_table_md(gt, trs) if gt else "_(本网格没有可配对的 Mx / M1@Mx)_\n")
    md.append("\n## W≥3 分解（方案 §7；物体贡献 = W≥3(e)，PA 贡献 = W≥3(g) − W≥3(u)）\n")
    md.append("[随机] = 每种子 5 个随机码本的平均再对种子平均；[oracle] = 1-bit accuracy oracle 码本。y = y₁ − y₂（线性功率）。"
              "W≥3(y) − W≥3(y_linref)：同码本、同 K(x)/C(x)、增益匹配线性 PA 的参考（PA 非线性本身对探头读数的高阶贡献）。"
              "rel_var K / C = 环路耦合与发射→探头系数对 x 的相对依赖（> 0 即物体在中继发射波束/环路里）。"
              "n = 2 时 W≥3 ≡ 0（最高 2 阶），看 W≥2。\n")
    md.append(w_table_md(table))
    md.append("\n## 实际工作点（PA 输入回退、源功率、体素、标记/自激比例）\n")
    md.append("closed-loop |u|² 相对 P1dB_in（未标记样本；种子平均的 均值 [min,max]）；开环 |u_in|² 与物体驱动 |u_in − u0|² "
              "说明背景偏置 u0 占主导的程度（方案推论 3）。1-bit oracle / 连续 oracle 码本是结论所依据的码本（审查 Q-3）。\n")
    md.append(oppoint_table_md(table))
    md.append("\n## 图\n")
    for p in paths["figs"]:
        md.append(f"- `{os.path.basename(p)}`")
    md.append("")
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return {"table": table, "gain": gt, "g1": g1, "plateau": pl, "paths": paths}


# ============================================================ CLI
def parse_args(argv=None):
    p = common.base_argparser("实验 A：奇偶校验阶梯（方案 §6.1）")
    p.add_argument("--n", type=int, nargs="+", default=None, help="奇偶位数（默认 2 3 4；quick 2 3）")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="条件标签，例如 M0 M1@M3 M1@M4 M1b@M4 M2 M3 M4（后缀 :noTx = 物体不受中继照射）")
    p.add_argument("--modes", nargs="+", default=None, choices=list(ALL_MODES), help="物体模式")
    p.add_argument("--materials", nargs="+", default=None, choices=list(ALL_MATERIALS) + ["metal"],
                   help="材料（默认 full 四种介质 / quick 水）；metal 只能显式给出（intra/full 下为 scalar_rr 软球极限，"
                        "born 下是 εr_eff = 4 的介质，见 materials）")
    p.add_argument("--n-relays", type=int, nargs="+", default=None, help="中继数 N_r（默认 1 2 4；quick 1）")
    p.add_argument("--spacing", type=float, nargs="+", default=None, help="槽间距（单位 λ，默认 0.5）")
    p.add_argument("--preset", default="L90")
    p.add_argument("--scenario", default=None, choices=["baseline", "informative", "informative_probe"],
                   help="命名场景（system.SCENARIOS，INTEGRATION S）；默认 quick → informative，full → baseline")
    p.add_argument("--dspsa-variant", default="auto", choices=["auto", "sign_acc", "subset", "faithful"],
                   help="DSPSA 变体（默认 auto = 1-bit sign_acc，INTEGRATION T；subset = 旧默认）")
    p.add_argument("--cont-plus", dest="cont_plus", action="store_true", default=None,
                   help="连续 oracle 加设计初值 + soft-min 精修（informative 场景默认开，INTEGRATION S）")
    p.add_argument("--no-cont-plus", dest="cont_plus", action="store_false")
    p.add_argument("--cont-refine-steps", type=int, default=300, help="cont-plus 的 soft-min 精修步数")
    p.add_argument("--cont-init-restarts", type=int, default=None,
                   help=f"cont-plus 的设计初值重启次数（默认 {CONT_PLUS_INIT_RESTARTS}；单次成功率仅 ~20–60%%，见 INTEGRATION S）")
    p.add_argument("--design", choices=("blocks", "factorial"), default=None,
                   help="blocks（full 默认）：主块 n×条件×模式（水、N_r=1）+ RQ2 材料/间距块 + N_r 块；factorial：全交叉")
    p.add_argument("--dspsa-iters", type=int, default=None, help="DSPSA 迭代数（quick 1500 / full 4000）")
    p.add_argument("--oracle-restarts", type=int, default=8)
    p.add_argument("--oracle-sweeps", type=int, default=30)
    p.add_argument("--oracle-cont", dest="oracle_cont", action="store_true", default=None)
    p.add_argument("--no-oracle-cont", dest="oracle_cont", action="store_false")
    p.add_argument("--cont-restarts", type=int, default=None, help="连续 oracle 重启数（quick 2 / full 4）")
    p.add_argument("--cont-steps", type=int, default=None, help="连续 oracle Adam 步数（quick 400 / full 800）")
    p.add_argument("--n-random", type=int, default=5, help="非线性度随机码本数")
    p.add_argument("--no-g1-controls", dest="g1_controls", action="store_false", default=True,
                   help="不加 M1@M3:noTx + Born 的线性中继门 G1 对照")
    p.add_argument("--polish-sweeps", type=int, default=POLISH_SWEEPS,
                   help="1-bit accuracy oracle 之后的最小裕量 polish 轮数（0 = 关；审查 Q-1）")
    p.add_argument("--oracle-loss", dest="oracle_loss", action="store_true", default=None,
                   help="另跑默认 softplus 损失 oracle 作诊断（审查 S0-2；full 默认开、quick 默认关）")
    p.add_argument("--no-oracle-loss", dest="oracle_loss", action="store_false")
    p.add_argument("--power-sweep", type=float, nargs="+", default=None, metavar="DB",
                   help="功率扫描块的回退 dB（审查 Q-3；n ≥ 3 × intra/full，M3@DB 配 M1@DB）；"
                        f"默认 full {list(POWER_SWEEP_FULL)} / quick {list(POWER_SWEEP_QUICK)}")
    p.add_argument("--no-power-sweep", dest="power_sweep", action="store_const", const=[])
    p.add_argument("--power-sweep-m1b", action="store_true", default=None,
                   help="功率扫描块另加 M1b@DB（Bussgang 对照；full 默认开）")
    p.add_argument("--cfg", nargs="+", default=[], metavar="KEY=VALUE",
                   help="SystemConfig 字段覆盖（全部 job 共用），例如 --cfg bg_cancel=null cal_ref=object")
    p.add_argument("--resummarize", type=str, default=None, help="只对已有输出目录的 runs.jsonl 重新汇总/作图")
    a = p.parse_args(argv)
    q = a.quick
    a.scenario = a.scenario or ("informative" if q else "baseline")
    a.cfg_over = {**scenario_overrides(a.scenario), **parse_cfg_overrides(a.cfg)}
    info = a.scenario != "baseline"
    a.n = a.n or ([2, 3] if q else [2, 3, 4])
    a.conditions = a.conditions or list(INFORMATIVE_QUICK_CONDITIONS if (q and info) else
                                        (QUICK_CONDITIONS if q else ALL_CONDITIONS))
    for c in a.conditions:
        parse_condition(c)
    a.modes = a.modes or (["full"] if (q and info) else list(("born", "intra", "full")))
    if a.cont_plus is None:
        a.cont_plus = info
    a.cont_init_restarts = CONT_PLUS_INIT_RESTARTS if a.cont_init_restarts is None else a.cont_init_restarts
    a.materials = a.materials or (["water"] if q else list(ALL_MATERIALS))
    a.n_relays = a.n_relays or ([1] if q else list(NR_LEVELS))
    a.spacing = a.spacing or [DEFAULT_SPACING_LAM]
    a.design = a.design or ("factorial" if q else "blocks")
    a.dspsa_iters = a.dspsa_iters or (1500 if q else 4000)
    if a.oracle_cont is None:
        a.oracle_cont = True
    if a.cont_plus:                     # cont-plus：随机重启几乎从不解出 3 位奇偶，预算给设计初值重启（INTEGRATION S）
        a.cont_restarts = a.cont_restarts or 1
        a.cont_steps = a.cont_steps or 600
    a.cont_restarts = a.cont_restarts or (2 if q else 4)
    a.cont_steps = a.cont_steps or (300 if q else 800)
    if a.oracle_loss is None:
        a.oracle_loss = not q            # quick 关（每个 Rapp run 省 ~7 s，保持 ≤ 10 分钟）；E 判据的 L* 则只取 1-bit oracle
    if a.power_sweep is None:           # informative quick 的条件集已含 Rapp@+20…+30，不再加功率扫描块
        a.power_sweep = [] if (q and info) else list(POWER_SWEEP_QUICK if q else POWER_SWEEP_FULL)
    if a.power_sweep_m1b is None:
        a.power_sweep_m1b = not q
    for b in a.power_sweep:
        parse_condition(f"M3@{float(b):+g}")
    return a


def main(argv=None) -> str:
    args = parse_args(argv)
    # 以规范模块名导入 job 函数：作为 __main__ 运行时，spawn 子进程按 "ris_nlsim.experiments.exp_a_parity" 反序列化
    from ris_nlsim.experiments.exp_a_parity import run_job as job_fn

    if args.resummarize:
        rows = common.load_jsonl(os.path.join(args.resummarize, "runs.jsonl"))
        import json
        with open(os.path.join(args.resummarize, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        res = summarize(rows, args.resummarize, config)
        print(f"重新汇总完成：{args.resummarize}（G1 {'PASS' if res['g1']['pass'] else 'FAIL/未检查'}）")
        return args.resummarize
    outdir = common.make_outdir(EXP, args.out)
    jobs = build_jobs(args)
    config = {k: v for k, v in vars(args).items() if k != "resummarize"}
    config.update(exp="A", n_jobs=len(jobs), started=_dt.datetime.now().isoformat(timespec="seconds"),
                  python=sys.version.split()[0],
                  quick_note=("场景 %s；n∈{2,3}，水，N_r=1，间距 λ/2，%d 个种子，DSPSA %d 迭代；物体模式 %s"
                              "（baseline 场景按 experiments.common.EXP_A_OBJECT_MODES 含 intra 作为 RQ2 强度匹配对照；"
                              "informative 场景只跑 full + 门 G1 的 born 格点）；连续 oracle %s；功率扫描块 %s；loss-oracle 诊断%s。"
                              % (args.scenario, args.seeds, args.dspsa_iters, "/".join(args.modes),
                                 ("开（%d 重启 × %d 步，少于 full 的 4 × 800%s）" % (
                                     args.cont_restarts, args.cont_steps,
                                     "；cont-plus 设计初值 + soft-min 精修" if args.cont_plus else ""))
                                 if args.oracle_cont else "关",
                                 ("回退 " + ", ".join(f"{float(b):+g}" for b in args.power_sweep) + " dB（n=3，intra/full，"
                                  "M3@ 配 M1@）") if args.power_sweep else "关",
                                 "开" if args.oracle_loss else "关（--oracle-loss 打开；E 判据的 L* 只取 1-bit accuracy oracle）")))
    print(f"实验 A：{len(jobs)} 个 run → {outdir}（workers={args.workers}）", flush=True)
    t0 = time.perf_counter()
    rows = common.run_jobs(job_fn, jobs, workers=args.workers, progress=True)
    config["wall_s"] = time.perf_counter() - t0
    res = summarize(rows, outdir, config)
    print(f"完成：{config['wall_s']:.0f} s；G1 自检 {'PASS' if res['g1']['pass'] else 'FAIL/未检查'}；输出 {outdir}",
          flush=True)
    return outdir


if __name__ == "__main__":
    main()
