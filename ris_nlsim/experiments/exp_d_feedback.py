"""实验 D：反馈环（RQ5；方案 §2 推论 4、§3 RQ5、§4.5、§6.2 表"D 反馈环"；INTEGRATION G "D"、G2、B3）。

假设（方案 RQ5）：环路增益接近 1 时类别间距变大，但噪声鲁棒性和稳定性变差；预测：类间距在 |Gκ| → 1 时增大，
M4 自激比例明显低于 M1（PA 饱和把自激钳成有限幅度的稳态）。

设置：flat0 预设（ψ = 0，收发共面；物体在 ROI = 中继中心 + (d, 0, 0)，同时处于收、发波束中 → **物体在放大环路里**），
3 位奇偶（水球 λ/3，自动体素，full 耦合）。网格：端口隔离 isolation_db × ROI 距离 d × 条件（M1 / M3 / M4，同一 P_feed
标定规则：M1 = 小信号增益、与 M3 同源功率）× 种子。κ0 相位：默认每个种子随机 U[0°, 360°)（同一种子在所有格点共用，
公共随机数），也可用 --kappa-phases 给定一组值作为网格因素。
环路里的泄漏 κ_eff = κ0_port·10^{−(loss_rx+loss_tx)/20}（INTEGRATION B2，RelayParams.kappa0）；名义小信号环路增益
|G κ_eff| (dB) = G_PA + isolation − loss_rx − loss_tx（system.bare_loop_gain_db）。

隔离网格按名义环路增益取点（审查 R-D2，INTEGRATION G "D" 注）：默认 |Gκ| nom = −15 / −3 / −1 / +1 / +10 dB
（system.isolation_for_loop_gain_db：默认参数下隔离 −40 / −28 / −26 / −24 / −15 dB）。方案原网格的 −25 dB 恰好
|G κ_eff| = 1：无物体样本 x = 000 的 K = κ_eff 与码本无关，ρ_ss(x=000) = 1，线性中继按谐振容差（ρ ≥ 1 − 1e-9）
**按构造必被标记**——它不是 RQ5 的证据。默认网格用 ±1 dB 两个偏离临界的点代替它；若用 --isolations 显式加入，
该行标记"临界（按构造）"（CRITICAL_TOL_DB 内），不进入配对差分的判读与趋势相关。

自激判据（审查 R-D1，INTEGRATION B3，relay.OSC_CRITERIA）：系统按 "cw"（~converged | ρ_op ≥ 1，DESIGN §3.4）训练与
评估；另报 "robust"（再或上 ρ_rob = ‖K‖₂·max(|a|+|b|) ≥ 1：任意泄漏相位 / 带内频率不起振的充分条件，R = 1 时即最坏
泄漏相位下的 ρ_op）与 "small_signal"（再或上 ρ_ss = ρ(G_v K) ≥ 1：源关闭即起振，最保守）。
注意：线性中继（M1）ρ_op = ρ_ss，ρ_ss ≥ 1 即被标记 → 名义 |Gκ| > 0 dB 时 M1 按定义 100% 自激（x = 000 的
ρ_ss = |Gκ_eff| > 1）；small_signal 判据下 M3/M4 在这些点同样按定义 100%。所以超临界点上"M4 自激比例低于 M1"
只在 robust 判据下是非平凡的结论（CW 判据会把只靠 0.01–0.04 dB 压缩维持的工作点也判为稳定，审查 R-D1）。

每个 run（一个格点的一个种子）：
  1. 训练无关的环路诊断：n_rand 个随机码本（同一种子在所有格点共用）上三种判据的自激比例（任一样本被标记）、
     被标记样本比例、小信号环路增益 ρ(G_v K)、物体灵敏度 rand_sens_db（未标记码本上 RMS_x[Δy(x) − Δy(000)]，Δy = y1 − y2，dB）。
  2. DSPSA（train.run_dspsa 默认变体：1-bit sign_acc = 判对优先可测目标 + sign-SPSA，被标记样本算错，输出 pocket 码本，
     INTEGRATION T；旧 subset 变体按 chance + osc_penalty 计）→ 训练中被标记测量的比例
     flagged_meas_frac（三种判据，_CriteriaCounter 统计每次测量；"中间码本自激比例"）。
  3. 1-bit oracle（train.run_oracle_bits(objective="accuracy")，审查 R-D3）：旧 subset DSPSA 400 次迭代在本任务上停在 y1 ≈ y2
     的平台（acc ≈ 0.5、裕量全负），其码本的类间距是噪声；"结构最多能做到多好"的码本才能回答"类间距是否随 |Gκ| → 1 增大"。
     oracle 码本的 orc_acc / orc_sep_class_db / orc_margin 是趋势相关的主数据；DSPSA 码本的趋势只作参考（判读里自动注明
     是否在 chance 附近）。
  4. 最终码本 evaluate：oscillating_run（cw）/ osc_robust_run / osc_ss_run、准确率（*_valid 剔除 cw 自激 run，方案 §4.5）、
     裕量 margin、类间距 sep_class_db = mean(Δy | 奇) − mean(Δy | 偶)（未标记样本）、环路诊断（INTEGRATION G2）：
     ρ_ss、ρ_op、ρ_rob、Gershgorin 上界、|det(I − G_v K)| 的最小值，物体对环路的调制 gk_obj_db = 20log10 max_x |G_v(K(x) − K(000))|
     与 1 − G_v K 的最小模（谐振距离）。
实际工作点：pa_in_rel_p1db_db（闭环 |u|²，未标记样本）min/mean/max、P_feed、体素尺寸（每行都记录）。
  5. κ0 相位扫描（训练无关，相对 INTEGRATION G 的增补）：每个 (距离, 隔离, 条件) 在 κ0 相位 0°, 45°, …, 315° 上用同一组
     随机码本做 codebook_diagnostics：奇偶判决量 Δy 对物体的灵敏度、物体对 PA 输入的相对调制 rel_du、中继支路 / 直达支路之比、
     谐振增强 enh = −20log10 min|1 − G_v K|、三种判据的自激比例。|Gκ| → 1 只有在同相（G_v K ≈ 1）时才是谐振；随机相位的训练 run
     很少落在谐振附近，扫描把相位依赖直接暴露出来。
CW 稳态近似（方案 §4.5）：被标记 = 不收敛或 ρ_op ≥ 1，只是"可能自激"的必要条件，不是时域预测。
场景（--scenario，system.SCENARIOS / apply_scenario，INTEGRATION S；默认 baseline）：几何始终是 flat0 + ROI 距离网格，
只沿用场景里的直达路径开关 D_SCENARIO_FIELDS（block_direct / skip_gain_db）。L90 场景的 ROI 偏移（0.35 m 相对接收板中心）
在 flat0 下由 --distances 网格代替，探头方位覆盖（informative_probe 的 45° / 75°）是 L90 几何专用，都不沿用
→ informative_probe 在 D 里等同 informative。实测（flat0，d = 0.25 m，M3）：block_direct 把馈源→探头视距项置 0，
探头读数与随机码本灵敏度变化 ~0.1–0.7 dB，环路诊断（ρ_ss、ρ_rob、自激比例、rel_du）不变——视距项不进环路，只改读出偏置。

用法：python -m ris_nlsim.experiments.exp_d_feedback [--quick] [--workers N] [--seeds S] [--out DIR]
      [--isolations ...] [--distances 0.25 0.4 0.8] [--conditions M1 M3 M4] [--kappa-phases ...] [--n-iter N]
      [--scenario baseline|informative|informative_probe]
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from . import common

EXP = "exp_d_feedback"
ALL_CONDITIONS = ("M1", "M3", "M4")
DEFAULT_LOOP_GAINS_DB = (-15.0, -3.0, -1.0, 1.0, 10.0)   # 名义 |Gκ_eff| (dB)：避开 0 dB（审查 R-D2）
FULL_DISTANCES = (0.25, 0.4, 0.8)
QUICK_DISTANCES = (0.25, 0.8)
N_BITS = 3
SCAN_PHASES = tuple(float(v) for v in range(0, 360, 45))   # κ0 相位扫描（训练无关的随机码本诊断）
CRITICAL_TOL_DB = 0.25                 # |名义环路增益| < 它 → "临界（按构造）"：x = 000 的 ρ_ss ≈ 1
CRITERIA = ("cw", "robust", "small_signal")
CRIT_SUFFIX = {"cw": "", "robust": "_robust", "small_signal": "_ss"}
CRIT_SHORT = {"cw": "cw", "robust": "rob", "small_signal": "ss"}
CHANCE_ACC = 0.5
CHANCE_BAND = 0.15                     # 平均准确率 < chance + 它 且全对比例 0 → 码本"在 chance 附近"
D_SCENARIO_FIELDS = ("block_direct", "skip_gain_db")   # 场景里 D 沿用的字段（直达路径开关；几何保持 flat0 + 距离网格）


def parse_condition(label: str) -> dict:
    """条件标签 → SystemConfig 字段（"M1@M4" → M1 + M4 的回退；其余按条件表）。"""
    from ..system import CONDITION_BACKOFF_DB, CONDITIONS

    lab = str(label).strip()
    if "@" in lab:
        base, ref = lab.split("@", 1)
        if base not in ("M1", "M1b") or ref not in CONDITION_BACKOFF_DB:
            raise ValueError(f"无法解析条件 {label!r}")
        return {"condition": base, "backoff_db": float(CONDITION_BACKOFF_DB[ref])}
    if lab not in CONDITIONS:
        raise ValueError(f"未知条件 {label!r}")
    return {"condition": lab}


def kappa_phase_for_seed(seed: int) -> float:
    """每个种子的 κ0 相位（度）：U[0, 360)，只依赖种子（所有格点、条件共用）。"""
    return float(common.seed_rng(EXP, "kappa_phase", int(seed)).uniform(0.0, 360.0))


def nominal_loop_gain_db(cfg) -> float:
    """名义小信号环路增益 |G_v κ_eff| (dB) = G_PA + isolation − loss_rx − loss_tx（不含经场景的耦合）。"""
    from ..system import bare_loop_gain_db

    return bare_loop_gain_db(cfg)


def default_isolations(loop_gains_db=DEFAULT_LOOP_GAINS_DB) -> list[float]:
    """默认隔离网格：按名义环路增益取点（system.isolation_for_loop_gain_db，默认 SystemConfig 的 PA 增益与网络损耗）。"""
    from ..system import SystemConfig, isolation_for_loop_gain_db

    cfg = SystemConfig(preset="flat0")
    return [round(isolation_for_loop_gain_db(cfg, float(g)), 6) for g in loop_gains_db]


def is_critical(gk_nominal_db) -> bool:
    """名义环路增益在 0 dB 的 ±CRITICAL_TOL_DB 内：x = 000 的 K = κ_eff 使 ρ_ss ≈ 1，线性中继按构造被标记（审查 R-D2）。"""
    return gk_nominal_db is not None and abs(float(gk_nominal_db)) < CRITICAL_TOL_DB


def regime(gk_nominal_db) -> str:
    """名义环路增益区间："critical"（临界，按构造）/ "sub"（< 0 dB）/ "super"（> 0 dB）。"""
    if is_critical(gk_nominal_db):
        return "critical"
    return "sub" if float(gk_nominal_db) < 0 else "super"


# ============================================================ 自激判据
def criterion_flags(m) -> dict:
    """一次测量 m 在三种判据下的逐样本标记（审查 R-D1，relay.OSC_CRITERIA）：
    cw = m.flags（系统按 osc_criterion="cw" 构造）；robust = cw | ρ_rob ≥ 1；small_signal = cw | ρ_ss ≥ 1。"""
    cw = np.asarray(m.flags, dtype=bool)
    rob = cw if m.robust_unstable is None else cw | np.asarray(m.robust_unstable, dtype=bool)
    ss = cw if m.ss_unstable is None else cw | np.asarray(m.ss_unstable, dtype=bool)
    return {"cw": cw, "robust": rob, "small_signal": ss}


class _CriteriaCounter:
    """PhysicalSystem 的转发代理：统计 DSPSA 每次测量（noisy=True 的 forward）在三种判据下是否有样本被标记。

    train.run_dspsa 只按系统的 osc_criterion（cw）计 n_flagged_measurements；本代理只拦截 forward，其余属性 / 方法
    原样转发到被代理的系统（evaluate / clean_metrics 内部调用的是原系统的 forward，不计入）。"""

    def __init__(self, system):
        self._sys = system
        self.counts = {"n": 0, **{c: 0 for c in CRITERIA}}

    def __getattr__(self, name):
        return getattr(self._sys, name)

    def forward(self, phases, rng=None, noisy=True, sample_idx=None):
        m = self._sys.forward(phases, rng, noisy=noisy, sample_idx=sample_idx)
        if noisy:
            self.counts["n"] += 1
            for c, f in criterion_flags(m).items():
                self.counts[c] += int(f.any())
        return m


# ============================================================ 诊断
def class_separation(system, readings: np.ndarray, flags: np.ndarray) -> dict:
    """奇偶的类间距（未标记样本）：sep_class_db = mean(Δy|奇) − mean(Δy|偶)，Δy = y1 − y2；
    sep_signed_mean_db = mean sgn·Δy；sep_abs_mean_db = mean |Δy|。任一类没有未标记样本时为 None。"""
    ok = ~np.asarray(flags, dtype=bool)
    dy = readings[:, 0] - readings[:, 1]
    lab = np.asarray(system.labels)
    odd, even = ok & (lab == 1), ok & (lab == 0)
    out = {"sep_class_db": None, "sep_signed_mean_db": None, "sep_abs_mean_db": None}
    if odd.any() and even.any():
        out["sep_class_db"] = float(dy[odd].mean() - dy[even].mean())
    if ok.any():
        sgn = np.where(lab[ok] == 1, 1.0, -1.0)
        out["sep_signed_mean_db"] = float(np.mean(sgn * dy[ok]))
        out["sep_abs_mean_db"] = float(np.mean(np.abs(dy[ok])))
    return out


def loop_extras(system, m) -> dict:
    """单中继环路的补充诊断：每样本 ρ_ss/ρ_op 的中位数、物体对环路的调制 |G(K(x) − K(000))|、谐振距离 min|1 − G K|。"""
    out = {"loop_gain_ss_med": None, "loop_gain_op_med": None, "gk_obj_db": None, "one_minus_gk_min": None}
    if system.is_passive or m.K is None:
        return out
    G = system.small_signal_gain()
    GK = G * m.K                                          # (B,R,R)
    ss = np.asarray(m.loop_gain_ss, dtype=float)
    op = np.asarray(m.loop_gain_op, dtype=float)
    out["loop_gain_ss_med"] = common.to_jsonable(float(np.nanmedian(ss)))
    out["loop_gain_op_med"] = common.to_jsonable(float(np.nanmedian(op))) if np.isfinite(op).any() else None
    x0 = _x0(system)
    d = np.abs(GK - GK[x0][None]).reshape(GK.shape[0], -1).max(axis=1)
    d = np.delete(d, x0)
    out["gk_obj_db"] = float(20.0 * np.log10(max(float(d.max()), 1e-300)))
    eye = np.eye(GK.shape[-1])
    smin = np.linalg.svd(eye[None] - GK, compute_uv=False)[..., -1]
    out["one_minus_gk_min"] = float(np.min(smin))
    return out


def _x0(system) -> int:
    return int(np.nonzero([len(s.objects) == 0 for s in system.task.samples])[0][0])     # x = 000（无物体）


def codebook_diagnostics(system, m) -> dict:
    """单个码本（无噪声测量 m）的环路 / 灵敏度诊断（中继条件）：
      flags_any / robust_any / ss_any（三种判据下任一样本被标记）、flag_frac / flag_frac_robust / flag_frac_ss、
      rho_ss_max = max_x ρ(G_v K(x))、rho_rob_max = max_x ρ_rob、enh_db = −20log10 min_x σ_min(I − G_v K(x))（线性环路对 u 的
      放大；|Gκ| → 1 且同相时才大）；x = 000 未被（cw）标记时再给
      sens_dy_db = RMS_x[Δy(x) − Δy(000)]（Δy = y1 − y2，奇偶判决量对物体的灵敏度，dB）、sens_y1_db（探头 1 单独）、
      rel_du = RMS_x|u(x)/u(000) − 1|（物体对 PA 输入的相对调制）、relay_over_direct_db = median 20log10(|C g|/|d|)
      （中继支路相对直达支路；中继支路 rank-1，方案推论 1：它占主导时 y1/y2 → |c1/c2|²，与输入无关）。
      灵敏度量只用 cw 未标记样本。"""
    cf = criterion_flags(m)
    fl = cf["cw"]
    rob = np.asarray(m.loop_gain_robust, dtype=float) if m.loop_gain_robust is not None else np.zeros(fl.shape)
    out = {"flags_any": bool(fl.any()), "flag_frac": float(fl.mean()),
           "robust_any": bool(cf["robust"].any()), "flag_frac_robust": float(cf["robust"].mean()),
           "ss_any": bool(cf["small_signal"].any()), "flag_frac_ss": float(cf["small_signal"].mean()),
           "rho_ss_max": float(np.nanmax(m.loop_gain_ss)) if np.size(m.loop_gain_ss) else 0.0,
           "rho_rob_max": float(np.nanmax(rob)) if rob.size and np.isfinite(rob).any() else None,
           "enh_db": None, "sens_dy_db": None, "sens_y1_db": None, "rel_du": None, "relay_over_direct_db": None}
    if system.is_passive or m.K is None:
        return out
    GK = system.small_signal_gain() * m.K
    smin = np.linalg.svd(np.eye(GK.shape[-1])[None] - GK, compute_uv=False)[..., -1]
    out["enh_db"] = float(-20.0 * np.log10(max(float(np.min(smin)), 1e-300)))
    x0 = _x0(system)
    if fl[x0]:
        return out
    ok = ~fl
    ok[x0] = False
    if not ok.any():
        return out
    y = system.clean_readings(m)
    dy = y[:, 0] - y[:, 1]
    out["sens_dy_db"] = float(np.sqrt(np.mean((dy[ok] - dy[x0]) ** 2)))
    out["sens_y1_db"] = float(np.sqrt(np.mean((y[ok, 0] - y[x0, 0]) ** 2)))
    u = m.u[:, 0]
    out["rel_du"] = float(np.sqrt(np.mean(np.abs(u[ok] / u[x0] - 1.0) ** 2)))
    # 直达支路 d = s_b·H[探头, 馈源]（与码本无关；PhysicalSystem.direct_field，INTEGRATION B3）
    d = system.direct_field()
    rel = np.abs(m.z - d) / np.maximum(np.abs(d), 1e-300)
    okk = ~fl
    out["relay_over_direct_db"] = float(np.median(20.0 * np.log10(np.maximum(rel[okk], 1e-300))))
    return out


def _med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def random_codebook_diagnostics(system, rng: np.random.Generator, n_rand: int) -> dict:
    """训练无关的诊断：n_rand 个随机码本（无噪声）上 codebook_diagnostics 的汇总——rand_osc_frac（cw 判据下任一样本被
    标记的码本比例）、rand_osc_frac_robust / rand_osc_frac_ss（另两种判据，审查 R-D1）、rand_sample_flag_frac（及 _robust / _ss）、
    各量的中位数（灵敏度类只在 x = 000 未被标记的码本上取）。"""
    ds = [codebook_diagnostics(system, system.forward(system.phases_from_params(system.random_params(rng)),
                                                      noisy=False)) for _ in range(int(n_rand))]
    sens = [d["sens_dy_db"] for d in ds if d["sens_dy_db"] is not None]
    return {"rand_osc_frac": float(np.mean([d["flags_any"] for d in ds])),
            "rand_osc_frac_robust": float(np.mean([d["robust_any"] for d in ds])),
            "rand_osc_frac_ss": float(np.mean([d["ss_any"] for d in ds])),
            "rand_sample_flag_frac": float(np.mean([d["flag_frac"] for d in ds])),
            "rand_sample_flag_frac_robust": float(np.mean([d["flag_frac_robust"] for d in ds])),
            "rand_sample_flag_frac_ss": float(np.mean([d["flag_frac_ss"] for d in ds])),
            "rand_loop_gain_ss_med": _med([d["rho_ss_max"] for d in ds]),
            "rand_loop_gain_rob_med": _med([d["rho_rob_max"] for d in ds]),
            "rand_enh_db_med": _med([d["enh_db"] for d in ds]),
            "rand_sens_db": _med(sens), "rand_sens_y1_db": _med([d["sens_y1_db"] for d in ds]),
            "rand_rel_du": _med([d["rel_du"] for d in ds]),
            "rand_relay_over_direct_db": _med([d["relay_over_direct_db"] for d in ds]), "rand_n_sens": len(sens)}


# ============================================================ job
_EVAL_KEYS = ("acc_clean", "acc_clean_valid", "all_correct", "all_correct_valid", "acc_noisy", "acc_noisy_valid",
              "margin_db", "margin_db_unflagged", "oscillating_run", "n_flagged", "n_not_converged", "osc_criterion",
              "robust_unstable_run", "n_robust_unstable", "ss_unstable_run", "n_ss_unstable",
              "loss_clean", "loss_meas_clean", "loop_gain_ss_max", "loop_gain_op_max", "loop_gain_robust_max",
              "gershgorin_max", "det_min",
              "pa_in_rel_p1db_db_min", "pa_in_rel_p1db_db_mean", "pa_in_rel_p1db_db_max",
              "pa_in_open_rel_p1db_db_mean", "pa_in_delta_rel_p1db_db_mean", "null_depth_db", "noisy_flagged_frac",
              "p_feed_dbm", "backoff_db", "voxel_d", "effective_eps", "object_model")
_ORC_KEYS = ("acc_clean", "acc_clean_valid", "all_correct", "all_correct_valid", "margin_db_unflagged",
             "oscillating_run", "n_flagged", "robust_unstable_run", "ss_unstable_run", "loop_gain_ss_max",
             "loop_gain_op_max", "loop_gain_robust_max", "pa_in_rel_p1db_db_mean", "null_depth_db", "loss_meas_clean")


def criterion_runs(ev: dict) -> dict:
    """evaluate 输出 → 三种判据下的"自激 run"（审查 R-D1）：osc_robust_run = oscillating_run(cw) | robust_unstable_run，
    osc_ss_run = oscillating_run | ss_unstable_run；以及 robust 判据下剔除自激 run 的准确率 acc_clean_valid_robust。"""
    osc = bool(ev.get("oscillating_run"))
    rob = osc or bool(ev.get("robust_unstable_run"))
    ss = osc or bool(ev.get("ss_unstable_run"))
    return {"osc_robust_run": rob, "osc_ss_run": ss,
            "acc_clean_valid_robust": None if rob else ev.get("acc_clean")}


def scenario_fields(name: str) -> dict:
    """命名场景（system.apply_scenario）里 D 沿用的字段 D_SCENARIO_FIELDS（直达路径开关）；baseline → 默认值。"""
    from ..system import apply_scenario

    over = apply_scenario({}, name)
    return {k: over[k] for k in D_SCENARIO_FIELDS}


def _make_cfg(job: dict, kph: float):
    from ..system import SystemConfig

    return SystemConfig(preset="flat0", object_mode=job.get("object_mode", "full"),
                        roi_offset=(float(job["dist"]), 0.0, 0.0), isolation_db=float(job["iso"]),
                        kappa0_phase_deg=float(kph), osc_criterion="cw", **parse_condition(job["cond"]),
                        **scenario_fields(job.get("scenario", "baseline")))


def run_d_job(job: dict) -> dict:
    from ..system import make_system
    from ..train import run_dspsa, run_oracle_bits

    t0 = time.perf_counter()
    seed = int(job["seed"])
    kph = float(job["kappa_phase_deg"]) if job.get("kappa_phase_deg") is not None else kappa_phase_for_seed(seed)
    cfg = _make_cfg(job, kph)
    s = make_system(cfg, "parity", n=N_BITS)
    t_build = time.perf_counter() - t0
    gk = nominal_loop_gain_db(cfg)
    row = {"dist": float(job["dist"]), "iso": float(job["iso"]), "cond": job["cond"], "seed": seed,
           "scenario": job.get("scenario", "baseline"), "kappa_phase_deg": kph,
           "kappa_phase_mode": "grid" if job.get("kappa_phase_deg") is not None else "random",
           "gk_nominal_db": gk, "critical": is_critical(gk), "regime": regime(gk),
           "condition_internal": s.label, "build_time_s": t_build}
    # 1. 训练无关：随机码本（所有格点、条件共用同一组码本）
    row.update(random_codebook_diagnostics(s, common.seed_rng(EXP, "rand", seed), int(job["n_rand"])))
    # 2. DSPSA（公共随机数：同种子 → 同初始码本与扰动序列）；代理统计三种判据下被标记的测量
    n_iter = int(job["n_iter"])
    proxy = _CriteriaCounter(s)
    res = run_dspsa(proxy, n_iter, common.seed_rng(EXP, "dspsa", seed), eval_every=max(25, n_iter // 8),
                    n_noise_eval=int(job["n_noise"]), pocket=True)
    ev = res["eval"]
    row.update({k: ev.get(k) for k in _EVAL_KEYS})
    row.update(criterion_runs(ev))
    nm = max(1, proxy.counts["n"])
    row.update(n_measurements=int(res["n_measurements"]), n_flagged_measurements=int(res["n_flagged_measurements"]),
               flagged_meas_frac=proxy.counts["cw"] / nm, flagged_meas_frac_robust=proxy.counts["robust"] / nm,
               flagged_meas_frac_ss=proxy.counts["small_signal"] / nm,
               dspsa_a=res["a"], a_auto_degenerate=bool(np.asarray(res["history"].get("a_auto_degenerate", [False]))[0]),
               curve=[{k: c.get(k) for k in ("iter", "loss_meas_clean", "acc_clean", "n_flagged")} for c in res["curve"]],
               curve_flagged_frac=float(np.mean([c.get("n_flagged", 0) > 0 for c in res["curve"]])),
               params=[int(v) for v in res["params"]], train_time_s=res["time_s"])
    pev = (res.get("pocket") or {}).get("eval") or {}
    row.update(pk_acc_clean=pev.get("acc_clean"), pk_acc_clean_valid=pev.get("acc_clean_valid"),
               pk_margin_db_unflagged=pev.get("margin_db_unflagged"), pk_oscillating_run=pev.get("oscillating_run"))
    # 3. 最终 DSPSA 码本的类间距与环路补充诊断
    m = s.forward(s.phases_from_params(res["params"]), noisy=False)
    row.update(class_separation(s, s.clean_readings(m), m.flags))
    row.update(loop_extras(s, m))
    row.update({f"fin_{k}": v for k, v in codebook_diagnostics(s, m).items()})
    # 4. 1-bit oracle（准确率目标，审查 R-D3 / B3）：结构最多能做到的码本上的类间距 / 裕量 / 环路
    t1 = time.perf_counter()
    orc = run_oracle_bits(s, common.seed_rng(EXP, "oracle", seed), n_restarts=int(job["oracle_restarts"]),
                          max_sweeps=int(job["oracle_sweeps"]), n_noise_eval=0, objective="accuracy")
    eo = orc["eval"]
    row.update({f"orc_{k}": eo.get(k) for k in _ORC_KEYS})
    row.update({f"orc_{k}": v for k, v in criterion_runs(eo).items()})
    mo = s.forward(s.phases_from_params(orc["params"]), noisy=False)
    row.update({f"orc_{k}": v for k, v in class_separation(s, s.clean_readings(mo), mo.flags).items()})
    row.update({f"orc_{k}": v for k, v in loop_extras(s, mo).items()})
    row.update(orc_objective=orc["objective"], orc_n_evals=int(orc["n_evals"]), orc_time_s=time.perf_counter() - t1,
               orc_params=[int(v) for v in orc["params"]])
    row["job_time_s"] = time.perf_counter() - t0
    return row


def run_d_scan_job(job: dict) -> dict:
    """κ0 相位扫描（训练无关）：给定 (距离, 隔离, 条件, 相位)，在同一组 n_rand 个随机码本上做 codebook_diagnostics。
    回答"类间距 / 灵敏度在 |Gκ| → 1（且同相，真正谐振）时是否增大"，不受训练质量影响。"""
    from ..system import make_system

    t0 = time.perf_counter()
    cfg = _make_cfg(job, float(job["kappa_phase_deg"]))
    s = make_system(cfg, "parity", n=N_BITS)
    gk = nominal_loop_gain_db(cfg)
    row = {"kind": "scan", "dist": float(job["dist"]), "iso": float(job["iso"]), "cond": job["cond"],
           "scenario": job.get("scenario", "baseline"), "kappa_phase_deg": float(job["kappa_phase_deg"]),
           "gk_nominal_db": gk, "critical": is_critical(gk),
           "regime": regime(gk), "p_feed_dbm": float(10 * np.log10(s.p_feed_w) + 30), "voxel_d": float(s.task.voxel_d)}
    row.update(random_codebook_diagnostics(s, common.seed_rng(EXP, "scan"), int(job["n_rand"])))
    row["job_time_s"] = time.perf_counter() - t0
    return row


def run_any_job(job: dict) -> dict:
    """run_jobs 的顶层入口：job["kind"] == "scan" → 相位扫描，否则训练 run。"""
    return run_d_scan_job(job) if job.get("kind") == "scan" else run_d_job(job)


# ============================================================ 网格
def build_config(args) -> dict:
    quick = bool(args.quick)
    seeds = args.seeds if args.seeds is not None else (3 if quick else 5)
    isos = [float(v) for v in (args.isolations or default_isolations())]
    cfg = {"exp": EXP, "quick": quick, "seeds": int(seeds),
           "isolations": isos,
           "distances": [float(v) for v in (args.distances or (QUICK_DISTANCES if quick else FULL_DISTANCES))],
           "conditions": list(args.conditions or ALL_CONDITIONS),
           "kappa_phases": None if not args.kappa_phases else [float(v) for v in args.kappa_phases],
           "n_iter": int(args.n_iter or (400 if quick else 1500)),
           "n_rand": int(args.n_rand or (8 if quick else 16)),
           "oracle_restarts": int(args.oracle_restarts or (2 if quick else 4)), "oracle_sweeps": 30,
           "n_noise": 10 if quick else 20, "object_mode": "full", "preset": "flat0", "n_bits": N_BITS,
           "osc_criterion": "cw", "criteria": list(CRITERIA),
           "scan_phases": [] if args.no_scan else [float(v) for v in (args.scan_phases or SCAN_PHASES)],
           "scenario": getattr(args, "scenario", None) or "baseline"}
    cfg["scenario_fields"] = scenario_fields(cfg["scenario"])
    from ..system import SystemConfig
    cfg["loop_gains_db"] = [nominal_loop_gain_db(SystemConfig(preset="flat0", isolation_db=v)) for v in isos]
    for c in cfg["conditions"]:
        parse_condition(c)
    return cfg


def make_jobs(cfg: dict) -> list[dict]:
    phases = cfg["kappa_phases"] or [None]
    jobs = []
    for dist in cfg["distances"]:
        for iso in cfg["isolations"]:
            for cond in cfg["conditions"]:
                for kph in phases:
                    for seed in range(cfg["seeds"]):
                        jobs.append({"dist": dist, "iso": iso, "cond": cond, "seed": seed, "kappa_phase_deg": kph,
                                     "n_iter": cfg["n_iter"], "n_rand": cfg["n_rand"], "n_noise": cfg["n_noise"],
                                     "oracle_restarts": cfg.get("oracle_restarts", 2),
                                     "oracle_sweeps": cfg.get("oracle_sweeps", 30),
                                     "object_mode": cfg["object_mode"], "scenario": cfg.get("scenario", "baseline")})
    for dist in cfg["distances"]:
        for iso in cfg["isolations"]:
            for cond in cfg["conditions"]:
                for kph in cfg.get("scan_phases") or []:
                    jobs.append({"kind": "scan", "dist": dist, "iso": iso, "cond": cond, "kappa_phase_deg": kph,
                                 "n_rand": cfg["n_rand"], "object_mode": cfg["object_mode"],
                                 "scenario": cfg.get("scenario", "baseline")})
    # 估计耗时：Rapp 中继在小信号环路增益 ≥ 1 时不动点难收敛（每次前向 ~0.1 s）→ 先提交；扫描 job 最后
    jobs.sort(key=lambda j: (j.get("kind") == "scan", -(j["iso"] + (100.0 if not j["cond"].startswith("M1") else 0.0))))
    return jobs


# ============================================================ 汇总
METRICS = ["oscillating_run", "osc_robust_run", "osc_ss_run",
           "flagged_meas_frac", "flagged_meas_frac_robust", "flagged_meas_frac_ss",
           "rand_osc_frac", "rand_osc_frac_robust", "rand_osc_frac_ss",
           "rand_sample_flag_frac", "rand_sample_flag_frac_robust", "rand_sample_flag_frac_ss", "acc_clean",
           "acc_clean_valid", "acc_clean_valid_robust", "all_correct_valid", "acc_noisy_valid", "margin_db",
           "margin_db_unflagged", "sep_class_db", "sep_signed_mean_db", "pk_acc_clean", "pk_acc_clean_valid",
           "orc_acc_clean", "orc_acc_clean_valid", "orc_acc_clean_valid_robust", "orc_all_correct_valid",
           "orc_margin_db_unflagged", "orc_sep_class_db", "orc_sep_signed_mean_db", "orc_oscillating_run",
           "orc_osc_robust_run", "orc_osc_ss_run", "orc_loop_gain_ss_max", "orc_one_minus_gk_min",
           "orc_pa_in_rel_p1db_db_mean",
           "rand_sens_db", "rand_sens_y1_db", "rand_rel_du",
           "rand_relay_over_direct_db", "rand_enh_db_med", "fin_sens_dy_db", "fin_rel_du", "fin_relay_over_direct_db",
           "fin_enh_db", "loop_gain_ss_max", "loop_gain_op_max", "loop_gain_robust_max",
           "loop_gain_ss_med", "loop_gain_op_med", "gershgorin_max", "det_min", "one_minus_gk_min", "gk_obj_db",
           "rand_loop_gain_ss_med", "rand_loop_gain_rob_med", "pa_in_rel_p1db_db_min", "pa_in_rel_p1db_db_mean",
           "pa_in_rel_p1db_db_max", "null_depth_db", "p_feed_dbm", "voxel_d", "gk_nominal_db", "job_time_s"]
PAIRED_METRICS = [f"d_{base}{CRIT_SUFFIX[c]}" for base in ("osc", "rand_osc", "flagged_meas_frac") for c in CRITERIA]
_OSC_KEY = {"cw": "oscillating_run", "robust": "osc_robust_run", "small_signal": "osc_ss_run"}


def _ci(t: dict, m: str, f: str = ".2f") -> str:
    mean, half, n = t.get(f"{m}_mean"), t.get(f"{m}_ci95"), t.get(f"{m}_n", 0)
    if not n or mean is None or not math.isfinite(mean):
        return "—"
    if half is None or not math.isfinite(half):
        return format(mean, f)
    return f"{format(mean, f)} ± {format(half, f)}"


def _m(t: dict, m: str, f: str = ".2f") -> str:
    v = t.get(f"{m}_mean")
    return "—" if v is None or not math.isfinite(v) else format(v, f)


def _tri(t: dict, base: str, f: str = ".2f", keys: dict | None = None) -> str:
    """三种判据的均值 "cw/rob/ss"。base 为 cw 的键名；其余判据键名 = base + 后缀（或 keys 显式给出）。"""
    ks = keys or {c: base + CRIT_SUFFIX[c] for c in CRITERIA}
    return "/".join(_m(t, ks[c], f) for c in CRITERIA)


def paired_osc(rows: list[dict]) -> list[dict]:
    """同 (距离, 隔离, κ0 相位, 种子) 配对：条件 − M1 的自激（最终码本、随机码本、训练中被标记测量比例），三种判据。
    临界行（名义 |Gκ| ≈ 0 dB，按构造被标记，审查 R-D2）不参与配对。"""
    idx = {(r["dist"], r["iso"], r["seed"], r.get("kappa_phase_deg"), r["cond"]): r for r in rows
           if "error" not in r and not r.get("critical", is_critical(r.get("gk_nominal_db")))}
    out = []
    for (d, iso, seed, kph, cond), r in idx.items():
        if cond == "M1":
            continue
        ref = idx.get((d, iso, seed, kph, "M1"))
        if ref is None:
            continue
        p = {"dist": d, "iso": iso, "cond": cond, "seed": seed, "gk_nominal_db": r.get("gk_nominal_db"),
             "regime": regime(r.get("gk_nominal_db"))}
        for c in CRITERIA:
            sfx = CRIT_SUFFIX[c]
            p[f"d_osc{sfx}"] = float(bool(r.get(_OSC_KEY[c]))) - float(bool(ref.get(_OSC_KEY[c])))
            for base in ("rand_osc_frac", "flagged_meas_frac"):
                a, b = r.get(base + sfx), ref.get(base + sfx)
                key = "d_rand_osc" if base == "rand_osc_frac" else "d_flagged_meas_frac"
                p[key + sfx] = None if a is None or b is None else float(a) - float(b)
        out.append(p)
    return out


def summary_markdown(table: list[dict], pt: list[dict], cfg: dict, rows: list[dict]) -> str:
    seeds = cfg["seeds"]
    ph = ("每个种子随机 U[0°,360°)（同种子所有格点共用）" if not cfg["kappa_phases"]
          else "网格 " + ", ".join(f"{v:g}°" for v in cfg["kappa_phases"]))
    lines = [f"# 实验 D：反馈环（RQ5）{'— quick' if cfg['quick'] else ''}", "",
             f"- flat0 预设（物体在环路里），场景 {cfg.get('scenario', 'baseline')}（只沿用直达路径开关 "
             f"{cfg.get('scenario_fields') or scenario_fields(cfg.get('scenario', 'baseline'))}），"
             f"3 位奇偶，水球 λ/3、自动体素、full 耦合；DSPSA {cfg['n_iter']} 次迭代（默认 sign_acc "
             f"变体：判对优先可测目标 + sign-SPSA，输出 pocket 码本，INTEGRATION T）；1-bit oracle（准确率目标，{cfg.get('oracle_restarts', '?')} 次重启）。",
             f"- 隔离 {', '.join(f'{v:g}' for v in cfg['isolations'])} dB（名义 |Gκ| "
             f"{', '.join(f'{v:+g}' for v in cfg.get('loop_gains_db', []))} dB）× ROI 距离 "
             f"{', '.join(f'{v:g}' for v in cfg['distances'])} m × 条件 {', '.join(cfg['conditions'])}；κ0 相位：{ph}；"
             f"每格 {seeds} 个种子" + ("（quick 模式少于 5 个种子）" if seeds < 5 else "") + "。",
             "- |Gκ| nom = G_PA + isolation − loss_rx − loss_tx（dB，不含经物体的耦合）。网格避开 0 dB（审查 R-D2）："
             "|Gκ_eff| = 1 时无物体样本 x = 000 的 ρ_ss 恰为 1，线性中继按构造必被标记；若网格含这样的点，表中标 ‡，"
             "且不参与配对差分与趋势相关。",
             "- **三种自激判据**（审查 R-D1，INTEGRATION B3）：cw = 不收敛或 ρ_op ≥ 1（训练与 *_valid 用它）；"
             "rob = cw 或 ρ_rob = ‖K‖·max(|a|+|b|) ≥ 1（任意泄漏相位 / 带内频率都不起振的充分条件）；"
             "ss = cw 或 ρ_ss = ρ(G_v K) ≥ 1（源关闭即起振）。线性中继 ρ_op = ρ_rob = ρ_ss，三者相同；"
             "名义 |Gκ| > 0 dB 时 x = 000 的 ρ_ss > 1，M1 按定义 100% 自激、ss 判据下所有条件按定义 100% →"
             "超临界点上 \"M4 自激比例低于 M1\" 只在 rob 判据下非平凡。",
             "- osc = 最终 DSPSA 码本自激 run 比例；flag_meas = 训练中被标记测量比例；rand_osc = 随机码本自激比例"
             "（训练无关，同一组码本）；均写作 cw/rob/ss。",
             "- acc_valid = DSPSA 最终码本剔除 cw 自激 run 后的准确率（括号内有效 run 数）；orc_* = 准确率目标 oracle 码本"
             "（审查 R-D3：DSPSA 在本任务停在 y1 ≈ y2 平台，其类间距是噪声）：orc_acc_valid、orc_all_ok、"
             "orc_sep = 类间距 mean(Δy|奇) − mean(Δy|偶)、orc_margin = 未标记样本的最小带符号裕量；sens = 随机码本物体灵敏度。",
             "- 环路：ρ_ss / ρ_rob = 最终 DSPSA 码本各样本小信号 / 相位稳健环路增益的最大值；"
             "PA in = 闭环 PA 输入相对 P1dB（未标记样本，dB）。", ""]
    lines += ["| d (m) | iso (dB) | |Gκ| nom (dB) | cond | n | osc cw/rob/ss | rand_osc cw/rob/ss | flag_meas cw/rob/ss | "
              "acc_valid | orc_acc_valid | orc_all_ok | orc_sep (dB) | orc_margin (dB) | sens (dB) | ρ_ss | ρ_rob | "
              "PA in min/mean/max | P_feed (dBm) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {c: i for i, c in enumerate(cfg["conditions"])}
    for t in sorted(table, key=lambda t: (t["dist"], t["iso"], order.get(t["cond"], 99))):
        pa = "/".join(_m(t, f"pa_in_rel_p1db_db_{k}", ".1f") for k in ("min", "mean", "max"))
        nv = t.get("acc_clean_valid_n", 0)
        nov = t.get("orc_acc_clean_valid_n", 0)
        crit = " ‡" if is_critical(t.get("gk_nominal_db_mean")) else ""
        lines.append(
            f"| {t['dist']:g} | {t['iso']:g}{crit} | {_m(t, 'gk_nominal_db', '+.0f')} | {t['cond']} | {t['n_runs']}"
            + (f" (err {t['n_errors']})" if t["n_errors"] else "") +
            f" | {_tri(t, 'oscillating_run', keys=_OSC_KEY)} | {_tri(t, 'rand_osc_frac')} | {_tri(t, 'flagged_meas_frac')} | "
            f"{_ci(t, 'acc_clean_valid')} ({nv}) | {_ci(t, 'orc_acc_clean_valid')} ({nov}) | "
            f"{_m(t, 'orc_all_correct_valid')} | {_ci(t, 'orc_sep_class_db')} | {_ci(t, 'orc_margin_db_unflagged')} | "
            f"{_ci(t, 'rand_sens_db')} | {_m(t, 'loop_gain_ss_max', '.3f')} | {_m(t, 'loop_gain_robust_max', '.3f')} | "
            f"{pa} | {_m(t, 'p_feed_dbm', '.1f')} |")
    if any(is_critical(t.get("gk_nominal_db_mean")) for t in table):
        lines += ["", "‡ 临界（按构造）：名义 |Gκ_eff| = 1，x = 000 的 ρ_ss(x=000) = 1，线性中继必被标记；不作 RQ5 证据。"]
    vd = [r.get("voxel_d") for r in rows if r.get("voxel_d") is not None]
    if vd:
        lines += ["", f"体素尺寸 voxel_d = {1e3 * float(np.median(vd)):.2f} mm（λ/{0.0856549880 / float(np.median(vd)):.1f}），"
                      f"材料 {next((r.get('effective_eps') for r in rows if r.get('effective_eps')), '?')}，"
                      f"极化率模型 {next((r.get('object_model') for r in rows if r.get('object_model')), '?')}。"]
    # 配对差分
    pa_t = common.aggregate(pt, ["dist", "iso", "cond"], PAIRED_METRICS)
    lines += ["", "## 配对差分（条件 − M1，同距离 / 隔离 / 种子；临界行已剔除）", "",
              "预测（方案 §6.2）：M4 自激比例明显低于 M1 → d_osc < 0。每格写作 cw / rob / ss 三种判据（均值 ± 95% 区间半宽）。", "",
              "| d (m) | iso (dB) | |Gκ| nom | cond | n | d_osc cw | d_osc rob | d_osc ss | d_rand_osc cw | d_rand_osc rob | "
              "d_rand_osc ss | d_flag_meas cw/rob/ss |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    gk_of = {(r["dist"], r["iso"]): r.get("gk_nominal_db") for r in pt}
    for t in sorted(pa_t, key=lambda t: (t["dist"], t["iso"], t["cond"])):
        g = gk_of.get((t["dist"], t["iso"]))
        lines.append(f"| {t['dist']:g} | {t['iso']:g} | {'—' if g is None else format(g, '+.0f')} | {t['cond']} | "
                     f"{t['n_runs']} | {_ci(t, 'd_osc')} | {_ci(t, 'd_osc_robust')} | {_ci(t, 'd_osc_ss')} | "
                     f"{_ci(t, 'd_rand_osc')} | {_ci(t, 'd_rand_osc_robust')} | {_ci(t, 'd_rand_osc_ss')} | "
                     f"{_tri(t, 'd_flagged_meas_frac')} |")
    lines += ["", osc_verdict_text(pt, cfg), "", trend_text(rows), ""]
    n_err = sum(1 for r in rows if "error" in r)
    if n_err:
        lines.append(f"{n_err} 个 run 出错（见 runs.jsonl 的 error 字段）。")
    return "\n".join(lines) + "\n"


def _ci_str(t: dict, m: str) -> tuple[str, bool | None]:
    """(“均值 [lo, hi]”, 区间是否不含 0 且 < 0)；n < 2 时只给均值。"""
    mean, lo, hi = t.get(f"{m}_mean"), t.get(f"{m}_lo"), t.get(f"{m}_hi")
    if mean is None or not math.isfinite(mean):
        return "—", None
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return f"{mean:+.2f}", None
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]", bool(hi < 0)


def osc_verdict_text(pt: list[dict], cfg: dict) -> str:
    """RQ5 自激预测的判读：按名义环路增益区间（sub < 0 dB / super > 0 dB，临界点已剔除）汇总配对 d_osc（最终码本）与
    d_rand_osc（随机码本，训练无关），三种判据分列；超临界点上 cw / ss 判据的结论按构造是什么，写在说明里。"""
    out = ["### 判读：M4（及 M3）自激比例是否低于 M1（RQ5，配对差分汇总）", "",
           "汇总 = 区间内全部 (距离, 隔离, 种子) 配对；方括号为 95% 区间。super（名义 |Gκ| > 0 dB）点上 M1 按定义 100% 自激，"
           "ss 判据下所有条件按定义 100%（d_osc_ss ≡ 0）；cw 判据可能乐观（只靠零点几 dB 压缩维持的工作点也算稳定），"
           "**rob 判据下的 d < 0 才说明 PA 饱和真的把环路钳住了**。sub 点上 M1 也可能因物体把 ρ_ss 推过 1 而被标记。", "",
           "| regime | cond | n | d_osc cw | d_osc rob | d_osc ss | d_rand_osc cw | d_rand_osc rob | d_rand_osc ss |",
           "|---|---|---|---|---|---|---|---|---|"]
    agg = common.aggregate(pt, ["regime", "cond"], PAIRED_METRICS)
    order = {"sub": 0, "super": 1}
    sig = []
    for t in sorted(agg, key=lambda t: (order.get(t["regime"], 9), t["cond"])):
        cells = []
        for m in ("d_osc", "d_osc_robust", "d_osc_ss", "d_rand_osc", "d_rand_osc_robust", "d_rand_osc_ss"):
            s, neg = _ci_str(t, m)
            cells.append(s + (" *" if neg else ""))
            if neg:
                sig.append((t["regime"], t["cond"], m))
        out.append(f"| {t['regime']} | {t['cond']} | {t['n_runs']} | " + " | ".join(cells) + " |")
    out += ["", "\\* 95% 区间整体 < 0（显著低于 M1）。"]
    rob_super = [x for x in sig if x[0] == "super" and x[2] in ("d_osc_robust", "d_rand_osc_robust")]
    cw_super = [x for x in sig if x[0] == "super" and x[2] in ("d_osc", "d_rand_osc")]
    if rob_super:
        out.append("- rob 判据下超临界点显著：" + "、".join(f"{c} {m}" for _, c, m in rob_super)
                   + " → PA 饱和在任意泄漏相位下都把环路钳住（RQ5 预测的非平凡支持）。")
    elif cw_super:
        out.append("- 只有 cw 判据下超临界点显著（" + "、".join(f"{c} {m}" for _, c, m in cw_super)
                   + "）；rob 判据下不显著 → \"M4 自激更少\" 依赖乐观的 CW 判据（只在当前泄漏相位、注入频率上稳定），"
                     "不能当作 PA 钳位的稳健证据。")
    else:
        out.append("- 超临界点上没有判据给出显著的 d < 0。")
    return "\n".join(out)


def _informative(rows: list[dict], acc_key: str, ok_key: str) -> tuple[bool, float | None]:
    """码本是否"在 chance 之上"：平均准确率 ≥ chance + CHANCE_BAND 或全对比例 > 0。"""
    acc = [r[acc_key] for r in rows if r.get(acc_key) is not None]
    allok = [bool(r[ok_key]) for r in rows if r.get(ok_key) is not None]
    if not acc:
        return False, None
    a = float(np.mean(acc))
    return (a >= CHANCE_ACC + CHANCE_BAND or (bool(allok) and any(allok))), a


def trend_text(rows: list[dict]) -> str:
    """类间距随环路增益的趋势：未自激 run、非临界格点上，sep / 裕量 / 灵敏度与 ρ_ss（dB）及谐振增强 enh 的 Spearman 相关。
    主数据 = oracle（准确率目标）码本；DSPSA 码本若在 chance 附近，判读里注明"相关是噪声"（审查 R-D3）。"""
    from scipy.stats import spearmanr

    out = ["## 类间距 vs 环路增益（未自激 run、非临界格点，Spearman）", "",
           "ρ_ss dB = 20log10 ρ(G_v K)（幅度判据）；enh dB = −20log10 min|1 − G_v K|（线性环路对 u 的实际放大，"
           "相位也要接近同相才谐振）。码本来源：oracle = 准确率目标 1-bit oracle（主数据）；DSPSA = 训练输出码本（sign_acc：pocket）。"
           "mean acc 是该来源在这些 run 上的平均准确率（chance 0.5）。", "",
           "| source | cond | n | mean acc | ρ(sep, ρ_ss dB) | ρ(sep, enh dB) | ρ(margin, ρ_ss dB) | ρ(margin, enh dB) | "
           "ρ(rand_sens, rand ρ_ss dB) |", "|---|---|---|---|---|---|---|---|---|"]
    ok_rows = [r for r in rows if "error" not in r and not r.get("critical", is_critical(r.get("gk_nominal_db")))]
    conds = list(dict.fromkeys(r["cond"] for r in ok_rows))
    notes = []

    def corr(a, b, src, sign=1.0):
        pts = [(x[a], x[b]) for x in src if x.get(a) is not None and x.get(b) is not None
               and np.isfinite(x[a]) and np.isfinite(x[b]) and x[b] > 0]
        if len(pts) < 4:
            return "—"
        u, v = np.array(pts).T
        if np.ptp(u) == 0 or np.ptp(v) == 0:
            return "—"
        r = spearmanr(u, sign * 20 * np.log10(v)).statistic
        return f"{r:+.2f} (n={len(pts)})"

    weak = []
    for src, pre, osc_key, acc_key, ok_key in (("oracle", "orc_", "orc_oscillating_run", "orc_acc_clean_valid",
                                                "orc_all_correct_valid"),
                                               ("DSPSA", "", "oscillating_run", "acc_clean_valid", "all_correct_valid")):
        for c in conds:
            allr = [r for r in ok_rows if r["cond"] == c]
            rr = [r for r in allr if not r.get(osc_key)]
            info, macc = _informative(rr, acc_key, ok_key)
            if rr and not info:
                (notes if src == "DSPSA" else weak).append(f"{src} / {c}")
            out.append(f"| {src} | {c} | {len(rr)} | {'—' if macc is None else format(macc, '.2f')} | "
                       f"{corr(pre + 'sep_class_db', pre + 'loop_gain_ss_max', rr)} | "
                       f"{corr(pre + 'sep_class_db', pre + 'one_minus_gk_min', rr, -1.0)} | "
                       f"{corr(pre + 'margin_db_unflagged', pre + 'loop_gain_ss_max', rr)} | "
                       f"{corr(pre + 'margin_db_unflagged', pre + 'one_minus_gk_min', rr, -1.0)} | "
                       f"{corr('rand_sens_db', 'rand_loop_gain_ss_med', allr) if src == 'oracle' else '（同上）'} |")
    if notes:
        out += ["", f"注意：{', '.join(notes)} 的码本平均准确率 < chance + {CHANCE_BAND:g} 且无全对 run —— 这些 DSPSA 码本"
                    "停在 y1 ≈ y2 的平台，类间距 / 裕量与环路增益的相关是噪声，不能当 RQ5 证据；RQ5 的类间距结论以 oracle 码本与"
                    "训练无关的随机码本相位扫描为准。"]
    if weak:
        out += ["", f"另：{', '.join(weak)} 的 oracle 码本平均准确率也 < chance + {CHANCE_BAND:g} 且无全对 run —— 结构本身"
                    "在这些格点几乎分不开 3 位奇偶，sep 的趋势只反映\"最好的码本\"的弱分离。"]
    return "\n".join(out)


def plot_all(table: list[dict], rows: list[dict], cfg: dict, outdir: str) -> list[str]:
    import matplotlib.pyplot as plt

    paths = []
    dists = sorted({t["dist"] for t in table})
    conds = [c for c in cfg["conditions"] if any(t["cond"] == c for t in table)]
    # 1) 自激比例 vs 名义环路增益（cw 实线、robust 虚线；随机码本点线；临界点灰带）
    fig, axes = plt.subplots(2, len(dists), figsize=(4.8 * len(dists), 7.6), squeeze=False, constrained_layout=True)
    for j, d in enumerate(dists):
        for i, (what, keys) in enumerate((("最终 DSPSA 码本", _OSC_KEY),
                                          ("随机码本（训练无关）", {c: "rand_osc_frac" + CRIT_SUFFIX[c] for c in CRITERIA}))):
            ax = axes[i][j]
            for k, c in enumerate(conds):
                ts = sorted([t for t in table if t["dist"] == d and t["cond"] == c],
                            key=lambda t: t["gk_nominal_db_mean"])
                x = [t["gk_nominal_db_mean"] for t in ts]
                for crit, ls, mk in (("cw", "-", "o"), ("robust", "--", "s"), ("small_signal", ":", "^")):
                    ax.plot(x, [t[f"{keys[crit]}_mean"] for t in ts], ls=ls, marker=mk, ms=4, color=f"C{k}",
                            alpha=1.0 if crit != "small_signal" else 0.45,
                            label=f"{c} {CRIT_SHORT[crit]}")
            ax.axvspan(-CRITICAL_TOL_DB, CRITICAL_TOL_DB, color="0.85", zorder=0)
            ax.axvline(0.0, color="k", lw=0.8, ls=":")
            ax.set_xlabel("名义 |G κ_eff| (dB)")
            ax.set_ylabel("可能自激比例")
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"flat0, d = {d:g} m：{what}", fontsize=9)
            if j == 0:
                ax.legend(fontsize=6, ncol=len(conds))
    paths.append(common.savefig(fig, os.path.join(outdir, "oscillation_vs_loopgain.png")))
    # 2) 类间距 / 灵敏度 vs 实际小信号环路增益与谐振增强（每个 run 一点；oracle 实心、DSPSA 空心；非临界）
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 3.8))
    okrows = [r for r in rows if "error" not in r and not r.get("critical", is_critical(r.get("gk_nominal_db")))]
    for k, c in enumerate(conds):
        rr = [r for r in okrows if r["cond"] == c]
        for pre, osck, filled in (("orc_", "orc_oscillating_run", True), ("", "oscillating_run", False)):
            kw = dict(s=14, color=f"C{k}") if filled else dict(s=14, facecolors="none", edgecolors=f"C{k}", alpha=0.5)
            lab = f"{c} {'oracle' if filled else 'DSPSA'}"
            ok = [r for r in rr if not r.get(osck) and r.get(pre + "sep_class_db") is not None
                  and (r.get(pre + "loop_gain_ss_max") or 0) > 0]
            axes[0].scatter([20 * np.log10(r[pre + "loop_gain_ss_max"]) for r in ok],
                            [r[pre + "sep_class_db"] for r in ok], label=lab, **kw)
            ok = [r for r in rr if not r.get(osck) and r.get(pre + "margin_db_unflagged") is not None
                  and (r.get(pre + "loop_gain_ss_max") or 0) > 0]
            axes[1].scatter([20 * np.log10(r[pre + "loop_gain_ss_max"]) for r in ok],
                            [r[pre + "margin_db_unflagged"] for r in ok], label=lab, **kw)
            ok = [r for r in rr if not r.get(osck) and r.get(pre + "sep_class_db") is not None
                  and (r.get(pre + "one_minus_gk_min") or 0) > 0]
            axes[3].scatter([-20 * np.log10(r[pre + "one_minus_gk_min"]) for r in ok],
                            [r[pre + "sep_class_db"] for r in ok], label=lab, **kw)
        ok = [r for r in rr if r.get("rand_sens_db") is not None and (r.get("rand_loop_gain_ss_med") or 0) > 0]
        axes[2].scatter([20 * np.log10(r["rand_loop_gain_ss_med"]) for r in ok], [r["rand_sens_db"] for r in ok],
                        s=14, color=f"C{k}", label=c)
    for ax, yl, tt, xl in zip(axes, ("类间距 sep (dB)", "最小带符号裕量 (dB)", "物体灵敏度 (dB)", "类间距 sep (dB)"),
                              ("训练码本：类间距", "训练码本：最小裕量", "随机码本", "训练码本：类间距"),
                              ["小信号环路增益 ρ(G K) (dB)"] * 3 + ["线性环路增强 −20log|1−GK| (dB)"]):
        ax.axvline(0.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(tt + "（未自激、非临界）", fontsize=9)
        ax.legend(fontsize=6)
    paths.append(common.savefig(fig, os.path.join(outdir, "separation_vs_loopgain.png")))
    # 3) 剔除自激后的准确率（oracle 实线、DSPSA 虚线）
    fig, axes = plt.subplots(1, len(dists), figsize=(4.6 * len(dists), 3.4), squeeze=False)
    for j, d in enumerate(dists):
        ax = axes[0][j]
        for k, c in enumerate(conds):
            ts = sorted([t for t in table if t["dist"] == d and t["cond"] == c], key=lambda t: t["gk_nominal_db_mean"])
            for key, fmt, lab in (("orc_acc_clean_valid", "o-", "oracle"), ("acc_clean_valid", "s--", "DSPSA")):
                ax.errorbar([t["gk_nominal_db_mean"] + 0.3 * (k - 1) for t in ts], [t[f"{key}_mean"] for t in ts],
                            yerr=[0.0 if not np.isfinite(t[f"{key}_ci95"]) else t[f"{key}_ci95"] for t in ts],
                            fmt=fmt, capsize=2, color=f"C{k}", alpha=1.0 if lab == "oracle" else 0.5,
                            label=f"{c} {lab}")
        ax.axvspan(-CRITICAL_TOL_DB, CRITICAL_TOL_DB, color="0.85", zorder=0)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("名义 |G κ_eff| (dB)")
        ax.set_ylabel("准确率（剔除 cw 自激 run）")
        ax.set_title(f"3 位奇偶，d = {d:g} m", fontsize=9)
        ax.legend(fontsize=6)
    paths.append(common.savefig(fig, os.path.join(outdir, "accuracy_valid.png")))
    return paths


SCAN_METRICS = ["rand_osc_frac", "rand_osc_frac_robust", "rand_osc_frac_ss", "rand_sample_flag_frac",
                "rand_loop_gain_ss_med", "rand_loop_gain_rob_med", "rand_enh_db_med", "rand_sens_db",
                "rand_sens_y1_db", "rand_rel_du", "rand_relay_over_direct_db", "rand_n_sens", "gk_nominal_db"]


def scan_markdown(stab: list[dict], srows: list[dict], cfg: dict) -> str:
    """相位扫描：每个 (距离, 条件) 一张透视表，行 = 隔离，列 = κ0 相位，格 = 灵敏度 sens（dB）/ rel_du / 自激比例（cw|rob）。"""
    from scipy.stats import spearmanr

    phases = sorted({t["kappa_phase_deg"] for t in stab})
    lines = ["## κ0 相位扫描（训练无关，随机码本）", "",
             f"同一组 {cfg['n_rand']} 个随机码本（所有格点共用），每格：sens = 奇偶判决量 Δy = y1 − y2 对物体的 RMS 变化（dB，"
             "cw 未标记码本中位数）/ rel_du = RMS|u(x)/u(000) − 1|（物体对 PA 输入的相对调制）/ osc = 自激码本比例"
             "（cw|rob 判据）。relay/direct = 中继支路相对直达支路（dB）；enh = −20log10 min|1 − G_v K|；ss 判据见 phase_scan.csv。"
             "‡ = 临界（按构造）。", ""]

    def cell(t):
        def f(k, fmt):
            v = t.get(f"{k}_mean")
            return "—" if v is None or not math.isfinite(v) else format(v, fmt)
        return (f"{f('rand_sens_db', '.2f')} / {f('rand_rel_du', '.3f')} / "
                f"{f('rand_osc_frac', '.2f')}\\|{f('rand_osc_frac_robust', '.2f')}")

    idx = {(t["dist"], t["iso"], t["cond"], t["kappa_phase_deg"]): t for t in stab}
    for d in sorted({t["dist"] for t in stab}):
        for c in [c for c in cfg["conditions"] if any(t["cond"] == c for t in stab)]:
            lines += [f"**d = {d:g} m，{c}**（sens / rel_du / osc cw\\|rob）", "",
                      "| iso (dB) | |Gκ| nom | " + " | ".join(f"{p:g}°" for p in phases) + " |",
                      "|---|---|" + "---|" * len(phases)]
            for iso in sorted({t["iso"] for t in stab}):
                ts = [idx.get((d, iso, c, p)) for p in phases]
                g = next((t.get("gk_nominal_db_mean") for t in ts if t), None)
                crit = " ‡" if is_critical(g) else ""
                lines.append(f"| {iso:g}{crit} | {'—' if g is None else format(g, '+.0f')} | "
                             + " | ".join("—" if t is None else cell(t) for t in ts) + " |")
            lines.append("")
    lines += ["相位扫描的 Spearman 相关（每格一点，全部距离 / 非临界隔离 / 相位；sens 只在有未标记码本时计入）：", "",
              "| cond | n | ρ(sens, enh dB) | ρ(rel_du, enh dB) | ρ(relay/direct, enh dB) | ρ(sens, relay/direct) |",
              "|---|---|---|---|---|---|"]
    for c in cfg["conditions"]:
        rr = [r for r in srows if "error" not in r and r["cond"] == c
              and not r.get("critical", is_critical(r.get("gk_nominal_db")))]

        def corr(a, b):
            pts = [(r[a], r[b]) for r in rr if r.get(a) is not None and r.get(b) is not None]
            if len(pts) < 4:
                return "—"
            u, v = np.array(pts, dtype=float).T
            if np.ptp(u) == 0 or np.ptp(v) == 0:
                return "—"
            return f"{spearmanr(u, v).statistic:+.2f}"

        n = sum(1 for r in rr if r.get("rand_sens_db") is not None)
        lines.append(f"| {c} | {n} | {corr('rand_sens_db', 'rand_enh_db_med')} | {corr('rand_rel_du', 'rand_enh_db_med')} | "
                     f"{corr('rand_relay_over_direct_db', 'rand_enh_db_med')} | "
                     f"{corr('rand_sens_db', 'rand_relay_over_direct_db')} |")
    return "\n".join(lines) + "\n"


def plot_scan(stab: list[dict], cfg: dict, outdir: str) -> str:
    import matplotlib.pyplot as plt

    dists = sorted({t["dist"] for t in stab})
    conds = [c for c in cfg["conditions"] if any(t["cond"] == c for t in stab)]
    isos = sorted({t["iso"] for t in stab})
    rows_spec = (("rand_sens_db_mean", "Δy 对物体的灵敏度 (dB)"),
                 ("rand_rel_du_mean", "物体对 u 的相对调制"),
                 ("rand_relay_over_direct_db_mean", "中继 / 直达支路 (dB)"),
                 ("rand_osc_frac_mean", "自激比例 cw"),
                 ("rand_osc_frac_robust_mean", "自激比例 robust"))
    fig, axes = plt.subplots(len(rows_spec), len(dists) * len(conds),
                             figsize=(3.3 * len(dists) * len(conds), 2.6 * len(rows_spec)), squeeze=False)
    for j, (d, c) in enumerate([(d, c) for d in dists for c in conds]):
        for i, (key, yl) in enumerate(rows_spec):
            ax = axes[i][j]
            for k, iso in enumerate(isos):
                ts = sorted([t for t in stab if t["dist"] == d and t["cond"] == c and t["iso"] == iso],
                            key=lambda t: t["kappa_phase_deg"])
                g = ts[0].get("gk_nominal_db_mean") if ts else None
                ax.plot([t["kappa_phase_deg"] for t in ts], [t.get(key, np.nan) for t in ts], "o-", ms=3,
                        color=f"C{k}", label=f"iso {iso:g} dB" + (" ‡" if is_critical(g) else ""))
            if i == 0:
                ax.set_title(f"d = {d:g} m, {c}", fontsize=9)
            if j == 0:
                ax.set_ylabel(yl, fontsize=8)
            if i == len(rows_spec) - 1:
                ax.set_xlabel("κ0 相位 (°)")
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    return common.savefig(fig, os.path.join(outdir, "kappa_phase_scan.png"))


def summarize(rows: list[dict], outdir: str, cfg: dict) -> dict:
    trows = [r for r in rows if r.get("kind", "train") != "scan"]
    srows = [r for r in rows if r.get("kind") == "scan"]
    keys = ["dist", "iso", "cond"] + (["kappa_phase_deg"] if cfg.get("kappa_phases") else [])
    table = common.aggregate(trows, keys, METRICS)
    pt = paired_osc(trows)
    paths = common.write_outputs(outdir, rows, table, config=cfg)
    common.save_csv(common.aggregate(pt, ["dist", "iso", "cond"], PAIRED_METRICS), os.path.join(outdir, "paired.csv"))
    md = summary_markdown(table, pt, cfg, trows)
    stab = []
    if srows:
        stab = common.aggregate(srows, ["dist", "iso", "cond", "kappa_phase_deg"], SCAN_METRICS)
        common.save_csv(stab, os.path.join(outdir, "phase_scan.csv"))
        md += "\n" + scan_markdown(stab, srows, cfg)
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(md)
    try:
        paths["figs"] = plot_all(table, trows, cfg, outdir)
        if stab:
            paths["figs"].append(plot_scan(stab, cfg, outdir))
    except Exception as e:          # noqa: BLE001 —— 作图失败不影响数据输出
        paths["fig_error"] = repr(e)
    return {"table": table, "scan": stab, "paired": pt, "paths": paths, "markdown": md}


def run(cfg: dict, workers: int = 1, out: str | None = None, progress: bool = False) -> dict:
    outdir = common.make_outdir(EXP, out)
    t0 = time.perf_counter()
    jobs = make_jobs(cfg)
    rows = common.run_jobs(run_any_job, jobs, workers=workers, progress=progress)
    for r in rows:                       # error 行补上格点键，aggregate 才能落到正确格点
        if "error" in r:
            j = r.get("job", {})
            r.update(dist=j.get("dist"), iso=j.get("iso"), cond=j.get("cond"), seed=j.get("seed"),
                     kappa_phase_deg=j.get("kappa_phase_deg"), kind=j.get("kind", "train"))
    cfg = dict(cfg, n_jobs=len(jobs), wall_time_s=time.perf_counter() - t0, workers=int(workers))
    res = summarize(rows, outdir, cfg)
    res.update(outdir=outdir, rows=rows)
    return res


def parse_args(argv=None):
    p = common.base_argparser("实验 D：反馈环（RQ5，方案 §6.2）")
    p.set_defaults(seeds=None)
    p.add_argument("--isolations", type=float, nargs="+", default=None,
                   help="端口隔离 dB（默认按名义 |Gκ| = −15 −3 −1 +1 +10 dB 取点，即 −40 −28 −26 −24 −15；"
                        "−25 = 临界，按构造标记）")
    p.add_argument("--distances", type=float, nargs="+", default=None, help="ROI 距离 m（默认 0.25 0.4 0.8；quick 0.25 0.8）")
    p.add_argument("--conditions", nargs="+", default=None, help="条件（默认 M1 M3 M4）")
    p.add_argument("--kappa-phases", type=float, nargs="+", default=None, help="κ0 相位网格（度）；缺省每种子随机")
    p.add_argument("--n-iter", type=int, default=None, help="DSPSA 迭代数（quick 400 / full 1500）")
    p.add_argument("--n-rand", type=int, default=None, help="随机码本诊断数（quick 8 / full 16）")
    p.add_argument("--oracle-restarts", type=int, default=None, help="准确率目标 oracle 重启数（quick 2 / full 4）")
    p.add_argument("--scan-phases", type=float, nargs="+", default=None,
                   help="κ0 相位扫描（度，训练无关的随机码本诊断；默认 0,45,…,315）")
    p.add_argument("--no-scan", action="store_true", help="不做 κ0 相位扫描")
    p.add_argument("--scenario", default=None, choices=["baseline", "informative", "informative_probe"],
                   help="命名场景（system.SCENARIOS / apply_scenario）；默认 baseline。D 的几何保持 flat0 + 距离网格，"
                        "只沿用直达路径开关 block_direct / skip_gain_db（informative_probe 的探头方位不沿用）")
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
