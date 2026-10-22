"""训练封装：DSPSA、1-bit 模型 oracle、连续相位 oracle（INTEGRATION D，DESIGN §6，方案 §4.7）。

- run_dspsa：每次迭代两次**有噪声测量**（system.forward(noisy=True) → system.loss_from(model_penalties=False)），
  与硬件流程一致：只用可测目标（任务损失 + 被标记样本替换 + 零陷项），**不含**模型内部环路谱半径的光滑自激惩罚
  （审查 INT-SEM-3；方案 §4.7 DSPSA 是 model-free 训练）。每 eval_every 次迭代记录无噪声 loss/准确率（不计入测量次数）。
- run_oracle_bits：无噪声**模型目标**（system.loss(noisy=False)，含光滑自激惩罚，与 TorchSystem 同定义）上的随机重启
  坐标下降（"这个离散结构最多能做到多好"）。
- run_oracle_continuous：torch 连续相位 + 隐函数梯度（oracle_torch，模型目标），再用 numpy 系统按连续相位复评。
每个返回字典都带 "objective"（"measurable" / "model" / "margin" / "accuracy"）。比较 DSPSA 与 oracle（实验 E 的 90% 判据）
时两边都取 eval["loss_meas_clean"]（同一可测目标的无噪声值），而不是各自优化的目标值。

审查 S0-2：二分类 softplus 损失在不可解任务上 ≈ τ·log2 − mean(s)/2，最小化它是最大化平均带符号裕量而不是最少错误，
oracle 的 acc_clean 可低于随机（M0 + Born 3 位奇偶实测 0.25），不能当"最好可达准确率"。要报后者请用
run_oracle_bits(objective="accuracy")（判对样本数 + 裕量平局打破）；objective="margin" 直接最大化最小带符号裕量
（门 G1 用：> 0 ⇔ 全对）。
审查 S1-0：subset DSPSA 在奇偶任务上常停在 y1 ≈ y2 的 τ·log2 平台或单比特翻转局部极小，最终迭代码本远差于训练中
测到过的码本。run_dspsa(pocket=True) 额外返回"测到过的最好码本"（pocket，model-free：硬件上同样可记录），
不改变 DSPSA 迭代本身与 params/eval。

训练器选择（INTEGRATION T，训练器探索后的默认）：1-bit 参数上 variant="auto" → "sign_acc"：
  可测"判对优先"目标 J = −(n_correct + ½·mean_{b,j} tanh(M_bj/τ_j))（M = Task.decision_units，逐判决单元自适应温度
  τ_j = max(1e-3, 0.5·exp(EMA_0.05 log 稳健离散度))）+ sign-SPSA（ĝ = sgn(J+ − J−)Δ，a_k = a/(1+k+A)^0.602，A = 400，
  首步 0.1、无标定）+ 子集扰动 m = p//8（±½ 中点，每迭代 2 次测量，方案 §4.7 骨架）+ pocket（键 = 判对优先目标 @τ = 1 dB，
  带噪测量），返回 pocket 码本；无重启。focus 任务退回系统可测损失。旧默认 "subset"（a="auto"、softplus τ = 1 dB）会停在
  τ·ln2 平台（a="auto" 两头都把步长定错，τ 损失奖励 y1 ≈ y2 的塌缩），保留作对照；多电平（M0 2-bit）仍用 "faithful"。
run_ils：迭代局部搜索（单比特首次改进爬山 + 2–6 位踢出 + 中性漂移），同一可测判对优先目标（样本裕量、τ = cfg.tau_db），
  每次尝试 1 次测量；extras=True 另加"拒绝时重测 x*（≤ 8 次）+ 末段 4×3 终选赛"。大预算（~3000 次测量）时略好于 sign_acc。
"""
from __future__ import annotations

import math
import time

import numpy as np

from .dspsa import decide, dspsa
from .oracle_bits import oracle_bits

DSPSA_VARIANTS = ("sign_acc", "subset", "faithful")
# sign_acc 变体的超参数（INTEGRATION T；= _scratch/trainers/dspsa_family.py 的 df_u_nores）
SIGN_ACC_STEP0 = 0.1             # 首步 |Δθ|（θ ∈ [0, 1]）
SIGN_ACC_ALPHA = 0.602           # 增益指数（Spall 标准值；旧变体用 DEFAULTS["dspsa_alpha"] = 0.501）
TAU_KAPPA = 0.5                  # τ_j = κ·稳健离散度
TAU_FLOOR_DB = 1e-3
TAU_EMA = 0.05                   # log 离散度的指数滑动平均系数
_MAD_K = 1.4826                  # MAD → σ（正态一致）


def _variant(system, variant: str) -> str:
    """'auto'：所有分量电平数 > 2 → 'faithful'（Wang & Spall / Song 原版），否则 'sign_acc'（INTEGRATION T 的 1-bit 默认；
    旧 1-bit 默认 'subset' 仍可显式选择）。"""
    if variant != "auto":
        if variant not in DSPSA_VARIANTS:
            raise ValueError(f"未知 DSPSA 变体 {variant!r}，可选 {DSPSA_VARIANTS} 或 'auto'")
        return variant
    return "faithful" if np.all(np.asarray(system.levels) > 2) else "sign_acc"


# ============================================================ 可测目标（只用一次测量的读数，INTEGRATION T）
class AccuracyObjective:
    """可测"判对优先"目标（越小越好）：J = −(n_correct + ½·mean tanh(M/τ))，由**带噪读数**算出，不额外耗测量。

    n_correct：Task.decide 判对的样本数（被标记可能自激的样本 / 平局算错，同 evaluate）；平局项 ∈ (−½, ½) 只在判对数相同时起作用。
    tie="sample"：M = 样本裕量 s_b = min_j M_bj（= oracle_objective("accuracy") 的原式，作用在带噪读数上）；
    tie="unit"：M = 逐判决单元裕量 M_bj（Task.decision_units：二分类 1 列、多分类每竞争类一列、loc3bit 每探头比特一列），
               mean 对 (b, j)。被标记样本的裕量记 −FLAG_MARGIN_DB。
    adaptive=True：τ_j = max(tau_floor, κ·exp(EMA log 稳健离散度_j))，每次 observe(读数) 更新（tie="unit" 时逐单元列，
               "sample" 时一个 τ）；未观测前 τ = 1。adaptive=False：固定 τ = tau。
    构造：from_task(task, labels, ...)（系统侧）或直接给 kind / labels / decide / units 回调（bench 的 MeasurementBudget 侧）。
    """

    def __init__(self, kind: str, labels, decide, units, tau: float = 1.0, tie: str = "sample",
                 adaptive: bool = False, kappa: float = TAU_KAPPA, tau_floor: float = TAU_FLOOR_DB,
                 tau_ema: float = TAU_EMA):
        if kind == "focus":
            raise ValueError("focus 任务没有判决单元，请用系统可测损失")
        if tie not in ("sample", "unit"):
            raise ValueError(f"tie 只能是 'sample' | 'unit'，得到 {tie!r}")
        self.kind = str(kind)
        self.labels = np.asarray(labels)
        self._decide, self._units = decide, units
        self.tau, self.tie, self.adaptive = float(tau), tie, bool(adaptive)
        self.kappa, self.tau_floor, self.tau_ema = float(kappa), float(tau_floor), float(tau_ema)
        self.log_s = None                       # log 离散度的 EMA（(J,) 或 (1,)）

    @classmethod
    def from_task(cls, task, labels=None, **kw) -> "AccuracyObjective":
        lab = np.asarray(task.labels if labels is None else labels)
        return cls(task.kind, lab, lambda y: task.decide(y, lab), lambda y: task.decision_units(y, lab), **kw)

    @staticmethod
    def spread(M: np.ndarray) -> np.ndarray:
        """(J,)：每列的样本间稳健离散度 √(MAD·max(sd, MAD))（MAD = 1.4826·median|M − median|；MAD = 0 时取 sd）。"""
        med = np.median(M, axis=0, keepdims=True)
        mad = _MAD_K * np.median(np.abs(M - med), axis=0)
        sd = np.std(M, axis=0)
        return np.where(mad > 0, np.sqrt(mad * np.maximum(sd, mad)), sd)

    def observe(self, y, fl) -> None:
        """自适应温度：用这次读数（未标记样本，≥ 2 个且有限）更新 log 离散度 EMA；adaptive=False 时不做事。"""
        if not self.adaptive:
            return
        ok = ~np.asarray(fl, dtype=bool)
        if ok.sum() < 2 or not np.all(np.isfinite(y[ok])):
            return
        try:
            M = self._units(np.asarray(y, dtype=float))[ok]
        except ValueError:
            return
        s = self.spread(M) if self.tie == "unit" else self.spread(np.min(M, axis=1, keepdims=True))
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            return
        ls = np.log(s)
        self.log_s = ls if self.log_s is None else self.log_s + self.tau_ema * (ls - self.log_s)

    def tau_now(self):
        """当前温度：adaptive 时 (J,) 数组（未观测前 1.0），否则固定 τ。"""
        if not self.adaptive:
            return self.tau
        if self.log_s is None:
            return 1.0
        return np.maximum(self.tau_floor, self.kappa * np.exp(self.log_s))

    def __call__(self, y, fl, tau=None) -> float:
        """J(读数, 标记)；tau=None 用 tau_now()。非有限读数（未标记样本）或判决单元无定义 → +inf。"""
        tau_u = self.tau_now() if tau is None else tau
        y = np.asarray(y, dtype=float)
        fl = np.asarray(fl, dtype=bool)
        if not np.all(np.isfinite(y[~fl])):
            return math.inf
        try:
            M = self._units(y)
        except ValueError:
            return math.inf
        pred = np.asarray(self._decide(y)).copy()
        pred[fl] = -1
        n_ok = float(np.sum(pred == self.labels))
        if self.tie == "unit":
            Mu = np.where(fl[:, None], -FLAG_MARGIN_DB, M)
            tb = float(np.mean(np.tanh(Mu / np.broadcast_to(tau_u, Mu.shape[1:]))))
        else:
            sm = np.where(fl, -FLAG_MARGIN_DB, np.min(M, axis=1))
            t = float(np.mean(tau_u)) if np.ndim(tau_u) else float(tau_u)
            tb = float(np.mean(np.tanh(sm / t)))
        f = -(n_ok + 0.5 * tb)
        return f if math.isfinite(f) else math.inf


def measurable_loss_from_readings(system, readings, flags) -> float:
    """系统可测损失（任务损失 τ = cfg.tau_db，被标记样本替换为 chance + osc_penalty）只由读数算出
    （= bench MeasurementBudget.loss_from_readings；bg_cancel="null" 的零陷项不在读数里，不计）。非有限 → +inf。"""
    from . import losses

    task = system.task
    labels = np.asarray(system.labels)
    tau = float(system.cfg.tau_db)
    y = np.asarray(readings, dtype=float)
    per = np.asarray(task.loss(y, labels, tau, reduction="none"), dtype=float)
    fl = np.asarray(flags, dtype=bool)
    if np.any(fl):
        ch = losses.chance_loss(task.kind, system.n_probes, tau, **task.task_kwargs)
        base = per if ch is None else np.full_like(per, ch)
        per = np.where(fl, base + float(system.cfg.osc_penalty), per)
    f = float(np.mean(per))
    return f if math.isfinite(f) else math.inf


def measurable_objective(system, objective: str = "accuracy", tie: str = "sample", adaptive: bool = False,
                         tau: float | None = None):
    """带噪读数 → 标量目标（越小越好）的可调用对象 f(readings (B,K), flags (B,))，不计测量次数（INTEGRATION T）。

    "accuracy"：AccuracyObjective（tie / adaptive 见其 docstring；tau=None → cfg.tau_db）——tie="sample"、固定 τ 时即
    oracle_objective("accuracy") 作用在测量读数上。"loss"：measurable_loss_from_readings。focus 任务一律 "loss"。
    """
    if system.task.kind == "focus" or objective == "loss":
        return lambda y, fl: measurable_loss_from_readings(system, y, fl)
    if objective != "accuracy":
        raise ValueError(f"objective 只能是 'accuracy' | 'loss'，得到 {objective!r}")
    t = float(system.cfg.tau_db) if tau is None else float(tau)
    return AccuracyObjective.from_task(system.task, system.labels, tau=t, tie=tie, adaptive=adaptive)


def acc_first_loss_fn(measure, objective, pocket: dict | None = None, counter: dict | None = None,
                      loss_of=None, pocket_tau: float = 1.0):
    """sign_acc 的一次测量 → dspsa 的 loss_fn(x, rng)（INTEGRATION T；与 _scratch df_u_nores 同一顺序）。

    measure(x) → (readings, flags)：一次带噪测量。objective：AccuracyObjective（自适应 τ，先 observe 本次读数再求值）；
    None（focus）→ loss_of(readings, flags)（系统可测损失）。pocket（dict，可选）：键 = objective(·, τ = pocket_tau)
    （focus：系统损失），严格更小才更新 {"params", "key", "at_measurement"}。counter["n"] 计测量次数。
    """
    cnt = {"n": 0} if counter is None else counter

    def loss_fn(x, _r):
        y, fl = measure(x)
        cnt["n"] = cnt.get("n", 0) + 1
        if objective is None:
            f = float(loss_of(y, fl))
            f = f if math.isfinite(f) else math.inf
            key = f
        else:
            objective.observe(y, fl)
            f = objective(y, fl)
            key = objective(y, fl, tau=pocket_tau) if pocket is not None else None
        if pocket is not None and key < pocket["key"]:
            pocket.update(params=np.asarray(x, dtype=np.int64).copy(), key=float(key), at_measurement=int(cnt["n"]))
        return f

    return loss_fn


def _measurable_loss_fn(system, counter: dict, pocket: dict | None = None, noise_rng=None):
    """DSPSA 的一次测量：有噪声前向 → 可测目标（审查 INT-SEM-3）；记录测量数、被标记测量数与 pocket（测到的最好码本）。
    noise_rng=None：测量噪声用 dspsa 传入的 Generator（= 算法随机流，原行为）。"""

    def loss_fn(x, r):
        m = system.forward(system.phases_from_params(x), r if noise_rng is None else noise_rng, noisy=True)
        counter["n"] += 1
        counter["flagged"] += int(m.flags.any())
        f = system.loss_from(m, model_penalties=False)
        if pocket is not None and np.isfinite(f) and f < pocket["loss_measured"]:
            pocket.update(params=np.asarray(x, dtype=np.int64).copy(), loss_measured=float(f),
                          at_measurement=int(counter["n"]))
        return f

    return loss_fn


def calibrate_dspsa_gain(system, rng: np.random.Generator, n_starts: int = 1, n_est: int = 10,
                         variant: str = "auto", subset_size: int | None = None, A: float | None = None,
                         alpha: float | None = None) -> dict:
    """在 system 上按 a="auto" 的规则标定 DSPSA 增益（dspsa.auto_gain，审查 S2-4），n_starts 个随机初始码本合并平均。

    用途：实验 E（RQ6）比较 M1/M2/M3/M4 的收敛速度时，在参考条件上标定一次，再把同一个 res["a"] 传给各条件的
    run_dspsa(a=...)——否则 a="auto" 会按各自的 mean|ĝ| 归一化，"深饱和使损失地形整体变平"在增益上被抵消。
    返回 {"a", "mean_abs_g", "n_measurements", "variant"}（这些测量不计入之后训练的测量数）。
    variant="auto" 在 1-bit 上解析为 sign_acc，但符号梯度无需标定，这里按 "subset" 扰动标定（返回 variant="subset"）。
    """
    from .constants import DEFAULTS
    from .dspsa import auto_gain

    A = DEFAULTS["dspsa_A"] if A is None else A
    alpha = DEFAULTS["dspsa_alpha"] if alpha is None else alpha
    var = _variant(system, variant)
    if var == "sign_acc":           # 符号梯度不需要标定；增益标定只对 raw 梯度变体有意义 → 按 subset 扰动标定
        var = "subset"
    counter = {"n": 0, "flagged": 0}
    loss_fn = _measurable_loss_fn(system, counter)
    gs = []
    for _ in range(max(1, int(n_starts))):
        cal = auto_gain(loss_fn, np.asarray(system.levels), rng, variant=var, subset_size=subset_size, A=A,
                        alpha=alpha, n_est=n_est)
        if not cal["degenerate"]:
            gs.append(cal["mean_abs_g"])
    mean_g = float(np.mean(gs)) if gs else float("nan")
    a_val = 0.3 * (1.0 + A) ** alpha / (mean_g if gs else 1.0)
    return {"a": float(a_val), "mean_abs_g": mean_g, "n_measurements": int(counter["n"]), "variant": var}


def run_dspsa(system, n_iter: int, rng: np.random.Generator, variant: str = "auto", subset_size: int | None = None,
              a="auto", eval_every: int = 100, theta0=None, n_noise_eval: int = 20, A: float | None = None,
              alpha: float | None = None, pocket: bool = False, noise_rng: np.random.Generator | None = None,
              step0: float | None = None) -> dict:
    """DSPSA 训练（INTEGRATION D；1-bit 默认变体见 INTEGRATION T）。

    variant："auto"（1-bit → "sign_acc"，多电平 → "faithful"）| "sign_acc" | "subset"（旧 1-bit 默认）| "faithful"。
    返回 {"params", "eval", "history", "n_measurements", "variant", "curve", "n_flagged_measurements", "time_s",
          "objective"="measurable", "train_objective", "a"}：
      curve：[{iter, loss_clean, loss_meas_clean, acc_clean, n_flagged}]，k = 0、eval_every 的倍数与 n_iter
      （无噪声，不计入测量次数；loss_meas_clean 是系统可测损失去掉噪声，loss_clean 是模型目标）；
      n_flagged_measurements：训练中出现任一样本被标记可能自激的测量次数（实验 D 的"中间码本自激比例"）。
    noise_rng：测量噪声的 Generator（None → 与算法共用 rng，原行为；给定时算法 / 噪声随机流分开，便于与 bench 逐位对照）。
    subset / faithful（旧变体，行为逐位不变）：softplus 可测损失（train_objective="loss"），a = 数值（例如
      calibrate_dspsa_gain 在参考条件上的结果，审查 S2-4）或 "auto"；params / eval = 最终判决 round(θ)（Song 的输出）。
      pocket=True（审查 S1-0，默认关）：同时记录全部测量（含 a="auto" 标定的 20 次）中**测到的**可测目标最小的码本，
      返回 "pocket" = {"params", "loss_measured"（那次带噪测量值）, "at_measurement"（第几次测量）, "eval"}，
      curve 每点另加 pocket_loss_meas_clean / pocket_acc_clean / pocket_at_measurement（到该迭代为止的 pocket 码本，
      无噪声复评，不计入测量次数）。选择依据是带噪测量：VNA 级噪声下可忽略；低 SNR 时 pocket 有"赢家诅咒"偏差，
      其 eval（独立复评）才是可报告的值。
    sign_acc（INTEGRATION T，仅 1-bit）：判对优先可测目标（逐单元自适应 τ；focus → 系统可测损失，train_objective="loss"）
      + sign-SPSA（step0 = 0.1、α = SIGN_ACC_ALPHA、A = 400；a 为数值时代替 step0）+ 子集 ±½ 扰动；pocket 是算法的一部分：
      params / eval = pocket 码本（判对优先目标 @τ = cfg.tau_db 的带噪测量最小者，严格改进才替换），"decision_params" =
      最终 round(θ)；curve 记录当时的输出码本（pocket，尚无测量时为 round(θ)）。pocket=True 时另返回同形的 "pocket"
      （loss_measured = pocket 键值，eval 与主 eval 相同）。测量数 = 2·n_iter（无标定）。
    """
    from .constants import DEFAULTS

    t0 = time.perf_counter()
    var = _variant(system, variant)
    if var == "sign_acc":
        return _run_sign_acc(system, int(n_iter), rng, subset_size=subset_size, a=a, eval_every=eval_every,
                             theta0=theta0, n_noise_eval=n_noise_eval, A=A, alpha=alpha, pocket=pocket,
                             noise_rng=noise_rng, step0=step0, t0=t0)
    counter = {"n": 0, "flagged": 0}
    pk = {"params": None, "loss_measured": np.inf, "at_measurement": None} if pocket else None
    loss_fn = _measurable_loss_fn(system, counter, pk, noise_rng)
    levels = np.asarray(system.levels)

    def callback(k, theta):
        d = system.clean_metrics(decide(theta, levels))
        d["iter"] = int(k)
        if pk is not None:
            if pk["params"] is None:
                d.update(pocket_loss_meas_clean=None, pocket_acc_clean=None, pocket_at_measurement=None)
            else:
                dp = system.clean_metrics(pk["params"])
                d.update(pocket_loss_meas_clean=dp["loss_meas_clean"], pocket_acc_clean=dp["acc_clean"],
                         pocket_at_measurement=pk["at_measurement"])
        return d

    res = dspsa(loss_fn, levels, int(n_iter), rng, a=a, A=DEFAULTS["dspsa_A"] if A is None else A,
                alpha=DEFAULTS["dspsa_alpha"] if alpha is None else alpha, variant=var, subset_size=subset_size,
                theta0=theta0, callback=callback, callback_every=max(1, int(eval_every)))
    params = np.asarray(res.bits, dtype=np.int64)
    ev = system.evaluate(params, rng, n_noise=n_noise_eval)
    out = {"params": params, "eval": ev, "history": res.history, "n_measurements": int(res.n_measurements),
           "variant": var, "curve": list(res.history.get("callback", [])),
           "n_flagged_measurements": int(counter["flagged"]), "a": float(res.history["a"][0]),
           "objective": "measurable", "train_objective": "loss", "time_s": time.perf_counter() - t0}
    if pk is not None:
        pev = None if pk["params"] is None else system.evaluate(pk["params"], rng, n_noise=n_noise_eval)
        out["pocket"] = {"params": pk["params"], "loss_measured": _finite_or_none(pk["loss_measured"]),
                         "at_measurement": pk["at_measurement"], "eval": pev}
        out["time_s"] = time.perf_counter() - t0
    return out


def _run_sign_acc(system, n_iter: int, rng: np.random.Generator, *, subset_size, a, eval_every, theta0,
                  n_noise_eval, A, alpha, pocket, noise_rng, step0, t0) -> dict:
    """run_dspsa(variant="sign_acc") 的实现（INTEGRATION T；= _scratch dspsa_family df_u_nores，逐位一致见 tests）。"""
    from .constants import DEFAULTS

    levels = np.asarray(system.levels)
    if np.any(levels != 2):
        raise ValueError("sign_acc 只支持 1-bit 参数（levels 全为 2）；多电平请用 'faithful'")
    nrng = rng if noise_rng is None else noise_rng
    counter = {"n": 0, "flagged": 0}
    focus = system.task.kind == "focus"
    obj = None if focus else AccuracyObjective.from_task(system.task, system.labels, tie="unit", adaptive=True)
    pk = {"params": None, "key": math.inf, "at_measurement": None}

    def measure(x):
        m = system.forward(system.phases_from_params(x), nrng, noisy=True)
        counter["flagged"] += int(np.any(m.flags))
        return m.readings_dbm, m.flags

    loss_fn = acc_first_loss_fn(measure, obj, pk, counter,
                                loss_of=lambda y, fl: measurable_loss_from_readings(system, y, fl),
                                pocket_tau=float(system.cfg.tau_db))

    def current(theta):
        return pk["params"] if pk["params"] is not None else decide(theta, levels)

    def callback(k, theta):
        d = system.clean_metrics(current(theta))
        d.update(iter=int(k), n_meas=int(counter["n"]))
        if pocket:
            d.update(pocket_loss_meas_clean=d["loss_meas_clean"], pocket_acc_clean=d["acc_clean"],
                     pocket_at_measurement=pk["at_measurement"])
        return d

    use_step0 = isinstance(a, str) or a is None
    res = dspsa(loss_fn, levels, int(n_iter), rng, a=1.0 if use_step0 else float(a),
                A=DEFAULTS["dspsa_A"] if A is None else A, alpha=SIGN_ACC_ALPHA if alpha is None else alpha,
                variant="subset", subset_size=subset_size, theta0=theta0, callback=callback,
                callback_every=max(1, int(eval_every)), grad="sign",
                step0=(SIGN_ACC_STEP0 if step0 is None else float(step0)) if use_step0 else None,
                gain_index="update")
    decision = np.asarray(res.bits, dtype=np.int64)
    params = decision if pk["params"] is None else np.asarray(pk["params"], dtype=np.int64)
    ev = system.evaluate(params, rng, n_noise=n_noise_eval)
    tau = None if obj is None else obj.tau_now()
    out = {"params": params, "eval": ev, "history": res.history, "n_measurements": int(res.n_measurements),
           "variant": "sign_acc", "curve": list(res.history.get("callback", [])),
           "n_flagged_measurements": int(counter["flagged"]), "a": float(res.history["a"][0]),
           "objective": "measurable", "train_objective": "loss" if focus else "accuracy_first",
           "decision_params": decision, "pocket_key": _finite_or_none(pk["key"]),
           "pocket_at_measurement": pk["at_measurement"],
           "tau_final": None if tau is None else [float(v) for v in np.atleast_1d(tau)],
           "time_s": time.perf_counter() - t0}
    if pocket:
        out["pocket"] = {"params": pk["params"], "loss_measured": _finite_or_none(pk["key"]),
                         "at_measurement": pk["at_measurement"], "eval": None if pk["params"] is None else ev}
    return out


def run_ils(system, n_meas: int, rng: np.random.Generator, extras: bool = False, kick=(2, 6),
            noise_rng: np.random.Generator | None = None, n_noise_eval: int = 20, eval_every: int = 200) -> dict:
    """迭代局部搜索训练（INTEGRATION T；ils.ils_bits，model-free：只用 system.forward(noisy=True) 的读数与标记）。

    目标 = measurable_objective(system, "accuracy")（判对优先、样本裕量、τ = cfg.tau_db；focus → 系统可测损失）。
    n_meas：测量预算（每次尝试 1 次测量，与 run_dspsa 的 2·n_iter 可比）。extras：拒绝时重测 x* + 末段 4×3 终选赛
    （= _scratch ao_ils4_k0_drift；False = ao_ils_acc）。noise_rng=None 时测量噪声与算法共用 rng（同 run_dspsa）。
    返回与 run_dspsa 同形：{"params", "eval", "history", "curve", "n_measurements", "n_flagged_measurements", "variant",
      "objective"="measurable", "train_objective", "J", "n_ils_iter", "n_accept", "n_star_remeasure", "time_s"}；
      curve：测量数 0、eval_every 的倍数与最终处，当时输出码本（pocket）的无噪声 clean_metrics（不计测量次数），
      每点含 n_meas 与 iter = n_meas // 2（等测量 DSPSA 迭代数，便于与 run_dspsa 的曲线对齐）；
      history["pocket_n"]：pocket 改变时的已用测量数。
    """
    from .ils import ils_bits

    t0 = time.perf_counter()
    levels = np.asarray(system.levels)
    if np.any(levels != 2):
        raise ValueError("run_ils 只支持 1-bit 参数（levels 全为 2）")
    nrng = rng if noise_rng is None else noise_rng
    score = measurable_objective(system, "accuracy", tie="sample", adaptive=False)
    counter = {"n": 0, "flagged": 0}

    def measure(x):
        m = system.forward(system.phases_from_params(x), nrng, noisy=True)
        counter["n"] += 1
        counter["flagged"] += int(np.any(m.flags))
        return m.readings_dbm, m.flags

    changes: list = []
    res = ils_bits(measure, score, int(system.n_params), int(n_meas), rng, kick=kick, extras=bool(extras),
                   callback=lambda n, x: changes.append((int(n), np.asarray(x, dtype=np.int64).copy())))
    params = np.asarray(res["params"], dtype=np.int64)
    n_used = int(res["n_measurements"])
    # ---- 学习曲线（无噪声复评，相同码本只评一次）
    cache: dict = {}

    def cm(x):
        k = x.tobytes()
        if k not in cache:
            cache[k] = system.clean_metrics(x)
        return cache[k]

    step = max(1, int(eval_every))
    cps = sorted(set(list(range(0, n_used, step)) + [n_used]))
    curve = []
    for c in cps:
        if c >= n_used:
            x = params
        else:
            x = None
            for n, xx in changes:
                if n <= c:
                    x = xx
                else:
                    break
        if x is None:
            continue
        d = dict(cm(x))
        d.update(n_meas=int(c), iter=int(c) // 2)
        curve.append(d)
    ev = system.evaluate(params, rng, n_noise=n_noise_eval)
    return {"params": params, "eval": ev, "history": {"pocket_n": [n for n, _ in changes]}, "curve": curve,
            "n_measurements": n_used, "n_flagged_measurements": int(counter["flagged"]),
            "variant": "ils_extras" if extras else "ils", "objective": "measurable",
            "train_objective": "loss" if system.task.kind == "focus" else "accuracy_first",
            "J": res["J"], "n_ils_iter": int(res["n_ils_iter"]), "n_accept": int(res["n_accept"]),
            "n_star_remeasure": int(res["n_star_remeasure"]), "time_s": time.perf_counter() - t0}


def _finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


FLAG_MARGIN_DB = 1.0e3          # oracle 的裕量 / 准确率目标里，被标记可能自激的样本按裕量 −FLAG_MARGIN_DB 计（= 判错）


def oracle_objective(system, objective: str):
    """run_oracle_bits 的无噪声目标函数 f(params_int) → float（越小越好；审查 S0-2）。

    "loss"：模型目标 system.loss(noisy=False)（DESIGN §6.2，默认）。
    "margin"：−min_b s_b，s_b = task.sample_margins_db（检波器无噪声读数 clean_readings）；被标记样本 s_b = −FLAG_MARGIN_DB。
              最大化最小带符号裕量（> 0 ⇔ 全对，门 G1 的判据量）。
    "accuracy"：−(判对样本数 + ½·mean_b tanh(s_b/τ))：判对数优先（与 evaluate 的 acc_clean 同一判决：被标记 / 平局算错），
              tanh 项 ∈ (−½, ½) 只在判对数相同时打破平局（推开已对的、拉近错的）。给出"这个离散结构最多判对几个"。
    """
    if objective == "loss":
        return lambda x: system.loss(x, None, noisy=False)
    if objective not in ("margin", "accuracy"):
        raise ValueError(f"objective 只能是 'loss' | 'margin' | 'accuracy'，得到 {objective!r}")
    task = system.task
    if task.kind == "focus":
        raise ValueError("focus 任务没有判决裕量，请用 objective='loss'")
    labels = np.asarray(system.labels)
    tau = float(system.cfg.tau_db)

    def f(x):
        m = system.forward(system.phases_from_params(x), noisy=False)
        y = system.clean_readings(m)
        flags = np.asarray(m.flags, dtype=bool)
        if not np.all(np.isfinite(y[~flags])):
            return np.inf
        s = np.asarray(task.sample_margins_db(y, labels), dtype=float)
        s = np.where(flags, -FLAG_MARGIN_DB, s)
        if objective == "margin":
            return -float(np.min(s))
        pred = np.asarray(task.decide(y, labels)).copy()
        pred[flags] = -1
        n_ok = float(np.sum(pred == labels))
        return -(n_ok + 0.5 * float(np.mean(np.tanh(s / tau))))

    return f


def run_oracle_bits(system, rng: np.random.Generator, n_restarts: int = 8, max_sweeps: int = 30,
                    anneal_steps: int = 0, n_noise_eval: int = 20, objective: str = "loss") -> dict:
    """1-bit（或多电平）模型 oracle：无噪声目标上的随机重启坐标下降（DESIGN §6.2）。

    objective（oracle_objective，审查 S0-2）："loss"（默认，模型目标，返回 "objective"="model"）、"margin"（最大化最小
    带符号裕量）、"accuracy"（最大化判对样本数）。"loss" 在不可解任务上的 acc_clean 不是"最好可达准确率"（见模块 docstring）。
    返回 "loss" = 所选目标的最优值（margin：−最小裕量 dB；accuracy：−(判对数 + 平局项)）。
    """
    t0 = time.perf_counter()
    n_calls = {"n": 0}
    obj = oracle_objective(system, objective)

    def f(x):
        n_calls["n"] += 1
        return obj(x)

    params, best = oracle_bits(f, system.levels, rng, n_restarts=n_restarts, max_sweeps=max_sweeps,
                               anneal_steps=anneal_steps)
    params = np.asarray(params, dtype=np.int64)
    ev = system.evaluate(params, rng, n_noise=n_noise_eval)
    return {"params": params, "loss": float(best), "eval": ev, "n_evals": n_calls["n"],
            "objective": "model" if objective == "loss" else objective, "time_s": time.perf_counter() - t0}


def run_oracle_continuous(system, n_restarts: int = 4, steps: int = 800, lr: float = 0.05, seed: int = 0,
                          n_noise_eval: int = 20, init: dict | None = None) -> dict:
    """连续相位 oracle（DESIGN §6.3）：torch Adam 优化 system.to_torch()，再用 system.evaluate_phases 复评。"""
    from .oracle_torch import oracle_continuous

    t0 = time.perf_counter()
    ts = system.to_torch()
    res = oracle_continuous(ts, n_restarts=n_restarts, steps=steps, lr=lr, seed=seed, init=init)
    ev = system.evaluate_phases(res["phases"], np.random.default_rng(seed), n_noise=n_noise_eval)
    return {"phases": res["phases"], "loss": float(res["loss"]), "eval": ev, "history": res["history"],
            "restart_best": res["restart_best"], "objective": "model", "time_s": time.perf_counter() - t0}
