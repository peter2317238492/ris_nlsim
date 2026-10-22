"""零阶训练器基准（RQ6 工具；方案 §4.7、§6.2 表"E 可训练性"、§7"收敛迭代数"；DESIGN §6；INTEGRATION D、G2、B3）。

目的：在**同一预算、同一噪声、只用硬件可得信息**的条件下比较 1-bit 中继的 model-free 训练器，并据此选默认训练器。

公平性约定（model-free，方案 §4.7）：
- 训练器只拿到 MeasurementBudget：参数个数 / 电平数、训练样本标签、任务类型、探头数、损失温度 τ，以及测量接口：
    measure(params) → (readings_dbm (B,K) 带噪读数（系统检波器原样输出）, unstable_flags (B,) bool)
    measure_loss(params) → 系统自己的带噪可测目标（PhysicalSystem.loss_from(model_penalties=False)：任务损失 + 被标记
                          样本替换为 chance + osc_penalty + 零陷项；与 train.run_dspsa 的一次测量完全相同）
  **一次测量 = 对全部训练样本过一遍**（方案 §4.7：DSPSA 每次迭代 2 次测量）。两者都计 1 次，硬预算用尽抛 BudgetExhausted。
  另有不计测量次数的纯数据函数（loss_from_readings / predict / accuracy_from_readings：任务定义本身，可换 τ）
  与几何先验码本 focus_params（量化共轭，由已知馈源 / 探头位置算出，硬件上同样可得，0 次测量）。
- report(params) 记录训练器"当前判决"，由基准在无噪声模型上复评画学习曲线（不计测量次数，训练器拿不到结果）。
- 内置训练器（INTEGRATION T）：dspsa_sign_acc（= train.run_dspsa 的新 1-bit 默认 sign_acc）、ils / ils_extras
  （= train.run_ils）、旧默认 dspsa_subset / dspsa_subset_pocket、dspsa_faithful、coord_descent、random_search、focus_init。
- 训练器接口：fn(meas, rng) → params_int。随机流（公共随机数，配对比较）：同一 (任务, 种子) 下所有训练器的
  rng 种子相同、测量噪声流种子相同；噪声 Generator 训练器拿不到。训练器超预算（BudgetExhausted）或返回 None 时，
  最终判决 = 最后一次 report（记 overran / fallback）。

任务（每个 (任务, 种子) 在进程内只构造一次；本版物理实例与种子无关，种子只改变训练器 / 噪声 / oracle 重启的随机流，
构造期标定随机流（logdet / M1b）按 (任务, 种子) 派生）：
  focus         M3 full，FocusTask（单样本无物体，最大化探头 0 功率）——DSPSA 已知能学会的对照
  letters       M3 full，金属薄板字母 A/B/C（3 样本、3 探头 argmax）
  loc3bit       M3 full，Song 2026 式 8 格定位（水球 λ/3，一套码本、3 探头阈值 → 3-bit）
  parity2_born  M3 born（DWBA，水散射被高估 ~30 dB），2 位奇偶
  parity3_born  M3 born，3 位奇偶
  parity2_full  M3 full，2 位奇偶（默认场景物体调制 ~ −30 dB，判决裕量 ≲ 0.1 dB；默认 8 次重启的 loss oracle 解不出，
                但 accuracy 目标 oracle 与逐单元贪心偶尔能全对，基线结果见 results/bench_trainers/baseline_b3000_s5）

oracle 参照（每个 (任务, 种子)，缓存为 JSON：_scratch/bench_cache/<任务>_s<种子>_<实例哈希>_r<重启>w<轮数>.json，
探索者可直接复用 oracle_refs()）：train.run_oracle_bits(objective="accuracy") 与 (objective="loss")，另记几何聚焦码本。
实例哈希 = BENCH_VERSION + 任务规格 + 物理指纹（几个固定码本的无噪声读数，物理代码改动后自动失效）。
参照值 ref（审查 S0-2 / 实验 E 同做法）：同一实例哈希下**全部种子、两种目标**（及聚焦码本）的最好值——
  "全部种子"= 本次运行的种子 + 缓存目录里同一实例哈希的**全部**缓存 JSON（任意种子、任意 oracle 设置；INTEGRATION T，
  参照不再随本次 --seeds 变化而变弱），n_pooled_seeds 记录合并的种子数；
  ref_acc = 最高无噪声准确率，ref_power = focus 的最高探头 0 功率，ref_loss = 最小 loss_meas_clean。
  （实测：字母任务 accuracy 目标的坐标下降会停在 2/3，loss 目标才到 3/3，所以只取单一目标会低估上界。）

指标（每个 (训练器, 任务, 种子)）：最终码本的无噪声准确率 / 全对 / 可测目标 loss_meas_clean（focus：探头 0 功率与
ref 之差 Δp dB）、与 ref 的差距、所用测量数、学习曲线（report 的判决在 curve_points+1 个等距测量检查点上的无噪声
acc / loss / 功率）、训练器墙钟时间。达标判据：分类 acc ≥ 0.9·ref_acc（另报 chance 归一化分数
(acc − chance)/(ref_acc − chance)）；focus 功率 ≥ ref_power − 1 dB。measurements-to-90% = 学习曲线首个达标检查点
（分辨率 = budget/curve_points；未达标记 None = 删失）。
汇总（每个 (训练器, 任务)）：均值 [95% t 区间]、达标比例、达标测量数中位数（删失按 +∞，过半删失记 ">budget"）。

用法：python -m ris_nlsim.experiments.bench_trainers [--trainers ...] [--plugin path.py ...] [--tasks ...]
        [--seeds 5] [--budget 3000] [--workers 3] [--out DIR] [--include-runs OLD/runs.jsonl ...]
插件：一个 .py 文件定义 TRAINERS = {名字: fn(meas, rng) → params}（可选 BENCH_TASKS = {名字: TaskSpec 或 dict}），
按路径 importlib 加载（多进程下各 worker 各自加载，fn 无需可 pickle）。--include-runs 合并旧 runs.jsonl 中实例哈希与预算
都相同的行（例如基线），避免重跑。输出：runs.jsonl、summary.md、summary.csv、curves.png、references.json、config.json。
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import common

EXP = "bench_trainers"
BENCH_VERSION = 1                       # 改变任务定义 / 参照计算方式时 +1（使旧缓存失效）
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PKG_ROOT, "_scratch", "bench_cache")
DEFAULT_BUDGET = 3000
DEFAULT_WORKERS = 3                     # 多 agent 共用笔记本：默认 3 个进程
DEFAULT_CURVE_POINTS = 100
REACH_FRAC = 0.9                        # 分类：acc ≥ 0.9·ref_acc（方案 §7"达到 oracle 90%"）
FOCUS_TOL_DB = 1.0                      # focus：功率 ≥ ref_power − 1 dB
DSPSA_AUTO_GAIN_MEAS = 20               # a="auto" 标定用的测量数（dspsa.auto_gain：n_est = 10，每次 2 次测量）
_EPS = 1e-9


class BudgetExhausted(RuntimeError):
    """测量预算用尽（硬上限）。训练器可不捕获：基准把最后一次 report 当最终判决。"""


# ============================================================ 测量预算包装（只暴露硬件可得信息）
class MeasurementBudget:
    """把 PhysicalSystem 包成"硬件"：只给测量与任务定义，不给模型（方案 §4.7 model-free）。

    公开属性（只读副本）：n_params、levels (p,)、labels (B,)、kind、n_probes、n_samples、tau_db、task_name、
    budget、n_used、remaining、n_flagged_measurements。
    计数接口（各 1 次测量，超预算抛 BudgetExhausted）：measure(params) → (readings_dbm (B,K), flags (B,))；
    measure_loss(params) → float。
    不计数：loss_from_readings / predict / accuracy_from_readings（任务定义）、focus_params（几何先验）、
    random_params、report（学习曲线记录）。带下划线 / 名字改编的属性（系统、噪声流）不属于接口，训练器不得使用。
    """

    def __init__(self, system, budget: int, noise_rng: np.random.Generator, task_name: str = ""):
        if not isinstance(noise_rng, np.random.Generator):
            raise TypeError("noise_rng 必须是 np.random.Generator")
        self.__sys = system
        self.__rng = noise_rng
        self.budget = int(budget)
        self.task_name = str(task_name)
        self._n_used = 0
        self._n_flagged = 0
        self._reports: list[tuple[int, np.ndarray]] = []
        self._levels = np.asarray(system.levels, dtype=np.int64).copy()
        self._labels = np.asarray(system.labels, dtype=np.int64).copy()
        self.n_params = int(self._levels.size)
        self.kind = str(system.task.kind)
        self.n_probes = int(system.n_probes)
        self.n_samples = int(system.B)
        self.tau_db = float(system.cfg.tau_db)
        self._osc_penalty = float(system.cfg.osc_penalty)

    # ---------------------------------------------------------------- 只读信息
    @property
    def levels(self) -> np.ndarray:
        return self._levels.copy()

    @property
    def labels(self) -> np.ndarray:
        return self._labels.copy()

    @property
    def n_used(self) -> int:
        return self._n_used

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self._n_used)

    @property
    def n_flagged_measurements(self) -> int:
        """出现任一样本被标记可能自激的测量次数（硬件上可由频谱 / 功率跳变看到）。"""
        return self._n_flagged

    @property
    def reports(self) -> list[tuple[int, np.ndarray]]:
        """[(report 时已用测量数, 码本副本)]（基准用；训练器自己知道自己报过什么）。"""
        return [(n, x.copy()) for n, x in self._reports]

    # ---------------------------------------------------------------- 参数
    def check_params(self, params) -> np.ndarray:
        """(p,) 整数电平，越界 / 形状不符抛 ValueError；返回 int64 副本。"""
        x = np.asarray(params)
        if x.shape != (self.n_params,):
            raise ValueError(f"参数形状应为 ({self.n_params},)，得到 {x.shape}")
        xf = x.astype(float)
        if not np.all(np.isfinite(xf)) or np.any(xf != np.round(xf)):
            raise ValueError("参数必须是整数电平")
        x = xf.astype(np.int64)
        if np.any(x < 0) or np.any(x >= self._levels):
            raise ValueError("参数电平越界")
        return x

    def random_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self._levels).astype(np.int64)

    def focus_params(self, probe: int = 0) -> np.ndarray:
        """几何先验码本（PhysicalSystem.focus_params：接收对准馈源 + 发射对准探头的量化共轭，空场景）。
        由已知馈源 / 探头 / 板位置算出，硬件上同样可得；不计测量次数。"""
        return np.asarray(self.__sys.focus_params(int(probe)), dtype=np.int64).copy()

    # ---------------------------------------------------------------- 测量（计数）
    def _forward(self, params):
        x = self.check_params(params)
        if self._n_used >= self.budget:
            raise BudgetExhausted(f"测量预算 {self.budget} 已用尽")
        self._n_used += 1
        s = self.__sys
        m = s.forward(s.phases_from_params(x), self.__rng, noisy=True)
        if np.any(m.flags):
            self._n_flagged += 1
        return m

    def measure(self, params) -> tuple[np.ndarray, np.ndarray]:
        """一次测量：全部训练样本的带噪检波器读数 (B,K) dBm 与自激标记 (B,) bool。"""
        m = self._forward(params)
        return np.array(m.readings_dbm, dtype=float, copy=True), np.array(m.flags, dtype=bool, copy=True)

    def measure_loss(self, params) -> float:
        """一次测量：系统自己的带噪可测目标（= train.run_dspsa 的一次测量，审查 INT-SEM-3）。"""
        m = self._forward(params)
        return float(self.__sys.loss_from(m, model_penalties=False))

    # ---------------------------------------------------------------- 任务定义（不计数）
    def loss_from_readings(self, readings_dbm, flags=None, tau: float | None = None, reduction: str = "mean"):
        """任务损失（losses.task_loss）作用在给定读数上；被标记样本替换为 chance + osc_penalty（同 system.sample_losses）。
        tau=None 用系统 τ（此时与 measure_loss 同一读数下相等；bg_cancel="null" 的零陷项除外，本基准任务均为 "free"）。"""
        from .. import losses

        t = self.tau_db if tau is None else float(tau)
        y = np.asarray(readings_dbm, dtype=float)
        task = self.__sys.task
        per = np.asarray(task.loss(y, self._labels, t, reduction="none"), dtype=float)
        if flags is not None and np.any(flags):
            fl = np.asarray(flags, dtype=bool)
            ch = losses.chance_loss(task.kind, self.n_probes, t, **task.task_kwargs)
            base = per if ch is None else np.full_like(per, ch)
            per = np.where(fl, base + self._osc_penalty, per)
        if reduction == "none":
            return per
        return float(np.mean(per))

    def predict(self, readings_dbm) -> np.ndarray:
        """任务判决（Task.decide：平局 −1）；loc3bit 的自动阈值由这些读数与训练标签求出。"""
        return np.asarray(self.__sys.task.decide(np.asarray(readings_dbm, dtype=float), self._labels)).copy()

    def decision_units(self, readings_dbm) -> np.ndarray:
        """逐判决单元带符号裕量 (B, J)（Task.decision_units，任务定义本身；loc3bit 的自动阈值由这些读数与训练标签求出）。"""
        return np.asarray(self.__sys.task.decision_units(np.asarray(readings_dbm, dtype=float), self._labels),
                          dtype=float).copy()

    def accuracy_from_readings(self, readings_dbm, flags=None) -> float:
        """判对比例（被标记样本算错，同 PhysicalSystem._accuracy）；focus 为 nan。"""
        if self.kind == "focus":
            return float("nan")
        pred = self.predict(readings_dbm)
        if flags is not None:
            pred[np.asarray(flags, dtype=bool)] = -1
        return float(np.mean(pred == self._labels))

    # ---------------------------------------------------------------- 学习曲线记录（不计数）
    def report(self, params) -> None:
        """记录训练器当前判决（基准在无噪声模型上复评，不计测量次数）。"""
        self._reports.append((self._n_used, self.check_params(params)))


# ============================================================ 内置训练器
def _dspsa_trainer(meas: MeasurementBudget, rng: np.random.Generator, variant: str, pocket: bool = False):
    """train.run_dspsa 默认行为经包装重现：a="auto"（20 次标定测量）、A、α 取 DEFAULTS、subset m = max(2, p//8)，
    n_iter = (预算 − 20)//2；每次迭代 report 当前判决 round(θ)（pocket=True 时 report / 返回测到的最好码本）。"""
    from ..constants import DEFAULTS
    from ..dspsa import decide, dspsa

    L = meas.levels
    if meas.remaining < DSPSA_AUTO_GAIN_MEAS + 2:
        x = meas.random_params(rng)
        meas.report(x)
        return x
    n_iter = (meas.remaining - DSPSA_AUTO_GAIN_MEAS) // 2
    pk = {"x": None, "f": np.inf}

    def loss_fn(x, _r):
        f = meas.measure_loss(x)
        if pocket and np.isfinite(f) and f < pk["f"]:
            pk.update(x=np.asarray(x, dtype=np.int64).copy(), f=f)
        return f

    def cb(_k, theta):
        meas.report(pk["x"] if (pocket and pk["x"] is not None) else decide(theta, L))

    res = dspsa(loss_fn, L, int(n_iter), rng, a="auto", A=DEFAULTS["dspsa_A"], alpha=DEFAULTS["dspsa_alpha"],
                variant=variant, callback=cb, callback_every=1)
    if pocket and pk["x"] is not None:
        return pk["x"]
    return np.asarray(res.bits, dtype=np.int64)


def dspsa_subset(meas: MeasurementBudget, rng: np.random.Generator) -> np.ndarray:
    """旧 1-bit 默认：随机子集 DSPSA（DESIGN §6.1，train.run_dspsa(variant="subset")；a="auto"、softplus τ = 1 dB）。"""
    return _dspsa_trainer(meas, rng, "subset")


def _budget_objective(meas: MeasurementBudget, tie: str, adaptive: bool):
    """MeasurementBudget 侧的判对优先目标（train.AccuracyObjective；focus → None）。"""
    from .. import train

    if meas.kind == "focus":
        return None
    return train.AccuracyObjective(meas.kind, meas.labels, meas.predict, meas.decision_units, tau=meas.tau_db, tie=tie,
                                   adaptive=adaptive)


def dspsa_sign_acc(meas: MeasurementBudget, rng: np.random.Generator) -> np.ndarray:
    """新 1-bit 默认（INTEGRATION T）：train.run_dspsa(variant="sign_acc") 经包装重现——判对优先可测目标（逐单元自适应 τ）
    + sign-SPSA（首步 0.1、α = 0.602、A = 400，无标定）+ 子集 ±½ 扰动 + pocket（返回 / report pocket 码本）；
    n_iter = 预算 // 2。非 1-bit 参数退回 dspsa_subset。"""
    from .. import train
    from ..constants import DEFAULTS
    from ..dspsa import decide, dspsa

    L = meas.levels
    if np.any(L != 2):
        return dspsa_subset(meas, rng)
    n_iter = meas.remaining // 2
    pk = {"params": None, "key": np.inf, "at_measurement": None}
    loss_fn = train.acc_first_loss_fn(meas.measure, _budget_objective(meas, "unit", True), pk,
                                      loss_of=meas.loss_from_readings, pocket_tau=meas.tau_db)

    def cb(_k, theta):
        meas.report(pk["params"] if pk["params"] is not None else decide(theta, L))

    res = dspsa(loss_fn, L, int(n_iter), rng, a=1.0, A=DEFAULTS["dspsa_A"], alpha=train.SIGN_ACC_ALPHA,
                variant="subset", callback=cb, callback_every=1, grad="sign", step0=train.SIGN_ACC_STEP0,
                gain_index="update")
    return pk["params"] if pk["params"] is not None else np.asarray(res.bits, dtype=np.int64)


def _ils_trainer(meas: MeasurementBudget, rng: np.random.Generator, extras: bool) -> np.ndarray:
    from ..ils import ils_bits

    if np.any(meas.levels != 2):
        return dspsa_subset(meas, rng)
    obj = _budget_objective(meas, "sample", False)
    score = obj if obj is not None else (lambda y, fl: meas.loss_from_readings(y, fl))
    res = ils_bits(meas.measure, score, meas.n_params, meas.remaining, rng, extras=extras,
                   callback=lambda _n, x: meas.report(x))
    return res["params"]


def ils(meas: MeasurementBudget, rng: np.random.Generator) -> np.ndarray:
    """迭代局部搜索（INTEGRATION T，train.run_ils）：判对优先可测目标（样本裕量、τ = 1 dB）+ 单比特首次改进爬山 +
    2–6 位踢出 + 中性漂移；返回 pocket（测到的单次目标最小者）。"""
    return _ils_trainer(meas, rng, False)


def ils_extras(meas: MeasurementBudget, rng: np.random.Generator) -> np.ndarray:
    """ILS + 拒绝时重测 x*（≤ 8 次）+ 末段 4×3 终选赛（train.run_ils(extras=True)）。"""
    return _ils_trainer(meas, rng, True)


def dspsa_faithful(meas: MeasurementBudget, rng: np.random.Generator) -> np.ndarray:
    """Wang & Spall 原版（1-bit 下退化：x± 恒为互补码本，README §3.5）。"""
    return _dspsa_trainer(meas, rng, "faithful")


def dspsa_subset_pocket(meas: MeasurementBudget, rng: np.random.Generator) -> np.ndarray:
    """subset DSPSA + pocket（train.run_dspsa(pocket=True)，审查 S1-0）：返回训练中测到的可测目标最小的码本。"""
    return _dspsa_trainer(meas, rng, "subset", pocket=True)


def random_search(meas: MeasurementBudget, rng: np.random.Generator) -> np.ndarray:
    """随机码本，各测一次，取测到的最好者（每次改进 report）。"""
    best, best_f = None, np.inf
    while meas.remaining > 0:
        x = meas.random_params(rng)
        f = meas.measure_loss(x)
        if best is None or (np.isfinite(f) and f < best_f):
            best, best_f = x, f
            meas.report(best)
    return best


def coord_descent(meas: MeasurementBudget, rng: np.random.Generator) -> np.ndarray:
    """经典逐单元贪心（测量版 oracle_bits，DESIGN §6.2 的坐标下降）：随机初始码本，随机顺序逐分量试其它电平，
    测到更小损失即接受；一整轮无改进则随机重启。report / 返回测到的最好码本。每次尝试 1 次测量。"""
    L = meas.levels
    best, best_f = None, np.inf
    while meas.remaining > 0:
        x = meas.random_params(rng)
        f = meas.measure_loss(x)
        if best is None or f < best_f:
            best, best_f = x.copy(), f
            meas.report(best)
        improved = True
        while improved and meas.remaining > 0:
            improved = False
            for i in rng.permutation(L.size):
                for lv in range(int(L[i])):
                    if lv == x[i] or meas.remaining <= 0:
                        continue
                    y = x.copy()
                    y[i] = lv
                    fy = meas.measure_loss(y)
                    if fy < f:
                        x, f, improved = y, fy, True
                        if f < best_f:
                            best, best_f = x.copy(), f
                            meas.report(best)
    return best


def focus_init(meas: MeasurementBudget, rng: np.random.Generator) -> np.ndarray:
    """几何聚焦码本，0 次测量（仅作健全性对照）。"""
    x = meas.focus_params()
    meas.report(x)
    return x


TRAINERS: dict[str, Callable] = {
    "dspsa_sign_acc": dspsa_sign_acc,
    "ils": ils,
    "ils_extras": ils_extras,
    "dspsa_subset": dspsa_subset,
    "dspsa_faithful": dspsa_faithful,
    "random_search": random_search,
    "focus_init": focus_init,
    "dspsa_subset_pocket": dspsa_subset_pocket,
    "coord_descent": coord_descent,
}
BUILTIN_TRAINERS = tuple(TRAINERS)


# ============================================================ 任务
@dataclass
class TaskSpec:
    """基准任务：SystemConfig 字段 + make_task 的 kind / kwargs（INTEGRATION E）。"""

    cfg: dict
    kind: str
    kwargs: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {"cfg": dict(self.cfg), "kind": self.kind, "kwargs": dict(self.kwargs)}


_M3_FULL = {"preset": "L90", "condition": "M3", "object_mode": "full"}
_M3_BORN = {"preset": "L90", "condition": "M3", "object_mode": "born"}
TASKS: dict[str, TaskSpec] = {
    "focus": TaskSpec(_M3_FULL, "focus", {}, "M3 full，单样本聚焦（最大化探头 0 功率）"),
    "letters": TaskSpec(_M3_FULL, "letter", {}, "M3 full，金属薄板字母 A/B/C，3 探头 argmax"),
    "loc3bit": TaskSpec(_M3_FULL, "loc3bit", {}, "M3 full，8 格定位（水球 λ/3），3 探头阈值 → 3-bit"),
    "parity2_born": TaskSpec(_M3_BORN, "parity", {"n": 2}, "M3 born（DWBA，水），2 位奇偶"),
    "parity3_born": TaskSpec(_M3_BORN, "parity", {"n": 3}, "M3 born（DWBA，水），3 位奇偶"),
    "parity2_full": TaskSpec(_M3_FULL, "parity", {"n": 2}, "M3 full（水），2 位奇偶（物体调制 ~−30 dB，判决裕量 ≲ 0.1 dB）"),
}
CHANCE_ACC = {"binary": lambda K: 0.5, "multiclass": lambda K: 1.0 / K, "loc3bit": lambda K: 1.0 / 2 ** K}


def _as_taskspec(v) -> TaskSpec:
    if isinstance(v, TaskSpec):
        return v
    if isinstance(v, dict):
        return TaskSpec(cfg=dict(v["cfg"]), kind=str(v["kind"]), kwargs=dict(v.get("kwargs", {})),
                        note=str(v.get("note", "")))
    raise TypeError(f"任务规格应为 TaskSpec 或 dict，得到 {type(v).__name__}")


def chance_acc(kind: str, n_probes: int) -> float | None:
    f = CHANCE_ACC.get(kind)
    return None if f is None else float(f(int(n_probes)))


# ============================================================ 插件
_PLUGINS: dict[str, object] = {}


def load_plugin(path: str):
    """按路径 importlib 加载插件模块（同一进程只加载一次）。模块须定义 TRAINERS dict，可选 BENCH_TASKS dict。"""
    ap = os.path.abspath(path)
    if ap in _PLUGINS:
        return _PLUGINS[ap]
    if not os.path.isfile(ap):
        raise FileNotFoundError(f"插件文件不存在：{ap}")
    name = "bench_plugin_" + hashlib.sha256(ap.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, ap)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if not isinstance(getattr(mod, "TRAINERS", None), dict) and not isinstance(getattr(mod, "BENCH_TASKS", None), dict):
        raise ValueError(f"插件 {ap} 既没有 TRAINERS 也没有 BENCH_TASKS（dict）")
    _PLUGINS[ap] = mod
    return mod


def registry(plugins=()) -> tuple[dict, dict]:
    """内置 + 插件的 (训练器, 任务) 注册表；插件名与已有名字冲突抛 ValueError。"""
    trainers, tasks = dict(TRAINERS), dict(TASKS)
    for p in plugins or ():
        mod = load_plugin(p)
        for k, fn in (getattr(mod, "TRAINERS", None) or {}).items():
            if k in trainers and trainers[k] is not fn:
                raise ValueError(f"插件 {p} 的训练器名 {k!r} 与已有训练器冲突")
            if not callable(fn):
                raise TypeError(f"插件 {p} 的训练器 {k!r} 不可调用")
            trainers[k] = fn
        for k, v in (getattr(mod, "BENCH_TASKS", None) or {}).items():
            spec = _as_taskspec(v)
            if k in tasks and tasks[k].to_dict() != spec.to_dict():
                raise ValueError(f"插件 {p} 的任务名 {k!r} 与已有任务冲突")
            tasks[k] = spec
    return trainers, tasks


# ============================================================ 系统构造与指纹（进程内缓存）
_SYSTEMS: dict = {}


def _spec_json(spec: TaskSpec) -> str:
    return json.dumps(common.to_jsonable(spec.to_dict()), sort_keys=True, ensure_ascii=False)


def build_system(task: str, seed: int, tasks: dict | None = None):
    """(任务, 种子) → PhysicalSystem（进程内缓存）。构造期随机流 seed_rng(EXP, "calib", 任务, 种子)。"""
    from ..system import SystemConfig, make_system

    spec = (tasks or TASKS)[task]
    key = (task, int(seed), _spec_json(spec))
    if key not in _SYSTEMS:
        cfg = SystemConfig(**spec.cfg)
        _SYSTEMS[key] = make_system(cfg, spec.kind, common.seed_rng(EXP, "calib", task, int(seed)),
                                    **dict(spec.kwargs))
    return _SYSTEMS[key]


def physics_fingerprint(system) -> str:
    """物理指纹：聚焦码本 + 2 个固定随机码本的无噪声检波器读数（6 位小数）与标签 / 电平的哈希。
    物理代码改动（读数变化 > 1e-6 dB）后 oracle 缓存自动失效；文档 / 重构改动不影响。"""
    rng = np.random.default_rng(20261019)
    xs = [system.focus_params()] + [system.random_params(rng) for _ in range(2)]
    vals = []
    for x in xs:
        m = system.forward(system.phases_from_params(x), noisy=False)
        y = np.round(np.asarray(system.clean_readings(m), dtype=float), 6) + 0.0
        vals.append([[None if not math.isfinite(v) else v for v in row] for row in y.tolist()])
    payload = {"readings": vals, "labels": np.asarray(system.labels).tolist(),
               "levels": np.asarray(system.levels).tolist(), "kind": system.task.kind}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def instance_hash(task: str, seed: int, tasks: dict | None = None, system=None) -> str:
    """实例哈希 = BENCH_VERSION + 任务规格 + 物理指纹（与种子无关，除非构造期随机流改变了物理）。"""
    spec = (tasks or TASKS)[task]
    system = build_system(task, seed, tasks) if system is None else system
    payload = {"v": BENCH_VERSION, "spec": json.loads(_spec_json(spec)), "fp": physics_fingerprint(system)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


# ============================================================ 无噪声复评
def clean_eval(system, params) -> dict:
    """无噪声复评（一次前向）：acc（被标记 / 平局算错）、all_correct（acc = 1 且裕量 > 0）、margin_db、
    loss（可测目标 loss_meas_clean）、loss_model（模型目标）、power（focus：探头 0 dBm，被标记为 None）、n_flagged。"""
    task = system.task
    m = system.forward(system.phases_from_params(params), noisy=False)
    y = np.asarray(system.clean_readings(m), dtype=float)
    flags = np.asarray(m.flags, dtype=bool)
    labels = np.asarray(system.labels)
    out = {"loss": common.to_jsonable(float(system.loss_meas_clean(m))),
           "loss_model": common.to_jsonable(float(system.loss_from(m))), "n_flagged": int(flags.sum())}
    if task.kind == "focus":
        p = float(y[0, 0])
        out.update(acc=None, all_correct=None, margin_db=None,
                   power=None if (flags[0] or not math.isfinite(p)) else p)
        return out
    pred = np.asarray(task.decide(y, labels)).copy()
    pred[flags] = -1
    acc = float(np.mean(pred == labels))
    ok = ~flags
    margin = None
    if ok.any() and np.all(np.isfinite(y[ok])):
        mg = float(task.margin_db(y, labels, mask=ok))
        margin = mg if math.isfinite(mg) else None
    out.update(acc=acc, all_correct=bool(acc == 1.0 and margin is not None and margin > 0.0), margin_db=margin,
               power=None)
    return out


# ============================================================ oracle 参照（JSON 缓存）
def _atomic_write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable(obj), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def oracle_cache_path(task: str, seed: int, ihash: str, n_restarts: int, max_sweeps: int,
                      cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"{task}_s{int(seed)}_{ihash}_r{int(n_restarts)}w{int(max_sweeps)}.json")


def oracle_refs(task: str, seed: int, *, tasks: dict | None = None, cache_dir: str = CACHE_DIR, n_restarts: int = 8,
                max_sweeps: int = 30, recompute: bool = False) -> dict:
    """(任务, 种子) 的 oracle 参照（缓存 JSON，键 = 任务 + 种子 + 实例哈希 + oracle 设置）。

    返回 {"task","seed","instance_hash","kind","n_probes","chance_acc","spec","oracle": {"accuracy": d|None, "loss": d},
          "focus_params": d, "cached", ...}；d = clean_eval + {"objective_value","n_evals","time_s","params"}。
    oracle 随机流 seed_rng(EXP, "oracle", 任务, 种子, 目标)。focus 没有 accuracy 目标（None）。
    """
    from .. import train

    spec = (tasks or TASKS)[task]
    system = build_system(task, seed, tasks)
    ih = instance_hash(task, seed, tasks, system)
    path = oracle_cache_path(task, seed, ih, n_restarts, max_sweeps, cache_dir)
    if not recompute and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            if doc.get("instance_hash") == ih:
                doc["cached"] = True
                doc["cache_path"] = path
                return doc
        except (OSError, ValueError):
            pass                                 # 损坏的缓存：重算覆盖
    orc = {}
    for obj in ("accuracy", "loss"):
        if system.task.kind == "focus" and obj == "accuracy":
            orc[obj] = None
            continue
        r = train.run_oracle_bits(system, common.seed_rng(EXP, "oracle", task, int(seed), obj), n_restarts=n_restarts,
                                  max_sweeps=max_sweeps, n_noise_eval=0, objective=obj)
        d = clean_eval(system, r["params"])
        d.update(objective_value=float(r["loss"]), n_evals=int(r["n_evals"]), time_s=float(r["time_s"]),
                 params=[int(v) for v in r["params"]])
        orc[obj] = d
    fp = clean_eval(system, system.focus_params())
    fp["params"] = [int(v) for v in system.focus_params()]
    doc = {"bench_version": BENCH_VERSION, "task": task, "seed": int(seed), "instance_hash": ih,
           "kind": system.task.kind, "n_probes": int(system.n_probes), "n_samples": int(system.B),
           "n_params": int(system.n_params), "chance_acc": chance_acc(system.task.kind, system.n_probes),
           "spec": spec.to_dict(), "note": spec.note, "n_restarts": int(n_restarts), "max_sweeps": int(max_sweeps),
           "oracle": orc, "focus_params": fp, "created": _dt.datetime.now().isoformat(timespec="seconds")}
    _atomic_write_json(doc, path)
    doc["cached"] = False
    doc["cache_path"] = path
    return doc


def _candidates(doc: dict) -> list[dict]:
    c = [d for d in (doc["oracle"].get("accuracy"), doc["oracle"].get("loss")) if d]
    if doc.get("focus_params"):
        c.append(doc["focus_params"])
    return c


def _max_or_none(vals):
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return max(v) if v else None


def cached_docs(ihash: str, cache_dir: str = CACHE_DIR) -> list[dict]:
    """缓存目录里实例哈希为 ihash 的全部 oracle 参照 JSON（任意种子、任意 oracle 设置；损坏 / 版本不符的跳过）。"""
    import glob

    out = []
    for path in sorted(glob.glob(os.path.join(cache_dir, f"*_{ihash}_r*w*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            continue
        if doc.get("instance_hash") == ihash and doc.get("bench_version") == BENCH_VERSION:
            doc.setdefault("cache_path", path)
            out.append(doc)
    return out


def _doc_id(d: dict):
    cp = d.get("cache_path")
    if cp:
        return os.path.normcase(os.path.abspath(cp))
    return (d.get("task"), d.get("seed"), d.get("n_restarts"), d.get("max_sweeps"))


def build_references(docs: list[dict], cache_dir: str | None = None) -> dict:
    """每个 (任务, 种子) 的参照：同一实例哈希下全部种子、两种 oracle 目标与聚焦码本的最好值（见模块 docstring）。
    cache_dir 给定时另合并该目录里同一实例哈希的全部缓存 JSON（INTEGRATION T：池化全部已缓存种子）。"""
    by_hash: dict = {}
    for d in docs:
        by_hash.setdefault(d["instance_hash"], []).append(d)
    if cache_dir:
        for ih, grp in by_hash.items():
            seen = {_doc_id(g) for g in grp}
            for c in cached_docs(ih, cache_dir):
                if _doc_id(c) not in seen:
                    seen.add(_doc_id(c))
                    grp.append(c)
    refs = {}
    for d in docs:
        grp = by_hash[d["instance_hash"]]
        cands = [c for g in grp for c in _candidates(g)]
        losses_ = [c["loss"] for c in cands if c.get("loss") is not None]
        orc_acc = [g["oracle"]["accuracy"]["acc"] for g in grp if g["oracle"].get("accuracy")]
        orc_loss_acc = [g["oracle"]["loss"]["acc"] for g in grp if g["oracle"].get("loss")]
        refs[(d["task"], int(d["seed"]))] = {
            "task": d["task"], "seed": int(d["seed"]), "kind": d["kind"], "instance_hash": d["instance_hash"],
            "chance_acc": d.get("chance_acc"), "n_pooled_seeds": len({int(g["seed"]) for g in grp}),
            "n_pooled_docs": len(grp),
            "ref_acc": _max_or_none(c.get("acc") for c in cands),
            "ref_all_correct": any(bool(c.get("all_correct")) for c in cands),
            "ref_power": _max_or_none(c.get("power") for c in cands),
            "ref_loss": min(losses_) if losses_ else None,
            "oracle_acc_objective_best": _max_or_none(orc_acc),
            "oracle_loss_objective_best_acc": _max_or_none(orc_loss_acc),
            "oracle_acc_seed": (d["oracle"]["accuracy"] or {}).get("acc"),
            "oracle_loss_seed": (d["oracle"]["loss"] or {}).get("loss"),
            "oracle_power_seed": (d["oracle"]["loss"] or {}).get("power"),
            "focus_params_acc": d["focus_params"].get("acc"), "focus_params_power": d["focus_params"].get("power"),
            "focus_params_loss": d["focus_params"].get("loss"),
        }
    return refs


# ============================================================ 单个 run
def curve_checkpoints(budget: int, curve_points: int = DEFAULT_CURVE_POINTS) -> np.ndarray:
    """0..budget 上 curve_points+1 个等距整数检查点（去重）。"""
    n = max(1, int(curve_points))
    return np.unique(np.round(np.linspace(0, int(budget), n + 1)).astype(np.int64))


def run_trainer(fn: Callable, task: str, seed: int, budget: int = DEFAULT_BUDGET, *, name: str = "custom",
                tasks: dict | None = None, curve_points: int = DEFAULT_CURVE_POINTS, n_noise_eval: int = 20) -> dict:
    """在 (任务, 种子) 上跑一个训练器并复评（同进程，探索者调试用；run_benchmark 的 worker 也调它）。

    随机流（公共随机数）：训练器 rng = seed_rng(EXP, "trainer", 任务, 种子)，测量噪声 = seed_rng(EXP, "noise", 任务, 种子)，
    最终评估噪声 = seed_rng(EXP, "eval", 任务, 种子)，均与训练器无关。返回一行 dict（不含参照相关指标，见 add_metrics）。
    学习曲线：检查点 c 上的判决 = n_used ≤ c 时最后一次 report（无 report 为 None）；c ≥ 最终已用测量数时为最终返回码本。
    """
    system = build_system(task, seed, tasks)
    meas = MeasurementBudget(system, budget, common.seed_rng(EXP, "noise", task, int(seed)), task_name=task)
    rng = common.seed_rng(EXP, "trainer", task, int(seed))
    overran = fallback = False
    t0 = time.perf_counter()
    try:
        out = fn(meas, rng)
    except BudgetExhausted:
        overran, out = True, None
    wall = time.perf_counter() - t0
    reports = meas.reports
    if out is None:
        fallback = True
        if not reports:
            raise RuntimeError(f"训练器 {name!r} 没有返回码本{'（超预算）' if overran else ''}，也从未 report")
        out = reports[-1][1]
    final = meas.check_params(out)
    n_used = meas.n_used
    # ---- 学习曲线（无噪声复评，不计测量次数；相同码本只评一次）
    cache: dict = {}

    def ev(x):
        k = x.tobytes()
        if k not in cache:
            cache[k] = clean_eval(system, x)
        return cache[k]

    cps = curve_checkpoints(budget, curve_points)
    n_at = np.array([n for n, _ in reports], dtype=np.int64)
    curve = {"n": [], "acc": [], "loss": [], "power": []}
    for c in cps:
        if c >= n_used:
            x = final
        else:
            j = int(np.searchsorted(n_at, c, side="right")) - 1
            x = reports[j][1] if j >= 0 else None
        e = ev(x) if x is not None else {"acc": None, "loss": None, "power": None}
        curve["n"].append(int(c))
        curve["acc"].append(e["acc"])
        curve["loss"].append(e["loss"])
        curve["power"].append(e["power"])
    if n_used not in curve["n"]:                     # 最终点（已用测量数处）
        e = ev(final)
        k = int(np.searchsorted(curve["n"], n_used))
        for key, v in (("n", int(n_used)), ("acc", e["acc"]), ("loss", e["loss"]), ("power", e["power"])):
            curve[key].insert(k, v)
    # ---- 最终码本：无噪声 + n_noise_eval 次带噪评估
    fe = ev(final)
    full = system.evaluate(final, common.seed_rng(EXP, "eval", task, int(seed)), n_noise=int(n_noise_eval))
    return {"trainer": name, "task": task, "seed": int(seed), "budget": int(budget), "kind": system.task.kind,
            "instance_hash": instance_hash(task, seed, tasks, system),
            "n_meas_used": int(n_used), "overran": bool(overran), "fallback_to_report": bool(fallback),
            "n_reports": len(reports), "n_flagged_meas": int(meas.n_flagged_measurements), "wall_s": float(wall),
            "acc_final": fe["acc"], "all_correct": fe["all_correct"], "margin_db": fe["margin_db"],
            "loss_final": fe["loss"], "loss_model_final": fe["loss_model"], "power_final": fe["power"],
            "n_flagged_final": fe["n_flagged"], "acc_noisy": full.get("acc_noisy"),
            "oscillating_final": bool(full.get("oscillating_run")),
            "pa_in_rel_p1db_mean": full.get("pa_in_rel_p1db_db_mean"),
            "params": [int(v) for v in final], "curve": curve}


# ============================================================ 参照相关指标
def _reached(kind: str, acc, power, ref: dict) -> bool:
    if kind == "focus":
        rp = ref.get("ref_power")
        return power is not None and rp is not None and power >= rp - FOCUS_TOL_DB - _EPS
    ra = ref.get("ref_acc")
    return acc is not None and ra is not None and acc >= REACH_FRAC * ra - _EPS


def add_metrics(rows: list[dict], refs: dict) -> list[dict]:
    """给每行加参照指标（原地，返回 rows）：ref_acc、gap_acc = ref_acc − acc、norm_score = (acc − chance)/(ref − chance)、
    ref_power、power_gap_db = power − ref_power（focus）、reached（最终码本达标）、meas_to_crit（学习曲线首个达标点，
    未达标 None）、meas_to_settle（此后一直达标的最早检查点，None = 最终未达标）。"""
    for r in rows:
        if "error" in r:
            continue
        ref = refs.get((r["task"], int(r["seed"])))
        if ref is None:
            continue
        kind = r["kind"]
        acc, pw = r.get("acc_final"), r.get("power_final")
        ra, rp, ch = ref.get("ref_acc"), ref.get("ref_power"), ref.get("chance_acc")
        r["ref_acc"], r["ref_power"], r["ref_loss"] = ra, rp, ref.get("ref_loss")
        r["gap_acc"] = None if (acc is None or ra is None) else float(ra - acc)
        r["norm_score"] = (None if (acc is None or ra is None or ch is None or ra - ch <= _EPS)
                           else float((acc - ch) / (ra - ch)))
        r["power_gap_db"] = None if (pw is None or rp is None) else float(pw - rp)
        r["reached"] = bool(_reached(kind, acc, pw, ref))
        cv = r.get("curve") or {}
        hit = [(int(n), _reached(kind, a, p, ref))
               for n, a, p in zip(cv.get("n", []), cv.get("acc", []), cv.get("power", []))]
        r["meas_to_crit"] = next((n for n, h in hit if h), None)
        settle = None                                # 达标并保持到预算结束的最早检查点
        for n, h in reversed(hit):
            if not h:
                break
            settle = n
        r["meas_to_settle"] = settle
    return rows


# ============================================================ worker 入口（模块顶层，spawn 安全）
def _oracle_job(job: dict) -> dict:
    _, tasks = registry(job.get("plugins", ()))
    return oracle_refs(job["task"], int(job["seed"]), tasks=tasks, cache_dir=job["cache_dir"],
                       n_restarts=int(job["oracle_restarts"]), max_sweeps=int(job["oracle_sweeps"]),
                       recompute=bool(job.get("recompute", False)))


def _trainer_job(job: dict) -> dict:
    base = {"trainer": job["trainer"], "task": job["task"], "seed": int(job["seed"]), "budget": int(job["budget"])}
    try:
        trainers, tasks = registry(job.get("plugins", ()))
        return run_trainer(trainers[job["trainer"]], job["task"], int(job["seed"]), int(job["budget"]),
                           name=job["trainer"], tasks=tasks, curve_points=int(job["curve_points"]),
                           n_noise_eval=int(job["n_noise_eval"]))
    except KeyboardInterrupt:
        raise
    except BaseException:      # noqa: BLE001 —— 训练器任何异常记为该行 error，不中断其它 run
        return dict(base, error=traceback.format_exc())


# ============================================================ 汇总与输出
SUMMARY_METRICS = ["acc_final", "all_correct", "loss_final", "gap_acc", "norm_score", "power_final", "power_gap_db",
                   "reached", "n_meas_used", "wall_s", "acc_noisy"]


def _median_meas(vals: list, budget: int):
    """删失（None）按 +∞ 的中位数；过半删失 → None（显示为 ">budget"）。"""
    if not vals:
        return None
    a = np.array([np.inf if v is None else float(v) for v in vals])
    med = float(np.median(a))
    return med if math.isfinite(med) else None


def summarize(rows: list[dict], trainers: list[str], tasks: list[str]) -> list[dict]:
    """每个 (训练器, 任务) 一行：common.aggregate 的均值 / 95% 区间 + 达标测量数中位数 + 超预算 / 错误计数。"""
    ok = [r for r in rows if r.get("trainer") in trainers and r.get("task") in tasks]
    table = common.aggregate(ok, ["trainer", "task"], SUMMARY_METRICS)
    by = {}
    for r in ok:
        if "error" not in r:
            by.setdefault((r["trainer"], r["task"]), []).append(r)
    order = {t: i for i, t in enumerate(trainers)}, {t: i for i, t in enumerate(tasks)}
    for row in table:
        rs = by.get((row["trainer"], row["task"]), [])
        budget = rs[0]["budget"] if rs else None
        row["kind"] = rs[0]["kind"] if rs else None
        row["budget"] = budget
        row["n_reached"] = sum(1 for r in rs if r.get("reached"))
        row["median_meas_to_crit"] = _median_meas([r.get("meas_to_crit") for r in rs], budget) if rs else None
        row["median_meas_to_settle"] = _median_meas([r.get("meas_to_settle") for r in rs], budget) if rs else None
        row["n_overran"] = sum(1 for r in rs if r.get("overran"))
        row["ref_acc"] = rs[0].get("ref_acc") if rs else None
        row["ref_power"] = rs[0].get("ref_power") if rs else None
    table.sort(key=lambda r: (order[1].get(r["task"], 99), order[0].get(r["trainer"], 99)))
    return table


def _ci(row: dict, m: str, fmt: str = ".2f") -> str:
    mean, lo, hi, n = row.get(f"{m}_mean"), row.get(f"{m}_lo"), row.get(f"{m}_hi"), row.get(f"{m}_n", 0)
    if not n or mean is None or not math.isfinite(mean):
        return "—"
    s = format(mean, fmt)
    if n > 1 and lo is not None and math.isfinite(lo):
        s += f" [{format(lo, fmt)}, {format(hi, fmt)}]"
    return s


def _meas_str(row: dict, key: str = "median_meas_to_crit") -> str:
    v = row.get(key)
    return f">{row.get('budget')}" if v is None else str(int(v))


def summary_markdown(table: list[dict], refs: dict, trainers: list[str], tasks: list[str], cfg: dict) -> str:
    lines = [f"# 训练器基准（bench_trainers，预算 {cfg['budget']} 次测量，{cfg['n_seeds']} 个种子）", ""]
    lines += ["一次测量 = 全部训练样本过一遍（方案 §4.7：DSPSA 每次迭代 2 次）。训练器只用 MeasurementBudget 的硬件可得量；"
              "准确率 / 损失 / 功率均为最终码本的**无噪声**复评（损失 = 可测目标 loss_meas_clean）。",
              f"达标：分类 acc ≥ {REACH_FRAC}·ref_acc；focus 探头 0 功率 ≥ ref − {FOCUS_TOL_DB:g} dB。"
              "ref = 同一实例全部已缓存种子 × {accuracy, loss} 两种 1-bit oracle 目标（及几何聚焦码本）的最好值"
              "（池化种子数见 references.json 的 n_pooled_seeds）。"
              f"「达标测量数」= 学习曲线首个达标检查点（首达）/ 此后一直达标的最早检查点（保持）的种子中位数"
              f"（分辨率 {cfg['budget'] // max(1, cfg['curve_points'])} 次测量，删失按 +∞）。区间为 95% t 区间（种子间）。", ""]
    # ---- 参照
    lines += ["## oracle 参照", "",
              "| 任务 | 类型 | chance | ref_acc | acc 目标 oracle | loss 目标 oracle 的 acc | ref_loss | ref_power (dBm) | "
              "聚焦码本 acc / 功率 | 说明 |", "|---|---|---|---|---|---|---|---|---|---|"]
    seen = set()
    for (t, s), ref in sorted(refs.items(), key=lambda kv: (tasks.index(kv[0][0]) if kv[0][0] in tasks else 99,
                                                             kv[0][1])):
        if t in seen or t not in tasks:
            continue
        seen.add(t)

        def f(v, fmt=".3f"):
            return "—" if v is None else format(v, fmt)

        fp = (f(ref["focus_params_power"], ".2f") + " dBm" if ref["kind"] == "focus"
              else f(ref["focus_params_acc"]))
        lines.append(f"| {t} | {ref['kind']} | {f(ref['chance_acc'])} | {f(ref['ref_acc'])} | "
                     f"{f(ref['oracle_acc_objective_best'])} | {f(ref['oracle_loss_objective_best_acc'])} | "
                     f"{f(ref['ref_loss'], '.4f')} | {f(ref['ref_power'], '.2f')} | {fp} | "
                     f"{cfg['task_notes'].get(t, '')} |")
    # ---- 透视表
    lines += ["", "## 透视表：最终准确率均值（focus：Δp = 功率 − ref，dB）· 达标种子数", "",
              "| 训练器 | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    cell = {(r["trainer"], r["task"]): r for r in table}
    for tr in trainers:
        cs = []
        for t in tasks:
            r = cell.get((tr, t))
            if r is None or not r.get("n_runs"):
                cs.append("—" if r is None else f"error×{r.get('n_errors', 0)}")
                continue
            m = "power_gap_db" if r.get("kind") == "focus" else "acc_final"
            v = r.get(f"{m}_mean")
            vs = "—" if v is None or not math.isfinite(v) else (f"{v:+.2f} dB" if m == "power_gap_db" else f"{v:.2f}")
            cs.append(f"{vs} · {r['n_reached']}/{r['n_runs']}")
        lines.append(f"| {tr} | " + " | ".join(cs) + " |")
    # ---- 明细
    lines += ["", "## 明细（均值 [95% 区间]）", "",
              "| 任务 | 训练器 | n | acc | 全对比例 | Δp (dB) | loss | 与 ref 差距 | chance 归一化 | 达标比例 | "
              "达标测量数中位数（首达 / 保持） | 所用测量 | 墙钟 (s) | 超预算 / 错误 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        if not r.get("n_runs"):
            lines.append(f"| {r['task']} | {r['trainer']} | 0 | | | | | | | | | | | 0 / {r.get('n_errors', 0)} |")
            continue
        foc = r.get("kind") == "focus"
        lines.append(
            f"| {r['task']} | {r['trainer']} | {r['n_runs']} | {'—' if foc else _ci(r, 'acc_final')} | "
            f"{'—' if foc else _ci(r, 'all_correct')} | {_ci(r, 'power_gap_db') if foc else '—'} | "
            f"{_ci(r, 'loss_final', '.4f')} | {'—' if foc else _ci(r, 'gap_acc')} | "
            f"{'—' if foc else _ci(r, 'norm_score')} | {_ci(r, 'reached')} | "
            f"{_meas_str(r)} / {_meas_str(r, 'median_meas_to_settle')} | "
            f"{_ci(r, 'n_meas_used', '.0f')} | {_ci(r, 'wall_s', '.1f')} | {r['n_overran']} / {r.get('n_errors', 0)} |")
    n_err = sum(int(r.get("n_errors", 0)) for r in table)
    if n_err:
        lines += ["", f"{n_err} 个 run 出错（见 runs.jsonl 的 error 字段）。"]
    if cfg.get("included_runs"):
        lines += ["", f"合并的旧结果：{', '.join(cfg['included_runs'])}（{cfg.get('n_included_rows', 0)} 行）。"]
    return "\n".join(lines) + "\n"


# 参照调色板（dataviz references/palette.md 浅色 8 色，按训练器固定顺序分配，不循环）
_SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def plot_curves(rows: list[dict], refs: dict, trainers: list[str], tasks: list[str], path: str) -> str:
    """每个任务一幅小图：种子平均的无噪声准确率（focus：功率 − ref，dB）vs 测量数；虚线 = ref，点线 = 达标线。"""
    import matplotlib.pyplot as plt

    ncol = min(3, max(1, len(tasks)))
    nrow = int(math.ceil(len(tasks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False)
    handles = {}
    for ti, t in enumerate(tasks):
        ax = axes[ti // ncol][ti % ncol]
        ref = next((v for (tt, _s), v in refs.items() if tt == t), None)
        kind = ref["kind"] if ref else None
        for ci, tr in enumerate(trainers):
            rs = [r for r in rows if r.get("trainer") == tr and r.get("task") == t and "error" not in r]
            if not rs:
                continue
            n = np.asarray(rs[0]["curve"]["n"], dtype=float)
            ys = []
            for r in rs:
                cv = r["curve"]
                if len(cv["n"]) != len(n) or not np.allclose(cv["n"], n):
                    continue
                if kind == "focus":
                    rp = (refs.get((t, int(r["seed"]))) or {}).get("ref_power")
                    ys.append([np.nan if (p is None or rp is None) else p - rp for p in cv["power"]])
                else:
                    ys.append([np.nan if a is None else a for a in cv["acc"]])
            if not ys:
                continue
            with warnings.catch_warnings():            # 某检查点全部种子尚无 report → nan（不画）
                warnings.simplefilter("ignore", RuntimeWarning)
                y = np.nanmean(np.asarray(ys, dtype=float), axis=0)
            color = _SERIES_COLORS[ci] if ci < len(_SERIES_COLORS) else "#888888"
            ls = "-" if ci < len(_SERIES_COLORS) else ":"
            (h,) = ax.plot(n, y, color=color, lw=1.8, ls=ls, label=tr)
            handles.setdefault(tr, h)
        if ref is not None:
            if kind == "focus":
                ax.axhline(0.0, color="#52514e", lw=1.0, ls="--")
                ax.axhline(-FOCUS_TOL_DB, color="#52514e", lw=0.8, ls=":")
                ax.set_ylabel("探头 0 功率 − ref (dB)")
            else:
                if ref.get("ref_acc") is not None:
                    ax.axhline(ref["ref_acc"], color="#52514e", lw=1.0, ls="--")
                    ax.axhline(REACH_FRAC * ref["ref_acc"], color="#52514e", lw=0.8, ls=":")
                if ref.get("chance_acc") is not None:
                    ax.axhline(ref["chance_acc"], color="#b5b4ae", lw=0.8, ls="-.")
                ax.set_ylabel("无噪声准确率")
                ax.set_ylim(-0.03, 1.03)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("测量次数")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for k in range(len(tasks), nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)
    if handles:
        fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=min(6, len(handles)),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("训练器学习曲线（种子平均；虚线 = oracle ref，点线 = 达标线，点划线 = chance）", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return common.savefig(fig, path)


def _load_included(paths, refs: dict, budget: int, exclude_trainers) -> tuple[list[dict], int]:
    """旧 runs.jsonl（或其所在目录）中实例哈希、预算都与本次相同、训练器不在本次运行中的行。"""
    out, n_skip = [], 0
    for p in paths or ():
        fp = os.path.join(p, "runs.jsonl") if os.path.isdir(p) else p
        for r in common.load_jsonl(fp):
            if "error" in r or r.get("trainer") in exclude_trainers:
                continue
            ref = refs.get((r.get("task"), int(r.get("seed", -1))))
            if ref is None or r.get("instance_hash") != ref["instance_hash"] or int(r.get("budget", -1)) != budget:
                n_skip += 1
                continue
            r = dict(r, included_from=fp)
            out.append(r)
    return out, n_skip


def run_benchmark(trainers=None, tasks=None, seeds: int = 5, seed0: int = 0, budget: int = DEFAULT_BUDGET,
                  workers: int = DEFAULT_WORKERS, out: str | None = None, plugins=(), cache_dir: str = CACHE_DIR,
                  curve_points: int = DEFAULT_CURVE_POINTS, n_noise_eval: int = 20, oracle_restarts: int = 8,
                  oracle_sweeps: int = 30, recompute_oracle: bool = False, include_runs=(), run_new: bool = True,
                  progress: bool = False) -> dict:
    """两阶段：(1) 每个 (任务, 种子) 的 oracle 参照（缓存）；(2) 每个 (训练器, 任务, 种子) 一个 run。写出全部输出文件。

    trainers / tasks：名字列表，None 或含 "all" → 全部已注册（内置 + 插件）。返回 {"rows","table","refs","outdir","paths"}。
    include_runs：合并旧 runs.jsonl（实例哈希与预算相同；本次也跑的训练器以本次为准）。
    run_new=False：不跑新 run，只用 include_runs 的行重新汇总（trainers 给定时只保留这些训练器）。
    """
    plugins = [os.path.abspath(p) for p in (plugins or ())]
    reg_tr, reg_tk = registry(plugins)
    all_req = not trainers or "all" in trainers
    keep = None if all_req else list(trainers)
    trainers = (list(reg_tr) if all_req else list(trainers)) if run_new else []
    tasks = list(reg_tk) if (not tasks or "all" in tasks) else list(tasks)
    if not run_new and not include_runs:
        raise ValueError("run_new=False 时必须给 include_runs")
    for t in trainers:
        if t not in reg_tr:
            raise ValueError(f"未知训练器 {t!r}，可选 {list(reg_tr)}")
    for t in tasks:
        if t not in reg_tk:
            raise ValueError(f"未知任务 {t!r}，可选 {list(reg_tk)}")
    seed_list = list(range(int(seed0), int(seed0) + int(seeds)))
    outdir = common.make_outdir(EXP, out)
    t0 = time.perf_counter()
    # ---- (1) oracle 参照
    ojobs = [{"task": t, "seed": s, "plugins": plugins, "cache_dir": cache_dir, "oracle_restarts": oracle_restarts,
              "oracle_sweeps": oracle_sweeps, "recompute": recompute_oracle} for t in tasks for s in seed_list]
    docs = common.run_jobs(_oracle_job, ojobs, workers=workers, progress=progress)
    bad = [d for d in docs if "error" in d]
    if bad:
        raise RuntimeError("oracle 参照计算失败：\n" + bad[0]["error"])
    refs = build_references(docs, cache_dir)
    t_oracle = time.perf_counter() - t0
    # ---- (2) 训练器 run
    jobs = [{"trainer": tr, "task": t, "seed": s, "budget": int(budget), "plugins": plugins,
             "curve_points": int(curve_points), "n_noise_eval": int(n_noise_eval)}
            for t in tasks for s in seed_list for tr in trainers]
    rows = common.run_jobs(_trainer_job, jobs, workers=workers, progress=progress) if jobs else []
    for r in rows:                                   # run_jobs 层面的错误行补上分组键
        if "error" in r and "trainer" not in r:
            j = r.get("job", {})
            r.update(trainer=j.get("trainer"), task=j.get("task"), seed=j.get("seed"), budget=j.get("budget"))
    inc, n_skip = _load_included(include_runs, refs, int(budget), set(trainers))
    if not run_new and keep is not None:
        inc = [r for r in inc if r["trainer"] in keep]
    inc_names = list(dict.fromkeys(r["trainer"] for r in inc))
    if not run_new and keep is not None:
        inc_names = [t for t in keep if t in inc_names]
    all_trainers = trainers + [t for t in inc_names if t not in trainers]
    rows_all = add_metrics(rows + inc, refs)
    table = summarize(rows_all, all_trainers, tasks)
    reg_notes = {t: reg_tk[t].note for t in tasks}
    cfg = {"exp": EXP, "bench_version": BENCH_VERSION, "trainers": all_trainers, "trainers_run": trainers,
           "tasks": tasks, "seeds": seed_list,
           "n_seeds": len(seed_list), "budget": int(budget), "workers": int(workers), "plugins": plugins,
           "cache_dir": cache_dir, "curve_points": int(curve_points), "n_noise_eval": int(n_noise_eval),
           "oracle_restarts": int(oracle_restarts), "oracle_sweeps": int(oracle_sweeps),
           "task_specs": {t: reg_tk[t].to_dict() for t in tasks}, "task_notes": reg_notes,
           "reach_frac": REACH_FRAC, "focus_tol_db": FOCUS_TOL_DB,
           "included_runs": [str(p) for p in include_runs or ()], "n_included_rows": len(inc),
           "n_included_skipped": n_skip, "oracle_time_s": t_oracle, "wall_time_s": time.perf_counter() - t0,
           "oracle_cache_files": sorted({d.get("cache_path") for d in docs})}
    md = summary_markdown(table, refs, all_trainers, tasks, cfg)
    paths = {"runs": common.save_jsonl(rows if run_new else rows_all, os.path.join(outdir, "runs.jsonl")),
             "csv": common.save_csv(table, os.path.join(outdir, "summary.csv"))}
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    paths["md"] = os.path.join(outdir, "summary.md")
    paths["refs"] = common.save_json({f"{t}/{s}": v for (t, s), v in refs.items()},
                                     os.path.join(outdir, "references.json"))
    paths["config"] = common.save_json(cfg, os.path.join(outdir, "config.json"))
    paths["curves"] = plot_curves(rows_all, refs, all_trainers, tasks, os.path.join(outdir, "curves.png"))
    return {"rows": rows, "rows_all": rows_all, "table": table, "refs": refs, "outdir": outdir, "paths": paths,
            "markdown": md, "config": cfg}


# ============================================================ CLI
def parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="零阶训练器基准（RQ6 工具，方案 §4.7）")
    p.add_argument("--trainers", nargs="+", default=None,
                   help=f"训练器名（默认 all = 内置 {list(BUILTIN_TRAINERS)} + 插件）")
    p.add_argument("--plugin", nargs="+", action="extend", default=[],
                   help="插件 .py（定义 TRAINERS dict，可选 BENCH_TASKS），可多次给出")
    p.add_argument("--tasks", nargs="+", default=None, help=f"任务（默认 all = {list(TASKS)} + 插件任务）")
    p.add_argument("--seeds", type=int, default=5, help="种子数（种子 seed0 .. seed0+seeds−1）")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="每个 run 的测量预算")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并行进程数（默认 3）")
    p.add_argument("--out", type=str, default=None, help="输出目录（默认 results/bench_trainers/<时间戳>/）")
    p.add_argument("--cache-dir", type=str, default=CACHE_DIR, help="oracle 参照缓存目录")
    p.add_argument("--curve-points", type=int, default=DEFAULT_CURVE_POINTS, help="学习曲线检查点数")
    p.add_argument("--n-noise-eval", type=int, default=20, help="最终码本的带噪评估次数")
    p.add_argument("--oracle-restarts", type=int, default=8)
    p.add_argument("--oracle-sweeps", type=int, default=30)
    p.add_argument("--recompute-oracle", action="store_true", help="忽略缓存重算 oracle")
    p.add_argument("--include-runs", nargs="+", default=[],
                   help="合并旧 runs.jsonl（或其目录）中实例哈希与预算相同的行（如基线），不重跑")
    p.add_argument("--no-run", action="store_true",
                   help="不跑新 run，只把 --include-runs 的行（可用 --trainers 过滤）按当前参照重新汇总")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    res = run_benchmark(a.trainers, a.tasks, seeds=a.seeds, seed0=a.seed0, budget=a.budget, workers=a.workers,
                        out=a.out, plugins=a.plugin, cache_dir=a.cache_dir, curve_points=a.curve_points,
                        n_noise_eval=a.n_noise_eval, oracle_restarts=a.oracle_restarts,
                        oracle_sweeps=a.oracle_sweeps, recompute_oracle=a.recompute_oracle,
                        include_runs=a.include_runs, run_new=not a.no_run, progress=True)
    print(res["markdown"])
    print(f"输出目录：{res['outdir']}（{res['config']['wall_time_s']:.0f} s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
