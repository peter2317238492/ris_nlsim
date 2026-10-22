"""任务定义（DESIGN §7.3，INTEGRATION E；方案 §6）。

一个任务 = 若干样本（物体节点集合 + 标签 + meta）+ 判决/损失。损失一律复用 losses.py（numpy 与 torch 同源，
INTEGRATION B2），本模块只负责样本构造、判决（predict）与裕量。

- ParityTask（kind="binary"）：n 个槽位放/不放小散射体，样本 i ↔ x_j = (i>>j)&1（与 metrics 的 Walsh 编码一致），
  标签 = 奇偶；两个探头 y1 > y2 判为奇（方案 §6.1）。
- LetterTask（kind="multiclass"）：字母 A/B/C 薄板（INTEGRATION B2：厚度 1 个体素、d = λ/10），3 探头 argmax；
  可选距离 / 功率抖动（方案 §6.2 实验 C，功率抖动写进 meta["power_db"]，系统层按它缩放馈源功率）。
- LocalizationTask（kind="loc3bit"）：Song 2026 的 8 格定位，一套码本、3 个探头各自阈值判决 → 3-bit 格号。
- FocusTask（kind="focus"）：无物体，最大化探头 0 功率（实验 F）。

体素尺寸（INTEGRATION B2）：voxel_d=None 按材料自动——介质 d = min(λ/10, λ/(π|√ε_eff|))（水 ≈ λ/28），金属 λ/20；
字母任务默认 λ/10 薄板。所有任务记录实际 voxel_d、极化率模型 model 与 effective_eps。

极化率模型与物体模式（审查 R2-INT-1）：scalar_rr 的局域场分母 1 − χk²a²/2 + jχk³V/(4π) 是体素自作用（体内多次散射），
只有在体素互相耦合（full / intra）时才与离散 LS 方程一致；born（DWBA，体素互不耦合、p = α·E_bg）下它不被抵消，
高对比度材料在自动体素附近近谐振（水 λ/28：|α/(χV)| = 3.2，单物体场 +9.6 dB；λ/20 → λ/80 非单调），金属随 d→0 发散。
born 的一致离散是一阶 Born p = χV·E_bg（再做辐射修正）= volume_rr，λ/20 起收敛到 0.3 dB 内。
object_model(mode) 给出各模式的默认模型（born → BORN_MODEL = volume_rr，full / intra → scalar_rr）；
system.make_task 按 cfg.object_mode 自动选用。体素网格不随这一替换改变（_grid_model），三种物体模式同一离散。

金属 + born（审查 S1-1）：volume_rr / cm / cm_rr 对金属用 CM 极限 α0 = 3V，连续极限是 εr_eff = 4 的介质（effective_eps
记为 '4+0j'），不是导体；高对比度物体本来就没有收敛的一阶 Born 模型（scalar_rr 体素互不耦合时随 d→0 发散）。
所以 born 模式下的"金属"字母 / 物体实为 εr = 4 薄板（线性理论参考），构造任务时发 RuntimeWarning；
要比较导体请用 full / intra（scalar_rr 软球极限）。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from . import losses
from .constants import K0, LAMBDA
from .geometry import parity_slots, voxelize_letter, voxelize_sphere
from .materials import DEFAULT_MODEL, effective_eps, voxels_to_nodes
from .metrics import parity_labels
from .scene import NodeSet

KINDS = ("binary", "multiclass", "loc3bit", "focus")
BORN_MODEL = "volume_rr"        # born（DWBA）物体的体素极化率模型（审查 R2-INT-1，见模块 docstring）


def object_model(object_mode: str | None, model: str | None = None) -> str:
    """物体模式对应的体素极化率模型（审查 R2-INT-1）。

    model 显式给出时原样返回（调用方的选择优先）；否则 born → BORN_MODEL（一阶 Born α = χV/(1 + jχk³V/(4π))，
    体素尺寸收敛），full / intra / None → DEFAULT_MODEL（scalar_rr，耦合求解的一致离散）。
    """
    if model is not None:
        return model
    return BORN_MODEL if object_mode == "born" else DEFAULT_MODEL


def _grid_model(model: str) -> str:
    """体素网格判据用的模型：born 替身 BORN_MODEL 沿用物理模型 DEFAULT_MODEL 的网格（金属仍 λ/20，介质 ε_eff 相同），
    保证同一任务在 born / full / intra 下是同一组体素（审查 R2-INT-1）；其余模型按自身 ε_eff。"""
    return DEFAULT_MODEL if model == BORN_MODEL else model


def resolve_voxel_d(material, voxel_d: float | None = None, k: float = K0, model: str = DEFAULT_MODEL) -> float:
    """实际体素尺寸（INTEGRATION B2，DESIGN §1.3）。

    voxel_d 给定时原样返回；None → 介质 d = min(λ/10, λ/(π|√ε_eff|))（判据 |√ε|·k·d ≲ 2，水 ≈ λ/28），金属 λ/20。
    ε_eff 取 materials.effective_eps(material, model)（scalar_rr 下即 εr 本身）。
    """
    if voxel_d is not None:
        d = float(voxel_d)
        if not d > 0:
            raise ValueError("voxel_d 必须为正")
        return d
    lam = 2.0 * np.pi / k
    eps = effective_eps(material, model)
    if isinstance(eps, str):                    # "metal"
        return lam / 20.0
    return float(min(lam / 10.0, lam / (np.pi * abs(np.sqrt(complex(eps))))))


def _warn_metal_model(material, model: str) -> None:
    """金属在非 scalar_rr 模型下退化为 εr_eff = 4 介质（审查 S1-1，模块 docstring）→ RuntimeWarning。"""
    if isinstance(material, str) and material == "metal" and model != "scalar_rr":
        warnings.warn(f"material='metal' 在 model={model!r} 下是 εr_eff = 4 的介质（CM 极限 α0 = 3V），不是导体："
                      "object_mode='born'（DWBA，volume_rr）的金属物体请按 εr = 4 板解读，导体请用 full / intra"
                      "（审查 S1-1）", RuntimeWarning, stacklevel=3)


def eps_repr(material, model: str = DEFAULT_MODEL) -> str:
    """effective_eps 的 JSON 友好字符串（金属为 'metal'）。"""
    e = effective_eps(material, model)
    return e if isinstance(e, str) else f"{complex(e).real:g}{complex(e).imag:+g}j"


@dataclass
class Sample:
    """一个样本：物体节点（体素，各向同性，alpha 已算好，group = 物体编号）、标签、附加信息。"""

    objects: NodeSet
    label: int
    meta: dict = field(default_factory=dict)


def _as_vec(x) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.shape != (3,):
        raise ValueError(f"需要 3 维向量，得到 {np.shape(x)}")
    return v


def _unit(x) -> np.ndarray:
    v = _as_vec(x)
    n = np.linalg.norm(v)
    if n < 1e-15:
        raise ValueError("零向量")
    return v / n


class Task:
    """任务基类（INTEGRATION E）：name、kind、n_probes、samples、labels、task_kwargs、loss、predict、margin_db。"""

    name: str = "task"
    kind: str = "binary"
    n_probes: int = 2

    def __init__(self, samples: list[Sample]):
        if not samples:
            raise ValueError("任务至少需要一个样本")
        self.samples = list(samples)
        self.labels = np.array([s.label for s in self.samples], dtype=np.int64)

    # ---- 损失 / 判决 ----
    @property
    def task_kwargs(self) -> dict:
        """传给 losses.task_loss / oracle_torch.TorchSystem 的额外参数（loc3bit 的阈值与权重）。"""
        return {}

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def power_db(self) -> np.ndarray:
        """每样本馈源功率抖动 (dB)，meta["power_db"]，缺省 0（INTEGRATION C）。"""
        return np.array([float(s.meta.get("power_db", 0.0)) for s in self.samples])

    def loss(self, readings_db, labels=None, tau: float = 1.0, reduction: str = "mean"):
        """任务损失，委托 losses.task_loss（DESIGN §7.3）。labels=None → 全部样本的标签。"""
        labels = self.labels if labels is None else labels
        return losses.task_loss(readings_db, labels, self.kind, tau, reduction=reduction, **self.task_kwargs)

    def predict(self, readings_db, labels=None) -> np.ndarray:
        raise NotImplementedError

    def ties(self, readings_db, labels=None) -> np.ndarray:
        """(B,) bool：判决量恰在判决边界上（平局）的样本。默认无平局；子类实现。"""
        return np.zeros(np.shape(readings_db)[0], dtype=bool)

    def decide(self, readings_db, labels=None) -> np.ndarray:
        """计分用判决（审查 R2-INT-2）：predict 的结果，平局记为 −1（未决，计为错）。

        logdet 检波器把两个探头钳在同一量程边沿时 y1 == y2，predict 的 y1 > y2 规则会把它判成"偶"并对全部偶样本计对
        （字母 argmax 判类 0、定位 y > t 判比特 0 同理），虚抬准确率。钳位 + 公共 offset 是单调不减变换，二分类与
        argmax 的严格判决下"钳位读数判对 ⇒ 理想读数也判对"，所以把平局计为未决后 acc(检波器读数) ≤ acc(理想读数)，
        门 G1（M0 + Born 3 位奇偶不可解）不会被钳位平局"补"出来。（loc3bit 的自动阈值由读数本身决定，无此保证。）
        """
        pred = np.asarray(self.predict(readings_db, labels)).astype(np.int64)
        pred[np.asarray(self.ties(readings_db, labels), dtype=bool)] = -1
        return pred

    def sample_margins_db(self, readings_db, labels=None) -> np.ndarray:
        """逐样本带符号判决裕量 (B,) dB：> 0 ⇔ 该样本判对，平局为 0（审查 S0-2：oracle 的"准确率 / 裕量"目标用它）。
        判决规则（loc3bit 的自动阈值）由全部样本决定，与 predict 一致。子类实现。"""
        raise NotImplementedError

    def decision_units(self, readings_db, labels=None) -> np.ndarray:
        """逐判决单元带符号裕量 M (B, J) dB（INTEGRATION T：可测"判对优先"目标的逐单元平局项与逐单元温度用它）。
        样本 b 判对 ⇔ min_j M_bj > 0，且 min_j M_bj = sample_margins_db。默认 J = 1（= sample_margins_db）；
        多分类为每个竞争类一列（J = K − 1），loc3bit 为每个探头比特一列（J = K）。"""
        return np.asarray(self.sample_margins_db(readings_db, labels), dtype=float)[:, None]

    def margin_db(self, readings_db, labels=None, mask=None) -> float:
        """带符号的最小判决裕量 (dB)：> 0 ⇔ 全部判对（平局裕量为 0）。mask (B,) bool：只在这些样本上取最小，
        判决规则本身（loc3bit 的自动阈值）仍由全部样本决定，与 predict 一致（审查 R2-INT-3）。
        = min(sample_margins_db[mask])；无样本时 nan。"""
        labels = self.labels if labels is None else np.asarray(labels)
        y = np.asarray(readings_db, dtype=float)
        s = self.sample_margins_db(y, labels)
        if mask is not None:
            m = np.asarray(mask, dtype=bool).ravel()
            if m.shape != (y.shape[0],):
                raise ValueError(f"mask 形状应为 ({y.shape[0]},)，得到 {np.shape(mask)}")
            s = s[m]
        return float(np.min(s)) if len(s) else float("nan")

    @staticmethod
    def _masked(y: np.ndarray, labels: np.ndarray, mask) -> tuple[np.ndarray, np.ndarray]:
        if mask is None:
            return y, labels
        m = np.asarray(mask, dtype=bool).ravel()
        if m.shape != (y.shape[0],):
            raise ValueError(f"mask 形状应为 ({y.shape[0]},)，得到 {np.shape(mask)}")
        return y[m], labels[m]

    def describe(self) -> dict:
        """JSON 友好的任务摘要（写入实验结果）。"""
        return {"task": self.name, "kind": self.kind, "n_probes": self.n_probes, "n_samples": self.n_samples}


# ============================================================ 奇偶
class ParityTask(Task):
    """n 位奇偶校验（方案 §6.1，INTEGRATION E）。

    槽位 parity_slots(n, roi_center, spacing, axis)；每槽一个直径 diameter 的球（体素化，group = 槽号）。
    样本 i 的 x_j = (i>>j)&1，label = popcount(i) mod 2；n_probes = 2；predict：y1 > y2 → 1（奇）；y1 == y2 为平局。
    model：体素极化率模型（born 请用 object_model("born") = volume_rr，审查 R2-INT-1；make_task 自动选）。
    """

    name = "parity"
    kind = "binary"
    n_probes = 2

    def __init__(self, n: int, roi_center, spacing: float = LAMBDA / 2, diameter: float = LAMBDA / 3,
                 material="water", voxel_d: float | None = None, axis=(0.0, 0.0, 1.0), k: float = K0,
                 model: str = DEFAULT_MODEL):
        if n < 1:
            raise ValueError("n ≥ 1")
        self.n = int(n)
        self.material = material
        self.model = model
        _warn_metal_model(material, model)
        self.k = float(k)
        self.voxel_d = resolve_voxel_d(material, voxel_d, k, _grid_model(model))
        self.diameter = float(diameter)
        self.spacing = float(spacing)
        self.slots = parity_slots(self.n, _as_vec(roi_center), spacing, axis)
        vox = [voxelize_sphere(c, diameter, self.voxel_d) for c in self.slots]
        self.n_vox_per_object = int(len(vox[0][0]))
        samples = []
        labels = parity_labels(self.n)
        for i in range(2 ** self.n):
            x = tuple((i >> j) & 1 for j in range(self.n))
            sets = [voxels_to_nodes(vox[j][0], vox[j][1], material, k, model, group=j)
                    for j in range(self.n) if x[j]]
            obj = NodeSet.concat(sets) if sets else NodeSet.empty()
            samples.append(Sample(obj, int(labels[i]), {"x": x}))
        super().__init__(samples)

    def predict(self, readings_db, labels=None) -> np.ndarray:
        y = np.asarray(readings_db, dtype=float)
        return (y[:, 0] > y[:, 1]).astype(np.int64)

    def ties(self, readings_db, labels=None) -> np.ndarray:
        y = np.asarray(readings_db, dtype=float)
        return y[:, 0] == y[:, 1]

    def sample_margins_db(self, readings_db, labels=None) -> np.ndarray:
        """s_b = sgn(label_b)·(y1 − y2)，sgn = +1 奇 / −1 偶（metrics.binary_margin_db = min_b s_b）。"""
        labels = self.labels if labels is None else np.asarray(labels)
        y = np.asarray(readings_db, dtype=float)
        return np.where(np.asarray(labels).ravel() != 0, 1.0, -1.0) * (y[:, 0] - y[:, 1])

    def describe(self) -> dict:
        d = super().describe()
        d.update(n=self.n, material=str(self.material), model=self.model, voxel_d=self.voxel_d,
                 voxel_d_over_lambda=LAMBDA / self.voxel_d,
                 effective_eps=eps_repr(self.material, self.model), n_vox_per_object=self.n_vox_per_object,
                 diameter=self.diameter, spacing=self.spacing)
        return d


# ============================================================ 字母
class LetterTask(Task):
    """字母 A/B/C 识别（方案 §6.2 实验 C，INTEGRATION E / B2）。

    字母平面由 plane_u（列方向）与 plane_v（行"向上"）张成，法向 n̂ = û×v̂（默认 +x，正对 L90 接收板）。
    默认薄板：voxel_d = λ/10、厚度 = 1 个体素（INTEGRATION B2），像素 pixel×pixel 再细分。
    抖动（每个样本独立）：沿 n̂ 位移 U(−dist_jitter, dist_jitter)；meta["power_db"] = U(−power_jitter_db, power_jitter_db)。
    样本顺序：rep 外层、字母内层；label = 字母在 letters 中的序号；n_probes = 3；predict = argmax（最大值不唯一为平局）。
    """

    name = "letter"
    kind = "multiclass"
    n_probes = 3

    def __init__(self, roi_center, letters: str = "ABC", pixel: float = LAMBDA / 2, voxel_d: float | None = None,
                 thickness: float | None = None, material="metal", plane_u=(0.0, 1.0, 0.0), plane_v=(0.0, 0.0, 1.0),
                 n_per_class: int = 1, dist_jitter: float = 0.0, power_jitter_db: float = 0.0,
                 rng: np.random.Generator | None = None, k: float = K0, model: str = DEFAULT_MODEL):
        if (dist_jitter > 0 or power_jitter_db > 0) and rng is None:
            raise ValueError("有抖动时必须显式传入 rng")
        self.letters = str(letters)
        self.n_probes = len(self.letters)
        self.material = material
        self.model = model
        _warn_metal_model(material, model)
        self.dist_jitter = float(dist_jitter)
        self.power_jitter_db = float(power_jitter_db)
        self.k = float(k)
        lam = 2.0 * np.pi / k
        self.voxel_d = float(voxel_d) if voxel_d is not None else lam / 10.0      # 薄板默认 λ/10
        self.thickness = float(thickness) if thickness is not None else self.voxel_d
        self.pixel = float(pixel)
        u, v = _unit(plane_u), _unit(plane_v)
        nrm = np.cross(u, v)
        center = _as_vec(roi_center)
        self.n_vox = {}
        samples = []
        for rep in range(int(n_per_class)):
            for c, letter in enumerate(self.letters):
                shift = float(rng.uniform(-dist_jitter, dist_jitter)) if dist_jitter > 0 else 0.0
                pdb = float(rng.uniform(-power_jitter_db, power_jitter_db)) if power_jitter_db > 0 else 0.0
                pos, vol, _pix = voxelize_letter(letter, center + shift * nrm, u, v, pixel, self.thickness,
                                                 self.voxel_d)
                self.n_vox[letter] = int(len(pos))
                obj = voxels_to_nodes(pos, vol, material, k, model, group=0)
                samples.append(Sample(obj, c, {"letter": letter, "rep": rep, "dist_shift": shift, "power_db": pdb}))
        super().__init__(samples)

    def predict(self, readings_db, labels=None) -> np.ndarray:
        return np.argmax(np.asarray(readings_db, dtype=float), axis=1).astype(np.int64)

    def ties(self, readings_db, labels=None) -> np.ndarray:
        y = np.asarray(readings_db, dtype=float)
        return np.sum(y == y.max(axis=1, keepdims=True), axis=1) > 1

    def sample_margins_db(self, readings_db, labels=None) -> np.ndarray:
        """y_true − max_{k≠true} y_k：正确时即 top1−top2，错误时为负（INTEGRATION C 的多分类裕量；margin_db 取最小）。"""
        labels = self.labels if labels is None else np.asarray(labels)
        y = np.asarray(readings_db, dtype=float)
        idx = np.arange(len(y))
        yt = y[idx, labels]
        other = y.copy()
        other[idx, labels] = -np.inf
        return yt - other.max(axis=1)

    def decision_units(self, readings_db, labels=None) -> np.ndarray:
        """(B, K−1)：y_true − y_k，k ≠ true 按类序号升序（每个竞争类一个判决单元；最小值 = sample_margins_db）。"""
        labels = self.labels if labels is None else np.asarray(labels)
        y = np.asarray(readings_db, dtype=float)
        B, K = y.shape
        oh = np.zeros((B, K), dtype=bool)
        oh[np.arange(B), labels] = True
        yt = y[np.arange(B), labels]
        return (yt[:, None] - y)[~oh].reshape(B, K - 1)

    def describe(self) -> dict:
        d = super().describe()
        d.update(letters=self.letters, material=str(self.material), model=self.model, voxel_d=self.voxel_d,
                 effective_eps=eps_repr(self.material, self.model), thickness=self.thickness, n_vox=self.n_vox,
                 dist_jitter=self.dist_jitter, power_jitter_db=self.power_jitter_db)
        return d


# ============================================================ 8 格定位（Song 2026）
class LocalizationTask(Task):
    """Song 2026 式 8 格定位（INTEGRATION E）：一套码本、3 个探头各自阈值判决 → 3-bit 格号。

    格子在 plane_u / plane_v 平面内排成 grid = (行数 n_v, 列数 n_u) = (2,4)；格 c = row·n_u + col，
    中心 roi + (col − (n_u−1)/2)·cell·û + (row − (n_v−1)/2)·cell·v̂。每个样本 = 一个直径 obj_diameter 的球位于某格
    （格内抖动 U(−jitter, jitter)² 沿 û、v̂），每格 n_per_cell 个样本（rep 外层、格内层）。
    实验 C 的抖动（方案 §6.2，与 LetterTask 同一约定，审查 S2-0）：沿平面法向 n̂ = û×v̂ 位移 U(−dist_jitter, dist_jitter)、
    meta["power_db"] = U(−power_jitter_db, power_jitter_db)（系统层按它缩放馈源功率）；每样本抽样顺序：格内偏移
    （jitter>0 时）→ 法向位移（dist_jitter>0 时）→ 功率（power_jitter_db>0 时），不启用的项不消耗随机数。
    标签 = 格号 c；探头 j 的比特 = (c>>j)&1，读数 > t_j 判 1。thresholds_db="auto"：t_j = 两类中位数中点
    （losses.loc3bit_thresholds，不回传梯度），或长度 n_probes 的固定数组。损失 = losses.loc3bit_loss × loss_weight。
    "auto" 阈值总是由**被评估样本自身的标签**求出：在留出集上评估请先用训练集求阈值
    （PhysicalSystem.calibrate_thresholds）再传 evaluate(..., thresholds_db=...)（审查 S2-1）。
    """

    name = "loc3bit"
    kind = "loc3bit"
    n_probes = 3

    def __init__(self, roi_center, cell: float = LAMBDA / 2, grid=(2, 4), obj_diameter: float = LAMBDA / 3,
                 material="water", n_per_cell: int = 1, jitter: float = 0.0, plane_u=(0.0, 1.0, 0.0),
                 plane_v=(0.0, 0.0, 1.0), thresholds_db="auto", loss_weight: float = 1.0,
                 rng: np.random.Generator | None = None, voxel_d: float | None = None, k: float = K0,
                 model: str = DEFAULT_MODEL, dist_jitter: float = 0.0, power_jitter_db: float = 0.0):
        if (jitter > 0 or dist_jitter > 0 or power_jitter_db > 0) and rng is None:
            raise ValueError("有抖动（格内 / 法向距离 / 功率）时必须显式传入 rng")
        n_v, n_u = int(grid[0]), int(grid[1])
        n_cells = n_v * n_u
        self.n_probes = max(1, int(np.ceil(np.log2(n_cells))))
        if 2 ** self.n_probes != n_cells:
            raise ValueError(f"格子数 {n_cells} 须为 2 的幂（每个探头编码 1 bit）")
        self.grid = (n_v, n_u)
        self.cell = float(cell)
        self.material = material
        self.model = model
        _warn_metal_model(material, model)
        self.k = float(k)
        self.voxel_d = resolve_voxel_d(material, voxel_d, k, _grid_model(model))
        self.thresholds_db = thresholds_db
        self.loss_weight = float(loss_weight)
        self.jitter = float(jitter)
        self.dist_jitter = float(dist_jitter)
        self.power_jitter_db = float(power_jitter_db)
        u, v = _unit(plane_u), _unit(plane_v)
        nrm = _unit(np.cross(u, v))
        center = _as_vec(roi_center)
        rows, cols = np.divmod(np.arange(n_cells), n_u)
        self.cell_centers = (center + ((cols - (n_u - 1) / 2.0) * cell)[:, None] * u
                             + ((rows - (n_v - 1) / 2.0) * cell)[:, None] * v)
        samples = []
        n_vox = None
        for rep in range(int(n_per_cell)):
            for c in range(n_cells):
                off = rng.uniform(-jitter, jitter, size=2) if jitter > 0 else np.zeros(2)
                shift = float(rng.uniform(-dist_jitter, dist_jitter)) if dist_jitter > 0 else 0.0
                pdb = float(rng.uniform(-power_jitter_db, power_jitter_db)) if power_jitter_db > 0 else 0.0
                pc = self.cell_centers[c] + off[0] * u + off[1] * v + shift * nrm
                pos, vol = voxelize_sphere(pc, obj_diameter, self.voxel_d)
                n_vox = len(pos)
                obj = voxels_to_nodes(pos, vol, material, k, model, group=0)
                samples.append(Sample(obj, c, {"cell": c, "rep": rep, "offset": off.tolist(), "dist_shift": shift,
                                               "power_db": pdb}))
        self.n_vox_per_object = int(n_vox)
        super().__init__(samples)

    @property
    def task_kwargs(self) -> dict:
        return {"thresholds_db": self.thresholds_db, "loss_weight": self.loss_weight}

    def thresholds(self, readings_db, labels=None) -> np.ndarray:
        """t_j：固定阈值，或 "auto" 时按（给定 / 本任务）标签取两类中位数中点。"""
        labels = self.labels if labels is None else labels
        return losses._loc3bit_t(np.asarray(readings_db, dtype=float), labels, self.thresholds_db)

    def bits(self, readings_db, labels=None) -> np.ndarray:
        """(B, n_probes) 判决比特：y_j > t_j → 1。"""
        y = np.asarray(readings_db, dtype=float)
        return (y > self.thresholds(y, labels)[None, :]).astype(np.int64)

    def predict(self, readings_db, labels=None) -> np.ndarray:
        """3 个比特拼成格号 c = Σ_j bit_j·2^j。"""
        b = self.bits(readings_db, labels)
        return (b << np.arange(b.shape[1])[None, :]).sum(axis=1).astype(np.int64)

    def _bit_ties(self, readings_db, labels=None) -> np.ndarray:
        """(B, n_probes)：读数恰等于阈值（logdet 两类中位数都钳在同一边沿时 t 就是边沿值）。"""
        y = np.asarray(readings_db, dtype=float)
        return y == self.thresholds(y, labels)[None, :]

    def ties(self, readings_db, labels=None) -> np.ndarray:
        return np.any(self._bit_ties(readings_db, labels), axis=1)

    def bit_accuracy(self, readings_db, labels=None) -> np.ndarray:
        """逐比特（逐探头）准确率 (n_probes,)；读数恰在阈值上的比特计为错（审查 R2-INT-2）。"""
        labels = self.labels if labels is None else np.asarray(labels)
        truth = (labels[:, None] >> np.arange(self.n_probes)[None, :]) & 1
        ok = (self.bits(readings_db, labels) == truth) & ~self._bit_ties(readings_db, labels)
        return np.mean(ok, axis=0)

    def sample_margins_db(self, readings_db, labels=None) -> np.ndarray:
        """min_j sgn(bit_bj)·(y_bj − t_j)：> 0 ⇔ 样本 b 的全部比特判对；margin_db 取 min_b。

        阈值 t 始终由全部样本（与 predict 相同）求出，margin_db 的 mask 只限定取最小的样本（审查 R2-INT-3：在子集上重算
        "auto" 阈值时，被标记样本一去掉，某一位可能只剩一类 → losses._median_midpoint 抛 ValueError，整次评估崩溃）。
        """
        return np.min(self.decision_units(readings_db, labels), axis=1)

    def decision_units(self, readings_db, labels=None) -> np.ndarray:
        """(B, n_probes)：sgn(bit_bj)·(y_bj − t_j)，每个探头比特一个判决单元（阈值同 predict，由全部样本求出）。"""
        labels = self.labels if labels is None else np.asarray(labels)
        y = np.asarray(readings_db, dtype=float)
        t = self.thresholds(y, labels)
        sgn = 2.0 * ((labels[:, None] >> np.arange(self.n_probes)[None, :]) & 1) - 1.0
        return sgn * (y - t[None, :])

    def describe(self) -> dict:
        d = super().describe()
        d.update(grid=list(self.grid), cell=self.cell, material=str(self.material), model=self.model,
                 voxel_d=self.voxel_d,
                 effective_eps=eps_repr(self.material, self.model), n_vox_per_object=self.n_vox_per_object,
                 thresholds_db=self.thresholds_db if isinstance(self.thresholds_db, str)
                 else list(map(float, self.thresholds_db)), loss_weight=self.loss_weight, jitter=self.jitter,
                 dist_jitter=self.dist_jitter, power_jitter_db=self.power_jitter_db)
        return d


# ============================================================ 聚焦
class FocusTask(Task):
    """无物体单样本，最大化探头 0 功率（实验 F；losses.focus_loss）。"""

    name = "focus"
    kind = "focus"

    def __init__(self, n_probes: int = 1):
        self.n_probes = int(n_probes)
        self.material = None
        self.model = None
        self.voxel_d = None
        super().__init__([Sample(NodeSet.empty(), 0, {})])

    def predict(self, readings_db, labels=None) -> np.ndarray:
        return np.zeros(np.shape(readings_db)[0], dtype=np.int64)

    def sample_margins_db(self, readings_db, labels=None) -> np.ndarray:
        return np.full(np.shape(readings_db)[0], np.nan)

    def decision_units(self, readings_db, labels=None) -> np.ndarray:
        raise ValueError("focus 任务没有判决单元（训练目标退回系统损失，INTEGRATION T）")

    def margin_db(self, readings_db, labels=None, mask=None) -> float:
        return float("nan")
