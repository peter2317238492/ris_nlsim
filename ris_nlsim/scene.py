"""场景数据结构与矩阵拼装（DESIGN §2.1）。

NodeSet：共享同一方向图的一组节点（源 / 散射体 / 接收端）。
Scene：一次仿真的全部固定节点集合（馈源、中继收发单元、探头、环境、无源面）。
文件后半部分是把 list[NodeSet] 拼成大矩阵的辅助函数（源列、接收行、散射体耦合）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import wavelength, wavenumber
from .patterns import Pattern


def _as_points(x, name: str) -> np.ndarray:
    """转成 (N,3) float64 副本；单个点 (3,) → (1,3)。"""
    a = np.array(x, dtype=np.float64, copy=True)
    if a.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if a.ndim == 1 and a.shape[0] == 3:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} 形状应为 (N,3)，得到 {a.shape}")
    return a


@dataclass(eq=False)   # 含数组字段：按对象身份比较
class NodeSet:
    """一组共享方向图的节点。

    pos (N,3)；normal (N,3) 或 (3,)（广播到全部节点；各向同性节点可为 None/0）；
    alpha (N,) complex（散射体极化率 m³，可为标量广播）；group (N,) int（物体编号，其余 -1）。
    构造时：非零法向被归一化；非各向同性方向图要求全部法向非零。
    """

    pos: np.ndarray
    normal: np.ndarray
    pattern: Pattern
    alpha: np.ndarray | None = None
    group: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.pattern, Pattern):
            raise TypeError("pattern 必须是 Pattern")
        self.pos = _as_points(self.pos, "pos")
        n = len(self.pos)
        # 法向：None → 0；(3,) → 广播；非零行归一化
        if self.normal is None:
            nrm = np.zeros((n, 3))
        else:
            nrm = np.array(self.normal, dtype=np.float64, copy=True)
            if nrm.ndim == 1 and nrm.shape == (3,):
                nrm = np.broadcast_to(nrm, (n, 3)).copy()
            elif nrm.size == 0 and n == 0:
                nrm = np.zeros((0, 3))
        if nrm.shape != (n, 3):
            raise ValueError(f"normal 形状应为 ({n},3)，得到 {nrm.shape}")
        length = np.linalg.norm(nrm, axis=1)
        nz = length > 1e-12
        nrm[nz] /= length[nz, None]
        if not self.pattern.isotropic and not np.all(nz):
            raise ValueError("有方向性的节点集合要求每个节点法向非零")
        self.normal = nrm
        if self.alpha is not None:
            a = np.asarray(self.alpha, dtype=np.complex128)
            a = np.full(n, a, dtype=np.complex128) if a.ndim == 0 else a.copy()
            if a.shape != (n,):
                raise ValueError(f"alpha 形状应为 ({n},)，得到 {a.shape}")
            self.alpha = a
        if self.group is not None:
            g = np.asarray(self.group)
            g = np.full(n, int(g), dtype=np.int64) if g.ndim == 0 else g.astype(np.int64)
            if g.shape != (n,):
                raise ValueError(f"group 形状应为 ({n},)，得到 {g.shape}")
            self.group = g

    @staticmethod
    def empty(pattern: Pattern = Pattern()) -> "NodeSet":
        """空集合（0 个节点）。"""
        return NodeSet(np.zeros((0, 3)), np.zeros((0, 3)), pattern)

    def __len__(self) -> int:
        return int(self.pos.shape[0])

    def take(self, idx) -> "NodeSet":
        """取子集（idx 为整数索引或布尔掩码），方向图不变。"""
        idx = np.asarray(idx)
        return NodeSet(
            self.pos[idx], self.normal[idx], self.pattern,
            None if self.alpha is None else self.alpha[idx],
            None if self.group is None else self.group[idx],
        )

    @staticmethod
    def concat(sets: list["NodeSet"]) -> "NodeSet":
        """按顺序拼接；要求 pattern 相同（不同方向图请分开存放为 list[NodeSet]）。

        alpha：非空集合要么全有、要么全无（混合时报错）；group：缺失的集合填 -1。
        空列表返回各向同性空集合。
        """
        sets = list(sets)
        if not sets:
            return NodeSet.empty()
        pat = sets[0].pattern
        for s in sets[1:]:
            if s.pattern != pat:
                raise ValueError(f"concat 要求相同 pattern：{pat} vs {s.pattern}")
        nonempty = [s for s in sets if len(s) > 0]
        has_alpha = [s.alpha is not None for s in nonempty]
        if any(has_alpha) and not all(has_alpha):
            raise ValueError("concat：部分集合缺少 alpha")
        pos = np.concatenate([s.pos for s in sets], axis=0)
        nrm = np.concatenate([s.normal for s in sets], axis=0)
        if nonempty and all(has_alpha):
            alpha = np.concatenate([s.alpha for s in nonempty])
        elif not nonempty and any(s.alpha is not None for s in sets):
            alpha = np.zeros(0, dtype=np.complex128)
        else:
            alpha = None
        if any(s.group is not None for s in sets):
            group = np.concatenate([
                s.group if s.group is not None else np.full(len(s), -1, dtype=np.int64)
                for s in sets
            ])
        else:
            group = None
        return NodeSet(pos, nrm, pat, alpha, group)


def _as_list(x) -> list[NodeSet]:
    if x is None:
        return []
    if isinstance(x, NodeSet):
        return [x]
    out = list(x)
    if not all(isinstance(s, NodeSet) for s in out):
        raise TypeError("期望 NodeSet 列表")
    return out


@dataclass(eq=False)   # 含数组字段：按对象身份比较
class Scene:
    """仿真场景（DESIGN §2.1）。

    源列顺序：[feed, tx_0..tx_{NT-1}]（S = 1 + N_T 列，所有馈源按 feed_weights 合成第 0 列）；
    接收行顺序：[rx_0..rx_{NRX-1}, probe_0..probe_{K-1}]。
    除 freq_hz、feeds 外均有默认值（空集合 / 空列表 / 权重全 1）。
    """

    freq_hz: float
    feeds: NodeSet
    feed_weights: np.ndarray | None = None
    tx: list[NodeSet] = field(default_factory=list)
    rx: list[NodeSet] = field(default_factory=list)
    probes: NodeSet = field(default_factory=NodeSet.empty)
    env: list[NodeSet] = field(default_factory=list)
    passive: NodeSet = field(default_factory=NodeSet.empty)
    block_direct_feed_probe: bool = False
    skip_gain_db: float = 0.0
    include_direct_tx_rx: bool = False

    def __post_init__(self):
        if not (np.isfinite(self.freq_hz) and self.freq_hz > 0):
            raise ValueError("freq_hz 必须为正")
        if not isinstance(self.feeds, NodeSet):
            raise TypeError("feeds 必须是 NodeSet")
        nf = len(self.feeds)
        if self.feed_weights is None:
            self.feed_weights = np.ones(nf, dtype=np.complex128)
        else:
            w = np.asarray(self.feed_weights, dtype=np.complex128).reshape(-1)
            if w.shape != (nf,):
                raise ValueError(f"feed_weights 长度应为 {nf}，得到 {w.shape}")
            self.feed_weights = w
        self.tx = _as_list(self.tx)
        self.rx = _as_list(self.rx)
        self.env = _as_list(self.env)
        if self.probes is None:
            self.probes = NodeSet.empty()
        if self.passive is None:
            self.passive = NodeSet.empty()

    # ---- 便捷量 ----
    @property
    def k(self) -> float:
        return wavenumber(self.freq_hz)

    @property
    def lam(self) -> float:
        return wavelength(self.freq_hz)

    @property
    def n_tx(self) -> int:
        return sum(len(s) for s in self.tx)

    @property
    def n_rx(self) -> int:
        return sum(len(s) for s in self.rx)

    @property
    def n_probe(self) -> int:
        return len(self.probes)

    @property
    def n_src(self) -> int:
        """源列数 S = 1 + N_T。"""
        return 1 + self.n_tx

    @property
    def n_rcv(self) -> int:
        """接收行数 N_RCV = N_RX + K。"""
        return self.n_rx + self.n_probe

    @property
    def n_env(self) -> int:
        return sum(len(s) for s in self.env)

    @property
    def n_passive(self) -> int:
        return len(self.passive)


# ===========================================================================
# 矩阵拼装辅助（DESIGN §2.1–§2.2，方案 §4.1–§4.3）
#
# 源列顺序 [feed, tx_0..tx_{NT-1}]：所有馈源按 feed_weights 线性合成第 0 列，每个发射单元一列。
# 接收行顺序 [rx_0..rx_{NRX-1}, probe_0..probe_{K-1}]。
# 散射体列表 list[NodeSet] 按顺序拼接；同一 NodeSet 对象自身块的对角（自作用）为 0。
# 注意：greens 在模块级 `from .scene import NodeSet`，为避免循环导入，这里在函数内延迟导入 greens。
# ===========================================================================

def receiver_sets(scene: Scene) -> list[NodeSet]:
    """接收端集合，行顺序 [rx_0.., rx_{R-1}, probes]（DESIGN §2.1）。"""
    return [*scene.rx, scene.probes]


def _check_unique(sets: list[NodeSet], what: str) -> None:
    """同一 NodeSet 对象在列表里出现两次会被误当成"跨集合"耦合（重合节点），直接报错。

    注意反过来也成立：assemble_scat_to_scat 把 rows 与 cols 里的同一对象当作"同一集合"（对角置 0），
    所以调用方须保证物体 / 环境 / 无源面互不相同（solver 在入口统一检查）。
    """
    ids = [id(s) for s in sets if len(s) > 0]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{what}：同一个 NodeSet 对象出现了多次")


def check_scatterer_sets(scene: Scene, objects: NodeSet) -> None:
    """物体、各环境集合、无源面必须是互不相同的 NodeSet 对象（空集合除外）。

    否则同一对象会在 born/intra 的 C_BO 或约化表示的 C_XP 里被当成"同一集合"、对角置 0，
    把散射体同时算作背景和物体却不报错。
    """
    _check_unique([objects, *scene.env, scene.passive], "objects/env/passive")


def stack_alpha(sets: list[NodeSet]) -> np.ndarray:
    """按顺序拼接各散射体集合的极化率 → (N,) complex128；非空集合缺 alpha 时报错。"""
    parts = []
    for s in sets:
        if len(s) == 0:
            continue
        if s.alpha is None:
            raise ValueError("散射体集合缺少 alpha")
        parts.append(s.alpha)
    if not parts:
        return np.zeros(0, dtype=np.complex128)
    return np.concatenate(parts).astype(np.complex128, copy=False)


def as_passive_phase(passive_phase, n_p: int) -> np.ndarray | None:
    """校验并返回无源面相位 (n_p,) float64（弧度）；None → 全 0。n_p == 0 时返回 None。

    复数输入（例如误传权重 e^{jφ} 而不是相位 φ）直接 TypeError，避免 float 转换静默丢掉虚部。
    """
    if passive_phase is not None and np.iscomplexobj(passive_phase):
        raise TypeError("passive_phase 应为实数相位（弧度），得到复数数组；若手上是权重 w=e^{jφ}，请传 np.angle(w)")
    if n_p == 0:
        if passive_phase is not None and np.size(passive_phase) != 0:
            raise ValueError("场景无无源面，但给了非空 passive_phase")
        return None
    if passive_phase is None:
        return np.zeros(n_p)
    phase = np.asarray(passive_phase, dtype=np.float64).reshape(-1)
    if phase.shape != (n_p,):
        raise ValueError(f"passive_phase 长度应为 {n_p}，得到 {phase.shape}")
    return phase


def passive_with_phase(scene: Scene, passive_phase) -> NodeSet:
    """返回带相位的无源面副本：α_P = alpha_base·exp(jφ)（DESIGN §2.2）。

    passive_phase=None 视为全 0；长度须等于 N_P；必须是实数。原 scene.passive 不被修改。
    """
    P = scene.passive
    phase = as_passive_phase(passive_phase, len(P))
    if phase is None:
        return P
    if P.alpha is None:
        raise ValueError("无源面 passive 缺少 alpha（应为 λΓ/k²）")
    return NodeSet(P.pos, P.normal, P.pattern, alpha=P.alpha * np.exp(1j * phase), group=P.group)


def assemble_scat_to_scat(rows: list[NodeSet], cols: list[NodeSet], k: float) -> np.ndarray:
    """散射体耦合块矩阵 C (ΣN_rows, ΣN_cols)：C[a,b] = scat_to_scat(a, b)（DESIGN §1.2）。

    rows[i] 与 cols[j] 为同一 NodeSet 对象时视为同一集合（对角自作用置 0）。
    """
    from .greens import scat_to_scat

    rows, cols = list(rows), list(cols)
    _check_unique(rows, "rows")
    _check_unique(cols, "cols")
    nr = sum(len(s) for s in rows)
    nc = sum(len(s) for s in cols)
    out = np.zeros((nr, nc), dtype=np.complex128)
    i0 = 0
    for a in rows:
        j0 = 0
        for b in cols:
            if len(a) and len(b):
                out[i0:i0 + len(a), j0:j0 + len(b)] = scat_to_scat(a, b, k, same=a is b)
            j0 += len(b)
        i0 += len(a)
    return out


def _feed_column(fn, targets: list[NodeSet], scene: Scene) -> np.ndarray:
    """第 0 列：Σ_f w_f · fn(target, feed_f)，所有馈源合成为一个源（DESIGN §2.1）。"""
    k = scene.k
    n = sum(len(t) for t in targets)
    if len(scene.feeds) == 0 or n == 0:
        return np.zeros(n, dtype=np.complex128)
    return np.concatenate([fn(t, scene.feeds, k) @ scene.feed_weights for t in targets])


def _tx_columns(fn, targets: list[NodeSet], scene: Scene) -> np.ndarray:
    """第 1..N_T 列：每个发射单元一列，按 scene.tx 列表顺序拼接。"""
    k = scene.k
    n = sum(len(t) for t in targets)
    out = np.zeros((n, scene.n_tx), dtype=np.complex128)
    i0 = 0
    for t in targets:
        j0 = 0
        for s in scene.tx:
            if len(t) and len(s):
                out[i0:i0 + len(t), j0:j0 + len(s)] = fn(t, s, k)
            j0 += len(s)
        i0 += len(t)
    return out


def assemble_src_to_scat(scene: Scene, scat: list[NodeSet]) -> np.ndarray:
    """源列在散射体处的局部场 E0 (ΣN_scat, S)，S = 1 + N_T（DESIGN §2.2 的 E0）。

    第 0 列 = Σ_f w_f·src_to_scat(·, feed_f)；第 1+m 列 = 发射单元 m（始终包含，经物体的反馈环靠它）。
    """
    from .greens import src_to_scat

    scat = list(scat)
    E0 = np.empty((sum(len(s) for s in scat), scene.n_src), dtype=np.complex128)
    E0[:, 0] = _feed_column(src_to_scat, scat, scene)
    E0[:, 1:] = _tx_columns(src_to_scat, scat, scene)
    return E0


def assemble_scat_to_rcv(scene: Scene, scat: list[NodeSet]) -> np.ndarray:
    """散射体偶极矩 → 接收量 C_RS (N_RCV, ΣN_scat)，行顺序 [rx.., probes]（DESIGN §2.2）。"""
    from .greens import scat_to_rcv

    scat = list(scat)
    rcv = receiver_sets(scene)
    out = np.zeros((scene.n_rcv, sum(len(s) for s in scat)), dtype=np.complex128)
    i0 = 0
    for a in rcv:
        j0 = 0
        for b in scat:
            if len(a) and len(b):
                out[i0:i0 + len(a), j0:j0 + len(b)] = scat_to_rcv(a, b, scene.k)
            j0 += len(b)
        i0 += len(a)
    return out


def feed_probe_los_factor(scene: Scene) -> float:
    """馈源→探头视距项的幅度系数：吸波屏遮挡时为 0，否则 10^(skip_gain_db/20)（方案 §5，DESIGN §2.1）。"""
    if scene.block_direct_feed_probe:
        return 0.0
    return float(10.0 ** (scene.skip_gain_db / 20.0))


def assemble_src_to_rcv(scene: Scene) -> np.ndarray:
    """直达项 E0_R (N_RCV, S)：源不经任何散射体直接到接收端（DESIGN §2.1）。

    - 馈源→rx：始终包含；馈源→探头：乘 feed_probe_los_factor（遮挡/skip 旋钮）。
    - 发射单元→探头：始终包含。
    - 发射单元→rx：同一中继（scene.tx[r]→scene.rx[r]）仅 include_direct_tx_rx=True 时包含（板间泄漏由 κ0 统一表示）；
      不同中继之间（r≠s，N_r>1）的自由空间耦合始终包含（κ0 只描述同一中继收发板的泄漏；审查 PHY-R2-4）。
    """
    from .greens import src_to_rcv

    n_rx = scene.n_rx
    H0 = np.zeros((scene.n_rcv, scene.n_src), dtype=np.complex128)
    H0[:n_rx, 0] = _feed_column(src_to_rcv, scene.rx, scene)
    H0[n_rx:, 0] = feed_probe_los_factor(scene) * _feed_column(src_to_rcv, [scene.probes], scene)
    tx_rx = _tx_columns(src_to_rcv, scene.rx, scene)
    if not scene.include_direct_tx_rx and len(scene.rx) == len(scene.tx):
        i0 = j0 = 0
        for a, b in zip(scene.rx, scene.tx):            # 第 r 个 rx 集合与第 r 个 tx 集合属于同一中继
            tx_rx[i0:i0 + len(a), j0:j0 + len(b)] = 0.0
            i0 += len(a)
            j0 += len(b)
    elif not scene.include_direct_tx_rx:
        tx_rx[:] = 0.0                                   # 无法按中继配对时退回旧行为：全部交给 κ0
    H0[:n_rx, 1:] = tx_rx
    H0[n_rx:, 1:] = _tx_columns(src_to_rcv, [scene.probes], scene)
    return H0
