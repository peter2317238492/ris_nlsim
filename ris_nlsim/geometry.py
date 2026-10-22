"""几何：中继板、观测点、物体体素化、字母位图、奇偶槽位（DESIGN §1.4）。

坐标单位米；方位角 az 在 xy 平面从 +x 转向 +y，仰角 el 从 xy 平面向 +z（DESIGN §1）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import LAMBDA, PITCH


def _vec3(x, name: str = "vector") -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.shape != (3,):
        raise ValueError(f"{name} 必须是 3 维向量，得到形状 {np.shape(x)}")
    return v


def _unit(x, name: str = "vector") -> np.ndarray:
    v = _vec3(x, name)
    n = np.linalg.norm(v)
    if n < 1e-15:
        raise ValueError(f"{name} 不能为零向量")
    return v / n


# ---------------------------------------------------------------- 板与中继

def board_elements(origin_corner, u_axis, v_axis, nu: int = 5, nv: int = 5,
                   pitch: float = PITCH) -> np.ndarray:
    """矩形板单元位置（单元在格心）(nu*nv, 3)。

    第 (i,j) 个单元（索引 i*nv + j）位于 corner + (i+½)·pitch·û + (j+½)·pitch·v̂；û、v̂ 自动归一化。
    """
    c = _vec3(origin_corner, "origin_corner")
    u = _unit(u_axis, "u_axis")
    v = _unit(v_axis, "v_axis")
    if nu < 1 or nv < 1 or pitch <= 0:
        raise ValueError("nu、nv ≥ 1 且 pitch > 0")
    i, j = np.meshgrid(np.arange(nu) + 0.5, np.arange(nv) + 0.5, indexing="ij")
    return c + pitch * (i.reshape(-1, 1) * u + j.reshape(-1, 1) * v)


@dataclass(eq=False)   # 含数组字段：按对象身份比较
class RelayGeom:
    """一块折叠中继（收、发两块 n×n 板）的几何。

    DESIGN §1.4 字段：rx_pos (n²,3)、rx_normal (3,)、tx_pos (n²,3)、tx_normal (3,)、
    rx_center、tx_center、center（= origin，即名义合页参考点）、fold_deg。
    附加字段：rx_u / tx_u 为各板"从合页向外"的板内方向，v_axis = +z，n、pitch、gap。
    单元索引 i*n + j：i 沿 *_u（从合页向外），j 沿 +z；收、发板同索引互为镜像。
    """

    rx_pos: np.ndarray
    rx_normal: np.ndarray
    tx_pos: np.ndarray
    tx_normal: np.ndarray
    rx_center: np.ndarray
    tx_center: np.ndarray
    center: np.ndarray
    fold_deg: float
    rx_u: np.ndarray | None = None
    tx_u: np.ndarray | None = None
    v_axis: np.ndarray | None = None
    n: int = 5
    pitch: float = PITCH
    gap: float = 0.01

    @property
    def width(self) -> float:
        """板边长 W = n·pitch。"""
        return self.n * self.pitch


def make_relay(fold_deg: float = 90.0, origin=(0.0, 0.0, 0.0), gap: float = 0.01, n: int = 5,
               pitch: float = PITCH, *, back_offset: float | None = None) -> RelayGeom:
    """折叠中继几何（合页轴平行 z 轴，DESIGN §1.4）。

    接收板：平面 x = origin.x，y ∈ [−gap/2 − W, −gap/2]，z ∈ [−W/2, W/2]（相对 origin），法向 +x。
    发射板：沿 e_t = (−sinψ, cosψ, 0) 从合页展开（内边缘偏移 gap/2），法向 n_t = (cosψ, sinψ, 0)。

    取舍（偏离 DESIGN 字面公式）：DESIGN 的发射板在 ψ=180° 时与接收板**完全重合**（单元位置相同），
    ψ 接近 180° 时两板内边缘距离 gap·cos(ψ/2) → 0，物理上板会相撞。这里把真实合页轴放在接收板
    平面之后 b = back_offset（默认 gap/2）处，发射板 = 接收板关于 y 镜像后绕该轴转 ψ，
    等价于 DESIGN 公式再平移 b·(n_t − x̂)：
      ψ=0 与 DESIGN 完全一致（共面，内边缘相距 gap）；ψ=90° 平移 (−b, b, 0)；
      ψ=180° 背靠背、两板平面相距 2b（默认 = gap）。两板内边缘距离 = gap·sqrt(1+sinψ) ≥ gap。
    back_offset=0 可恢复 DESIGN 字面公式（ψ=180° 时单元重合，下游耦合会报错）。
    """
    psi_deg = float(fold_deg)
    if not (0.0 <= psi_deg <= 180.0):
        raise ValueError(f"fold_deg 应在 [0, 180]，得到 {fold_deg}")
    if gap < 0:
        raise ValueError("gap 不能为负")
    o = _vec3(origin, "origin")
    b = gap / 2.0 if back_offset is None else float(back_offset)
    psi = np.deg2rad(psi_deg)
    W = n * pitch
    ex = np.array([1.0, 0.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    # 接收板（方案 §2；DESIGN §1.4）
    rx_u = np.array([0.0, -1.0, 0.0])
    rx_corner = o + np.array([0.0, -gap / 2.0, -W / 2.0])
    rx_pos = board_elements(rx_corner, rx_u, ez, n, n, pitch)
    # 发射板：e_t = (−sinψ, cosψ, 0)，n_t = (cosψ, sinψ, 0)
    e_t = np.array([-np.sin(psi), np.cos(psi), 0.0])
    n_t = np.array([np.cos(psi), np.sin(psi), 0.0])
    tx_corner = o + (gap / 2.0) * e_t + b * (n_t - ex) + np.array([0.0, 0.0, -W / 2.0])
    tx_pos = board_elements(tx_corner, e_t, ez, n, n, pitch)
    return RelayGeom(
        rx_pos=rx_pos, rx_normal=ex.copy(), tx_pos=tx_pos, tx_normal=n_t,
        rx_center=rx_pos.mean(axis=0), tx_center=tx_pos.mean(axis=0), center=o.copy(),
        fold_deg=psi_deg, rx_u=rx_u, tx_u=e_t, v_axis=ez.copy(), n=int(n), pitch=float(pitch),
        gap=float(gap),
    )


def make_relays(n_relays: int, fold_deg: float, spacing_z: float = 0.30, origin=(0.0, 0.0, 0.0),
                **kwargs) -> list[RelayGeom]:
    """沿 z 等距排列 n_relays 块中继，整体以 origin 居中；其余参数传给 make_relay。"""
    if n_relays < 1:
        raise ValueError("n_relays ≥ 1")
    o = _vec3(origin, "origin")
    n = kwargs.get("n", 5)
    pitch = kwargs.get("pitch", PITCH)
    if n_relays > 1 and spacing_z < n * pitch - 1e-12:
        raise ValueError(f"spacing_z={spacing_z} 小于板宽 {n * pitch}，板会重叠")
    zs = (np.arange(n_relays) - (n_relays - 1) / 2.0) * spacing_z
    return [make_relay(fold_deg, o + np.array([0.0, 0.0, z]), **kwargs) for z in zs]


def point_from(ref, az_deg: float, el_deg: float, dist: float) -> np.ndarray:
    """ref + dist·(cos el·cos az, cos el·sin az, sin el)。"""
    r = _vec3(ref, "ref")
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return r + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


# ---------------------------------------------------------------- 体素化

def voxelize_sphere(center, diameter: float, d: float) -> tuple[np.ndarray, float]:
    """球体体素化（DESIGN §1.4）：以 center 为格点、名义间距 d 的立方网格，取 |g| ≤ D/2 的格点（N 个），
    再把格点偏移整体缩放 s = ((πD³/6)/(N·d³))^{1/3}，使实际格距 s·d 满足 (s·d)³ = vol_each = (πD³/6)/N。

    即 N·vol_each = πD³/6 精确成立，且"格胞体积 = 偶极子体积"（DDSCAT 惯例 N·d_eff³ = V）。中心格点总在内
    ⇒ 至少 1 个体素（N = 1 时偏移全为 0，s 无影响）。返回的 vol_each^{1/3} 就是实际格距。

    取舍（PHY-R2-2，偏离"格距恰为 d"的字面读法）：小球（D/d ≈ 3~7）的壳层台阶使 N 偏离理想值 πD³/(6d³)
    达 ±20%（λ/3 球 λ/15：81 vs 65.4），只改 vol_each 而保留格距 d 时，格胞体积 d³ 与偶极子体积不一致，
    相当于把物体在空间上放大/缩小，DDA 远场误差随网格非单调（scalar_rr εr=10 vs 标量 Mie：λ/10 3.6%、
    λ/15 25.6%、λ/20 14.7%）。缩放格距后误差随网格平稳（同例 2.9%、3.1%、3.6%；tests/test_solver.py 有回归）。
    代价：实际格距在 d 附近 ±10% 浮动；s > 1 时最外层格点可略超出球面，|pos − c| ≤ s·D/2。
    """
    c = _vec3(center, "center")
    if diameter <= 0 or d <= 0:
        raise ValueError("diameter、d 必须为正")
    r = diameter / 2.0
    m = int(np.floor(r / d + 1e-9))
    idx = np.arange(-m, m + 1, dtype=np.float64)
    g = np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), axis=-1).reshape(-1, 3)   # 单位格距的整数格点
    inside = np.einsum("ij,ij->i", g, g) * d * d <= r * r * (1.0 + 1e-12)
    n_vox = int(np.count_nonzero(inside))
    vol_each = (np.pi * diameter**3 / 6.0) / n_vox             # 体积修正：N·vol_each = πD³/6
    pitch_eff = np.cbrt(vol_each)                               # 实际格距 s·d，格胞体积 = vol_each
    pos = c + pitch_eff * g[inside]
    return pos, float(vol_each)


def _box_offsets(size, d: float) -> np.ndarray:
    """长方体（以原点为中心、边长 size）的体素中心偏移，每轴格数 ceil(size/d)（体素边长 ≤ d）。"""
    size = np.asarray(size, dtype=np.float64)
    if np.any(size <= 0) or d <= 0:
        raise ValueError("尺寸与 d 必须为正")
    counts = np.maximum(1, np.ceil(size / d - 1e-9)).astype(int)
    axes = [(np.arange(nc) - (nc - 1) / 2.0) * (s / nc) for nc, s in zip(counts, size)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def voxelize_box(center, size_xyz, d: float) -> tuple[np.ndarray, float]:
    """轴对齐长方体体素化：每轴 ceil(size/d) 格均分（体素边长 ≤ d），vol_each = 体积/N。"""
    c = _vec3(center, "center")
    size = _vec3(size_xyz, "size_xyz")
    off = _box_offsets(size, d)
    return c + off, float(np.prod(size) / len(off))


# ---------------------------------------------------------------- 字母

def _bitmap(rows: list[str]) -> np.ndarray:
    a = np.array([[int(ch) for ch in r] for r in rows], dtype=np.int64)
    a.setflags(write=False)
    return a


# 7 行 × 5 列，行 0 在上、列 0 在左（经典 5×7 点阵字体）
LETTERS: dict[str, np.ndarray] = {
    "A": _bitmap(["01110", "10001", "10001", "11111", "10001", "10001", "10001"]),
    "B": _bitmap(["11110", "10001", "10001", "11110", "10001", "10001", "11110"]),
    "C": _bitmap(["01110", "10001", "10000", "10000", "10000", "10001", "01110"]),
}


def voxelize_letter(letter: str, center, plane_u, plane_v, pixel: float = LAMBDA / 2,
                    thickness: float = LAMBDA / 2, d: float = LAMBDA / 5
                    ) -> tuple[np.ndarray, float, np.ndarray]:
    """字母（7×5 位图）体素化。

    plane_u：列方向（左→右），plane_v：行"向上"方向（行 0 在 +v 一侧），二者须正交；
    厚度沿 n̂ = û×v̂。位图中心（行 3、列 2）位于 center。
    每个亮像素是 pixel×pixel×thickness 的方块，按 _box_offsets 再细分（体素边长 ≤ d）。
    返回 (pos (N,3), vol_each, pixel_ids (N,) int)，pixel_id = 行*5 + 列。
    """
    key = letter.upper()
    if key not in LETTERS:
        raise ValueError(f"未知字母 {letter!r}，可选 {list(LETTERS)}")
    bm = LETTERS[key]
    c = _vec3(center, "center")
    u = _unit(plane_u, "plane_u")
    v = _unit(plane_v, "plane_v")
    if abs(np.dot(u, v)) > 1e-9:
        raise ValueError("plane_u 与 plane_v 必须正交")
    nrm = np.cross(u, v)
    nrow, ncol = bm.shape
    rc = np.argwhere(bm > 0)                                   # (P,2) 亮像素 (行, 列)
    pix_c = (c + ((rc[:, 1] - (ncol - 1) / 2.0) * pixel)[:, None] * u
             + (((nrow - 1) / 2.0 - rc[:, 0]) * pixel)[:, None] * v)
    local = _box_offsets([pixel, pixel, thickness], d)        # (M,3) 局部 (u,v,n) 坐标
    world = local @ np.stack([u, v, nrm])                      # (M,3)
    pos = (pix_c[:, None, :] + world[None, :, :]).reshape(-1, 3)
    pixel_ids = np.repeat(rc[:, 0] * ncol + rc[:, 1], len(local))
    vol_each = pixel * pixel * thickness / len(local)
    return pos, float(vol_each), pixel_ids.astype(np.int64)


def parity_slots(n: int, center, spacing: float, axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    """奇偶任务槽位 (n,3)：沿 axis 等距 spacing，以 center 居中。"""
    if n < 1:
        raise ValueError("n ≥ 1")
    c = _vec3(center, "center")
    a = _unit(axis, "axis")
    offs = (np.arange(n) - (n - 1) / 2.0) * spacing
    return c + offs[:, None] * a
