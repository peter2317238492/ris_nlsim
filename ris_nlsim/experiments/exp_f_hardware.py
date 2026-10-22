"""实验 F：硬件对齐（方案 §6.2 表 F 行、§8 门 G1；INTEGRATION G "F"；DESIGN §9）。

问题：仿真器里"中继相对同尺寸金属板的增益提升"能否落在实测区间（实测表 测试数据.xlsx，不随仓库发布，用 --xlsx 指定）±3 dB 内？
任务：单探头聚焦、无物体输入（方案 §6.2）。

测量数据（load_measurements）：Sheet1 两块——馈源喇叭在方位 30°、距阵列 1.8 m，以及方位 50°、1.5 m；每块有绝对增益表
（含"金属反射板增益"列）与"增益相较金属板增益提升"表（3.45–3.55 GHz × 各配置）。只用后者做对比；顺带核对
提升 = 绝对增益 − 金属板增益（报告最大偏差）。

列名解读（parse_label；依据：项目总结 pptx 第 4–6 页 "(0,0) 指向和 (90,15) 指向"、"结构上采用可旋转的物理合页，
0°–180° 折叠，搭配波束控制实现方位面 360° 覆盖"、"分别通过超表面波束指向和结构的旋转"）：
- "收a b-发c d" / "接收a b 发c d"：接收板、发射板的波束指向 (φ, θ)（HFSS 约定，θ 为偏离板法向角，φ 为方位平面角），
  "00" = (0,0) 法向。φ = 90° 是方位（水平）面、偏向馈源一侧——两块的最后一列"共面、仅靠波束控制"恰是接收 (90, Δ)、
  发射 (0,0)，与馈源方位 Δ = 30°/50° 吻合。
- 两块板相同、同向安装 ⇒ φ = 90° 对两块板是同一物理侧（偏向馈源）。由此，接收波束在方位面偏 α_r、发射波束偏 α_t
  （α = θ 的方位分量）时，"其余角度由结构旋转补齐"给出折叠角 ψ = Δ − α_r + α_t（make_relay 的 ψ：0 共面、90 L 形）。
  核对：这样每块里只有标注"无夹角"的那一列 ψ = 0（若取 ψ = Δ − α_r − α_t，30° 块的"收00-发90 30"、"接收90 15 发90 15"
  和 50° 块的"收00-发90 50"也会共面，与表注矛盾）。
- 30° 块的"接收15 0"、"接收15 15"不在 φ ∈ {0, 90} 上，有歧义：按 (φ, θ) 字面，(15,0) 就是法向（与"00"同码本，
  但实测比"收00-发00"低约 4 dB），(15,15) 主要是俯仰 14.5°；另一读法是 (方位, 俯仰) = (15°,0°) / (15°,15°)。
  两种读法都算（reading = "phi_theta" / "az_el"），标记 ambiguous，门 G1 的计数只用无歧义列（另行报告）。
- 俯仰分量不能靠合页（只转方位）补齐：几何（折叠、馈源/探头位置）只用方位分量；俯仰只影响"steer"码本。

仿真几何（make_geometry）：make_relay(ψ)；接收板法向 +x，"偏向馈源一侧"单位向量 h_rx = −ŷ，发射板 h_tx = h_rx 绕 z 转 ψ。
馈源 = 接收板中心 + d_feed·(方位 −α_r 的单位向量)，喇叭（q=10）对准接收板中心；探头 = 接收喇叭（q=10），位于发射板中心
+ d_probe·(方位 ψ − α_t)（即发射波束方向；"探头在发射板法向 1.8 m"，有发射偏转时随波束），对准发射板中心。
50° 块的探头距离表里没写，默认同 1.8 m（--probe-dist）。馈源→探头视距项保留（两喇叭都对准阵列，远低于反射路径）。

金属板参考（plate_nodes）：与一块板等面积（--plate both：两块板，2W×W）的 PEC 板，λ0/4 的 PO 单元作为 env：
Γ = −1，α = alpha_passive(−1)，方向图 Pattern(q=1, d0 = 4πA_cell/λ², 后瓣 −60 dB)（DESIGN §1.1 / §1.3）。
板心 = 收、发板中心的中点，法向 = 指向馈源与探头的单位向量的角平分线（镜面最佳取向，对应实测"等面积金属板最大增益"）。
远场单站校验：该 PO 板的 RCS = 4πA²/λ²（tests/test_exp_f.py）。改善量 = 10log10(P_probe,relay / P_probe,plate)，同一 P_feed。

中继：H = solver.solve_scene（无散射体时即直达项），relay.relay_linear → solve_fixed_point（RappPA）→ probe_fields。
κ 在环路里是 RelayParams.kappa0（端口隔离 × 收发网络损耗，INTEGRATION B2）；同一中继板间自由空间耦合不进 H
（include_direct_tx_rx=False，由 κ0 统一表示），另报"几何隔离" iso_geo_*（用单元方向图算的板间自由空间耦合，供折叠扫描参考）。
码本（全部在 f0 = 3.5 GHz 设计、再在各频点评估——硬件也是一套码本扫频）：
- conj：量化共轭（接收对馈源、发射对探头，各取 16 个全局偏移中合成幅度最大者，INTEGRATION B / G "F"）；不知道通道相位误差。
- steer：对标签方向 (φ,θ) 的远场导向矢量量化共轭（模拟 matlab 预计算"目标偏转角"码本）；只对实测配置。
- oracle：1-bit 模型 oracle（oracle_bits 随机重启坐标下降）最大化探头功率；搜索用线性闭式 z = d + C·G·u_in/(1 − Gκ)
  （深回退下与 Rapp 不动点同最优，见 fast_probe_field），含通道相位误差；结果再与 conj 取优（oracle ≥ conj）。
源功率（工作点）：每个增益、频点下令 conj 码本的开环 PA 输入 = P1dB_in·10^{backoff/10}（默认 −20 dB，深回退 ≈ 线性），
报告 p_feed_dbm 与闭环工作点 pa_in_rel_p1db_db。
自激标记（审查 S0-0，relay.OSC_CRITERIA）：flagged = CW 判据（~converged | ρ_op ≥ 1）或相位稳健环路增益
ρ_rob = |K|(|a|+|b|) ≥ 1（任意泄漏相位 / 带内频率可能起振）——物理判据，improvement_db 在 flagged 时置空；
flagged_ss = CW 或小信号 |Gκ| ≥ 1（源关闭即起振）——保守列，improvement_ss_db 在 flagged_ss 时置空。
深回退下两者几乎一致；∠κ0 = 0° 且 |Gκ| 略大于 1 时强注入可把 PA 压缩成注入锁定的稳定饱和态（只有保守列标记）。
随机种子：每个种子 = 一组通道相位误差（N(0, σ²)，σ = --phase-err-deg，默认 10°）+ oracle 的随机重启。

κ0 相位系统误差（审查 P-1）：空场景下 K = κ_eff 与码本无关，环路因子 1/|1 − G_v κ_eff| 只依赖未测的 ∠κ0；
默认 κ0 = (−30 dB, 90°) 是假设值（INTEGRATION B2），G = 30 dB 时环路因子在 −3.9（180°）… +7.2 dB（0°）之间。
所以 config job 另在 f0、conj 码本上扫 κ0 = −30 dB × ∠κ0 ∈ {0, 45, …, 315}° 与开环参考（κ0 → 0，
OPEN_LOOP_ISO_DB），全部增益点都算：门 G1 计数、G_match_00、G_band 逐 ∠κ0 报告（gate_kappa.csv、
gain_fit_kappa.csv，gain_fit.csv 里给开环值与 ∠κ0 扫描的 min–max），summary.md 的门 G1 结论写成"以 ∠κ0 为条件"，
开环预测作为与 κ0 无关的主结果。其他码本的 κ0 依赖与 conj 相同（同一个码本无关的环路因子，只差视距项干涉）。

扫描：PA 小信号增益（15–40 dB，另在 |Gκ| = 1 的 35 dB 下方加 33.5/34.5 dB）；折叠角 0–180°（Δ 固定，转角差由波束补：ψ<Δ 时接收偏 Δ−ψ，ψ>Δ 时发射偏 ψ−Δ）；
隔离度 × κ0 相位（共面与"收00-发00"两种几何）；频点（full：3.45/3.50/3.55 GHz，quick 只 3.50）。
输出：results/exp_f/<时间戳>/ 下 runs.jsonl、summary.csv（预测 vs 实测主表）、summary.md（全部表格）、gain_sweep.csv、
gain_fit.csv、gain_fit_kappa.csv、gate_kappa.csv、fold_sweep.csv、isolation_sweep.csv、measured.csv、config.json、
fig_*.png（common 导入时已设中文字体回退，图中直接用中文列名，审查 S0-1）。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

from ..constants import DEFAULTS, F0, w2dbm, wavelength, wavenumber
from ..geometry import board_elements, make_relay
from ..materials import alpha_passive
from ..oracle_bits import oracle_bits
from ..pa import RappPA
from ..patterns import element_pattern, horn_pattern, po_cell_pattern
from ..relay import RelayParams, probe_fields, relay_linear, solve_fixed_point, weights
from ..scene import NodeSet, Scene, assemble_src_to_rcv
from ..solver import solve_scene
from ..system import quantized_conjugate
from . import common

EXP_NAME = "exp_f"
N_ELEM = 25
DEFAULT_XLSX = os.path.join(os.path.dirname(os.path.dirname(common.RESULTS_ROOT)), "测试数据.xlsx")
GATE_TOL_DB = 3.0                       # 方案 §8 门 G1：预测落在实测区间 ±3 dB 内
DEFAULT_GAIN_DB = float(DEFAULTS["pa_gain_db"])
GAINS_FULL = tuple(float(g) for g in np.arange(15.0, 40.01, 2.5))
FREQS_FULL = (3.45e9, 3.50e9, 3.55e9)
ISO_SWEEP_DB = (-50.0, -40.0, -35.0, -32.5, -30.0, -27.5, -25.0, -20.0, -15.0, -10.0)
KAPPA_PHASES_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
PLOT_PHASES_DEG = (0.0, 45.0, 90.0, 135.0, 180.0)   # |1 − x·e^{jφ}| 关于 φ ↔ −φ 对称，只画一半
EZ = np.array([0.0, 0.0, 1.0])
# κ0 相位系统误差（审查 P-1）：默认 κ0 是假设值，∠κ0 从未测过
DEFAULT_KAPPA = (float(DEFAULTS["isolation_db"]), float(DEFAULTS["kappa0_phase_deg"]))
KAPPA_SYS_ISO_DB = float(DEFAULTS["isolation_db"])  # 扫 ∠κ0 时的 |κ0|（端口隔离）
KAPPA_SYS_PHASES_DEG = KAPPA_PHASES_DEG              # 45° 步长全扫（AM-PM 使 φ ↔ −φ 不严格对称）
OPEN_LOOP_ISO_DB = -200.0                           # "开环"参考 κ0 → 0：|G_v κ_eff| ≤ −165 dB（有限值，JSON/CSV 友好）
# 默认 κ0 下小信号 |Gκ_eff| = 1 的 PA 增益（30 dB 隔离 + 5 dB 网络损耗 ⇒ 35 dB）；config job 在其下方加密两点，
# 使 ∠κ0 ≈ 135°–225°（负反馈，需要更高 G 才能对上实测）的 G_match_00 能在稳定域内解出（审查 P-1）
G_LOOP_LIMIT_DB = float(-DEFAULTS["isolation_db"] + DEFAULTS["loss_rx_db"] + DEFAULTS["loss_tx_db"])
GAINS_NEAR_LIMIT = (G_LOOP_LIMIT_DB - 1.5, G_LOOP_LIMIT_DB - 0.5)


# ============================================================ 测量数据
@dataclass
class MeasBlock:
    """一块实测数据（一个馈源位置）。impr/abs_gain 形状 (F, C)，列顺序同 labels。"""

    name: str
    feed_az_deg: float
    feed_dist_m: float
    freqs_ghz: np.ndarray
    labels: list
    coplanar_note: dict
    impr: np.ndarray
    abs_gain: np.ndarray | None = None
    plate_gain: np.ndarray | None = None
    raw_headers: dict = field(default_factory=dict)

    def stats(self, label: str) -> dict:
        y = self.impr[:, self.labels.index(label)]
        i0 = int(np.argmin(np.abs(self.freqs_ghz - F0 / 1e9)))
        return {"meas_mean": float(np.mean(y)), "meas_min": float(np.min(y)), "meas_max": float(np.max(y)),
                "meas_std": float(np.std(y, ddof=1)), "meas_f0": float(y[i0]), "meas_n_freq": int(y.size)}

    def consistency_db(self) -> float | None:
        """max |提升表 − (绝对增益 − 金属板增益)|（数据自洽性）。"""
        if self.abs_gain is None or self.plate_gain is None or self.abs_gain.shape != self.impr.shape:
            return None
        return float(np.max(np.abs(self.impr - (self.abs_gain - self.plate_gain[:, None]))))


_BLOCK_RE = re.compile(r"方位\s*(\d+(?:\.\d+)?)\s*°.*?距离阵列\s*(\d+(?:\.\d+)?)\s*m")


def _label_key(header: str) -> str:
    return str(header).split("\n")[0].strip()


def _read_table(grid: list, i: int, j: int) -> dict:
    """从表头单元 (i, j)（"频点/GHz"）向右读列、向下读数值行（遇到非数值即止）。"""
    hdr = grid[i]
    cols = []
    for c in range(j + 1, len(hdr)):
        h = hdr[c]
        if not isinstance(h, str) or not h.strip() or h.strip().startswith("说明"):
            continue
        cols.append((c, h))
    freqs, rows = [], []
    for r in range(i + 1, len(grid)):
        f = grid[r][j] if j < len(grid[r]) else None
        if not isinstance(f, (int, float)):
            break
        freqs.append(float(f))
        rows.append([float(grid[r][c]) if isinstance(grid[r][c], (int, float)) else np.nan for c, _ in cols])
    return {"freqs": np.array(freqs), "headers": [h for _, h in cols], "values": np.array(rows, dtype=float)}


def load_measurements(path: str = DEFAULT_XLSX) -> list[MeasBlock]:
    """读取 测试数据.xlsx（openpyxl，只读）。返回按出现顺序的 MeasBlock 列表。"""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        ws = wb.worksheets[0]
        grid = [list(r) for r in ws.iter_rows(values_only=True)]
    finally:
        wb.close()
    width = max(len(r) for r in grid)
    grid = [r + [None] * (width - len(r)) for r in grid]
    blocks, cur, pending_impr = [], None, False
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if not isinstance(v, str):
                continue
            m = _BLOCK_RE.search(v)
            if m:
                cur = {"az": float(m.group(1)), "dist": float(m.group(2)), "abs": None, "impr": None}
                blocks.append(cur)
            elif "相较金属板" in v:
                pending_impr = True
            elif v.strip().startswith("频点") and cur is not None:
                tab = _read_table(grid, i, j)
                if pending_impr:
                    cur["impr"], pending_impr = tab, False
                else:
                    cur["abs"] = tab
    out = []
    for b in blocks:
        if b["impr"] is None:
            continue
        ti = b["impr"]
        labels = [_label_key(h) for h in ti["headers"]]
        note = {_label_key(h): ("无夹角" in h) for h in ti["headers"]}
        abs_g = plate = None
        if b["abs"] is not None:
            ta = b["abs"]
            akeys = [_label_key(h) for h in ta["headers"]]
            pc = [k for k, h in enumerate(ta["headers"]) if "金属反射板" in h]
            if pc:
                plate = ta["values"][:, pc[0]]
            if all(lb in akeys for lb in labels) and len(ta["freqs"]) == len(ti["freqs"]):
                abs_g = np.stack([ta["values"][:, akeys.index(lb)] for lb in labels], axis=1)
        out.append(MeasBlock(name=f"az{b['az']:g}", feed_az_deg=b["az"], feed_dist_m=b["dist"],
                             freqs_ghz=ti["freqs"], labels=labels, coplanar_note=note, impr=ti["values"],
                             abs_gain=abs_g, plate_gain=plate,
                             raw_headers={_label_key(h): h for h in ti["headers"]}))
    if not out:
        raise ValueError(f"{path} 中没有找到'增益相较金属板增益提升'表")
    return out


# ============================================================ 列名 → 几何
def _parse_pointing(tok: str) -> tuple[float, float]:
    t = tok.replace("-", " ").split()
    if len(t) == 1 and t[0] in ("00", "0"):
        return (0.0, 0.0)
    if len(t) != 2:
        raise ValueError(f"无法解析波束指向 {tok!r}")
    return (float(t[0]), float(t[1]))


def parse_label(label: str) -> dict:
    """'收00-发90 15' / '接收90 15 发90 30' / '接收15 15发00' → {'rx': (a,b), 'tx': (c,d)}（原始两数）。"""
    s = re.sub(r"[（(].*?[)）]", "", _label_key(label))
    if "发" not in s:
        raise ValueError(f"列名缺少'发'：{label!r}")
    left, right = s.split("发", 1)
    left = left.replace("接收", " ").replace("收", " ")
    return {"rx": _parse_pointing(left), "tx": _parse_pointing(right)}


def is_ambiguous(p: tuple[float, float]) -> bool:
    """(a,b) 不在 (0,0) 或 φ = 90° 方位面上 → 读法有歧义（见模块 docstring）。"""
    a, b = p
    return not ((a == 0.0 and b == 0.0) or a == 90.0)


def ascii_label(label: str) -> str:
    """ASCII 简写：'接收90 15 发90 30' → 'rx90,15/tx90,30'（文件名 / 日志用；图里已改用中文列名，审查 S0-1）。"""
    p = parse_label(label)

    def f(t):
        return "00" if t == (0.0, 0.0) else f"{t[0]:g},{t[1]:g}"
    return f"rx{f(p['rx'])}/tx{f(p['tx'])}"


def pointing_angles(p: tuple[float, float], reading: str = "phi_theta") -> tuple[float, float]:
    """(a,b) → (α 方位分量, ε 俯仰分量)（度）。phi_theta：(φ,θ) HFSS 约定，φ=90° 为方位面；az_el：(方位, 俯仰)。"""
    a, b = p
    if reading == "az_el":
        return float(a), float(b)
    if reading != "phi_theta":
        raise ValueError(f"未知读法 {reading!r}")
    phi, th = np.deg2rad(a), np.deg2rad(b)
    alpha = np.degrees(np.arctan2(np.sin(th) * np.sin(phi), np.cos(th)))
    eps = np.degrees(np.arcsin(np.sin(th) * np.cos(phi)))
    return round(float(alpha), 9), round(float(eps), 9)          # 去掉 30.000000000000004 之类的舍入噪声


@dataclass(frozen=True)
class FConfig:
    """一个测量配置的仿真几何参数（角度：度，距离：米）。"""

    block: str
    label: str
    reading: str
    delta_deg: float
    feed_dist: float
    probe_dist: float
    alpha_rx: float
    eps_rx: float
    alpha_tx: float
    eps_tx: float
    coplanar_note: bool = False
    ambiguous: bool = False

    @property
    def fold_deg(self) -> float:
        """ψ = Δ − α_r + α_t（模块 docstring 的"结构补齐"）。"""
        return float(self.delta_deg - self.alpha_rx + self.alpha_tx)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["fold_deg"] = self.fold_deg
        return d


def configs_for_block(blk: MeasBlock, probe_dist: float = 1.8) -> list[FConfig]:
    """一块实测数据的全部配置（歧义列各出两种读法）。"""
    out = []
    for lb in blk.labels:
        p = parse_label(lb)
        amb = is_ambiguous(p["rx"]) or is_ambiguous(p["tx"])
        for reading in (("phi_theta", "az_el") if amb else ("phi_theta",)):
            ar, er = pointing_angles(p["rx"], reading)
            at, et = pointing_angles(p["tx"], reading)
            fc = FConfig(blk.name, lb, reading, blk.feed_az_deg, blk.feed_dist_m, probe_dist, ar, er, at, et,
                         bool(blk.coplanar_note.get(lb, False)), amb)
            if not (0.0 <= fc.fold_deg <= 180.0):
                raise ValueError(f"{blk.name}/{lb}：折叠角 {fc.fold_deg} 超出 [0,180]")
            out.append(fc)
    return out


def fold_sweep_config(delta_deg: float, fold_deg: float, feed_dist: float, probe_dist: float,
                      block: str = "") -> FConfig:
    """折叠扫描：Δ 固定，转角差由波束补（ψ<Δ 接收偏 Δ−ψ；ψ>Δ 发射偏 ψ−Δ）。"""
    ar = max(delta_deg - fold_deg, 0.0)
    at = max(fold_deg - delta_deg, 0.0)
    return FConfig(block, f"fold{fold_deg:g}", "split", delta_deg, feed_dist, probe_dist, ar, 0.0, at, 0.0)


# ============================================================ 场景
def _azv(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r), 0.0])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def make_geometry(fc: FConfig) -> dict:
    """中继 + 馈源 + 探头几何（模块 docstring）。"""
    psi = fc.fold_deg
    relay = make_relay(psi)
    feed = relay.rx_center + fc.feed_dist * _azv(-fc.alpha_rx)
    probe = relay.tx_center + fc.probe_dist * _azv(psi - fc.alpha_tx)
    h_rx = np.array([0.0, -1.0, 0.0])                                  # 接收板"偏向馈源一侧"
    p = np.deg2rad(psi)
    h_tx = np.array([np.sin(p), -np.cos(p), 0.0])                      # h_rx 绕 z 转 ψ（同向安装）
    return {"relay": relay, "feed": feed, "probe": probe, "h_rx": h_rx, "h_tx": h_tx, "fold_deg": psi}


def _horn_nodes(geo: dict) -> tuple[NodeSet, NodeSet]:
    r, horn = geo["relay"], horn_pattern()
    return (NodeSet(geo["feed"], r.rx_center - geo["feed"], horn),
            NodeSet(geo["probe"], r.tx_center - geo["probe"], horn))


def relay_scene(geo: dict, freq_hz: float, include_direct_tx_rx: bool = False) -> Scene:
    r, ep = geo["relay"], element_pattern()
    feed, probe = _horn_nodes(geo)
    return Scene(freq_hz, feed, tx=[NodeSet(r.tx_pos, r.tx_normal, ep)], rx=[NodeSet(r.rx_pos, r.rx_normal, ep)],
                 probes=probe, include_direct_tx_rx=include_direct_tx_rx)


def plate_nodes(geo: dict, freq_hz: float, size: str = "one", cells_per_side: int = 10) -> NodeSet:
    """镜面取向的 PEC 金属板 PO 单元（λ0/4，Γ=−1；DESIGN §1.1/§1.3）。size：'one'（W×W）| 'both'（2W×W）。"""
    if size not in ("one", "both"):
        raise ValueError("plate size 只能是 'one' 或 'both'")
    r = geo["relay"]
    pc = 0.5 * (r.rx_center + r.tx_center)
    n = _unit(_unit(geo["feed"] - pc) + _unit(geo["probe"] - pc))
    u = _unit(np.cross(EZ, n))
    cell = r.width / cells_per_side
    nu, nv = cells_per_side * (2 if size == "both" else 1), cells_per_side
    corner = pc - 0.5 * nu * cell * u - 0.5 * nv * cell * EZ
    pos = board_elements(corner, u, EZ, nu, nv, cell)
    lam, k = wavelength(freq_hz), wavenumber(freq_hz)
    return NodeSet(pos, n, po_cell_pattern(cell * cell, lam), alpha=alpha_passive(-1.0, k, lam))


def plate_response(geo: dict, freq_hz: float, size: str = "one", block_direct: bool = False) -> complex:
    """金属板场景的 馈源→探头 传输 H（√W/√W），含/不含视距项。"""
    feed, probe = _horn_nodes(geo)
    sc = Scene(freq_hz, feed, probes=probe, env=[plate_nodes(geo, freq_hz, size)],
               block_direct_feed_probe=block_direct)
    return complex(solve_scene(sc, None, None, "full")[0, 0])


# ============================================================ 码本
def _best_offset(v: np.ndarray) -> np.ndarray:
    """量化共轭 16 个全局偏移中 |Σ e^{jπb} v| 最大者（INTEGRATION B）。"""
    cand = quantized_conjugate(np.asarray(v)[None], 2)[:, 0]           # (16, N)
    s = np.abs(np.sum(np.exp(1j * np.pi * cand) * np.asarray(v)[None], axis=-1))
    return cand[int(np.argmax(s))].astype(np.int64)


def conj_bits(H: np.ndarray) -> np.ndarray:
    """conj 码本 (50,)：接收对馈源列 H[rx,0]、发射对探头行 H[probe, tx] 的量化共轭。"""
    return np.concatenate([_best_offset(H[:N_ELEM, 0]), _best_offset(H[N_ELEM, 1:1 + N_ELEM])])


def steer_bits(geo: dict, fc: FConfig, freq_hz: float = F0) -> np.ndarray:
    """steer 码本：对标签方向（方位 α、俯仰 ε，板坐标）的远场导向矢量 e^{jk û·(r−r_c)} 量化共轭。"""
    k = wavenumber(freq_hz)
    r = geo["relay"]
    out = []
    for pos, c, n, h, a, e in ((r.rx_pos, r.rx_center, r.rx_normal, geo["h_rx"], fc.alpha_rx, fc.eps_rx),
                               (r.tx_pos, r.tx_center, r.tx_normal, geo["h_tx"], fc.alpha_tx, fc.eps_tx)):
        a, e = np.deg2rad(a), np.deg2rad(e)
        u = np.cos(e) * (np.cos(a) * n + np.sin(a) * h) + np.sin(e) * EZ
        out.append(_best_offset(np.exp(1j * k * ((pos - c) @ u))))
    return np.concatenate(out)


def fast_probe_field(H: np.ndarray, bits: np.ndarray, gain_v: complex, params: RelayParams, s_feed: float) -> complex:
    """线性闭式探头场 z = s·d + C·G·u_in/(1 − G·κ)（R = 1，空场景 K = κ_eff，DESIGN §3.3–3.4）。

    只用于 oracle 搜索（深回退下 Rapp 与线性同最优码本）；与 relay 管线的一致性见 tests/test_exp_f.py。
    """
    w, ct = weights(np.pi * bits[None, :N_ELEM], np.pi * bits[None, N_ELEM:], params)
    u_in = s_feed * np.dot(w[0], H[:N_ELEM, 0])
    C = np.dot(ct[0], H[N_ELEM, 1:1 + N_ELEM])
    G = gain_v
    return s_feed * H[N_ELEM, 0] + C * G * u_in / (1.0 - G * params.kappa0)


def oracle_codebook(H: np.ndarray, params: RelayParams, gain_db: float, rng: np.random.Generator,
                    n_restarts: int = 4, max_sweeps: int = 30, start: np.ndarray | None = None) -> np.ndarray:
    """1-bit oracle：最大化探头功率（oracle_bits，线性闭式目标）；与 start（通常 conj）取优。"""
    gv = 10.0 ** (gain_db / 20.0)

    def loss(x):
        return -float(w2dbm(abs(fast_probe_field(H, x, gv, params, 1.0)) ** 2))

    best, fbest = oracle_bits(loss, np.full(2 * N_ELEM, 2), rng, n_restarts=n_restarts, max_sweeps=max_sweeps)
    if start is not None and loss(start) < fbest:
        best = np.asarray(start, dtype=np.int64).copy()
    return np.asarray(best, dtype=np.int64)


# ============================================================ 评估
def relay_params(phase_err_rx=None, phase_err_tx=None, isolation_db: float | None = None,
                 kappa0_phase_deg: float | None = None) -> RelayParams:
    kw = {}
    if isolation_db is not None:
        kw["isolation_db"] = float(isolation_db)
    if kappa0_phase_deg is not None:
        kw["kappa0_phase_deg"] = float(kappa0_phase_deg)
    return RelayParams(n_relays=1, n_elem=N_ELEM, phase_err_rx=phase_err_rx, phase_err_tx=phase_err_tx, **kw)


def open_loop_u_per_w(H: np.ndarray, bits: np.ndarray, params: RelayParams) -> float:
    """P_feed = 1 W 时码本的开环 PA 输入功率 |u_in|²（W）。"""
    w, _ = weights(np.pi * bits[None, :N_ELEM], np.pi * bits[None, N_ELEM:], params)
    return float(abs(np.dot(w[0], H[:N_ELEM, 0])) ** 2)


def evaluate_relay(H: np.ndarray, bits: np.ndarray, pa: RappPA, params: RelayParams, s_feed: float) -> dict:
    """完整中继管线（relay_linear → solve_fixed_point → probe_fields），返回探头功率、工作点与三种自激判据（审查 S0-0）。

    loop_factor_db = −20log10|1 − G_v K|：小信号环路因子（R = 1；空场景 K = κ_eff 与码本无关，审查 P-1）。
    """
    w, ct = weights(np.pi * bits[None, :N_ELEM], np.pi * bits[None, N_ELEM:], params)
    lin = relay_linear(H, w, ct, s_feed, params)
    fp = solve_fixed_point(lin.u_in, lin.K, pa)
    z = probe_fields(lin.d, lin.C, fp.g)
    p1 = pa.p1db_in_w()
    return {"p_probe_w": float(abs(z[0]) ** 2),
            "pa_in_rel_p1db_db": float(10.0 * np.log10(max(abs(fp.u[0]) ** 2, 1e-300) / p1)),
            "pa_in_open_rel_p1db_db": float(10.0 * np.log10(max(abs(lin.u_in[0]) ** 2, 1e-300) / p1)),
            "loop_gain_ss": float(fp.loop_gain_ss), "loop_gain_op": float(fp.loop_gain_op),
            "loop_gain_robust": float(fp.loop_gain_robust),
            "loop_factor_db": float(-20.0 * np.log10(max(abs(1.0 - pa.gain_v * lin.K[0, 0]), 1e-300))),
            "converged": bool(fp.converged), "possibly_oscillating": bool(fp.possibly_oscillating),
            # 相位稳健环路增益 ρ_rob ≥ 1：某个泄漏相位 / 带内频率上可能起振（relay.loop_gain_robust，同一容差）
            "robust_unstable": bool(fp.robust_unstable),
            # 小信号环路 |Gκ| ≥ 1：源关闭即起振（最保守；relay.gain_at_least_one 的容差）
            "ss_unstable": bool(fp.ss_unstable)}


def geometric_isolation_db(geo: dict, bits: np.ndarray, params: RelayParams, freq_hz: float) -> dict:
    """板间自由空间耦合（单元方向图，include_direct_tx_rx=True）折算到端口参考面的"几何隔离"（dB）：
    conj 码本下 |Σ w H_rx,tx ct|²·10^{(L_rx+L_tx)/10}，以及随机码本平均 Σ|H|²/N²（码本无关）。"""
    H = assemble_src_to_rcv(relay_scene(geo, freq_hz, include_direct_tx_rx=True))
    Hrt = H[:N_ELEM, 1:1 + N_ELEM]
    w, ct = weights(np.pi * bits[None, :N_ELEM], np.pi * bits[None, N_ELEM:], params)
    k_geo = complex(w[0] @ Hrt @ ct[0])
    net = 10.0 ** ((params.loss_rx_db + params.loss_tx_db) / 10.0)
    rms = float(np.sum(np.abs(Hrt) ** 2) / N_ELEM**2)
    return {"iso_geo_codebook_db": float(10.0 * np.log10(max(abs(k_geo) ** 2 * net, 1e-300))),
            "iso_geo_rms_db": float(10.0 * np.log10(max(rms, 1e-300)))}


def _phase_errors(rng: np.random.Generator, sigma_deg: float):
    if sigma_deg <= 0:
        return None, None
    s = np.deg2rad(sigma_deg)
    return s * rng.standard_normal((1, N_ELEM)), s * rng.standard_normal((1, N_ELEM))


def _eval_point(H: np.ndarray, Hp: complex, bits: np.ndarray, cal_bits: np.ndarray, gain_db: float,
                params: RelayParams, backoff_db: float, psat_out_dbm: float) -> dict:
    """一个 (几何, 频点, 增益, 码本) 点：按 cal_bits（conj）的开环 PA 输入标定 P_feed，算改善量与工作点。"""
    pa = RappPA(gain_db=gain_db, psat_out_dbm=psat_out_dbm)
    p_feed = pa.p1db_in_w() * 10.0 ** (backoff_db / 10.0) / open_loop_u_per_w(H, cal_bits, params)
    ev = evaluate_relay(H, bits, pa, params, float(np.sqrt(p_feed)))
    p_plate = p_feed * abs(Hp) ** 2
    # 审查 S0-0：物理判据 = CW 工作点或相位稳健环路增益 ≥ 1；保守列 = CW 或小信号 |Gκ| ≥ 1
    flagged = ev["possibly_oscillating"] or ev["robust_unstable"]
    flagged_ss = ev["possibly_oscillating"] or ev["ss_unstable"]
    raw = float(10.0 * np.log10(ev["p_probe_w"] / p_plate))
    ev.update(p_feed_dbm=float(w2dbm(p_feed)), relay_db=float(10.0 * np.log10(ev["p_probe_w"] / p_feed)),
              improvement_db=None if flagged else raw, improvement_ss_db=None if flagged_ss else raw,
              improvement_raw_db=raw, flagged=bool(flagged), flagged_ss=bool(flagged_ss))
    return ev


# ============================================================ job（模块顶层，可 pickle）
def kappa_case(isolation_db: float, kappa0_phase_deg: float) -> str:
    """κ0 情形标签：开环参考 → "open"，否则 "−30dB/90°"。"""
    if isolation_db <= OPEN_LOOP_ISO_DB + 1e-9:
        return "open"
    return f"{isolation_db:g}dB/{kappa0_phase_deg:g}°"


def kappa_sys_grid(iso_db: float = KAPPA_SYS_ISO_DB, phases=KAPPA_SYS_PHASES_DEG,
                   default: tuple = DEFAULT_KAPPA) -> list[tuple[float, float]]:
    """config job 的 κ0 系统误差网格（审查 P-1）：开环参考 + |κ0| = iso_db × ∠κ0 扫描（去掉默认点，它已在默认行里）。"""
    grid = [(OPEN_LOOP_ISO_DB, 0.0)]
    grid += [(float(iso_db), float(p)) for p in phases
             if not (abs(iso_db - default[0]) < 1e-12 and abs(p - default[1]) < 1e-12)]
    return grid


def run_job(job: dict) -> dict:
    """一个 job = 一个几何 × 一个种子；返回 {"rows": [...]}（每行一个 频点 × 增益 × 码本 × (隔离, 相位) 点）。

    行的 kappa_set："default"（默认 κ0，全部码本、频点）| "iso"（隔离 × 相位扫描 job）|
    "sys"（审查 P-1：job["kappa_sys"] 的 κ0 网格，只在 f0、conj 码本上算，全部增益）。"""
    t0 = time.perf_counter()
    fc = FConfig(**job["config"])
    rng = common.seed_rng("exp_f", job["kind"], fc.block, fc.label, fc.reading, int(job["seed"]))
    err_rx, err_tx = _phase_errors(rng, float(job.get("phase_err_deg", 0.0)))
    base = relay_params(err_rx, err_tx)
    geo = make_geometry(fc)
    plate = job.get("plate", "one")
    backoff = float(job.get("backoff_db", -20.0))
    psat = float(job.get("psat_out_dbm", DEFAULTS["psat_out_dbm"]))
    g_def = float(job.get("gain_default_db", DEFAULT_GAIN_DB))
    # ---- 码本：f0 设计（不知道通道相位误差），oracle 在含误差的模型上搜索
    H0 = solve_scene(relay_scene(geo, F0), None, None, "full")
    books = {"conj": conj_bits(H0)}
    if job.get("steer", False):
        books["steer"] = steer_bits(geo, fc, F0)
    if job.get("oracle", True):
        books["oracle"] = oracle_codebook(H0, base, g_def, rng, n_restarts=int(job.get("oracle_restarts", 4)),
                                          start=books["conj"])
    iso = geometric_isolation_db(geo, books["conj"], base, F0)
    common_fields = {"kind": job["kind"], "seed": int(job["seed"]), **fc.as_dict(), **iso,
                     "plate_size": plate, "backoff_db": backoff, "phase_err_deg": float(job.get("phase_err_deg", 0.0))}
    rows = []

    def add_row(fields: dict, g: float, name: str, prm: RelayParams, ev: dict, kset: str) -> None:
        rows.append({**common_fields, **fields, "gain_db": float(g), "codebook": name, "kappa_set": kset,
                     "isolation_db": float(prm.isolation_db), "kappa0_phase_deg": float(prm.kappa0_phase_deg),
                     "kappa_case": kappa_case(prm.isolation_db, prm.kappa0_phase_deg),
                     "net_gain_db": float(g - prm.loss_rx_db - prm.loss_tx_db),
                     "loop_gain_ss_db": float(20 * np.log10(max(ev["loop_gain_ss"], 1e-300))),
                     **{k: v for k, v in ev.items() if k != "p_probe_w"}})

    for f in job["freqs"]:
        H = H0 if f == F0 else solve_scene(relay_scene(geo, f), None, None, "full")
        Hp = plate_response(geo, f, plate)
        Hp_scat = plate_response(geo, f, plate, block_direct=True)
        geo_fields = {"freq_ghz": f / 1e9, "plate_db": float(10 * np.log10(abs(Hp) ** 2)),
                      "plate_scat_db": float(10 * np.log10(abs(Hp_scat) ** 2)),
                      "los_db": float(10 * np.log10(max(abs(H[N_ELEM, 0]) ** 2, 1e-300)))}
        for (iso_db, kph) in job.get("kappa_grid", [(None, None)]):
            prm = base if iso_db is None else relay_params(err_rx, err_tx, iso_db, kph)
            for g in job["gains"]:
                for name, bits in books.items():
                    add_row(geo_fields, g, name, prm, _eval_point(H, Hp, bits, books["conj"], g, prm, backoff, psat),
                            "default" if iso_db is None else "iso")
        if f == F0:
            # 审查 P-1：未测的 ∠κ0（及开环参考）系统误差，conj 码本、全部增益
            for (iso_db, kph) in job.get("kappa_sys", []):
                prm = relay_params(err_rx, err_tx, float(iso_db), float(kph))
                for g in job["gains"]:
                    add_row(geo_fields, g, "conj", prm,
                            _eval_point(H, Hp, books["conj"], books["conj"], g, prm, backoff, psat), "sys")
    return {"rows": rows, "n_rows": len(rows), "codebooks": {k: v.tolist() for k, v in books.items()},
            "time_s": time.perf_counter() - t0}


# ============================================================ 计划
def plan_jobs(blocks: list[MeasBlock], args) -> list[dict]:
    quick = bool(args.quick)
    gains = (15.0, 20.0, 25.0, 27.5, 30.0, 32.5, 35.0, 40.0) if quick else GAINS_FULL
    freqs = (F0,) if quick else FREQS_FULL
    common_kw = {"phase_err_deg": args.phase_err_deg, "plate": args.plate, "backoff_db": args.backoff_db,
                 "gain_default_db": args.gain_db, "oracle_restarts": 2 if quick else 6}
    jobs = []
    ksys = kappa_sys_grid()
    for blk in blocks:
        for fc in configs_for_block(blk, args.probe_dist):
            for s in range(args.seeds):
                jobs.append({"kind": "config", "config": fc.__dict__, "seed": s, "freqs": list(freqs),
                             "gains": sorted(set(gains) | {args.gain_db} | set(GAINS_NEAR_LIMIT)), "steer": True,
                             "kappa_sys": ksys, **common_kw})
    folds = np.arange(0.0, 180.01, 20.0 if quick else 10.0)
    for blk in blocks:
        for psi in sorted(set(folds.tolist()) | {float(blk.feed_az_deg)}):
            fc = fold_sweep_config(blk.feed_az_deg, float(psi), blk.feed_dist_m, args.probe_dist, blk.name)
            for s in range(args.seeds):
                jobs.append({"kind": "fold", "config": fc.__dict__, "seed": s, "freqs": [F0],
                             "gains": [args.gain_db], **common_kw})
    iso_grid = [(i, p) for i in ISO_SWEEP_DB for p in KAPPA_PHASES_DEG]
    for blk in blocks:
        cfgs = {c.label: c for c in configs_for_block(blk, args.probe_dist) if c.reading == "phi_theta"}
        pick = [lb for lb in blk.labels if lb.startswith("收00-发00")] + [lb for lb in blk.labels
                                                                         if blk.coplanar_note.get(lb)]
        for lb in pick:
            for s in range(args.seeds):
                jobs.append({"kind": "iso", "config": cfgs[lb].__dict__, "seed": s, "freqs": [F0],
                             "gains": [args.gain_db], "kappa_grid": iso_grid, **common_kw})
    return jobs


# ============================================================ 汇总
def _flatten(results: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, errors = [], []
    for r in results:
        if "error" in r:
            errors.append(r)
        else:
            rows.extend(r["rows"])
    return rows, errors


def _mean_of(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(v)) if v else None


def _interp_root(x: np.ndarray, y: np.ndarray, target: float) -> float | None:
    """y(x) 单调段上 y = target 的线性插值根（取第一个跨越点）。"""
    for i in range(len(x) - 1):
        a, b = y[i] - target, y[i + 1] - target
        if np.isfinite(a) and np.isfinite(b) and a * b <= 0 and a != b:
            return float(x[i] + (x[i + 1] - x[i]) * (-a) / (b - a))
    return None


def _kset(r: dict) -> str:
    return r.get("kappa_set", "default")


def _in_gate(pred, ms: dict) -> bool | None:
    """门 G1：预测落在实测区间 ±GATE_TOL_DB 内（pred 为 None/非有限 → None）。"""
    if pred is None or not np.isfinite(pred):
        return None
    return bool(ms["meas_min"] - GATE_TOL_DB <= pred <= ms["meas_max"] + GATE_TOL_DB)


def _phase_arcs(pass_phases, grid) -> str:
    """∠κ0 网格（圆周）上 PASS 相位的连续弧段，例如 "[90°, 270°]"；全部 → "全部"，无 → "无"。"""
    grid = sorted(float(p) for p in grid)
    ps = {round(float(p), 6) for p in pass_phases}
    ok = [round(p, 6) in ps for p in grid]
    if all(ok):
        return "全部"
    if not any(ok):
        return "无"
    n, arcs = len(grid), []
    for i in range(n):
        if ok[i] and not ok[i - 1]:                     # 弧段起点（i−1 在 i = 0 时回绕）
            j = i
            while ok[(j + 1) % n]:
                j = (j + 1) % n
            arcs.append(f"{grid[i]:g}°" if i == j else f"[{grid[i]:g}°, {grid[j]:g}°]")
    return " ∪ ".join(arcs)


def _curves(tab: list[dict], blk: MeasBlock, f0: float, **match) -> tuple[dict, dict, list]:
    """增益曲线 {label: (G, improvement)}（phi_theta 读法、f0）、实测统计与拟合用标签（非共面、无歧义）。"""
    curves, meas, labels_nc = {}, {}, []
    for lb in blk.labels:
        pts = [t for t in tab if t["block"] == blk.name and t["label"] == lb and t["reading"] == "phi_theta"
               and abs(t["freq_ghz"] - f0) < 1e-9 and all(t.get(k) == v for k, v in match.items())]
        if not pts:
            continue
        pts.sort(key=lambda t: t["gain_db"])
        y = np.array([np.nan if t["improvement_db_mean"] is None else t["improvement_db_mean"] for t in pts])
        curves[lb] = (np.array([t["gain_db"] for t in pts]), y)
        meas[lb] = blk.stats(lb)
        if not blk.coplanar_note.get(lb) and not is_ambiguous(parse_label(lb)["rx"]) \
                and not is_ambiguous(parse_label(lb)["tx"]):
            labels_nc.append(lb)
    return curves, meas, labels_nc


def _gain_fit(curves: dict, meas: dict, labels_nc: list, gfine: np.ndarray, g0: float) -> dict:
    """PA 增益反推：G_match_00（"收00-发00"预测 = 实测均值）、RMS 最优 G、全部拟合配置落在门限内的 G 区间。"""
    loss = DEFAULTS["loss_rx_db"] + DEFAULTS["loss_tx_db"]
    l00 = [lb for lb in curves if lb.startswith("收00-发00")]
    g00 = (_interp_root(*curves[l00[0]], meas[l00[0]]["meas_mean"]) if l00 else None)
    out = {"n_configs_fit": len(labels_nc), "G_match_00_db": g00,
           "net_G_match_00_db": None if g00 is None else g00 - loss}
    keys = ("G_rms_best_db", "rms_at_best_db", "bias_at_default_db", "rms_at_default_db", "G_band_lo_db",
            "G_band_hi_db", "net_G_band_lo_db", "net_G_band_hi_db")
    if not labels_nc:
        return {**out, **{k: None for k in keys}}
    Y = np.array([np.interp(gfine, *curves[lb], left=np.nan, right=np.nan) for lb in labels_nc])
    M = np.array([meas[lb]["meas_mean"] for lb in labels_nc])
    lo = np.array([meas[lb]["meas_min"] - GATE_TOL_DB for lb in labels_nc])
    hi = np.array([meas[lb]["meas_max"] + GATE_TOL_DB for lb in labels_nc])
    with np.errstate(invalid="ignore"):
        rms = np.sqrt(np.mean((Y - M[:, None]) ** 2, axis=0))
        bias = np.mean(Y - M[:, None], axis=0)
        inband = np.all((Y >= lo[:, None]) & (Y <= hi[:, None]), axis=0)
    ok = np.isfinite(rms)
    ib = gfine[inband & ok]
    i_best = int(np.nanargmin(np.where(ok, rms, np.nan))) if ok.any() else None
    out.update({"G_rms_best_db": None if i_best is None else float(gfine[i_best]),
                "rms_at_best_db": None if i_best is None else float(rms[i_best]),
                "bias_at_default_db": float(np.interp(g0, gfine, bias)) if np.isfinite(bias).any() else None,
                "rms_at_default_db": float(np.interp(g0, gfine, rms)) if np.isfinite(rms).any() else None,
                "G_band_lo_db": float(ib.min()) if ib.size else None,
                "G_band_hi_db": float(ib.max()) if ib.size else None,
                "net_G_band_lo_db": float(ib.min() - loss) if ib.size else None,
                "net_G_band_hi_db": float(ib.max() - loss) if ib.size else None})
    return out


def _first_flagged_gain(tab: list[dict], blk_name: str, f0: float, flag: str = "flagged", **match) -> float | None:
    g = [t["gain_db"] for t in tab if t["block"] == blk_name and abs(t["freq_ghz"] - f0) < 1e-9
         and all(t.get(k) == v for k, v in match.items()) and (t.get(f"{flag}_mean") or 0) > 0]
    return min(g) if g else None


def _gate_kappa(main: list[dict], nonc: list[dict], unamb: list[dict], cases: list, kap_g0: dict) -> dict:
    """审查 P-1：门 G1 计数逐 κ0 情形（开环 + |κ0| × ∠κ0，conj、G = g0、f0），以及"以 ∠κ0 为条件"的结论。

    verdict_code：PASS（扫描的全部 ∠κ0 都通过）| CONDITIONAL（部分通过）| FAIL（都不通过）| DEFAULT_ONLY（没有 sys 行，
    只有默认 ∠κ0）。开环（κ0 → 0）的计数单独给出，与 ∠κ0 无关。"""
    def cnt(sel, case):
        g = [r["kappa_gate"].get(case) for r in sel if case in r["kappa_gate"]]
        return int(sum(bool(x) for x in g)), len(g)

    def mean_of(case, key):
        v = [t[key] for k, t in kap_g0.items() if k[3] == case and t[key] is not None and np.isfinite(t[key])]
        return float(np.mean(v)) if v else None

    tab = []
    for case, iso, ph in cases:
        ok, n = cnt(nonc, case)
        devs = [r["kappa_pred"][case] - r["meas_mean"] for r in nonc if r["kappa_pred"].get(case) is not None]
        # 门限裕量：> 0 在实测区间 ±3 dB 内，< 0 在外（最差配置）
        marg = [min(r["kappa_pred"][case] - (r["meas_min"] - GATE_TOL_DB), (r["meas_max"] + GATE_TOL_DB)
                    - r["kappa_pred"][case]) for r in nonc if r["kappa_pred"].get(case) is not None]
        okc, nc = cnt([r for r in unamb if r["coplanar"]], case)
        oka, na = cnt([r for r in main if r["ambiguous"]], case)
        tab.append({"kappa_case": case, "isolation_db": iso, "kappa0_phase_deg": ph,
                    "is_default": bool(abs(iso - DEFAULT_KAPPA[0]) < 1e-9 and abs(ph - DEFAULT_KAPPA[1]) < 1e-9),
                    "loop_gain_ss_db": mean_of(case, "loop_gain_ss_db_mean"),
                    "loop_factor_db": mean_of(case, "loop_factor_db_mean"),
                    "n_pass_non_coplanar": ok, "n_non_coplanar": n, "pass_non_coplanar": bool(n > 0 and ok == n),
                    "mean_dev_db": float(np.mean(devs)) if devs else None,
                    "max_abs_dev_db": float(np.max(np.abs(devs))) if devs else None,
                    "min_gate_margin_db": float(np.min(marg)) if marg else None,
                    "n_pass_coplanar": okc, "n_coplanar": nc, "n_pass_ambiguous": oka, "n_ambiguous": na})
    ph_tab = [t for t in tab if t["kappa_case"] != "open"]
    op = next((t for t in tab if t["kappa_case"] == "open"), None)
    passed = [t["kappa0_phase_deg"] for t in ph_tab if t["pass_non_coplanar"]]
    failed = [t for t in ph_tab if not t["pass_non_coplanar"]]
    arcs = _phase_arcs(passed, [t["kappa0_phase_deg"] for t in ph_tab])
    iso_s = f"{ph_tab[0]['isolation_db']:g}" if ph_tab else "—"
    if len(ph_tab) <= 1:
        dflt = next((t for t in ph_tab if t["is_default"]), None)
        code = "DEFAULT_ONLY"
        verdict = (f"{'PASS' if dflt and dflt['pass_non_coplanar'] else 'FAIL'}（只算了默认 ∠κ0 = "
                   f"{DEFAULT_KAPPA[1]:g}°，未扫相位，结论依赖这个假设值）")
    elif not failed:
        code, verdict = "PASS", f"PASS（κ0 = {iso_s} dB 时扫描的全部 ∠κ0 都通过）"
    elif not passed:
        code, verdict = "FAIL", f"FAIL（κ0 = {iso_s} dB 时任何 ∠κ0 都不通过）"
    else:
        code = "CONDITIONAL"
        verdict = (f"有条件 PASS：κ0 = {iso_s} dB 时仅 ∠κ0 ∈ {arcs} 通过；"
                   + "、".join(f"{t['kappa0_phase_deg']:g}° {t['n_pass_non_coplanar']}/{t['n_non_coplanar']}"
                              f"（最差裕量 {_s(t['min_gate_margin_db'], '+.2f')} dB）" for t in failed) + " 不通过")
    if op is not None:
        verdict += f"；开环（κ0 → 0）{op['n_pass_non_coplanar']}/{op['n_non_coplanar']}"
    return {"kappa": tab, "kappa_iso_db": ph_tab[0]["isolation_db"] if ph_tab else None,
            "kappa_phases_pass": passed, "kappa_phases_fail": [t["kappa0_phase_deg"] for t in failed],
            "kappa_pass_arcs": arcs, "open_loop": None if op is None else (op["n_pass_non_coplanar"], op["n_non_coplanar"]),
            "pass_open_loop": None if op is None else op["pass_non_coplanar"],
            "pass_all_kappa_phases": bool(ph_tab) and not failed, "verdict_code": code, "verdict": verdict}


def summarize(rows: list[dict], blocks: list[MeasBlock], args) -> dict:
    """主表（默认增益、f0：预测 vs 实测 + 门 G1）、增益扫描、增益反推、κ0 相位系统误差、折叠扫描、隔离扫描。"""
    g0 = float(args.gain_db)
    f0 = F0 / 1e9
    by_block = {b.name: b for b in blocks}
    cfg_rows = [r for r in rows if r["kind"] == "config" and _kset(r) == "default"]
    gain_tab = common.aggregate(cfg_rows, ["block", "label", "reading", "codebook", "gain_db", "freq_ghz"],
                                ["improvement_db", "improvement_raw_db", "improvement_ss_db", "flagged", "flagged_ss",
                                 "pa_in_rel_p1db_db", "p_feed_dbm", "loop_gain_ss_db", "loop_gain_robust", "relay_db",
                                 "plate_db"])
    # 审查 P-1：κ0 相位系统误差（conj、f0；默认点来自 default 行，其余来自 sys 行）
    kap_rows = [r for r in rows if r["kind"] == "config" and r["codebook"] == "conj" and _kset(r) in ("default", "sys")
                and abs(r["freq_ghz"] - f0) < 1e-9]
    for r in kap_rows:
        r.setdefault("kappa_case", kappa_case(r["isolation_db"], r["kappa0_phase_deg"]))
    kappa_tab = common.aggregate(kap_rows, ["block", "label", "reading", "kappa_case", "isolation_db", "kappa0_phase_deg",
                                            "gain_db", "freq_ghz"],
                                 ["improvement_db", "improvement_raw_db", "improvement_ss_db", "flagged", "flagged_ss",
                                  "loop_factor_db", "loop_gain_ss_db", "pa_in_rel_p1db_db"])
    cases = sorted({(t["kappa_case"], t["isolation_db"], t["kappa0_phase_deg"]) for t in kappa_tab},
                   key=lambda c: (c[0] != "open", c[1], c[2]))
    kap_g0 = {(t["block"], t["label"], t["reading"], t["kappa_case"]): t for t in kappa_tab if t["gain_db"] == g0}
    # ---- 主表
    main = []
    keys = list(dict.fromkeys((r["block"], r["label"], r["reading"]) for r in cfg_rows))
    for blk_name, label, reading in keys:
        sel = [r for r in cfg_rows if (r["block"], r["label"], r["reading"]) == (blk_name, label, reading)
               and r["gain_db"] == g0 and abs(r["freq_ghz"] - f0) < 1e-9]
        if not sel:
            continue
        r0 = sel[0]
        ms = by_block[blk_name].stats(label)
        row = {"block": blk_name, "config": label, "reading": reading, "ambiguous": r0["ambiguous"],
               "coplanar": r0["coplanar_note"], "fold_deg": round(r0["fold_deg"], 2),
               "rx_az_deg": round(r0["alpha_rx"], 2), "rx_el_deg": round(r0["eps_rx"], 2),
               "tx_az_deg": round(r0["alpha_tx"], 2), "tx_el_deg": round(r0["eps_tx"], 2), **ms}
        for cb in ("conj", "steer", "oracle"):
            s = [r for r in sel if r["codebook"] == cb]
            if not s:
                continue
            agg = common.aggregate(s, ["codebook"], ["improvement_db", "flagged", "pa_in_rel_p1db_db", "p_feed_dbm"])[0]
            row[f"pred_{cb}"] = agg["improvement_db_mean"]
            row[f"pred_{cb}_ci95"] = agg["improvement_db_ci95"]
            row[f"flagged_{cb}"] = agg["flagged_mean"]
            if cb == "conj":
                row["pa_in_rel_p1db_db"] = agg["pa_in_rel_p1db_db_mean"]
                row["p_feed_dbm"] = agg["p_feed_dbm_mean"]
                row["n_seeds"] = agg["n_runs"]
        pc = row.get("pred_conj")
        row["dev_conj_db"] = None if pc is None else pc - ms["meas_mean"]
        row["gate_conj"] = _in_gate(pc, ms)
        row["gate_oracle"] = _in_gate(row.get("pred_oracle"), ms)
        row["iso_geo_rms_db"] = r0["iso_geo_rms_db"]
        # 审查 P-1：同一配置在各 κ0 情形下的 conj 预测（G = g0、f0）与门判定
        kp = {c[0]: kap_g0[(blk_name, label, reading, c[0])]["improvement_db_mean"]
              for c in cases if (blk_name, label, reading, c[0]) in kap_g0}
        kp = {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in kp.items()}
        row["kappa_pred"] = kp
        row["kappa_gate"] = {k: _in_gate(v, ms) for k, v in kp.items()}
        ph_vals = [v for k, v in kp.items() if k != "open"]
        fin = [v for v in ph_vals if v is not None]
        row["pred_conj_open"] = kp.get("open")
        row["gate_conj_open"] = row["kappa_gate"].get("open")
        row["pred_conj_kappa_min"] = min(fin) if fin else None
        row["pred_conj_kappa_max"] = max(fin) if fin else None
        row["n_kappa_gate_pass"] = (f"{sum(bool(row['kappa_gate'][k]) for k in kp if k != 'open')}/{len(ph_vals)}"
                                    if ph_vals else None)
        main.append(row)
    # ---- 门 G1（实验 F 侧）
    def gate_counts(sel, key="gate_conj"):
        ok = [r[key] for r in sel if r[key] is not None]
        return int(sum(ok)), len(ok)
    unamb = [r for r in main if not r["ambiguous"]]
    nonc = [r for r in unamb if not r["coplanar"]]
    gate = {"tol_db": GATE_TOL_DB, "gain_db": g0,
            "default_kappa": {"isolation_db": DEFAULT_KAPPA[0], "kappa0_phase_deg": DEFAULT_KAPPA[1]},
            "all_unambiguous": gate_counts(unamb),
            "non_coplanar": gate_counts(nonc),
            "coplanar": gate_counts([r for r in unamb if r["coplanar"]]),
            "ambiguous_rows": gate_counts([r for r in main if r["ambiguous"]])}
    gate["pass_non_coplanar"] = gate["non_coplanar"][0] == gate["non_coplanar"][1] > 0
    gate.update(_gate_kappa(main, nonc, unamb, cases, kap_g0))
    # ---- 增益反推（默认 κ0，各码本）
    fit = []
    gains = sorted({r["gain_db"] for r in cfg_rows})
    gfine = np.linspace(min(gains), max(gains), 501)
    for blk in blocks:
        for cb in ("conj", "steer", "oracle"):
            curves, meas, labels_nc = _curves(gain_tab, blk, f0, codebook=cb)
            if not curves:
                continue
            fit.append({"block": blk.name, "codebook": cb,
                        "G_unstable_from_db": _first_flagged_gain(gain_tab, blk.name, f0, codebook=cb),
                        "G_ss_unstable_from_db": _first_flagged_gain(gain_tab, blk.name, f0, "flagged_ss", codebook=cb),
                        **_gain_fit(curves, meas, labels_nc, gfine, g0)})
    # ---- 审查 P-1：增益反推逐 κ0 情形（conj），并把开环值与 ∠κ0 扫描的 min–max 写回 conj 行
    fit_kappa = []
    for blk in blocks:
        for case, iso, ph in cases:
            curves, meas, labels_nc = _curves(kappa_tab, blk, f0, kappa_case=case)
            if not curves:
                continue
            lf = [t["loop_factor_db_mean"] for k, t in kap_g0.items() if k[0] == blk.name and k[3] == case
                  and t["loop_factor_db_mean"] is not None and np.isfinite(t["loop_factor_db_mean"])]
            nb = [r for r in nonc if r["block"] == blk.name and case in r["kappa_gate"]]
            fit_kappa.append({"block": blk.name, "kappa_case": case, "isolation_db": iso, "kappa0_phase_deg": ph,
                              "loop_factor_at_default_db": float(np.mean(lf)) if lf else None,
                              "gate_non_coplanar": f"{sum(bool(r['kappa_gate'][case]) for r in nb)}/{len(nb)}",
                              "G_unstable_from_db": _first_flagged_gain(kappa_tab, blk.name, f0, kappa_case=case),
                              "G_ss_unstable_from_db": _first_flagged_gain(kappa_tab, blk.name, f0, "flagged_ss",
                                                                           kappa_case=case),
                              **_gain_fit(curves, meas, labels_nc, gfine, g0)})
    for f in fit:
        if f["codebook"] != "conj":
            continue
        fk = [t for t in fit_kappa if t["block"] == f["block"]]
        op = next((t for t in fk if t["kappa_case"] == "open"), None)
        phs = [t for t in fk if t["kappa_case"] != "open"]
        gm = [t["G_match_00_db"] for t in phs if t["G_match_00_db"] is not None]
        f.update({"G_match_00_open_db": None if op is None else op["G_match_00_db"],
                  "net_G_match_00_open_db": None if op is None else op["net_G_match_00_db"],
                  "G_band_open_lo_db": None if op is None else op["G_band_lo_db"],
                  "G_band_open_hi_db": None if op is None else op["G_band_hi_db"],
                  "G_match_00_kappa_min_db": min(gm) if gm else None,
                  "G_match_00_kappa_max_db": max(gm) if gm else None,
                  "kappa_phases_no_match_00": ",".join(f"{t['kappa0_phase_deg']:g}" for t in phs
                                                       if t["G_match_00_db"] is None) or None})
    # ---- 折叠扫描、隔离扫描
    fold_tab = common.aggregate([r for r in rows if r["kind"] == "fold"],
                                ["block", "delta_deg", "fold_deg", "codebook"],
                                ["improvement_db", "alpha_rx", "alpha_tx", "iso_geo_codebook_db", "iso_geo_rms_db",
                                 "flagged", "pa_in_rel_p1db_db", "plate_db"])
    iso_tab = common.aggregate([r for r in rows if r["kind"] == "iso"],
                               ["block", "label", "fold_deg", "codebook", "isolation_db", "kappa0_phase_deg"],
                               ["improvement_db", "improvement_raw_db", "improvement_ss_db", "flagged", "flagged_ss",
                                "loop_gain_ss_db", "loop_gain_robust", "pa_in_rel_p1db_db"])
    freq_tab = []
    freqs = sorted({r["freq_ghz"] for r in cfg_rows})
    if len(freqs) > 1:
        for blk_name, label, reading in keys:
            blk = by_block[blk_name]
            for fg in freqs:
                sel = [t for t in gain_tab if (t["block"], t["label"], t["reading"]) == (blk_name, label, reading)
                       and t["codebook"] == "conj" and t["gain_db"] == g0 and abs(t["freq_ghz"] - fg) < 1e-9]
                i = int(np.argmin(np.abs(blk.freqs_ghz - fg)))
                if sel and abs(blk.freqs_ghz[i] - fg) < 1e-6:
                    y = blk.impr[i, blk.labels.index(label)]
                    pc = sel[0]["improvement_db_mean"]
                    freq_tab.append({"block": blk_name, "config": label, "reading": reading, "freq_ghz": fg,
                                     "meas_db": float(y), "pred_conj_db": pc,
                                     "dev_db": None if pc is None or not np.isfinite(pc) else float(pc - y)})
    summ = {"main": main, "gate": gate, "gain_tab": gain_tab, "fit": fit, "fold_tab": fold_tab, "iso_tab": iso_tab,
            "freq_tab": freq_tab, "kappa_tab": kappa_tab, "fit_kappa": fit_kappa}
    summ["coplanar"] = coplanar_analysis(summ, blocks)
    return summ


def _points_str(pts: list[dict], target: float, max_list: int = 4) -> str:
    """稳定 (隔离, 相位) 点的简述：少则全列；多则给个数 + 最接近实测均值的点。"""
    if not pts:
        return ""
    fmt = [f"{t['isolation_db']:g}dB/{t['kappa0_phase_deg']:g}°" for t in pts]
    if len(pts) <= max_list:
        return "; ".join(fmt)
    best = min(range(len(pts)), key=lambda i: abs(pts[i]["improvement_db_mean"] - target))
    return f"{len(pts)} 个（最接近实测均值：{fmt[best]} → {pts[best]['improvement_db_mean']:.1f} dB）"


def coplanar_analysis(summ: dict, blocks: list[MeasBlock]) -> list[dict]:
    """共面退化（方案 §6.2 F"折叠角"；项目总结 pptx 第 7 页的原因列表）的定量拆解：
    实测"收00-发00"与共面列之差 vs 仿真同一差值（只含 1-bit 偏转与扫描损失）；再看隔离度 × κ0 相位扫描里
    **稳定**（保守判据：未标记且小信号 |Gκ| < 1，flagged_ss = 0）点能把共面预测压到多低——负反馈 1/|1 + |Gκ||
    在稳定域内最多 −6 dB（注入锁定的饱和态不算"稳定域"，审查 S0-0）。"""
    out = []
    for blk in blocks:
        cop = [lb for lb in blk.labels if blk.coplanar_note.get(lb)]
        l00 = [lb for lb in blk.labels if lb.startswith("收00-发00")]
        if not cop or not l00:
            continue
        mm = {r["config"]: r for r in summ["main"] if r["block"] == blk.name and r["reading"] == "phi_theta"}
        cp, z0 = mm.get(cop[0]), mm.get(l00[0])
        if cp is None or z0 is None or cp.get("pred_conj") is None:
            continue
        iso = [t for t in summ["iso_tab"] if t["block"] == blk.name and t["label"] == cop[0]
               and t["codebook"] == "conj"]
        stable = [t for t in iso if t.get("flagged_ss_mean", t["flagged_mean"]) == 0
                  and t["improvement_db_mean"] is not None and np.isfinite(t["improvement_db_mean"])]
        lo, hi = cp["meas_min"] - GATE_TOL_DB, cp["meas_max"] + GATE_TOL_DB
        reach = [t for t in stable if lo <= t["improvement_db_mean"] <= hi]
        worst = min(stable, key=lambda t: t["improvement_db_mean"]) if stable else None
        out.append({"block": blk.name, "coplanar_config": cop[0],
                    "meas_00_minus_coplanar_db": z0["meas_mean"] - cp["meas_mean"],
                    "pred_00_minus_coplanar_db": z0["pred_conj"] - cp["pred_conj"],
                    "unexplained_db": (z0["meas_mean"] - cp["meas_mean"]) - (z0["pred_conj"] - cp["pred_conj"]),
                    "pred_coplanar_default_db": cp["pred_conj"], "meas_coplanar_mean_db": cp["meas_mean"],
                    "min_stable_pred_db": None if worst is None else worst["improvement_db_mean"],
                    "min_stable_at_iso_db": None if worst is None else worst["isolation_db"],
                    "min_stable_at_phase_deg": None if worst is None else worst["kappa0_phase_deg"],
                    "n_stable_points": len(stable), "n_stable_points_in_gate": len(reach),
                    "stable_points_in_gate": _points_str(reach, cp["meas_mean"])})
    return out


MAIN_COLS = ["block", "config", "reading", "ambiguous", "coplanar", "fold_deg", "rx_az_deg", "rx_el_deg", "tx_az_deg",
             "meas_mean", "meas_min", "meas_max", "meas_f0", "pred_conj", "pred_conj_ci95", "pred_steer", "pred_oracle",
             "dev_conj_db", "gate_conj", "gate_oracle", "pred_conj_open", "gate_conj_open", "pred_conj_kappa_min",
             "pred_conj_kappa_max", "n_kappa_gate_pass", "flagged_conj", "pa_in_rel_p1db_db", "p_feed_dbm",
             "iso_geo_rms_db", "n_seeds"]
FIT_COLS = ["block", "codebook", "n_configs_fit", "G_unstable_from_db", "G_ss_unstable_from_db", "G_match_00_db",
            "net_G_match_00_db", "G_match_00_open_db", "net_G_match_00_open_db", "G_match_00_kappa_min_db",
            "G_match_00_kappa_max_db", "kappa_phases_no_match_00", "G_rms_best_db", "rms_at_best_db", "bias_at_default_db",
            "rms_at_default_db", "G_band_lo_db", "G_band_hi_db", "net_G_band_lo_db", "net_G_band_hi_db",
            "G_band_open_lo_db", "G_band_open_hi_db"]
FIT_KAPPA_COLS = ["block", "kappa_case", "isolation_db", "kappa0_phase_deg", "loop_factor_at_default_db", "gate_non_coplanar",
                  "G_match_00_db", "net_G_match_00_db", "G_rms_best_db", "rms_at_best_db", "bias_at_default_db",
                  "G_band_lo_db", "G_band_hi_db", "G_unstable_from_db", "G_ss_unstable_from_db", "n_configs_fit"]
GATE_KAPPA_COLS = ["kappa_case", "isolation_db", "kappa0_phase_deg", "is_default", "loop_gain_ss_db", "loop_factor_db",
                   "n_pass_non_coplanar", "n_non_coplanar", "pass_non_coplanar", "mean_dev_db", "max_abs_dev_db",
                   "min_gate_margin_db", "n_pass_coplanar", "n_coplanar", "n_pass_ambiguous", "n_ambiguous"]


def _fold_table_md(fold_tab: list[dict]) -> list[dict]:
    out = []
    for t in fold_tab:
        out.append({"block": t["block"], "fold_deg": t["fold_deg"], "codebook": t["codebook"],
                    "rx_steer_deg": t["alpha_rx_mean"], "tx_steer_deg": t["alpha_tx_mean"],
                    "pred_mean": t["improvement_db_mean"], "pred_ci95": t["improvement_db_ci95"],
                    "iso_geo_codebook_db": t["iso_geo_codebook_db_mean"], "iso_geo_rms_db": t["iso_geo_rms_db_mean"],
                    "flagged": t["flagged_mean"], "pa_in_rel_p1db_db": t["pa_in_rel_p1db_db_mean"], "n": t["n_runs"]})
    return out


def _iso_table_md(iso_tab: list[dict]) -> list[dict]:
    out = []
    for t in iso_tab:
        if t["codebook"] != "conj":
            continue
        out.append({"block": t["block"], "config": t["label"], "fold_deg": t["fold_deg"],
                    "isolation_db": t["isolation_db"], "kappa0_phase_deg": t["kappa0_phase_deg"],
                    "pred_mean": t["improvement_db_mean"], "pred_raw_mean": t["improvement_raw_db_mean"],
                    "flagged": t["flagged_mean"], "flagged_ss": t.get("flagged_ss_mean"),
                    "loop_gain_ss_db": t["loop_gain_ss_db_mean"], "loop_gain_robust": t.get("loop_gain_robust_mean"),
                    "pa_in_rel_p1db_db": t["pa_in_rel_p1db_db_mean"], "n": t["n_runs"]})
    return out


def _measured_table(blocks: list[MeasBlock]) -> list[dict]:
    out = []
    for b in blocks:
        for lb in b.labels:
            p = parse_label(lb)
            out.append({"block": b.name, "feed_az_deg": b.feed_az_deg, "feed_dist_m": b.feed_dist_m, "config": lb,
                        "rx": f"{p['rx'][0]:g},{p['rx'][1]:g}", "tx": f"{p['tx'][0]:g},{p['tx'][1]:g}",
                        "coplanar_note": b.coplanar_note.get(lb, False), **b.stats(lb),
                        "plate_gain_mean": None if b.plate_gain is None else float(np.mean(b.plate_gain)),
                        "impr_vs_abs_minus_plate_maxdev": b.consistency_db()})
    return out


# ============================================================ 作图
def _finite_minmax(vals) -> tuple[float, float]:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return (min(v), max(v)) if v else (np.nan, np.nan)


def make_figures(outdir: str, summ: dict, blocks: list[MeasBlock], args) -> list[str]:
    """图（中文标签：common 导入时已设字体回退，审查 S0-1）。"""
    import matplotlib.pyplot as plt

    paths = []
    f0 = F0 / 1e9
    ktab = summ.get("kappa_tab", [])
    kiso = summ["gate"].get("kappa_iso_db")
    # 1) 改善量 vs PA 增益；"收00-发00" 另画 ∠κ0 系统误差带与开环曲线（审查 P-1）
    fig, axes = plt.subplots(1, len(blocks), figsize=(6.4 * len(blocks), 4.6), squeeze=False)
    for ax, blk in zip(axes[0], blocks):
        nc_means = [blk.stats(lb)["meas_mean"] for lb in blk.labels if not blk.coplanar_note.get(lb)]
        ax.axhspan(min(nc_means), max(nc_means), color="tab:green", alpha=0.15, label="实测（非共面各列均值的范围）")
        used = set()
        for lb in blk.labels:
            cop = bool(blk.coplanar_note.get(lb))
            is00 = lb.startswith("收00-发00")
            kind = "共面" if cop else ("收00-发00" if is00 else "其他配置")
            col = {"共面": "tab:red", "收00-发00": "tab:blue"}.get(kind, "0.6")
            for cb, ls in (("conj", "-"), ("oracle", ":")):
                pts = sorted([t for t in summ["gain_tab"] if t["block"] == blk.name and t["label"] == lb
                              and t["codebook"] == cb and t["reading"] == "phi_theta"
                              and abs(t["freq_ghz"] - f0) < 1e-9], key=lambda t: t["gain_db"])
                if not pts:
                    continue
                x = [t["gain_db"] for t in pts]
                y = [np.nan if t["improvement_db_mean"] is None else t["improvement_db_mean"] for t in pts]
                key = f"预测 {cb}：{kind}"
                ax.plot(x, y, ls, color=col, lw=0.8 if kind == "其他配置" else 1.6,
                        label=None if key in used else key)
                used.add(key)
            if cop:
                ax.axhline(blk.stats(lb)["meas_mean"], color="tab:red", ls="--", lw=0.8,
                           label=None if "mc" in used else "实测共面列均值")
                used.add("mc")
            if is00 and ktab:
                pts = [t for t in ktab if t["block"] == blk.name and t["label"] == lb and t["reading"] == "phi_theta"]
                gs = sorted({t["gain_db"] for t in pts})
                band = [_finite_minmax([t["improvement_db_mean"] for t in pts
                                        if t["gain_db"] == g and t["kappa_case"] != "open"]) for g in gs]
                if gs and np.isfinite(np.array(band, dtype=float)).any():
                    ax.fill_between(gs, [b[0] for b in band], [b[1] for b in band], color="tab:blue", alpha=0.13,
                                    label=f"收00-发00 conj：∠κ0 扫描范围（κ0 = {_s(kiso, 'g')} dB）")
                op = sorted([t for t in pts if t["kappa_case"] == "open"], key=lambda t: t["gain_db"])
                if op:
                    ax.plot([t["gain_db"] for t in op],
                            [np.nan if t["improvement_db_mean"] is None else t["improvement_db_mean"] for t in op],
                            "--", color="navy", lw=1.2, label="收00-发00 conj：开环（κ0 → 0）")
        ax.axvline(args.gain_db, color="k", lw=0.6)
        ax.set_xlabel("PA 小信号增益 G (dB)  [净增益 = G − 5 dB]")
        ax.set_ylabel("中继 / 同尺寸 PEC 板 (dB)")
        ax.set_title(f"{blk.name}：馈源方位 {blk.feed_az_deg:g}°、{blk.feed_dist_m:g} m（f = 3.5 GHz，"
                     f"实线 κ0 = {DEFAULT_KAPPA[0]:g} dB/{DEFAULT_KAPPA[1]:g}°）", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    paths.append(common.savefig(fig, os.path.join(outdir, "fig_gain_sweep.png")))
    # 2) 预测 vs 实测（conj 另画 ∠κ0 扫描范围与开环预测）
    fig, axes = plt.subplots(1, len(blocks), figsize=(7.2 * len(blocks), 4.8), squeeze=False)
    for ax, blk in zip(axes[0], blocks):
        rows = [r for r in summ["main"] if r["block"] == blk.name]
        x = np.arange(len(rows))
        mm = np.array([r["meas_mean"] for r in rows])
        lo = np.array([r["meas_min"] for r in rows])
        hi = np.array([r["meas_max"] for r in rows])
        ax.fill_between(x, lo - GATE_TOL_DB, hi + GATE_TOL_DB, step="mid", color="tab:green", alpha=0.12,
                        label="实测范围 ±3 dB（门 G1）")
        ax.errorbar(x, mm, yerr=[mm - lo, hi - mm], fmt="o", color="k", ms=4, capsize=3, label="实测（3.45–3.55 GHz）")
        kmin = np.array([np.nan if r.get("pred_conj_kappa_min") is None else r["pred_conj_kappa_min"] for r in rows])
        kmax = np.array([np.nan if r.get("pred_conj_kappa_max") is None else r["pred_conj_kappa_max"] for r in rows])
        if np.isfinite(kmin).any():
            ax.vlines(x + 0.12, kmin, kmax, color="tab:blue", lw=2.5, alpha=0.35,
                      label=f"conj：∠κ0 扫描范围（κ0 = {_s(kiso, 'g')} dB）")
        for cb, mk, col in (("conj", "s", "tab:blue"), ("steer", "^", "tab:orange"), ("oracle", "D", "tab:purple")):
            y = [np.nan if r.get(f"pred_{cb}") is None else r[f"pred_{cb}"] for r in rows]
            ax.plot(x, y, mk, color=col, ms=5,
                    label=f"预测 {cb}（G = {args.gain_db:g} dB，κ0 = {DEFAULT_KAPPA[0]:g} dB/{DEFAULT_KAPPA[1]:g}°）")
        yo = np.array([np.nan if r.get("pred_conj_open") is None else r["pred_conj_open"] for r in rows], dtype=float)
        if np.isfinite(yo).any():
            ax.plot(x + 0.12, yo, "x", color="navy", ms=6, label="conj 开环（κ0 → 0）")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['config']}{'*' if r['reading'] == 'az_el' else ''}" for r in rows], rotation=60,
                           ha="right", fontsize=7)
        ax.set_ylabel("中继 / 同尺寸 PEC 板 (dB)")
        ax.set_title(f"{blk.name}（* = 歧义列的 方位/俯仰 读法）")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    paths.append(common.savefig(fig, os.path.join(outdir, "fig_pred_vs_meas.png")))
    # 3) 折叠扫描
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for blk, col in zip(blocks, ("tab:blue", "tab:orange", "tab:green")):
        for cb, ls in (("conj", "-"), ("oracle", ":")):
            pts = sorted([t for t in summ["fold_tab"] if t["block"] == blk.name and t["codebook"] == cb],
                         key=lambda t: t["fold_deg"])
            if pts:
                ax.plot([t["fold_deg"] for t in pts], [t["improvement_db_mean"] for t in pts], ls, color=col,
                        label=f"{blk.name} {cb}（仿真，Δ = {blk.feed_az_deg:g}°）")
        mr = [r for r in summ["main"] if r["block"] == blk.name and r["reading"] == "phi_theta"]
        ax.plot([r["fold_deg"] for r in mr], [r["meas_mean"] for r in mr], "o", color=col, mfc="none",
                label=f"{blk.name} 实测（折叠角由列名推出）")
    ax.set_xlabel("折叠角 ψ (°)  [0 = 共面]")
    ax.set_ylabel("中继 / 同尺寸 PEC 板 (dB)")
    ax.set_title("折叠角扫描（余角由波束偏转补齐）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    paths.append(common.savefig(fig, os.path.join(outdir, "fig_fold_sweep.png")))
    # 4) 隔离扫描
    labels = list(dict.fromkeys((t["block"], t["label"]) for t in summ["iso_tab"]))
    if labels:
        fig, axes = plt.subplots(1, len(labels), figsize=(4.6 * len(labels), 4.3), squeeze=False)
        for ax, (bn, lb) in zip(axes[0], labels):
            for ph, col in zip(PLOT_PHASES_DEG, ("tab:blue", "tab:cyan", "tab:orange", "tab:olive", "tab:green")):
                pts = sorted([t for t in summ["iso_tab"] if t["block"] == bn and t["label"] == lb
                              and t["codebook"] == "conj" and t["kappa0_phase_deg"] == ph],
                             key=lambda t: t["isolation_db"])
                x = np.array([t["isolation_db"] for t in pts])
                ok = np.array([t["flagged_mean"] == 0 for t in pts], dtype=bool)
                raw = np.array([t["improvement_raw_db_mean"] for t in pts], dtype=float)
                ax.plot(x[ok], raw[ok], "o-", ms=3, color=col, label=f"∠κ0 = {ph:g}°（稳定）")
                ax.plot(x[~ok], raw[~ok], "x", ms=5, color=col)
            st = by_name(blocks, bn).stats(lb)
            ax.axhspan(st["meas_min"], st["meas_max"], color="k", alpha=0.08, label="实测范围")
            ax.axhline(st["meas_mean"], color="k", ls="--", lw=0.8, label="实测均值")
            ax.set_ylim(min(0.0, st["meas_min"] - 3), 50)
            ax.set_title(f"{bn}：{lb}\n（x = 标记：CW 或相位稳健环路增益 ≥ 1，画 CW 原始值）", fontsize=8)
            ax.set_xlabel("端口隔离 κ0 (dB)")
            ax.set_ylabel("中继 / 同尺寸 PEC 板 (dB)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        paths.append(common.savefig(fig, os.path.join(outdir, "fig_isolation_sweep.png")))
    # 5) 审查 P-1：门 G1 计数与 G_match_00 随 ∠κ0
    gk = [t for t in summ["gate"].get("kappa", []) if t["kappa_case"] != "open"]
    if gk:
        op = next((t for t in summ["gate"]["kappa"] if t["kappa_case"] == "open"), None)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
        ph = np.array([t["kappa0_phase_deg"] for t in gk])
        npass = np.array([t["n_pass_non_coplanar"] for t in gk])
        ntot = max(t["n_non_coplanar"] for t in gk)
        a1.bar(ph, npass, width=30, color=["tab:green" if t["pass_non_coplanar"] else "tab:red" for t in gk],
               alpha=0.7, label="κ0 扫描（绿 = 全部通过）")
        a1.axhline(ntot, color="k", lw=0.8, label=f"配置数 {ntot}")
        if op is not None:
            a1.axhline(op["n_pass_non_coplanar"], color="navy", ls="--", lw=1.0,
                       label=f"开环 κ0 → 0：{op['n_pass_non_coplanar']}/{op['n_non_coplanar']}")
        for t in gk:
            if t["is_default"]:
                a1.annotate("默认", (t["kappa0_phase_deg"], t["n_pass_non_coplanar"]), ha="center", va="bottom",
                            fontsize=8)
        a1.set_xticks(ph)
        a1.set_xlabel(f"∠κ0 (°)，κ0 = {_s(kiso, 'g')} dB，G = {args.gain_db:g} dB")
        a1.set_ylabel("门 G1 通过的非共面无歧义配置数")
        a1.set_title("门 G1 随未测的 ∠κ0")
        a1.grid(alpha=0.3)
        a1.legend(fontsize=7, loc="lower right")
        for blk, col in zip(blocks, ("tab:blue", "tab:orange", "tab:green")):
            fk = sorted([t for t in summ.get("fit_kappa", []) if t["block"] == blk.name and t["kappa_case"] != "open"],
                        key=lambda t: t["kappa0_phase_deg"])
            if not fk:
                continue
            x = np.array([t["kappa0_phase_deg"] for t in fk])
            y = np.array([np.nan if t["G_match_00_db"] is None else t["G_match_00_db"] for t in fk])
            a2.plot(x, y, "o-", color=col, label=f"{blk.name}：κ0 = {_s(kiso, 'g')} dB")
            nm = ~np.isfinite(y)
            if nm.any():
                gtop = max(t["gain_db"] for t in summ["gain_tab"]) if summ["gain_tab"] else 40.0
                a2.plot(x[nm], np.full(nm.sum(), gtop), "v", color=col, mfc="none", ms=7,
                        label=f"{blk.name}：扫描范围内无解（画在上限）")
            fo = next((t for t in summ["fit_kappa"] if t["block"] == blk.name and t["kappa_case"] == "open"), None)
            if fo is not None and fo["G_match_00_db"] is not None:
                a2.axhline(fo["G_match_00_db"], color=col, ls="--", lw=1.0,
                           label=f"{blk.name}：开环 {fo['G_match_00_db']:.1f} dB")
        a2.set_xticks(ph)
        a2.set_xlabel("∠κ0 (°)")
        a2.set_ylabel("G_match_00 (dB)：收00-发00 预测 = 实测均值")
        a2.set_title("反推 PA 增益随 ∠κ0（conj）")
        a2.grid(alpha=0.3)
        a2.legend(fontsize=7)
        paths.append(common.savefig(fig, os.path.join(outdir, "fig_kappa_phase.png")))
    return paths


def by_name(blocks: list[MeasBlock], name: str) -> MeasBlock:
    return next(b for b in blocks if b.name == name)


# ============================================================ 主流程
def run(args, job_filter=None) -> dict:
    """完整流程。job_filter(jobs) -> jobs：可选的计划裁剪（单元测试用；正式运行不用）。"""
    t0 = time.perf_counter()
    blocks = load_measurements(args.xlsx)
    outdir = common.make_outdir(EXP_NAME, args.out)
    jobs = plan_jobs(blocks, args)
    if job_filter is not None:
        jobs = list(job_filter(jobs))
    results = common.run_jobs(run_job, jobs, args.workers)
    rows, errors = _flatten(results)
    summ = summarize(rows, blocks, args)
    meas_tab = _measured_table(blocks)
    fold_md = _fold_table_md(summ["fold_tab"])
    iso_md = _iso_table_md(summ["iso_tab"])
    # 运行行：job 级信息（码本、耗时、错误）+ 全部点
    run_rows = [{"job": r.get("job"), "codebooks": r.get("codebooks"), "time_s": r.get("time_s"),
                 "error": r["error"]} if "error" in r else
                {"job": r.get("job"), "codebooks": r.get("codebooks"), "time_s": r.get("time_s"), "rows": r["rows"]}
                for r in results]
    cfg = {"experiment": "F hardware alignment (方案 §6.2 F, §8 gate G1)", "quick": bool(args.quick),
           "xlsx": args.xlsx, "seeds": args.seeds, "workers": args.workers, "gain_default_db": args.gain_db,
           "plate": args.plate, "probe_dist_m": args.probe_dist, "phase_err_deg": args.phase_err_deg,
           "backoff_db": args.backoff_db, "psat_out_dbm": DEFAULTS["psat_out_dbm"],
           "loss_rx_db": DEFAULTS["loss_rx_db"], "loss_tx_db": DEFAULTS["loss_tx_db"],
           "isolation_db_default": DEFAULTS["isolation_db"], "kappa0_phase_deg_default": DEFAULTS["kappa0_phase_deg"],
           "kappa0_note": "∠κ0 未测（INTEGRATION B2 默认 90° 是假设值）；系统误差扫描见 kappa_sys_*（审查 P-1）",
           "kappa_sys_iso_db": KAPPA_SYS_ISO_DB, "kappa_sys_phases_deg": list(KAPPA_SYS_PHASES_DEG),
           "open_loop_iso_db": OPEN_LOOP_ISO_DB,
           "osc_flag": "flagged = CW (~converged | rho_op >= 1) or rho_robust >= 1 (physical, 审查 S0-0); "
                       "flagged_ss = CW or small-signal |G kappa| >= 1 (conservative)",
           "elem_pattern": "q=1, d0=pi, backlobe -25 dB", "horn_pattern": "q=10 (feed and receive horn)",
           "voxel_d": None, "voxel_note": "no objects in experiment F (focus task)",
           "n_jobs": len(jobs), "n_errors": len(errors), "gate": summ["gate"]}
    paths = common.write_outputs(outdir, run_rows, summ["main"], cfg, columns=MAIN_COLS,
                                 title="实验 F：中继相对金属板的增益提升（预测 vs 实测）")
    common.save_csv(summ["fit"], os.path.join(outdir, "gain_fit.csv"), FIT_COLS)
    common.save_csv(summ["fit_kappa"], os.path.join(outdir, "gain_fit_kappa.csv"), FIT_KAPPA_COLS)
    common.save_csv(summ["gate"]["kappa"], os.path.join(outdir, "gate_kappa.csv"), GATE_KAPPA_COLS)
    common.save_csv(summ["gain_tab"], os.path.join(outdir, "gain_sweep.csv"))
    common.save_csv(summ["kappa_tab"], os.path.join(outdir, "kappa_sweep.csv"))
    common.save_csv(fold_md, os.path.join(outdir, "fold_sweep.csv"))
    common.save_csv(iso_md, os.path.join(outdir, "isolation_sweep.csv"))
    common.save_csv(meas_tab, os.path.join(outdir, "measured.csv"))
    common.save_csv(summ["coplanar"], os.path.join(outdir, "coplanar_analysis.csv"))
    if summ["freq_tab"]:
        common.save_csv(summ["freq_tab"], os.path.join(outdir, "per_frequency.csv"))
    figs = make_figures(outdir, summ, blocks, args)
    md = _report_md(summ, meas_tab, fold_md, iso_md, cfg, args, time.perf_counter() - t0, len(errors))
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(md)
    return {"outdir": outdir, "summary": summ, "errors": errors, "figures": figs, "markdown": md,
            "elapsed_s": time.perf_counter() - t0}


def _gate_lines(g: dict) -> list[str]:
    """门 G1 段落（审查 P-1：结论以未测的 ∠κ0 为条件；开环预测与 ∠κ0 无关）。"""
    ok_nc, n_nc = g["non_coplanar"]
    dk = g["default_kappa"]
    kt = {t["kappa_case"]: t for t in g.get("kappa", [])}
    ph = [t for t in g.get("kappa", []) if t["kappa_case"] != "open"]
    lf = [t["loop_factor_db"] for t in ph if t["loop_factor_db"] is not None]
    lines = [f"**结论：{g['verdict']}**\n"]
    if ph:
        lg = ph[0]["loop_gain_ss_db"]
        lines.append(f"κ0 的相位 ∠κ0 从未测过（INTEGRATION B2 的 {dk['kappa0_phase_deg']:g}° 是假设值）。G = "
                     f"{g['gain_db']:g} dB、κ0 = {_s(g['kappa_iso_db'], 'g')} dB 时 |Gκ_eff| = {_s(lg)} dB，"
                     f"环路因子 1/|1 − Gκ_eff| 随 ∠κ0 在 [{_s(min(lf) if lf else None)}, {_s(max(lf) if lf else None)}] dB "
                     "之间变化，与码本无关地整体平移所有预测——所以单一 ∠κ0 下的 PASS/FAIL 不是关于硬件的结论；"
                     "与 κ0 无关的是开环（κ0 → 0）预测。\n")
    lines += [
        f"- 默认 κ0 = {dk['isolation_db']:g} dB/{dk['kappa0_phase_deg']:g}°：非共面、无歧义配置 **{ok_nc}/{n_nc} PASS**"
        f" → {'PASS' if g['pass_non_coplanar'] else 'FAIL'}",
    ]
    if ph:
        lines.append(f"- ∠κ0 扫描（κ0 = {_s(g['kappa_iso_db'], 'g')} dB，{len(ph)} 个相位）：PASS 的相位 "
                     f"{g['kappa_pass_arcs']}；逐相位 "
                     + "、".join(f"{t['kappa0_phase_deg']:g}° {t['n_pass_non_coplanar']}/{t['n_non_coplanar']}"
                                f"（偏差均值 {_s(t['mean_dev_db'], '+.1f')} dB）" for t in ph))
    if "open" in kt:
        t = kt["open"]
        lines.append(f"- 开环（κ0 → 0）：{t['n_pass_non_coplanar']}/{t['n_non_coplanar']} PASS（偏差均值 "
                     f"{_s(t['mean_dev_db'], '+.1f')} dB，最大 |偏差| {_s(t['max_abs_dev_db'])} dB）")
    lines += [
        f"- 共面配置（表注\"无夹角仅靠波束控制\"，默认 κ0）：{g['coplanar'][0]}/{g['coplanar'][1]} PASS"
        "（仿真几何不含共面时的馈源泄漏/隔离恶化，见隔离扫描）",
        f"- 歧义列（两种读法，默认 κ0）：{g['ambiguous_rows'][0]}/{g['ambiguous_rows'][1]} 行 PASS",
    ]
    return lines


def _report_md(summ, meas_tab, fold_md, iso_md, cfg, args, elapsed, n_err) -> str:
    g = summ["gate"]
    fmt = common.to_markdown
    lines = [
        "# 实验 F：硬件对齐（中继相对同尺寸金属板的增益提升）\n",
        f"- 模式：{'quick' if args.quick else 'full'}；种子 {args.seeds}/格点（每个种子 = 一组通道相位误差 σ = "
        f"{args.phase_err_deg:g}° + oracle 随机重启）{'（quick：少于 5 个种子）' if args.seeds < 5 else ''}；"
        f"频点 {'3.50 GHz' if args.quick else '3.45/3.50/3.55 GHz'}；耗时 {elapsed:.1f} s；出错 job {n_err}。",
        f"- 默认 PA 小信号增益 {args.gain_db:g} dB（净增益 = G − 5 dB 网络损耗），Psat_out {DEFAULTS['psat_out_dbm']:g} dBm；"
        f"κ0 端口隔离 {DEFAULTS['isolation_db']:g} dB、相位 {DEFAULTS['kappa0_phase_deg']:g}°（环路中乘收发网络损耗；"
        "**两者都是未测的假设值**，∠κ0 的系统误差见下）。",
        f"- 工作点：P_feed 使 conj 码本的开环 PA 输入 = P1dB_in{args.backoff_db:+g} dB（深回退≈线性）；表中 "
        "pa_in_rel_p1db_db 为闭环 |u|²/P1dB，p_feed_dbm 为对应馈源功率。无物体：voxel_d 不适用。",
        f"- 金属板：{'一块板' if args.plate == 'one' else '两块板'}等面积 PEC（λ/4 PO 单元，Γ=−1，镜面最佳取向）；"
        f"探头 = 接收喇叭（q=10），距发射板 {args.probe_dist:g} m。",
        "- 自激标记（审查 S0-0）：flagged = CW 判据或相位稳健环路增益 ≥ 1（物理判据，pred 置空）；flagged_ss = CW 或"
        "小信号 |Gκ| ≥ 1（保守列）。",
        "",
        "## 门 G1（实验 F 侧，conj 码本，f = 3.5 GHz，预测落在实测区间 ±3 dB）\n",
        *_gate_lines(g),
        "",
        "## 要点\n",
        *_key_findings(summ),
        "",
        "## 预测 vs 实测（默认增益）\n",
        "pred_*：默认 κ0；pred_conj_open：开环（κ0 → 0）；pred_conj_kappa_min/max：κ0 = "
        f"{_s(g.get('kappa_iso_db'), 'g')} dB 下 ∠κ0 扫描的最小/最大；n_kappa_gate_pass：通过门限的 ∠κ0 个数。\n",
        fmt(summ["main"], MAIN_COLS, ".3g"),
        "\n## ∠κ0 系统误差：门 G1 计数（conj，f = 3.5 GHz，默认增益；审查 P-1）\n",
        "loop_factor_db = −20log10|1 − G_v κ_eff|（小信号环路因子，空场景与码本无关）；mean_dev_db = 预测 − 实测均值"
        "（非共面无歧义配置平均）；min_gate_margin_db = 最差配置到门限边界的距离（> 0 在实测区间 ±3 dB 内）。\n",
        fmt(g.get("kappa", []), GATE_KAPPA_COLS, ".3g"),
        "\n## PA 增益反推（f = 3.5 GHz；拟合只用非共面、无歧义配置）\n",
        "G_match_00：使\"收00-发00\"预测 = 实测均值的 G；G_rms_best：预测与实测均值 RMS 最小的 G；"
        "G_band：全部拟合配置都落在门限内的 G 区间（净增益 = G − 5 dB）。默认 κ0 下各码本；*_open 为开环（κ0 → 0），"
        "*_kappa_min/max 为 ∠κ0 扫描的范围、kappa_phases_no_match_00 为扫描的 G 范围内（未标记点中）无解的 ∠κ0（这三类只对 conj 计算；"
        "空场景下环路因子与码本无关，steer/oracle 平移相同）。G_unstable_from：物理判据首个标记增益；"
        "G_ss_unstable_from：小信号 |Gκ| ≥ 1。\n",
        fmt(summ["fit"], FIT_COLS, ".3g"),
        "\n### 逐 κ0 情形（conj）\n",
        fmt(summ["fit_kappa"], FIT_KAPPA_COLS, ".3g"),
        "\n## 折叠角扫描（f = 3.5 GHz，默认增益；Δ 固定，余角由波束补）\n",
        "iso_geo_*：单元方向图算的同一中继板间自由空间耦合折算到端口（码本 conj / 随机码本平均），"
        "与默认 κ0 = −30 dB 对比；它不进入环路（include_direct_tx_rx=False）。\n",
        fmt(fold_md, None, ".3g"),
        "\n## 共面退化拆解（conj 码本，默认增益）\n",
        "pred_00_minus_coplanar：仿真里只有 1-bit 偏转 + 单元扫描损失造成的差；unexplained = 实测差 − 仿真差。"
        "min_stable_pred：隔离度 × κ0 相位扫描中稳定点（保守判据：|Gκ| < 1 且未标记）的最低共面预测——稳定负反馈最多压低 "
        "20log10(2) ≈ 6 dB（相对开环）；stable_points_in_gate：能让共面预测进入实测区间 ±3 dB 的稳定 (隔离, 相位)。\n",
        fmt(summ["coplanar"], None, ".3g"),
        "\n## 隔离度 × κ0 相位扫描（conj 码本，默认增益；flagged = CW 或相位稳健环路增益 ≥ 1，此时 pred_mean 置空、"
        "pred_raw_mean 为 CW 稳态算出的值，无测量意义；flagged_ss = 小信号 |Gκ| ≥ 1 的保守标记）\n",
        fmt(iso_md, None, ".3g"),
        *([f"\n## 逐频点（conj，G = {args.gain_db:g} dB，默认 κ0；码本在 3.5 GHz 设计）\n",
           fmt(summ["freq_tab"], None, ".3g")] if summ["freq_tab"] else []),
        "\n## 实测数据（提升表；impr_vs_abs_minus_plate_maxdev = 提升表与\"绝对增益 − 金属板增益\"的最大偏差）\n",
        fmt(meas_tab, None, ".4g"),
    ]
    return "\n".join(lines) + "\n"


def _key_findings(summ: dict) -> list[str]:
    """summary.md 的要点（全部由表中数值生成）。开环（κ0 → 0）预测在前，∠κ0 作为系统误差带（审查 P-1）。"""
    out = []
    g = summ["gate"]
    dk = g["default_kappa"]
    for f in summ["fit"]:
        if f["codebook"] != "conj":
            continue
        s = f"- {f['block']}："
        if f.get("G_match_00_open_db") is not None or f.get("G_band_open_lo_db") is not None:
            s += (f"开环（κ0 → 0，与 ∠κ0 无关）\"收00-发00\" 预测 = 实测均值所需 G = {_s(f.get('G_match_00_open_db'))} dB"
                  f"（净 {_s(f.get('net_G_match_00_open_db'))} dB），全部非共面配置落在门限内的 G 区间 "
                  f"[{_s(f.get('G_band_open_lo_db'))}, {_s(f.get('G_band_open_hi_db'))}] dB；")
        if f.get("G_match_00_kappa_min_db") is not None or f.get("kappa_phases_no_match_00"):
            s += (f"κ0 = {_s(g.get('kappa_iso_db'), 'g')} dB 时 G_match_00 随 ∠κ0 在 [{_s(f.get('G_match_00_kappa_min_db'))}, "
                  f"{_s(f.get('G_match_00_kappa_max_db'))}] dB 之间"
                  + (f"，∠κ0 = {f['kappa_phases_no_match_00']}° 时在扫描的 G 范围内（未标记点中）无解"
                     if f.get("kappa_phases_no_match_00") else "") + "；")
        s += (f"默认 κ0（{dk['isolation_db']:g} dB/{dk['kappa0_phase_deg']:g}°，假设值）下 G_match_00 = "
              f"{_s(f['G_match_00_db'])} dB（净 {_s(f['net_G_match_00_db'])} dB）、RMS 最小的 G = {_s(f['G_rms_best_db'])} dB"
              f"（RMS {_s(f['rms_at_best_db'])} dB）、门限内 G 区间 [{_s(f['G_band_lo_db'])}, {_s(f['G_band_hi_db'])}] dB；"
              + (f"G ≥ {_s(f['G_ss_unstable_from_db'])} dB 时小信号 |Gκ| ≥ 1。" if f.get("G_ss_unstable_from_db")
                 is not None else "扫描的 G 内小信号 |Gκ| 均 < 1。"))
        out.append(s)
    for c in summ["coplanar"]:
        out.append(f"- {c['block']} 共面列：实测比\"收00-发00\"低 {_s(c['meas_00_minus_coplanar_db'])} dB，"
                   f"仿真（1-bit 偏转 + 扫描损失）只低 {_s(c['pred_00_minus_coplanar_db'])} dB，未解释 "
                   f"{_s(c['unexplained_db'])} dB；稳定（|Gκ|<1）的隔离/相位最多把共面预测压到 "
                   f"{_s(c['min_stable_pred_db'])} dB（{_s(c['min_stable_at_iso_db'])} dB / {_s(c['min_stable_at_phase_deg'])}°），"
                   f"进入实测区间 ±3 dB 的稳定点：{c['stable_points_in_gate'] or '无'}。")
    return out


def _s(x, fmt: str = ".1f") -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(float(x), fmt)


def build_argparser():
    p = common.base_argparser("实验 F：硬件对齐（中继相对金属板增益，门 G1）")
    p.add_argument("--xlsx", type=str, default=DEFAULT_XLSX, help="实测数据 测试数据.xlsx 路径")
    p.add_argument("--plate", choices=("one", "both"), default="one", help="金属板面积 = 一块板 / 两块板")
    p.add_argument("--gain-db", type=float, default=DEFAULT_GAIN_DB, help="默认 PA 小信号增益 (dB)")
    p.add_argument("--phase-err-deg", type=float, default=10.0, help="通道相位误差标准差（度，每种子一组）")
    p.add_argument("--backoff-db", type=float, default=-20.0, help="conj 码本开环 PA 输入相对 P1dB (dB)")
    p.add_argument("--probe-dist", type=float, default=1.8, help="探头（接收喇叭）到发射板中心距离 (m)")
    return p


def parse_args(argv=None):
    """CLI 参数；--quick 且未显式给 --seeds 时种子数取 2（summary.md 里注明 quick 少于 5 个种子）。"""
    import sys

    args = build_argparser().parse_args(argv)
    if args.quick and "--seeds" not in (sys.argv[1:] if argv is None else list(argv)):
        args.seeds = 2
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    res = run(args)
    g = res["summary"]["gate"]
    print(res["markdown"])
    print(f"[exp_f] 输出目录：{res['outdir']}")
    print(f"[exp_f] 门 G1（非共面，默认 κ0）：{g['non_coplanar'][0]}/{g['non_coplanar'][1]}，"
          f"{'PASS' if g['pass_non_coplanar'] else 'FAIL'}；以 ∠κ0 为条件：{g['verdict']}；用时 {res['elapsed_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
