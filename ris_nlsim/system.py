"""物理系统：场景构建、源功率标定、条件 M0–M4、前向/测量/损失/评估（DESIGN §7，INTEGRATION A/B/B2/C/G2）。

数据流（方案 §2、§4）：
  任务样本（物体体素）→ solver.reduce_sample（每样本一次，缓存）→ 样本响应 H (N_RCV,S)
  中继条件：relay_linear → u_in、K、d、C（逐样本乘有效源幅度 s_b）→ PA 不动点 → z = d + C g(u) → |z|² → 读数
  M0：无源面相位 → 批量约化求值（_PassiveBatch，等价于逐样本 solver.eval_reduced）→ z = s_b·H[探头, 馈源]

条件（DESIGN §7.1 + INTEGRATION B）：
  M0  无源 1-bit 面，位置 = 所有中继收+发板的全部单元，Γ=1，无 PA；源功率与 M3 相同（backoff 0）。
  M1  LinearPA，增益 = Rapp 小信号增益（cfg.m1_gain="small_signal"）；或 Bussgang 等效复增益（"bussgang"，记 "M1b"，
      INTEGRATION B，Demir & Björnson）。源功率由 cfg.backoff_db 指定（默认 0，与 M3 同功率）。
  M2/M3/M4  RappPA，backoff −10 / 0 / +10 dB。
源功率标定（INTEGRATION B）：P_feed 使"参考 PA 输入功率" = P1dB_in·10^{backoff/10}；参考 = 中继版本空场景下、接收 1-bit
码本取对馈源的量化共轭（16 个全局偏移取 |u₀| 最大）时的 |u₀|²/P_feed。**这是开环量**（不含 κ 反馈与发射码本），
实际闭环工作点由 evaluate() 的 pa_in_rel_p1db_db 报告（每样本 10log10(|u|²/P1dB_in)）。背景相消实验可用
cfg.cal_ref = "nulled" / "object" 按零陷后的驱动标定（审查 INT-SEM-4）。

门 G1（方案 §6.1、§8）的可证范围：只有 M0 + Born 的探头场对物体存在向量 x 仿射。中继条件下物体处在中继发射波束中，
K(x)、C(x) 依赖 x（方案推论 4），M1 + Born 不保证解不出 3 位奇偶（审查 INT-PHYS-1）；需要"Born + 线性中继 = 二次"
的理想化对照时设 cfg.objects_see_relay_tx = False。

物体模式与极化率模型（审查 R2-INT-1）：make_task 按 cfg.object_mode 选体素极化率模型（tasks.object_model）——
born（DWBA，体素互不耦合）用 volume_rr（一阶 Born，体素尺寸收敛），full / intra 用 scalar_rr（耦合求解的一致离散）；
体素网格三种模式相同，evaluate / describe 记录 object_model 与 effective_eps。收敛后的 Born 对水（χ = 77、ka ≈ 1 的
λ/3 球）仍比完全耦合强约 27 dB（Born 近似在高对比度下失效，不是数值伪影），所以 RQ2（物体多次散射是否替代 PA 非线性）
在**强度匹配**下的对照是 "intra"（每个物体自身精确多次散射、物体之间不耦合；单物体与 full 完全相同，且 M0 下同样对 x
仿射）；"born" 保留给门 G1（M0 + Born 不可解）与"线性理论"参考。见 experiments.common.EXP_A_OBJECT_MODES。

训练目标：DSPSA 只用可测目标（loss_from(model_penalties=False)：任务损失 + 标记替换 + 零陷项），oracle / TorchSystem
用模型目标（再加 DESIGN §6.3 的光滑自激惩罚）（审查 INT-SEM-3）。

BLAS 线程（性能，非物理）：本机 numpy/scipy 自带的 OpenBLAS 多线程时小矩阵 LAPACK 慢约 100 倍（50×50 复数 solve：
3.8 ms vs 0.026 ms），M0 前向与 oracle 循环全是小矩阵。导入本模块时调用 limit_blas_threads() 把 OpenBLAS 设为单线程；
环境变量 RIS_NLSIM_BLAS_THREADS=N 改线程数，=keep 则不动（大 N_O 的 full 模式 LU 单线程最多慢 ~2 倍）。
"""
from __future__ import annotations

import copy
import dataclasses
import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from . import losses
from .constants import DEFAULTS, F0, w2dbm, wavelength, wavenumber
from .geometry import make_relays, point_from
from .materials import DEFAULT_MODEL, alpha_passive
from .metrics import walsh_coeffs, walsh_degree_profile, walsh_weight_ge
from .pa import LinearPA, RappPA
from .patterns import Pattern, element_pattern, horn_pattern, probe_pattern
from .readout import Detector, complex_awgn
from .relay import (OSC_CRITERIA, RelayParams, bits_to_phase, gain_at_least_one, pa_noise_power_w, probe_fields,
                    relay_linear, solve_fixed_point, weights)
from .scene import NodeSet, Scene
from .solver import ReducedSample, reduce_sample, solve_scene
from .tasks import BORN_MODEL, FocusTask, LetterTask, LocalizationTask, ParityTask, Task, eps_repr, object_model

def limit_blas_threads(n: int = 1) -> list[str]:
    """把 numpy / scipy 自带 OpenBLAS 的线程数设为 n（ctypes 调 openblas_set_num_threads*；无 threadpoolctl 时的替代）。

    返回成功设置的动态库文件名列表（找不到库或符号时静默跳过）。对已加载的库，ctypes.CDLL(完整路径) 返回同一句柄。
    """
    import ctypes
    import glob
    import importlib

    done = []
    for mod in ("numpy", "scipy"):
        try:
            pkg = importlib.import_module(mod)
        except ImportError:
            continue
        root = os.path.dirname(os.path.abspath(pkg.__file__))
        cands = (glob.glob(os.path.join(os.path.dirname(root), f"{mod}.libs", "*openblas*"))
                 + glob.glob(os.path.join(root, ".libs", "*openblas*")))
        for path in cands:
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            for sym in ("openblas_set_num_threads64_", "openblas_set_num_threads", "openblas_set_num_threads_64"):
                fn = getattr(lib, sym, None)
                if fn is not None:
                    fn(ctypes.c_int(int(n)))
                    done.append(os.path.basename(path))
                    break
    return done


_BLAS_ENV = os.environ.get("RIS_NLSIM_BLAS_THREADS", "1").strip().lower()
if _BLAS_ENV not in ("keep", ""):
    try:
        BLAS_LIMITED = limit_blas_threads(int(_BLAS_ENV))
    except (ValueError, OSError):               # 非法取值 / 库不可用：保持原状
        BLAS_LIMITED = []
else:
    BLAS_LIMITED = []

CONDITIONS = ("M0", "M1", "M1b", "M2", "M3", "M4")
CONDITION_BACKOFF_DB = {"M0": 0.0, "M1": 0.0, "M1b": 0.0, "M2": -10.0, "M3": 0.0, "M4": 10.0}
N_ELEM = 25                     # 每块板 5×5
N_FOCUS_OFFSETS = 16            # 量化共轭的全局偏移数（INTEGRATION B）
DETECTOR_SIGMA_DB = {"ideal": 0.0, "vna": 0.002, "logdet": 0.3}   # INTEGRATION C
OBJECT_MODES = ("full", "born", "intra")
CAL_REFS = ("focus", "nulled", "object")       # 源功率标定参考（审查 INT-SEM-4）
N_NULL_STARTS = 64                              # 参考零陷码本搜索的确定性随机起点数


# ============================================================ 配置
@dataclass
class SystemConfig:
    """系统配置（DESIGN §7.2 + INTEGRATION B/B2/C 增补字段）。不含随机种子：随机性一律由调用方的 Generator 提供。"""

    # 几何（DESIGN §1.5）
    preset: str = "L90"
    fold_deg: float | None = None           # None → 预设值
    n_relays: int = 1
    relay_spacing_z: float = 0.30           # 多中继沿 z 的间距（geometry.make_relays）
    roi_offset: tuple | None = None         # None → 预设 roi_offset；实验 D 用它扫物体距离
    # 馈源 / 探头几何覆盖（INTEGRATION A.3–A.4、S；None = 预设值）：馈源方位（相对接收板中心，度）/ 距离（m）；
    # 探头方位列表（相对发射板中心，前 n_probes 个，度）/ 距离（m）
    feed_az_deg: float | None = None
    feed_dist: float | None = None
    probe_az_deg: tuple | None = None
    probe_dist: float | None = None
    env_points: list | None = None          # [(pos(3,), alpha complex), ...] 固定散射点（INTEGRATION A.8）
    block_direct: bool = False              # 馈源→探头视距遮挡（方案 §5）
    skip_gain_db: float = 0.0
    freq_hz: float = F0
    # 条件（DESIGN §7.1）
    condition: str = "M3"                   # "M0" | "M1" | "M1b" | "M2" | "M3" | "M4"
    backoff_db: float | None = None         # None → 条件表（M2 −10 / M3 0 / M4 +10 / M0、M1 0）
    m1_gain: str = "small_signal"           # "small_signal" | "bussgang"（→ 标签 M1b）
    # 物体（DESIGN §2.2，INTEGRATION B2）
    # object_mode："full" 完全耦合；"intra" 物体内精确多次散射、物体间不耦合（RQ2 的强度匹配对照）；
    # "born" DWBA（门 G1 / 线性理论参考；水比 full 强 ~27 dB）。make_task 据此选极化率模型（审查 R2-INT-1，模块 docstring）。
    object_mode: str = "full"               # "full" | "born" | "intra"
    material: str = "water"                 # make_task 的默认材料
    voxel_d: float | None = None            # None → 按材料自动（tasks.resolve_voxel_d；三种物体模式同一网格）
    # 物体是否被中继发射场照射（审查 INT-PHYS-1；方案推论 4）。True（物理默认）：发射列 H[:, :, 1:] 含物体散射，
    # 环路 K(x) 与发射→探头 C(x) 依赖 x，线性中继的 u = u_in/(1 − G K(x)) 是 x 的有理函数 → M1+Born 不再是二次。
    # False（理想化对照，仅中继条件）：每个样本的发射列换成空场景发射列，物体只经馈源列进入
    # → 方案 §6.1 "Born + 线性中继 = 二次（3 位奇偶不可解）" 的前提成立。M0 不受影响。
    objects_see_relay_tx: bool = True
    # 背景相消（方案 §5）
    bg_cancel: str = "free"                 # "free" | "null"
    null_target_db: float = -20.0
    null_weight: float = 1.0
    # 源功率标定参考（审查 INT-SEM-4；INTEGRATION B 为 "focus"）：
    #   "focus"  空场景聚焦码本的开环 |u0|² = P1dB·10^{bo/10}（INTEGRATION B 原定义）
    #   "nulled" 参考零陷码本（空场景零陷深度最接近 null_target_db 的 1-bit 接收码本，即训练被推向的深度）下，
    #            各样本开环 |u_in(x)|² 的中位数 = P1dB·10^{bo/10}
    #   "object" 物体引起的开环驱动 max_x |u_in(x) − u0|² = P1dB·10^{bo/10}，参考码本 = 对物体散射场 δv(x) 的量化共轭
    #            （取 |δu| 最大的样本/偏移，即量化接收码本能给出的最强物体驱动；训练码本的物体驱动不超过它）
    # 后两者让 bg_cancel="null" 的条件标签（M2/M3/M4）对应零陷后的实际驱动（方案推论 3），而不是被零陷压低 17–27 dB。
    cal_ref: str = "focus"
    # 中继电参数（DESIGN §3.2；κ0 为端口隔离，环路里再乘收发网络损耗，INTEGRATION B2）
    isolation_db: float = DEFAULTS["isolation_db"]
    kappa0_phase_deg: float = DEFAULTS["kappa0_phase_deg"]
    loss_rx_db: float = DEFAULTS["loss_rx_db"]
    loss_tx_db: float = DEFAULTS["loss_tx_db"]
    nf_db: float = DEFAULTS["nf_db"]
    noise_bw_hz: float = DEFAULTS["noise_bw_hz"]
    pa_noise: bool = True                   # PA 输入等效噪声（noisy 前向时）
    # PA（DESIGN §3.1）
    ampm: bool = True
    pa_gain_db: float = DEFAULTS["pa_gain_db"]
    psat_out_dbm: float = DEFAULTS["psat_out_dbm"]
    rapp_p: float = DEFAULTS["rapp_p"]
    ampm_max_deg: float = DEFAULTS["ampm_max_deg"]
    ampm_beta: float = DEFAULTS["ampm_beta"]
    # 读出（DESIGN §4）
    detector: str = "vna"                   # "ideal" | "vna" | "logdet"
    detector_sigma_db: float | None = None  # None → vna 0.002 / logdet 0.3
    probe_noise_dbm: float | None = None
    # M0 无源面
    passive_levels: int = 2                 # 2 = 1-bit，4 = 2-bit
    # 中继收/发移相器电平数（2 = 1-bit 硬件现实；4/8/… = 多比特移相器，实验 G 的"相位精度阶梯"）。
    # 标定（源功率参考）始终用 1-bit 聚焦码本，所以改电平数不改 P_feed，条件之间可直接比较。
    relay_levels: int = 2
    # 损失（DESIGN §7.2–7.3、§6.3）
    tau_db: float = 1.0
    osc_penalty: float = 1.0                # 被标记样本：chance + osc_penalty（INTEGRATION C）
    osc_smooth_weight: float = 100.0        # DESIGN §6.3：weight·mean_b relu(ρ_b − 0.95)²（与 TorchSystem 同定义）
    # 自激标记判据（审查 S0-0，relay.OSC_CRITERIA）："cw"（默认，DESIGN §3.4：~converged | ρ_op ≥ 1，只看注入频率上的
    # CW 工作点，强注入可把 ρ_ss ≥ 1 的中继锁定成稳定饱和态）；"robust"（再要求相位稳健环路增益 ρ_rob < 1：任意泄漏相位 /
    # 带内频率都不起振）；"small_signal"（再要求 ρ(G_v K) < 1：源关闭也不起振，最保守；此时 M1 与 M2–M4 同一码本
    # 的标记相同，RQ5 "M4 自激比例低于 M1" 按构造不可能出现）。标记影响训练损失、准确率与 *_valid 字段；
    # evaluate 总会另报三种判据下的计数（n_ss_unstable、n_robust_unstable 等），可事后换判据统计。
    osc_criterion: str = "cw"
    n_bussgang_codebooks: int = 16          # M1b：随机码本数（INTEGRATION B）

    def __post_init__(self):
        if self.condition not in CONDITIONS:
            raise ValueError(f"未知条件 {self.condition!r}，可选 {CONDITIONS}")
        if self.preset not in DEFAULTS["presets"]:
            raise ValueError(f"未知预设 {self.preset!r}，可选 {list(DEFAULTS['presets'])}")
        if self.m1_gain not in ("small_signal", "bussgang"):
            raise ValueError("m1_gain 只能是 'small_signal' 或 'bussgang'")
        if self.object_mode not in OBJECT_MODES:
            raise ValueError(f"未知物体模式 {self.object_mode!r}")
        if self.bg_cancel not in ("free", "null"):
            raise ValueError("bg_cancel 只能是 'free' 或 'null'")
        if self.cal_ref not in CAL_REFS:
            raise ValueError(f"cal_ref 只能是 {CAL_REFS}")
        if self.detector not in DETECTOR_SIGMA_DB:
            raise ValueError(f"未知检波器 {self.detector!r}")
        if self.passive_levels < 2:
            raise ValueError("passive_levels ≥ 2")
        if self.relay_levels < 2:
            raise ValueError("relay_levels ≥ 2")
        if self.osc_criterion not in OSC_CRITERIA:
            raise ValueError(f"osc_criterion 只能是 {OSC_CRITERIA}")
        if self.n_relays < 1:
            raise ValueError("n_relays ≥ 1")
        for name in ("feed_dist", "probe_dist"):
            v = getattr(self, name)
            if v is not None and not float(v) > 0:
                raise ValueError(f"{name} 必须为正（m），得到 {v!r}")
        if self.probe_az_deg is not None:
            az = tuple(float(a) for a in np.atleast_1d(np.asarray(self.probe_az_deg, dtype=float)))
            if not az or not all(np.isfinite(az)):
                raise ValueError(f"probe_az_deg 应为非空有限角度列表，得到 {self.probe_az_deg!r}")
            self.probe_az_deg = az

    def to_dict(self) -> dict:
        """JSON 友好的字典。"""
        d = dataclasses.asdict(self)
        if d["roi_offset"] is not None:
            d["roi_offset"] = [float(x) for x in d["roi_offset"]]
        if d["env_points"] is not None:
            d["env_points"] = [[list(map(float, p)), [complex(a).real, complex(a).imag]] for p, a in d["env_points"]]
        if d["probe_az_deg"] is not None:
            d["probe_az_deg"] = [float(a) for a in d["probe_az_deg"]]
        return d

    @property
    def label(self) -> str:
        return condition_label(self)


def condition_label(cfg: SystemConfig) -> str:
    """条件标签：M1 + m1_gain="bussgang"（或直接写 "M1b"）→ "M1b"。"""
    if cfg.condition in ("M1", "M1b"):
        return "M1b" if (cfg.condition == "M1b" or cfg.m1_gain == "bussgang") else "M1"
    return cfg.condition


def effective_backoff_db(cfg: SystemConfig) -> float:
    """源功率标定用的回退量（INTEGRATION B）：cfg.backoff_db 优先，否则按条件表。"""
    if cfg.backoff_db is not None:
        return float(cfg.backoff_db)
    return CONDITION_BACKOFF_DB[cfg.condition]


def relay_params(cfg: SystemConfig) -> RelayParams:
    return RelayParams(n_relays=cfg.n_relays, n_elem=N_ELEM, loss_rx_db=cfg.loss_rx_db, loss_tx_db=cfg.loss_tx_db,
                       isolation_db=cfg.isolation_db, kappa0_phase_deg=cfg.kappa0_phase_deg, nf_db=cfg.nf_db,
                       noise_bw_hz=cfg.noise_bw_hz)


def bare_loop_gain_db(cfg: SystemConfig) -> float:
    """只含板间泄漏（无物体、无环境、单中继）时的小信号环路增益 20log10|G_v·κ_eff| (dB)（DESIGN §3.3，INTEGRATION B2）：
    = pa_gain_db + isolation_db − loss_rx_db − loss_tx_db。默认 30 − 30 − 5 = −5 dB。
    审查 S2-3：INTEGRATION G 实验 D 的 κ0 = −25 dB 恰好给出 0 dB（|Gκ_eff| = 1），无物体样本 K = κ_eff 处在临界上
    （线性中继按谐振容差必被标记），扫描请避开 0 dB 或用 isolation_for_loop_gain_db 按环路增益取点。"""
    return float(cfg.pa_gain_db + cfg.isolation_db - cfg.loss_rx_db - cfg.loss_tx_db)


def isolation_for_loop_gain_db(cfg: SystemConfig, loop_gain_db: float) -> float:
    """使 bare_loop_gain_db = loop_gain_db 的端口隔离 isolation_db (dB)（bare_loop_gain_db 的反函数，审查 S2-3）。"""
    return float(loop_gain_db - cfg.pa_gain_db + cfg.loss_rx_db + cfg.loss_tx_db)


def rapp_pa(cfg: SystemConfig) -> RappPA:
    """cfg 对应的 Rapp + Saleh PA（M2–M4 用它；M1 的增益与 P1dB 参考也取自它）。"""
    return RappPA(gain_db=cfg.pa_gain_db, psat_out_dbm=cfg.psat_out_dbm, p=cfg.rapp_p,
                  ampm_max_deg=cfg.ampm_max_deg, ampm_beta=cfg.ampm_beta, ampm=cfg.ampm)


class ComplexLinearPA(LinearPA):
    """复增益线性 PA g = G_c·u（M1b：Bussgang 等效增益含 AM-PM 相位）。

    relay.solve_fixed_point 的线性分支只把 pa.gain_v 当乘子（(I − G K)u = u_in），复数同样成立；
    loop_gain_ss = ρ(G_c K) = |G_c|ρ(K)。
    """

    def __init__(self, gain_c: complex):
        self.gain_c = complex(gain_c)
        self.gain_v = self.gain_c
        self.gain_db = float(20.0 * np.log10(abs(self.gain_c)))

    def __repr__(self) -> str:
        return f"ComplexLinearPA(|G|={self.gain_db:.3f} dB, ∠G={np.degrees(np.angle(self.gain_c)):.2f}°)"


# ============================================================ A. 场景构建
def scene_geometry(cfg: SystemConfig, n_probes: int = 2) -> dict:
    """几何参考点（INTEGRATION A.1–A.5，DESIGN §1.5），不建 NodeSet。馈源 / 探头方位与距离可由 cfg 覆盖（INTEGRATION S）。"""
    p = preset_geometry(cfg)
    fold = float(p["fold_deg"] if cfg.fold_deg is None else cfg.fold_deg)
    relays = make_relays(cfg.n_relays, fold, spacing_z=cfg.relay_spacing_z)
    rx_c = np.mean([r.rx_center for r in relays], axis=0)
    tx_c = np.mean([r.tx_center for r in relays], axis=0)
    c = np.mean([r.center for r in relays], axis=0)
    feed = point_from(rx_c, p["feed_az_deg"], 0.0, p["feed_dist"])
    az = list(p["probe_az_deg"])
    if n_probes > len(az):
        raise ValueError(f"预设 {cfg.preset} 最多 {len(az)} 个探头，需要 {n_probes}")
    probes = np.array([point_from(tx_c, a, 0.0, p["probe_dist"]) for a in az[:n_probes]]).reshape(-1, 3)
    ref = rx_c if p["roi_ref"] == "rx_center" else c
    off = np.asarray(p["roi_offset"] if cfg.roi_offset is None else cfg.roi_offset, dtype=float).reshape(3)
    return {"preset": cfg.preset, "fold_deg": fold, "relays": relays, "rx_c": rx_c, "tx_c": tx_c, "c": c,
            "roi_center": ref + off, "roi_offset": off, "feed_pos": feed, "probe_pos": probes,
            "probe_az_deg": az[:n_probes]}


def preset_geometry(cfg: SystemConfig) -> dict:
    """预设几何字典的副本，叠加 cfg 的馈源 / 探头覆盖（feed_az_deg / feed_dist / probe_az_deg / probe_dist，None = 预设）。"""
    p = dict(DEFAULTS["presets"][cfg.preset])
    for k in ("feed_az_deg", "feed_dist", "probe_dist"):
        if getattr(cfg, k) is not None:
            p[k] = float(getattr(cfg, k))
    if cfg.probe_az_deg is not None:
        p["probe_az_deg"] = [float(a) for a in cfg.probe_az_deg]
    return p


def roi_center(cfg: SystemConfig) -> np.ndarray:
    """物体感兴趣区中心（与探头数无关）。"""
    return scene_geometry(cfg, 1)["roi_center"]


def _env_sets(cfg: SystemConfig) -> list[NodeSet]:
    if not cfg.env_points:
        return []
    pos = np.array([np.asarray(p, dtype=float).reshape(3) for p, _ in cfg.env_points])
    alpha = np.array([complex(a) for _, a in cfg.env_points])
    return [NodeSet(pos, None, Pattern(), alpha=alpha)]


def build_scene(cfg: SystemConfig, n_probes: int = 2, relay: bool | None = None) -> tuple[Scene, dict]:
    """INTEGRATION A：按预设建场景，返回 (Scene, info)。

    relay=None → 按条件（M0 建无源面，其余建中继）；relay=True 强制中继版本（源功率标定用）。
    中继：tx/rx 各为每中继一个 NodeSet（单元方向图），passive 为空；
    M0：tx/rx 为空，passive = 所有中继收+发板全部单元（顺序：中继 0 的 rx 25、tx 25，中继 1 ……；与参数顺序一致），
        法向各自板法向，α = alpha_passive(Γ=1)。
    """
    geo = scene_geometry(cfg, n_probes)
    k, lam = wavenumber(cfg.freq_hz), wavelength(cfg.freq_hz)
    ep = element_pattern()
    relays = geo["relays"]
    use_relay = (cfg.condition != "M0") if relay is None else bool(relay)
    if use_relay:
        tx = [NodeSet(r.tx_pos, r.tx_normal, ep) for r in relays]
        rx = [NodeSet(r.rx_pos, r.rx_normal, ep) for r in relays]
        passive = NodeSet.empty(ep)
    else:
        tx, rx = [], []
        pos, nrm = [], []
        for r in relays:
            pos += [r.rx_pos, r.tx_pos]
            nrm += [np.broadcast_to(r.rx_normal, r.rx_pos.shape), np.broadcast_to(r.tx_normal, r.tx_pos.shape)]
        passive = NodeSet(np.concatenate(pos), np.concatenate(nrm), ep, alpha=alpha_passive(1.0, k, lam))
    feed = geo["feed_pos"]
    pp = geo["probe_pos"]
    scene = Scene(cfg.freq_hz, NodeSet(feed, geo["rx_c"] - feed, horn_pattern()), tx=tx, rx=rx,
                  probes=NodeSet(pp, geo["tx_c"] - pp, probe_pattern()), env=_env_sets(cfg), passive=passive,
                  block_direct_feed_probe=cfg.block_direct, skip_gain_db=cfg.skip_gain_db,
                  include_direct_tx_rx=False)
    info = dict(geo)
    info.update(relay=use_relay, n_rx=scene.n_rx, n_tx=scene.n_tx, n_passive=scene.n_passive,
                n_probe=scene.n_probe, n_env=scene.n_env)
    return scene, info


# ============================================================ B. 源功率标定
def quantized_conjugate(v: np.ndarray, levels: int = 2, n_offsets: int = N_FOCUS_OFFSETS) -> np.ndarray:
    """量化共轭码本（INTEGRATION B）：v (..., N, ) 为要相干合成的复系数，返回 (n_offsets, ..., N) 的电平。

    1-bit：b_n = 0 若 cos(−∠v_n + θ_j) ≥ 0 否则 1，θ_j = 2πj/n_offsets；多电平：l_n = round((θ_j − ∠v_n)/(2π/L)) mod L。
    """
    theta = 2.0 * np.pi * np.arange(n_offsets) / n_offsets
    ang = -np.angle(v)[None] + theta.reshape((-1,) + (1,) * np.ndim(v))
    if levels == 2:
        return (np.cos(ang) < 0.0).astype(np.int64)
    return np.mod(np.rint(ang / (2.0 * np.pi / levels)), levels).astype(np.int64)


def _best_codebook(v: np.ndarray, amp: float, err, levels: int = 2, ref=None) -> tuple[np.ndarray, np.ndarray]:
    """对每行 v[r] (R,N) 在量化共轭的 16 个偏移中取 |ref_r + Σ_n amp·e^{j(φ_n+err_n)} v_n| 最大者；
    ref=None（INTEGRATION B 的接收聚焦）即只看合成项。返回 (bits (R,N), 合成值 Σ (R,)，不含 ref)。"""
    cand = quantized_conjugate(v, levels)                               # (J,R,N)
    ph = bits_to_phase(cand, levels) + (0.0 if err is None else err)
    s = amp * np.sum(np.exp(1j * ph) * v[None], axis=-1)               # (J,R)
    tot = s if ref is None else s + np.asarray(ref).reshape(1, -1)
    best = np.argmax(np.abs(tot), axis=0)
    r = np.arange(v.shape[0])
    return cand[best, r], s[best, r]


def null_codebook(v: np.ndarray, amp: float, err=None, n_starts: int = N_NULL_STARTS, target_db: float | None = None,
                  ref_abs=None) -> tuple[np.ndarray, np.ndarray]:
    """参考零陷码本（审查 INT-SEM-4）：对每行 v[r] (R,N) 找 1-bit 码本使 u = Σ_n amp·e^{j(πb_n+err_n)} v_n 被压低。

    确定性多起点最速单比特翻转下降：起点 = 16 个量化共轭码本 + n_starts 个固定种子（default_rng(0)）随机码本；
    每步翻转使 |u| 下降最多的一位（翻转 n：u' = u − 2t_n），直到无改进。
    target_db=None：返回最深零陷；否则在下降途经的全部码本中取零陷深度 20log10(|u_r|/ref_abs_r) 最接近 target_db 者
    （ref_abs = 聚焦码本的 |u0_r|；训练的零陷惩罚把码本推向 cfg.null_target_db，参考码本取同一深度更有代表性）。
    返回 (bits (R,N), u (R,))。只作标定参考，不是全局最优保证。
    """
    v = np.asarray(v, dtype=np.complex128)
    R, N = v.shape
    rng = np.random.default_rng(0)                                      # 显式、固定：标定只依赖 cfg
    starts = np.concatenate([quantized_conjugate(v, 2), rng.integers(0, 2, size=(int(n_starts), R, N))])
    ph = bits_to_phase(starts, 2) + (0.0 if err is None else err)
    t = amp * np.exp(1j * ph) * v[None]                                 # (J,R,N) 各单元贡献
    bits = starts.copy()
    s = t.sum(axis=-1)                                                  # (J,R)
    r_idx = np.arange(R)
    if target_db is not None:
        ref = np.abs(np.asarray(ref_abs, dtype=float)).reshape(R)
        keep_bits, keep_s, keep_err = np.zeros((R, N), dtype=np.int64), np.zeros(R, complex), np.full(R, np.inf)

        def record():
            e = np.abs(20.0 * np.log10(np.maximum(np.abs(s), 1e-300) / ref[None]) - float(target_db))   # (J,R)
            jb = np.argmin(e, axis=0)
            eb = e[jb, r_idx]
            upd = eb < keep_err
            keep_err[upd] = eb[upd]
            keep_bits[upd] = bits[jb[upd], r_idx[upd]]
            keep_s[upd] = s[jb[upd], r_idx[upd]]
        record()
    for _ in range(50 * N):
        cand = np.abs(s[..., None] - 2.0 * t)                           # (J,R,N) 翻转每一位后的 |u|
        n_best = np.argmin(cand, axis=-1)
        best = np.take_along_axis(cand, n_best[..., None], axis=-1)[..., 0]
        imp = best < np.abs(s) * (1.0 - 1e-12)
        if not imp.any():
            break
        j, r = np.nonzero(imp)
        n = n_best[j, r]
        s[j, r] = s[j, r] - 2.0 * t[j, r, n]
        t[j, r, n] = -t[j, r, n]
        bits[j, r, n] = 1 - bits[j, r, n]
        if target_db is not None:
            record()
    if target_db is not None:
        return keep_bits, keep_s
    k = np.argmin(np.abs(s), axis=0)                                    # (R,)
    return bits[k, r_idx].astype(np.int64), s[k, r_idx]


def empty_relay_response(cfg: SystemConfig, n_probes: int = 1) -> np.ndarray:
    """中继版本空场景（无物体，含 env）的响应 H0 (N_RCV,S)（INTEGRATION B；零陷、发射列替换的基准）。"""
    scene, _ = build_scene(cfg, n_probes, relay=True)
    return solve_scene(scene, None, None, "full")


def _calibrate(cfg: SystemConfig, n_probes: int = 1, v_samples: np.ndarray | None = None,
               H0: np.ndarray | None = None) -> dict:
    """INTEGRATION B：中继版本空场景 → 聚焦码本 → p_ref_per_w、P_feed（cal_ref="focus"，默认）。

    cal_ref="nulled" / "object"（审查 INT-SEM-4）需要 v_samples (B,R,N)：各样本（含物体）在 P_feed=1 W 时各接收单元的
    馈源列场 H_b[rx, 0]（中继版本场景；不含每样本功率抖动）。
      nulled：参考接收码本 = null_codebook(target_db=cfg.null_target_db)（空场景零陷深度最接近训练目标深度的量化码本），
              p_ref = median_b mean_r |u_in,r(x_b)|²；
      object：参考接收码本 = 各样本物体场 δv_b = v_b − v 的量化共轭中 mean_r |δu|² 最大者，p_ref = 该最大值。
    三者都是开环量（与 INTEGRATION B 一致、对 P_feed 精确线性），实际闭环工作点由 evaluate() 报告。
    返回的 p_focus_per_w 始终是聚焦码本的开环 |u0|²/P_feed（零陷深度的归一化基准）。
    """
    if H0 is None:
        H0 = empty_relay_response(cfg, n_probes)                       # (N_RCV,S)，空场景（含 env）
    R, N = cfg.n_relays, N_ELEM
    prm = relay_params(cfg)
    v = H0[:R * N, 0].reshape(R, N)                                     # P_feed = 1 W 时各接收单元的 v_n
    amp = 10.0 ** (-prm.loss_rx_db / 20.0) / np.sqrt(N)                # w 的幅度（relay.weights）
    bits, u0 = _best_codebook(v, amp, prm.phase_err_rx)
    p_focus = float(np.mean(np.abs(u0) ** 2))                           # N_r>1：各中继 |u0|² 平均
    out = {"H0": H0, "focus_bits_rx": bits, "u0_focus_per_w": u0, "p_focus_per_w": p_focus,
           "cal_ref": cfg.cal_ref, "cal_bits_rx": bits, "cal_null_depth_db": 0.0}
    if cfg.cal_ref == "focus":
        p_ref = p_focus
    else:
        if v_samples is None:
            raise ValueError(f"cal_ref={cfg.cal_ref!r} 需要样本响应（请经 PhysicalSystem 构造）")
        vs = np.asarray(v_samples, dtype=np.complex128).reshape(-1, R, N)
        err = 0.0 if prm.phase_err_rx is None else prm.phase_err_rx
        if cfg.cal_ref == "nulled":
            nb, u0n = null_codebook(v, amp, prm.phase_err_rx, target_db=cfg.null_target_db, ref_abs=np.abs(u0))
            wn = amp * np.exp(1j * (bits_to_phase(nb, 2) + err))        # (R,N)
            u_s = np.sum(wn[None] * vs, axis=-1)                        # (B,R) 开环 u_in / √P_feed
            p_ref = float(np.median(np.mean(np.abs(u_s) ** 2, axis=-1)))
        else:                                                           # "object"：物体散射聚焦码本
            dv = vs - v[None]                                           # (B,R,N) 物体引起的接收单元场
            best = [_best_codebook(dv[b], amp, prm.phase_err_rx) for b in range(dv.shape[0])]
            p_b = np.array([float(np.mean(np.abs(sb) ** 2)) for _, sb in best])
            kb = int(np.argmax(p_b))
            nb = best[kb][0]
            p_ref = float(p_b[kb])
            u0n = np.sum(amp * np.exp(1j * (bits_to_phase(nb, 2) + err)) * v, axis=-1)
        if not p_ref > 0.0:
            raise ValueError(f"cal_ref={cfg.cal_ref!r}：参考功率为 0（任务没有物体？）")
        ratio = np.abs(u0n) ** 2 / np.abs(u0) ** 2                       # 各中继：参考码本的偏置 vs 聚焦
        out.update(cal_bits_rx=nb, cal_null_depth_db=float(10.0 * np.log10(max(float(np.max(ratio)), 1e-300))))
    p1db = rapp_pa(cfg).p1db_in_w()
    backoff = effective_backoff_db(cfg)
    p_feed = p1db * 10.0 ** (backoff / 10.0) / p_ref
    out.update(p_ref_per_w=p_ref, p_feed_w=p_feed, p1db_in_w=p1db, backoff_db=backoff)
    return out


def reference_feed_power(cfg: SystemConfig) -> tuple[float, float]:
    """INTEGRATION B：返回 (P_feed (W), p_ref_per_w = P_feed=1 W 时聚焦码本的开环 PA 输入功率)。

    只支持 cal_ref="focus"（与任务无关）；"nulled"/"object" 依赖样本，见 PhysicalSystem.p_feed_w。
    """
    cal = _calibrate(cfg)
    return cal["p_feed_w"], cal["p_ref_per_w"]


# ============================================================ M0 批量约化求值
def compress_reduced(rs: ReducedSample) -> ReducedSample:
    """born/intra 且 n_p>0 的 ReducedSample → 物体维压缩到 m = N_P + N_RCV 的等价表示（精确代数重写，DESIGN §2.3）。

    eval_reduced 只经 Cpo·pO 与 Hro·pO 用到物体偶极矩，pO = Lam(Eo + Cop·pP_bg)。令 T = [Cpo; Hro]·Lam (m×N_O)，则
      Eo′ = T·Eo，Cop′ = T·Cop，Lam′ = I_m，Cpo′ = [I_{N_P}, 0]，Hro′ = [0, I_{N_RCV}]
    给出逐位相同的公式结构（pO′ = [Cpo·pO; Hro·pO]）。水球 λ/28 体素 N_O ~ 10³，压缩后 m ≈ 50，
    torch.stack_reduced 不必零填充 (B,N_O,N_O) 的稠密 Lam；DWBA 下对物体存在向量 x 的仿射性不变（T·Eo = Σ_j 槽 j 的贡献）。
    """
    if rs.n_p == 0 or rs.mode == "full" or rs.Lam is None:
        return rs
    T = np.vstack([rs.Cpo, rs.Hro]) @ rs.Lam
    n_p = rs.n_p
    n_rcv = rs.H.shape[0]
    m = n_p + n_rcv
    return ReducedSample(mode=rs.mode, n_rx=rs.n_rx, n_probe=rs.n_probe, n_tx=rs.n_tx, n_p=n_p, H=rs.H,
                         alpha_p_base=rs.alpha_p_base, Cpp=rs.Cpp, Ep=rs.Ep, Hrp=rs.Hrp,
                         Eo=T @ rs.Eo, Cop=T @ rs.Cop,
                         Cpo=np.hstack([np.eye(n_p), np.zeros((n_p, n_rcv))]).astype(np.complex128),
                         Lam=np.eye(m, dtype=np.complex128),
                         Hro=np.hstack([np.zeros((n_rcv, n_p)), np.eye(n_rcv)]).astype(np.complex128))


class _PassiveBatch:
    """M0 的批量 eval_reduced（与 solver.eval_reduced 同公式，样本维向量化；born/intra 需先 compress_reduced）。

    M⁻¹·rhs 按 solver 的写法 (I − diag(α)Cpp)⁻¹(α⊙rhs)。born/intra 的 Cpp 只含环境、各样本相同 → 只分解一次。
    """

    def __init__(self, rss: list[ReducedSample]):
        r0 = rss[0]
        self.mode = r0.mode
        self.n_p = r0.n_p
        self.n_rx = r0.n_rx
        self.H = np.stack([r.H for r in rss])
        self.alpha = np.asarray(r0.alpha_p_base, dtype=np.complex128)
        self.Cpp = np.stack([r.Cpp for r in rss])
        self.Ep = np.stack([r.Ep for r in rss])
        self.Hrp = np.stack([r.Hrp for r in rss])
        self.shared = bool(all(np.array_equal(r.Cpp, r0.Cpp) for r in rss))
        if self.mode != "full":
            if any(r.Lam.shape[0] != self.n_p + r.H.shape[0] for r in rss):
                raise ValueError("born/intra 样本须先经 compress_reduced")
            self.Qe = np.stack([r.Eo for r in rss])         # [Cpo; Hro]·Lam·Eo
            self.Qc = np.stack([r.Cop for r in rss])        # [Cpo; Hro]·Lam·Cop

    def excitation(self) -> np.ndarray:
        """无源单元处的激励场（P_feed=1 W，不含无源面再辐射）(B,N_P)：full 为穿衣 Ep；born/intra 再加物体散射。"""
        e = self.Ep[:, :, 0].copy()
        if self.mode != "full":
            e = e + self.Qe[:, :self.n_p, 0]
        return e

    def eval(self, phase: np.ndarray, idx: np.ndarray) -> np.ndarray:
        a = self.alpha * np.exp(1j * np.asarray(phase, dtype=float))
        n_p = self.n_p
        Ep = self.Ep[idx]
        if self.shared:
            lu = lu_factor(np.eye(n_p) - a[:, None] * self.Cpp[0], check_finite=False)

            def solve(rhs):                                  # rhs (B,N_P,S')
                B, _, s = rhs.shape
                x = lu_solve(lu, (a[:, None] * rhs.transpose(1, 0, 2).reshape(n_p, B * s)), check_finite=False)
                return x.reshape(n_p, B, s).transpose(1, 0, 2)
        else:
            A = np.eye(n_p) - a[None, :, None] * self.Cpp[idx]

            def solve(rhs):
                return np.linalg.solve(A, a[None, :, None] * rhs)
        p_bg = solve(Ep)
        if self.mode == "full":
            return self.H[idx] + self.Hrp[idx] @ p_bg
        q = self.Qe[idx] + self.Qc[idx] @ p_bg                 # [Cpo·pO; Hro·pO]
        p_p = p_bg + solve(q[:, :n_p])
        return self.H[idx] + self.Hrp[idx] @ p_p + q[:, n_p:]


# ============================================================ 测量
@dataclass
class Measurement:
    """一次（批量）测量（DESIGN §7.2）。中继量在 M0 下为 None。"""

    readings_dbm: np.ndarray            # (B,K)
    p_w: np.ndarray                     # (B,K) 探头功率 |z|² (W)
    z: np.ndarray                       # (B,K) 探头复场 (√W)
    u: np.ndarray | None                # (B,R) 闭环 PA 输入
    g: np.ndarray | None                # (B,R) PA 输出
    u_in: np.ndarray | None             # (B,R) 开环 PA 输入（不含噪声）
    K: np.ndarray | None                # (B,R,R) 环路耦合
    flags: np.ndarray                   # (B,) possibly_oscillating
    loop_gain_op: np.ndarray            # (B,)
    loop_gain_ss: np.ndarray            # (B,)
    converged: np.ndarray               # (B,)
    sample_idx: np.ndarray              # (B,) 样本编号
    phases: dict = field(default_factory=dict)
    noisy: bool = False
    # 审查 S0-0 / S2-2 增补（M0 下环路量为 0、d / C 为 None）
    loop_gain_robust: np.ndarray | None = None   # (B,) 相位稳健工作点环路增益（relay.loop_gain_robust）
    ss_unstable: np.ndarray | None = None        # (B,) bool：小信号 ρ(G_v K) ≥ 1（源关闭即起振）
    robust_unstable: np.ndarray | None = None    # (B,) bool：ρ_rob ≥ 1（某泄漏相位 / 带内频率可能起振）
    d: np.ndarray | None = None                  # (B,K) 直达支路探头场 s_b·H[探头, 馈源]（方案 §2 的 d_k，无噪声）
    C: np.ndarray | None = None                  # (B,K,R) 发射阵 → 探头复增益（方案 §2 的 c_k）


def _softplus(x):
    return np.logaddexp(0.0, x)


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": float(x.min()), "mean": float(x.mean()), "max": float(x.max())}


def _f(x):
    """JSON 友好的 float（None / 非有限 → None）。"""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _walsh_rel_var(values, n: int) -> float:
    """非常数 Walsh 能量 / 常数项能量 Σ_{S≠∅}|f̂(S)|² / |f̂(∅)|²（向量值对分量求和）：f 对 x 的相对依赖强度。"""
    c = walsh_coeffs(values, n).reshape(2 ** n, -1)
    pw = np.sum(np.abs(c) ** 2, axis=1)
    return float(pw[1:].sum() / pw[0]) if pw[0] > 0 else float("inf")


# ============================================================ C. 物理系统
class PhysicalSystem:
    """条件 M0–M4 下的完整物理系统（DESIGN §7.2，INTEGRATION C）。

    参数（levels / params_int）：中继条件 = 2 × (50·N_r)，顺序为中继 0 的 rx 25、tx 25，中继 1 的 rx 25、tx 25……；
    M0 = passive_levels × (50·N_r)（无源单元顺序同上）。
    calib_rng：构造期随机量（M1b 的 16 个随机码本、logdet 标定码本）的 Generator；None → default_rng(0)（可复现）。
    """

    def __init__(self, cfg: SystemConfig, task: Task, calib_rng: np.random.Generator | None = None, *,
                 _reuse: list | None = None):
        if not isinstance(cfg, SystemConfig):
            raise TypeError("cfg 必须是 SystemConfig")
        t0 = time.perf_counter()
        self.cfg = cfg
        self.task = task
        self.label = condition_label(cfg)
        self.is_passive = cfg.condition == "M0"
        self.k, self.lam = wavenumber(cfg.freq_hz), wavelength(cfg.freq_hz)
        self.n_relays = int(cfg.n_relays)
        self.n_elem = N_ELEM
        self.n_probes = int(task.n_probes)
        self.params = relay_params(cfg)
        self.rapp = rapp_pa(cfg)
        self.p1db_in_w = self.rapp.p1db_in_w()
        self.scene, self.info = build_scene(cfg, self.n_probes)
        self.H0 = empty_relay_response(cfg, self.n_probes)          # 中继版本空场景（标定 / 零陷 / 发射列基准）
        self.object_model = getattr(task, "model", None)
        if cfg.object_mode == "born" and self.object_model == "scalar_rr":
            warnings.warn("object_mode='born' 的任务用了 scalar_rr 极化率：体素互不耦合时其局域场分母不被抵消，"
                          "高对比度材料随体素尺寸不收敛（水在自动体素 λ/28 处约 +10 dB 伪影，审查 R2-INT-1）。"
                          f"请用 make_task / make_system（自动选 {BORN_MODEL!r}）或显式 model={BORN_MODEL!r}。",
                          RuntimeWarning, stacklevel=2)
        self.labels = np.asarray(task.labels)
        self.B = len(task.samples)
        self.power_scale = 10.0 ** (task.power_db() / 10.0)         # 每样本功率缩放（INTEGRATION C）
        self._has_objects = np.array([len(s.objects) > 0 for s in task.samples], dtype=bool)
        # ---- 预计算（每样本一次，与源功率无关）
        if _reuse is not None:
            self._rs = _reuse                                   # 共享（只读）缓存（_scene_key 相同 ⇒ 已按同一规则处理）
        else:
            self._rs = [self._reduce(s.objects) for s in task.samples]
        if self.is_passive:
            self._pb = _PassiveBatch(self._rs)
            self.n_rx = 0
        else:
            self._H = np.stack([rs.H for rs in self._rs])          # (B,N_RCV,S)
            self.n_rx = self.n_relays * N_ELEM
        self.reduce_time_s = time.perf_counter() - t0
        # ---- B. 源功率标定（INTEGRATION B；cal_ref 见 SystemConfig，审查 INT-SEM-4）
        cal = _calibrate(cfg, self.n_probes, self._cal_samples() if cfg.cal_ref != "focus" else None, self.H0)
        self.focus_bits_rx = cal["focus_bits_rx"]
        self.cal_bits_rx = cal["cal_bits_rx"]
        self.cal_null_depth_db = cal["cal_null_depth_db"]
        self.backoff_db = cal["backoff_db"]
        self.p_ref_per_w = cal["p_ref_per_w"]
        self.p_focus_per_w = cal["p_focus_per_w"]
        self.p_feed_w = cal["p_feed_w"]
        self.s_feed = float(np.sqrt(self.p_feed_w))
        self.u0_focus_abs2 = self.p_feed_w * self.p_focus_per_w   # 名义 P_feed 下聚焦 |u0|²（零陷归一化）
        self.s_b = self.s_feed * np.sqrt(self.power_scale)
        # ---- 可训练参数
        self.n_params = 2 * N_ELEM * self.n_relays
        lv = cfg.passive_levels if self.is_passive else cfg.relay_levels
        self.levels = np.full(self.n_params, int(lv), dtype=np.int64)
        # ---- PA
        self.pa_noise_w = pa_noise_power_w(self.params) if cfg.pa_noise else 0.0
        self._calib_rng = calib_rng
        self.bussgang_gain = None
        if self.is_passive:
            self.pa = None
        elif self.label == "M1":
            self.pa = LinearPA(cfg.pa_gain_db)
        elif self.label == "M1b":
            self.bussgang_gain = self._bussgang_gain(cfg.n_bussgang_codebooks)
            self.pa = ComplexLinearPA(self.bussgang_gain)
        else:
            self.pa = self.rapp
        # ---- 检波器（INTEGRATION C）
        sigma = DETECTOR_SIGMA_DB[cfg.detector] if cfg.detector_sigma_db is None else float(cfg.detector_sigma_db)
        self.detector = Detector(cfg.detector, sigma, probe_noise_dbm=cfg.probe_noise_dbm)
        if cfg.detector == "logdet":                 # 随机码本 + 全部样本的无噪声功率标定一次后固定
            ref = self.forward(self.phases_from_params(self.random_params(self._crng())), noisy=False)
            self.detector.calibrate(ref.p_w)
        self.build_time_s = time.perf_counter() - t0

    # ---------------------------------------------------------------- 构造辅助
    def _crng(self) -> np.random.Generator:
        if self._calib_rng is None:
            self._calib_rng = np.random.default_rng(0)
        return self._calib_rng

    def _reduce(self, objects: NodeSet) -> ReducedSample:
        rs = reduce_sample(self.scene, objects, self.cfg.object_mode)
        if self.is_passive:
            return compress_reduced(rs)
        if not self.cfg.objects_see_relay_tx:
            # 审查 INT-PHYS-1：发射列换成空场景发射列 → 物体不被中继发射场照射，K、C 与 x 无关
            H = rs.H.copy()
            H[:, 1:] = self.H0[:, 1:]
            rs = dataclasses.replace(rs, H=H)
        return rs

    def _cal_samples(self) -> np.ndarray:
        """标定用各样本馈源列接收单元场 (B,R,N)（P_feed=1 W，中继版本场景）。M0 需另解中继版本场景。"""
        R, N = self.n_relays, N_ELEM
        if not self.is_passive:
            return self._H[:, :R * N, 0].reshape(self.B, R, N)
        scene_r, _ = build_scene(self.cfg, self.n_probes, relay=True)
        return np.stack([solve_scene(scene_r, s.objects, None, self.cfg.object_mode)[:R * N, 0]
                         for s in self.task.samples]).reshape(self.B, R, N)

    def _bussgang_gain(self, n_codebooks: int) -> complex:
        """INTEGRATION B：G_B = Σ g(u)·conj(u) / Σ|u|²，u = 同源功率、同几何、随机码本 × 全部样本下 Rapp 中继的
        不动点 PA 输入（无噪声；被标记可能自激的样本不计入）。全部被标记时退回小信号增益。"""
        rng = self._crng()
        num, den = 0j, 0.0
        for _ in range(int(n_codebooks)):
            ph = self.phases_from_params(rng.integers(0, 2, size=self.n_params))
            w, ct = weights(ph["phase_rx"], ph["phase_tx"], self.params)
            lin = relay_linear(self._H, w, ct, self.s_b, self.params)
            fp = solve_fixed_point(lin.u_in, lin.K, self.rapp, criterion=self.cfg.osc_criterion)
            ok = ~fp.possibly_oscillating
            num += complex(np.sum(fp.g[ok] * np.conj(fp.u[ok])))
            den += float(np.sum(np.abs(fp.u[ok]) ** 2))
        self.bussgang_valid = den > 0
        return num / den if den > 0 else complex(self.rapp.gain_v)

    def derive(self, calib_rng: np.random.Generator | None = None, **changes) -> "PhysicalSystem":
        """换条件 / PA / 读出等参数的新系统（审查 INT-SEM-2 / INT-NUM-2）。

        - 改变决定物体的字段（ROI 几何：preset/fold_deg/n_relays/relay_spacing_z/roi_offset；freq_hz；
          任务取自 cfg 的 material/voxel_d）时：make_task 构造的任务按其记录的 kind/kwargs 用新 cfg 重建
          （抖动 rng 用构造时的状态副本，样本抖动与原任务相同）；手工构造的任务无法重建 → ValueError。
        - 场景键与任务都不变时复用已缓存的样本响应。
        """
        cfg2 = dataclasses.replace(self.cfg, **changes)
        task = self.task
        if _task_inputs(cfg2, task) != _task_inputs(self.cfg, task):
            spec = getattr(task, "make_spec", None)
            if spec is None:
                raise ValueError("derive 不能改变物体几何/材料（任务不是 make_task 构造的，无法重建）；"
                                 "请用 make_system / make_task 重建任务")
            task = make_task(cfg2, spec["kind"], **_copy_task_kwargs(spec["kwargs"]))
        reuse = self._rs if (task is self.task and _scene_key(cfg2) == _scene_key(self.cfg)) else None
        return PhysicalSystem(cfg2, task, calib_rng, _reuse=reuse)

    # ---------------------------------------------------------------- 参数
    def random_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self.levels).astype(np.int64)

    def phases_from_params(self, params_int) -> dict:
        """整数电平 → 相位字典：中继 {'phase_rx','phase_tx'} (R,25)；M0 {'phase_p'} (50R,)。1-bit π·b，2-bit (π/2)·l。"""
        x = np.asarray(params_int).astype(np.int64).ravel()
        if x.size != self.n_params:
            raise ValueError(f"参数长度应为 {self.n_params}，得到 {x.size}")
        if np.any(x < 0) or np.any(x >= self.levels):
            raise ValueError("参数电平越界")
        if self.is_passive:
            return {"phase_p": bits_to_phase(x, int(self.levels[0]))}
        L = int(self.levels[0])
        x = x.reshape(self.n_relays, 2, N_ELEM)
        return {"phase_rx": bits_to_phase(x[:, 0], L), "phase_tx": bits_to_phase(x[:, 1], L)}

    def focus_params(self, probe: int = 0) -> np.ndarray:
        """量化共轭聚焦码本（空场景）：中继 = 接收对准馈源（标定码本）+ 发射对准探头 probe（只看中继路径 |C_k|）；
        M0 = 无源单元对"馈源→单元→探头"一阶路径相位的量化共轭，偏移按含直达项的总幅度选。"""
        if self.is_passive:
            # 一阶反射系数 v_n = Hrp[k,n]·α_n·Ep[n]；偏移按"直达项 + 反射和"的总幅度选（L90 中馈源→探头视距项
            # 比最佳相干反射强约 6 dB，只看反射和会选到与直达项反相的偏移）
            rs = compress_reduced(reduce_sample(self.scene, None, self.cfg.object_mode))
            v = (rs.Hrp[probe, :] * rs.Ep[:, 0] * rs.alpha_p_base)[None, :]
            bits, _ = _best_codebook(v, 1.0, None, int(self.levels[0]), ref=rs.H[probe, 0])
            return bits.ravel().astype(np.int64)
        R = self.n_relays
        c = self.H0[self.n_rx + probe, 1:].reshape(R, N_ELEM)          # 发射单元 → 探头
        amp = 10.0 ** (-self.params.loss_tx_db / 20.0) / np.sqrt(N_ELEM)
        L = int(self.levels[0])
        tx_bits, _ = _best_codebook(c, amp, self.params.phase_err_tx, L)
        # 标定用的接收聚焦码本是 1-bit（相位 0/π）；多电平时换成同相位的电平号（L 为偶数时 π ↔ L/2）
        rx = np.mod(np.rint(bits_to_phase(self.focus_bits_rx, 2) / (2.0 * np.pi / L)), L).astype(np.int64)
        return np.stack([rx, tx_bits], axis=1).ravel().astype(np.int64)

    # ---------------------------------------------------------------- 前向
    def _phase_arrays(self, phases: dict) -> dict:
        return {k: np.asarray(v.detach().numpy() if hasattr(v, "detach") else v, dtype=float)
                for k, v in phases.items()}

    def forward(self, phases: dict, rng: np.random.Generator | None = None, noisy: bool = True,
                sample_idx=None) -> Measurement:
        """INTEGRATION C 的前向。noisy=True：PA 输入噪声 + 探头加性噪声 + 检波器读数噪声（需 rng）；
        noisy=False：确定性，读数为理想 dBm（不加 logdet 的 offset/clip，与 TorchSystem 相同）。
        sample_idx：样本编号（可重复，用于一次批量生成多次独立噪声测量）。"""
        if noisy and rng is None:
            raise ValueError("noisy=True 需要显式 rng")
        idx = np.arange(self.B) if sample_idx is None else np.asarray(sample_idx, dtype=np.int64).ravel()
        ph = self._phase_arrays(phases)
        s_b = self.s_b[idx]
        Bn = idx.size
        if self.is_passive:
            Hb = self._pb.eval(ph["phase_p"], idx)
            z = s_b[:, None] * Hb[:, :, 0]                             # z = s_b·H[探头, 馈源]（M0 无 rx 行）
            zeros = np.zeros(Bn)
            u = g = u_in = K = d = C = None
            flags = np.zeros(Bn, dtype=bool)
            lg_op, lg_ss, lg_rob, conv = zeros, zeros.copy(), zeros.copy(), np.ones(Bn, dtype=bool)
        else:
            w, ct = weights(ph["phase_rx"], ph["phase_tx"], self.params)
            lin = relay_linear(self._H[idx], w, ct, s_b, self.params)
            noise = (complex_awgn(rng, lin.u_in.shape, self.pa_noise_w)
                     if noisy and self.pa_noise_w > 0 else None)
            fp = solve_fixed_point(lin.u_in, lin.K, self.pa, noise=noise, criterion=self.cfg.osc_criterion)
            z = probe_fields(lin.d, lin.C, fp.g)
            u, g, u_in, K, d, C = fp.u, fp.g, lin.u_in, lin.K, lin.d, lin.C
            flags = np.asarray(fp.possibly_oscillating, dtype=bool)
            lg_op, lg_ss = np.asarray(fp.loop_gain_op, float), np.asarray(fp.loop_gain_ss, float)
            lg_rob = np.asarray(fp.loop_gain_robust, float)
            conv = np.asarray(fp.converged, dtype=bool)
        if noisy:
            pn = self.detector.probe_noise(z.shape, rng)
            if pn is not None:
                z = z + pn
        p_w = np.abs(z) ** 2
        readings = self.detector.read(p_w, rng) if noisy else w2dbm(p_w)
        return Measurement(readings_dbm=readings, p_w=p_w, z=z, u=u, g=g, u_in=u_in, K=K, flags=flags,
                           loop_gain_op=lg_op, loop_gain_ss=lg_ss, converged=conv, sample_idx=idx, phases=ph,
                           noisy=bool(noisy), loop_gain_robust=lg_rob, ss_unstable=gain_at_least_one(lg_ss),
                           robust_unstable=gain_at_least_one(lg_rob), d=d, C=C)

    def measure(self, params_int, rng: np.random.Generator) -> Measurement:
        """= forward(phases_from_params(params), rng, noisy=True)（DESIGN §7.2）。"""
        return self.forward(self.phases_from_params(params_int), rng, noisy=True)

    def direct_field(self, sample_idx=None) -> np.ndarray:
        """不经中继 / 无源面的探头场 (B,K)（方案 §2 的 d_k：直达入射 + 物体散射 + 环境；无噪声，与码本无关；审查 S2-2）。

        中继条件：d = s_b·H[探头, 馈源]（= relay_linear 的 d，也在 Measurement.d 里）；z − d = C·g(u) 即中继支路。
        M0：无源面"关掉"（α_P → 0）时的探头场 s_b·(H + Hro·pO_bg)[探头, 馈源]（full 模式 H 已含物体精确效应；
        born/intra 加上物体对无无源面背景场的响应），z − d 即无源面的全部贡献（含面 ↔ 物体多次反射）。
        """
        idx = np.arange(self.B) if sample_idx is None else np.asarray(sample_idx, dtype=np.int64).ravel()
        if not self.is_passive:
            return self.s_b[idx, None] * self._H[idx, self.n_rx:, 0]
        pb = self._pb
        h = pb.H[idx][:, :, 0]
        if pb.mode != "full":
            h = h + pb.Qe[idx][:, pb.n_p:, 0]
        return self.s_b[idx, None] * h

    # ---------------------------------------------------------------- 损失
    def null_depth_db(self, phases: dict) -> float | None:
        """空场景、当前接收相位、名义 P_feed 下 10log10(|u0_r|²/u0_focus_abs2) 的最大值（N_r>1 取最差中继）；M0 为 None。"""
        if self.is_passive:
            return None
        ph = self._phase_arrays(phases)
        w, _ = weights(ph["phase_rx"], ph["phase_tx"], self.params)
        h0 = self.H0[:self.n_rx, 0].reshape(self.n_relays, N_ELEM)
        u0 = self.s_feed * np.sum(w * h0, axis=-1)
        nd = 10.0 * np.log10(np.maximum(np.abs(u0) ** 2, 1e-300) / self.u0_focus_abs2)
        return float(np.max(nd))

    def _null_penalty(self, phases: dict) -> float:
        if self.is_passive or self.cfg.bg_cancel != "null":
            return 0.0
        return float(self.cfg.null_weight * _softplus(self.null_depth_db(phases) - self.cfg.null_target_db))

    def _osc_smooth_penalty(self, m: Measurement) -> float:
        """DESIGN §6.3：weight·Σ_b relu(ρ_b − 0.95)²/B，ρ_b 收敛样本取 ρ_op、未收敛样本取 ρ(G_v K)（同 TorchSystem）。"""
        wgt = self.cfg.osc_smooth_weight
        if self.is_passive or wgt == 0.0:
            return 0.0
        rho = np.where(m.converged & np.isfinite(m.loop_gain_op), m.loop_gain_op, m.loop_gain_ss)
        rho = np.nan_to_num(rho, nan=0.0)
        return float(wgt * np.sum(np.maximum(rho - 0.95, 0.0) ** 2) / rho.size)

    def sample_losses(self, m: Measurement) -> np.ndarray:
        """逐样本任务损失（losses.py，reduction='none'）；被标记样本 → chance + osc_penalty（INTEGRATION C）。"""
        labels = self.labels[m.sample_idx]
        per = np.asarray(self.task.loss(m.readings_dbm, labels, self.cfg.tau_db, reduction="none"), dtype=float)
        if m.flags.any():
            ch = losses.chance_loss(self.task.kind, self.n_probes, self.cfg.tau_db, **self.task.task_kwargs)
            base = per if ch is None else np.full_like(per, ch)
            per = np.where(m.flags, base + self.cfg.osc_penalty, per)
        return per

    def loss_from(self, m: Measurement, model_penalties: bool = True) -> float:
        """训练目标（DESIGN §7.2、§6.3；审查 INT-SEM-3）。

        可测目标（model_penalties=False，DSPSA 用，方案 §4.7 "model-free"）：任务损失（被标记样本替换为
        chance + osc_penalty，INTEGRATION C）+ 零陷惩罚（bg_cancel='null'）。
        模型目标（True，默认；oracle_bits / TorchSystem 同定义）：再加光滑自激惩罚 osc_smooth_weight·mean relu(ρ−0.95)²
        ——ρ 是模型内部的工作点雅可比 / 小信号环路谱半径，探头读数测不到，不能交给 DSPSA。
        """
        total = float(np.mean(self.sample_losses(m)))
        if not self.is_passive:
            total += self._null_penalty(m.phases)
            if model_penalties:
                total += self._osc_smooth_penalty(m)
        return total

    def loss(self, params_int, rng: np.random.Generator | None = None, noisy: bool = True,
             model_penalties: bool = True) -> float:
        return self.loss_from(self.forward(self.phases_from_params(params_int), rng, noisy=noisy), model_penalties)

    def loss_phases(self, phases: dict, rng: np.random.Generator | None = None, noisy: bool = False,
                    model_penalties: bool = True) -> float:
        return self.loss_from(self.forward(phases, rng, noisy=noisy), model_penalties)

    def clean_readings(self, m: Measurement) -> np.ndarray:
        """无噪声的检波器读数（审查 INT-SEM-5）：logdet 时 Detector.read(p_w, None)（offset + 量程 clip，无读数噪声）；
        ideal / vna 即理想 dBm。forward(noisy=False) 的理想 dBm 只用于梯度 / oracle 路径（与 TorchSystem 相同）。"""
        if self.cfg.detector == "logdet":
            return self.detector.read(m.p_w, None)
        return m.readings_dbm

    def loss_meas_clean(self, m: Measurement) -> float:
        """无噪声的可测目标：检波器读数（logdet 含 offset + clip）上的 loss_from(model_penalties=False)。
        = DSPSA 训练目标去掉噪声；实验 E 比较 DSPSA 与 oracle 时两边都取它（同一目标）。"""
        if self.cfg.detector == "logdet":
            m = dataclasses.replace(m, readings_dbm=self.clean_readings(m))
        return self.loss_from(m, model_penalties=False)

    # ---------------------------------------------------------------- 评估
    def _accuracy(self, readings: np.ndarray, flags: np.ndarray, labels: np.ndarray) -> float:
        """被标记（可能自激）的样本一律算错（读数不可信）；判决平局（logdet 钳位使 y1 == y2 等）也算错
        （task.decide 记 −1，审查 R2-INT-2）。"""
        if self.task.kind == "focus":
            return float("nan")
        pred = np.asarray(self.task.decide(readings, labels)).copy()
        pred[np.asarray(flags, dtype=bool)] = -1
        return float(np.mean(pred == labels))

    def small_signal_gain(self) -> complex:
        """环路诊断用的小信号（复）增益：Rapp/M1 为 G_v，M1b 为 Bussgang 复增益。"""
        return complex(self.pa.gain_v) if self.pa is not None else 0j

    def loop_diagnostics(self, m: Measurement) -> dict:
        """INTEGRATION G2：谱半径、Gershgorin 上界 max_r Σ_s|G K_rs|（≥ ρ）与 |det(I − G K)|（→0 即谐振）；
        另报相位稳健环路增益与三种自激判据下的样本计数（审查 S0-0，relay.OSC_CRITERIA）：
          n_ss_unstable / ss_unstable_run：ρ(G_v K) ≥ 1（源关闭即起振；最保守）；
          n_robust_unstable / robust_unstable_run：ρ_rob ≥ 1（某泄漏相位 / 带内频率上可能起振）；
          n_flagged / oscillating_run（evaluate_phases）按 cfg.osc_criterion。"""
        keys = ("loop_gain_ss_max", "loop_gain_op_max", "loop_gain_robust_max", "gershgorin_max", "det_min",
                "n_ss_unstable", "ss_unstable_run", "n_robust_unstable", "robust_unstable_run")
        if self.is_passive:
            return {k: None for k in keys}
        GK = self.small_signal_gain() * m.K
        gersh = np.max(np.sum(np.abs(GK), axis=-1), axis=-1)
        det = np.abs(np.linalg.det(np.eye(self.n_relays) - GK))
        ss, rob = np.asarray(m.ss_unstable, dtype=bool), np.asarray(m.robust_unstable, dtype=bool)
        return {"loop_gain_ss_max": _f(np.nanmax(m.loop_gain_ss)), "loop_gain_op_max": _f(np.nanmax(m.loop_gain_op)),
                "loop_gain_robust_max": _f(np.nanmax(m.loop_gain_robust)),
                "gershgorin_max": _f(np.max(gersh)), "det_min": _f(np.min(det)),
                "n_ss_unstable": int(ss.sum()), "ss_unstable_run": bool(ss.any()),
                "n_robust_unstable": int(rob.sum()), "robust_unstable_run": bool(rob.any())}

    def evaluate(self, params_int, rng: np.random.Generator, n_noise: int = 20, thresholds_db=None) -> dict:
        out = self.evaluate_phases(self.phases_from_params(params_int), rng, n_noise, thresholds_db=thresholds_db)
        out["params"] = [int(v) for v in np.asarray(params_int).ravel()]
        return out

    def calibrate_thresholds(self, params_int=None, *, phases: dict | None = None) -> np.ndarray:
        """loc3bit 的判决阈值（审查 S2-1）：本系统（训练集）全部样本在给定码本下的无噪声检波器读数
        （clean_readings，logdet 含 offset + 量程钳位），按自动规则 t_j = 两类中位数中点（losses.loc3bit_thresholds）。
        与任务当前的 thresholds_db 设置无关。返回 (n_probes,)，传给留出集的 evaluate(..., thresholds_db=t)。"""
        if self.task.kind != "loc3bit":
            raise ValueError("calibrate_thresholds 只适用于 loc3bit 任务")
        if phases is None:
            phases = self.phases_from_params(params_int)
        m = self.forward(phases, noisy=False)
        return np.asarray(losses.loc3bit_thresholds(self.clean_readings(m), self.labels), dtype=float)

    def evaluate_phases(self, phases: dict, rng: np.random.Generator, n_noise: int = 20, thresholds_db=None) -> dict:
        """见 _evaluate_phases。thresholds_db（仅 loc3bit，审查 S2-1）：评估期间固定使用的判决阈值 (n_probes,)，
        例如训练集上 calibrate_thresholds 的结果——否则 "auto" 阈值由被评估样本自身的标签求出，在留出集上会泄漏
        测试标签。评估结束后恢复任务原设置；输出 thresholds_db（实际使用的阈值）与 thresholds_source。"""
        task = self.task
        if thresholds_db is None:
            out = self._evaluate_phases(phases, rng, n_noise)
            if task.kind == "loc3bit":
                out["thresholds_source"] = "auto" if isinstance(task.thresholds_db, str) else "task"
            return out
        if task.kind != "loc3bit":
            raise ValueError("thresholds_db 只适用于 loc3bit 任务")
        t = np.asarray(thresholds_db, dtype=float).ravel()
        if t.size != self.n_probes or not np.all(np.isfinite(t)):
            raise ValueError(f"thresholds_db 应为 {self.n_probes} 个有限值，得到 {thresholds_db!r}")
        old = task.thresholds_db
        task.thresholds_db = t
        try:
            out = self._evaluate_phases(phases, rng, n_noise)
        finally:
            task.thresholds_db = old
        out["thresholds_source"] = "given"
        return out

    def _evaluate_phases(self, phases: dict, rng: np.random.Generator, n_noise: int = 20) -> dict:
        """INTEGRATION C：无噪声准确率/裕量 + 有噪声平均准确率 + 自激标记 + 工作点 + G2 环路诊断。

        - 无噪声读数 = 检波器读数（clean_readings；logdet 含 offset + 量程 clip，审查 INT-SEM-5）；logdet 另报
          det_clipped_frac 与理想 dBm 下的 acc_clean_ideal / margin_db_ideal。
        - 被标记可能自激的样本一律算错（方案 §4.5）；判决平局（logdet 钳位 y1 == y2 等）算错（审查 R2-INT-2）。
          all_correct = (acc == 1) 且 margin > 0。margin_db 与"标记 = 错"一致：有标记时为 None
          （契约 margin > 0 ⇔ 全对）；margin_db_unflagged 只在未标记样本上取最小（审查 INT-NUM-3），判决规则
          （loc3bit 的自动阈值）与 predict 一样由全部样本决定（审查 R2-INT-3）。
        - *_valid（acc_clean_valid / all_correct_valid / acc_noisy_valid）：自激 run 为 None，
          experiments.common.aggregate 自动剔除 → "单独统计、不参与精度均值"（方案 §4.5，审查 INT-SEM-6）。
        - probe_dbm、pa_in_rel_p1db_db（闭环 |u|²）只统计未标记样本（被标记样本是最后迭代值/近谐振解，无物理意义）；
          pa_in_open_rel_p1db_db（开环 |u_in|²，与环路无关）统计全部样本；pa_in_delta_rel_p1db_db = 物体引起的
          开环驱动 |u_in(x) − u0|²（有物体的样本，审查 INT-SEM-4）。
        - loss_clean = 模型目标（= loss(noisy=False)，oracle 同定义）；loss_meas_clean = 可测目标（DSPSA 同定义）。
        """
        cfg, task = self.cfg, self.task
        clean = self.forward(phases, noisy=False)
        labels = self.labels
        flags = np.asarray(clean.flags, dtype=bool)
        ok = ~flags
        osc = bool(flags.any())
        y = self.clean_readings(clean)
        acc = self._accuracy(y, flags, labels)
        margin_unf = (_f(task.margin_db(y, labels, mask=ok)) if ok.any() and np.all(np.isfinite(y[ok])) else None)
        all_ok = bool(acc == 1.0 and margin_unf is not None and margin_unf > 0.0) if np.isfinite(acc) else None
        out = {"condition": self.label, "preset": cfg.preset, "object_mode": cfg.object_mode, "task": task.name,
               "n_samples": self.B, "material": None if getattr(task, "material", None) is None else str(task.material),
               "object_model": self.object_model,
               "voxel_d": _f(getattr(task, "voxel_d", None)),
               "effective_eps": (eps_repr(task.material, getattr(task, "model", DEFAULT_MODEL))
                                 if getattr(task, "material", None) is not None else None),
               "objects_see_relay_tx": bool(cfg.objects_see_relay_tx),
               "p_feed_dbm": float(w2dbm(self.p_feed_w)), "backoff_db": self.backoff_db, "cal_ref": cfg.cal_ref,
               "cal_null_depth_db": _f(self.cal_null_depth_db), "p1db_in_dbm": float(w2dbm(self.p1db_in_w)),
               "acc_clean": _f(acc), "all_correct": all_ok,
               "acc_clean_valid": None if osc else _f(acc), "all_correct_valid": None if osc else all_ok,
               "margin_db": None if osc else margin_unf, "margin_db_unflagged": margin_unf,
               "loss_clean": float(self.loss_from(clean)), "loss_meas_clean": float(self.loss_meas_clean(clean)),
               "oscillating_run": osc, "n_flagged": int(flags.sum()), "osc_criterion": cfg.osc_criterion,
               "n_not_converged": int((~clean.converged).sum()),
               "probe_dbm": _stats(y[ok]), "probe0_dbm_max": _f(np.max(y[ok, 0])) if ok.any() else None,
               "det_clipped_frac": None}
        if cfg.detector == "logdet":
            raw = w2dbm(clean.p_w) + self.detector.offset_db
            lo, hi = self.detector.range_dbm
            out["det_clipped_frac"] = float(np.mean((raw[ok] < lo) | (raw[ok] > hi))) if ok.any() else None
            y_id = clean.readings_dbm
            out["acc_clean_ideal"] = _f(self._accuracy(y_id, flags, labels))
            out["margin_db_ideal"] = (None if osc or not np.all(np.isfinite(y_id))
                                      else _f(task.margin_db(y_id, labels)))
        if task.kind == "loc3bit":
            out["bit_acc_clean"] = [float(v) for v in task.bit_accuracy(y, labels)]
            out["thresholds_db"] = [float(v) for v in task.thresholds(y, labels)]
        # ---- 有噪声（n_noise 次独立测量，一次批量求解）
        if n_noise > 0:
            idx = np.tile(np.arange(self.B), int(n_noise))
            m = self.forward(phases, rng, noisy=True, sample_idx=idx)
            yr = m.readings_dbm.reshape(n_noise, self.B, -1)
            fl = m.flags.reshape(n_noise, self.B)
            accs = np.array([self._accuracy(yr[i], fl[i], labels) for i in range(n_noise)])
            out["acc_noisy"] = _f(np.mean(accs))
            out["acc_noisy_std"] = _f(np.std(accs))
            out["noisy_flagged_frac"] = float(np.mean(fl))
        else:
            out["acc_noisy"] = out["acc_noisy_std"] = None
        out["acc_noisy_valid"] = None if osc else out["acc_noisy"]
        # ---- 中继工作点与环路诊断
        out.update(self.loop_diagnostics(clean))
        if not self.is_passive:
            p1 = self.p1db_in_w
            rel = 10.0 * np.log10(np.maximum(np.abs(clean.u[ok]) ** 2, 1e-300) / p1)
            rel_open = 10.0 * np.log10(np.maximum(np.abs(clean.u_in) ** 2, 1e-300) / p1)
            w, _ = weights(clean.phases["phase_rx"], clean.phases["phase_tx"], self.params)
            u0 = np.sum(w * self.H0[:self.n_rx, 0].reshape(self.n_relays, N_ELEM), axis=-1)       # (R,) / √W
            du = clean.u_in - self.s_b[:, None] * u0[None]
            rel_delta = 10.0 * np.log10(np.maximum(np.abs(du[self._has_objects]) ** 2, 1e-300) / p1)
            stats = {"pa_in_rel_p1db_db": _stats(rel),             # 实际闭环工作点（未标记样本×中继）
                     "pa_in_open_rel_p1db_db": _stats(rel_open),   # 开环 |u_in|²（标定参考的同类量）
                     "pa_in_delta_rel_p1db_db": _stats(rel_delta)}  # 物体引起的开环驱动 |u_in − u0|²
            out.update(stats)
            for key, st in stats.items():
                for kk, vv in st.items():
                    out[f"{key}_{kk}"] = vv
            out["null_depth_db"] = _f(self.null_depth_db(phases))
        else:
            out["pa_in_rel_p1db_db"] = out["pa_in_open_rel_p1db_db"] = out["pa_in_delta_rel_p1db_db"] = None
            out["null_depth_db"] = None
        if self.bussgang_gain is not None:
            out["bussgang_gain_db"] = float(20.0 * np.log10(abs(self.bussgang_gain)))
            out["bussgang_phase_deg"] = float(np.degrees(np.angle(self.bussgang_gain)))
        return out

    def clean_metrics(self, params_int) -> dict:
        """无噪声 loss / 准确率 / 标记数（训练曲线用，一次前向；不计入测量次数）。
        loss_clean = 模型目标，loss_meas_clean = 可测目标（DSPSA 实际优化的目标去掉噪声）；准确率按检波器读数。"""
        m = self.forward(self.phases_from_params(params_int), noisy=False)
        return {"loss_clean": float(self.loss_from(m)), "loss_meas_clean": float(self.loss_meas_clean(m)),
                "acc_clean": _f(self._accuracy(self.clean_readings(m), m.flags, self.labels[m.sample_idx])),
                "n_flagged": int(m.flags.sum())}

    # ---------------------------------------------------------------- 非线性度（方案 §7）
    def nonlinearity_profile(self, phases: dict) -> dict:
        """无噪声下各物理量的 Walsh 高阶权重 W≥2、W≥3（仅 ParityTask；样本 i ↔ x_j = (i>>j)&1）。

        键：e（中继：s·H[:, :N_RX, 0] 接收单元场；M0：无源单元处激励场）、u_in、u、g（中继）、z、y_diff_lin（P1−P2，W）、
        y_diff_db；每项为 {"W2", "W3", "profile"}（profile = 各阶归一化权重）。

        门 G1 的可证范围（审查 INT-PHYS-1）：只有 **M0 + Born** 的探头场对 x 仿射（y 二次，3 位奇偶不可解）。
        中继条件下 Born/intra 的物体处在中继发射波束中，环路 K(x) 与发射→探头 C(x) 对 x 仿射，线性中继的
        u = (I − G K(x))⁻¹u_in(x) 已是 x 的有理函数（方案推论 4），所以 M1 + Born 不保证不可解；"PA 贡献"不能用
        W(g) − W(e) 衡量（会把环路 / 再照射项记到 PA 头上）。中继条件因此另报：
          K、C：Walsh 权重 + rel_var（非常数 Walsh 能量 / 常数项能量，衡量对 x 的依赖强度）；
          *_linref：同一码本、同一 K(x)、C(x)、增益匹配线性 PA（small_signal_gain()）的 u、g、z、y；
          pa_excess_*：W(·) − W(·_linref)，即 PA 非线性本身加入的高阶权重（M1 恒为 0）；linref_flags_any：线性参考谐振。
        SystemConfig.objects_see_relay_tx=False 时 K、C 与 x 无关，恢复方案 §6.1 的"Born + 线性中继 = 二次"。
        """
        if not isinstance(self.task, ParityTask):
            raise TypeError("nonlinearity_profile 只适用于 ParityTask")
        n = self.task.n
        m = self.forward(phases, noisy=False)
        vals = {}
        if self.is_passive:
            vals["e"] = self.s_b[:, None] * self._pb.excitation()
        else:
            vals["e"] = self.s_b[:, None] * self._H[:, :self.n_rx, 0]
            vals["u_in"], vals["u"], vals["g"] = m.u_in, m.u, m.g
        vals["z"] = m.z
        vals["y_diff_lin"] = m.p_w[:, 0] - m.p_w[:, 1]
        vals["y_diff_db"] = m.readings_dbm[:, 0] - m.readings_dbm[:, 1]
        lin_ref = None
        if not self.is_passive:
            w, ct = weights(m.phases["phase_rx"], m.phases["phase_tx"], self.params)
            lin = relay_linear(self._H, w, ct, self.s_b, self.params)
            vals["K"], vals["C"] = lin.K, lin.C
            lin_ref = solve_fixed_point(lin.u_in, lin.K, ComplexLinearPA(self.small_signal_gain()))
            zl = probe_fields(lin.d, lin.C, lin_ref.g)
            pl = np.abs(zl) ** 2
            vals.update(u_linref=lin_ref.u, g_linref=lin_ref.g, z_linref=zl, y_diff_lin_linref=pl[:, 0] - pl[:, 1],
                        y_diff_db_linref=w2dbm(pl[:, 0]) - w2dbm(pl[:, 1]))
        out = {k: {"W2": walsh_weight_ge(v, n, 2), "W3": walsh_weight_ge(v, n, 3),
                   "profile": [float(p) for p in walsh_degree_profile(v, n)]} for k, v in vals.items()}
        if lin_ref is not None:
            for k in ("K", "C"):
                out[k]["rel_var"] = _walsh_rel_var(vals[k], n)
            for k in ("u", "g", "z", "y_diff_lin", "y_diff_db"):
                a, b = out[k], out[f"{k}_linref"]
                out[f"pa_excess_{k}"] = {"W2": a["W2"] - b["W2"], "W3": a["W3"] - b["W3"],
                                         "profile": [p - q for p, q in zip(a["profile"], b["profile"])]}
            out["linref_flags_any"] = bool(np.any(lin_ref.possibly_oscillating))
        out["flags_any"] = bool(m.flags.any())
        return out

    def nonlinearity_profile_random(self, rng: np.random.Generator, n_codebooks: int = 5) -> dict:
        """随机码本上 nonlinearity_profile 的平均 W≥2 / W≥3（及 K、C 的 rel_var）（方案 §7："在随机（未训练）配置上量"）。"""
        acc: dict = {}
        for _ in range(int(n_codebooks)):
            prof = self.nonlinearity_profile(self.phases_from_params(self.random_params(rng)))
            for k, v in prof.items():
                if isinstance(v, dict):
                    acc.setdefault(k, []).append(v)
        return {k: {f: float(np.mean([d[f] for d in v])) for f in ("W2", "W3", "rel_var") if f in v[0]}
                for k, v in acc.items()}

    # ---------------------------------------------------------------- torch
    def pa_dict(self) -> dict:
        """oracle_torch.TorchPA 的构造字典。"""
        cfg = self.cfg
        if self.label in ("M1", "M1b"):
            return {"type": "linear", "gain_db": cfg.pa_gain_db}
        return {"type": "rapp", "gain_db": cfg.pa_gain_db, "psat_out_dbm": cfg.psat_out_dbm, "p": cfg.rapp_p,
                "ampm_max_deg": cfg.ampm_max_deg, "ampm_beta": cfg.ampm_beta, "ampm": cfg.ampm}

    def to_torch(self):
        """打包为 oracle_torch.TorchSystem（INTEGRATION C；构造签名见其源码）。M1b 的复增益用 _ComplexLinearTorchPA 替换。"""
        from . import oracle_torch as ot

        cfg = self.cfg
        null_ref = None
        if not self.is_passive and cfg.bg_cancel == "null":
            null_ref = {"H0": self.H0, "u0_ref_abs2": self.u0_focus_abs2, "target_db": cfg.null_target_db,
                        "weight": cfg.null_weight}
        relay = {"loss_rx_db": cfg.loss_rx_db, "loss_tx_db": cfg.loss_tx_db, "isolation_db": cfg.isolation_db,
                 "kappa0_phase_deg": cfg.kappa0_phase_deg}
        ts = ot.TorchSystem(self._rs, kind="passive" if self.is_passive else "relay", s_feed=self.s_feed,
                            n_relays=self.n_relays, n_elem=N_ELEM, pa=self.pa_dict(), relay=relay,
                            labels=self.labels, task=self.task.kind, tau_db=cfg.tau_db, n_probes=self.n_probes,
                            power_scale=self.power_scale, null_ref=null_ref,
                            osc_penalty_weight=cfg.osc_smooth_weight, osc_flag_penalty=cfg.osc_penalty,
                            task_kwargs=self.task.task_kwargs, osc_criterion=cfg.osc_criterion)
        if self.label == "M1b":
            ts.pa = _complex_linear_torch_pa(ot, self.bussgang_gain)
        return ts

    def describe(self) -> dict:
        """JSON 友好的系统摘要（写入 config.json / runs.jsonl）。"""
        d = {"condition": self.label, "cfg": self.cfg.to_dict(), "task": self.task.describe(),
             "p_feed_dbm": float(w2dbm(self.p_feed_w)), "backoff_db": self.backoff_db,
             "p_ref_per_w_db": float(10.0 * np.log10(self.p_ref_per_w)), "p1db_in_dbm": float(w2dbm(self.p1db_in_w)),
             "cal_ref": self.cfg.cal_ref, "cal_null_depth_db": float(self.cal_null_depth_db),
             "objects_see_relay_tx": bool(self.cfg.objects_see_relay_tx), "object_model": self.object_model,
             "n_params": self.n_params, "build_time_s": self.build_time_s,
             "roi_center": [float(v) for v in self.info["roi_center"]]}
        if self.bussgang_gain is not None:
            d["bussgang_gain_db"] = float(20.0 * np.log10(abs(self.bussgang_gain)))
            d["bussgang_phase_deg"] = float(np.degrees(np.angle(self.bussgang_gain)))
        return d


def _complex_linear_torch_pa(ot, gain_c: complex):
    """TorchPA 的复增益线性版本（M1b）：g = G_c·u，to_numpy() 返回 ComplexLinearPA（同一不动点求解器）。"""

    class _ComplexLinearTorchPA(ot.TorchPA):
        def __init__(self, g):
            super().__init__(kind="linear", gain_db=float(20.0 * np.log10(abs(g))))
            self.gain_c = complex(g)
            self.gain_v = self.gain_c

        def to_numpy(self):
            return ComplexLinearPA(self.gain_c)

    return _ComplexLinearTorchPA(gain_c)


def _scene_key(cfg: SystemConfig) -> tuple:
    """决定样本响应 H 的配置字段（derive 复用缓存的依据）。"""
    env = None if not cfg.env_points else tuple((tuple(map(float, p)), complex(a)) for p, a in cfg.env_points)
    roi = None if cfg.roi_offset is None else tuple(map(float, cfg.roi_offset))
    probe_az = None if cfg.probe_az_deg is None else tuple(map(float, cfg.probe_az_deg))
    return (cfg.preset, cfg.fold_deg, cfg.n_relays, cfg.relay_spacing_z, roi, env, cfg.block_direct,
            cfg.skip_gain_db, cfg.freq_hz, cfg.condition == "M0", cfg.object_mode,
            bool(cfg.objects_see_relay_tx) or cfg.condition == "M0", cfg.feed_az_deg, cfg.feed_dist, probe_az,
            cfg.probe_dist)


_CFG_MATERIAL_KINDS = ("parity", "loc3bit", "loc", "localization")   # make_task 从 cfg 取 material/voxel_d 的任务


def _task_inputs(cfg: SystemConfig, task: Task) -> tuple:
    """决定任务物体（位置 / 材料 / 体素）的 cfg 输入（审查 INT-SEM-2 / INT-NUM-2；derive 据此判断是否重建任务）。

    无物体的任务（focus）→ ()；ROI 中心、频率总在内；material/voxel_d 仅当任务由 make_task 从 cfg 取用
    （parity / loc3bit 且调用方未显式给出）时在内；手工构造的任务保守地全部计入。
    极化率模型（审查 R2-INT-1）：make_task 未显式给 model 时它随 cfg.object_mode 变（born ↔ full/intra），计入；
    手工构造的任务模型固定（born + scalar_rr 时 PhysicalSystem 警告），不计入。
    """
    if task.kind == "focus":
        return ()
    spec = getattr(task, "make_spec", None)
    key = [("roi", tuple(np.round(roi_center(cfg), 12).tolist())), ("freq_hz", float(cfg.freq_hz))]
    kw = {} if spec is None else spec["kwargs"]
    if spec is None or spec["kind"] in _CFG_MATERIAL_KINDS:
        if "material" not in kw:
            key.append(("material", str(cfg.material)))
        if "voxel_d" not in kw:
            key.append(("voxel_d", None if cfg.voxel_d is None else float(cfg.voxel_d)))
    if spec is not None and "model" not in kw:
        key.append(("model", object_model(cfg.object_mode)))
    return tuple(key)


def _copy_task_kwargs(kw: dict) -> dict:
    """任务构造参数的副本：Generator 深拷贝（保留构造时状态 → 重建的抖动样本与原任务相同），其余浅拷贝。"""
    return {k: (copy.deepcopy(v) if isinstance(v, np.random.Generator) else v) for k, v in kw.items()}


# ============================================================ 便捷构造
def make_task(cfg: SystemConfig, kind: str = "parity", **kw) -> Task:
    """按 cfg 的几何（ROI 中心、频率）构造任务。kind：'parity'(需 n=) | 'letter' | 'loc3bit' | 'focus'。

    parity / loc3bit 的材料与体素默认取 cfg.material / cfg.voxel_d；letter 默认金属薄板（任务自身默认值）。
    体素极化率模型默认 object_model(cfg.object_mode)：born → volume_rr，full / intra → scalar_rr（审查 R2-INT-1）；
    显式 model= 优先。
    返回的任务带 make_spec = {"kind", "kwargs"}（调用方显式参数；Generator 为构造前状态的副本），
    供 PhysicalSystem.derive 在 ROI / 材料 / 物体模式等改变时重建（审查 INT-SEM-2）。
    """
    spec = {"kind": kind, "kwargs": _copy_task_kwargs(kw)}
    task = _make_task(cfg, kind, **kw)
    task.make_spec = spec
    return task


def _make_task(cfg: SystemConfig, kind: str, **kw) -> Task:
    k = wavenumber(cfg.freq_hz)
    if kind == "focus":
        return FocusTask(**kw)
    roi = roi_center(cfg)
    kw["model"] = object_model(cfg.object_mode, kw.get("model"))
    if kind == "parity":
        kw.setdefault("material", cfg.material)
        kw.setdefault("voxel_d", cfg.voxel_d)
        return ParityTask(roi_center=roi, k=k, **kw)
    if kind == "letter":
        return LetterTask(roi, k=k, **kw)
    if kind in ("loc3bit", "loc", "localization"):
        kw.setdefault("material", cfg.material)
        kw.setdefault("voxel_d", cfg.voxel_d)
        return LocalizationTask(roi, k=k, **kw)
    raise ValueError(f"未知任务 {kind!r}")


def make_system(cfg: SystemConfig, kind: str = "parity", calib_rng: np.random.Generator | None = None,
                **task_kw) -> PhysicalSystem:
    """make_task + PhysicalSystem。"""
    return PhysicalSystem(cfg, make_task(cfg, kind, **task_kw), calib_rng)


# ============================================================ 命名场景（INTEGRATION S）
# 只列与 SystemConfig 默认值不同的几何 / 场景字段（条件、物体模式、材料由实验网格决定；水球 λ/3、间距 λ/2 是任务默认）。
SCENARIOS: dict = {
    # 默认 L90 奇偶场景：ROI 距接收板 0.5 m，馈源→探头视距保留。偏置主导（方案推论 3）：物体对 PA 输入的调制 ~ −30 dB，
    # 1-bit 结构对 2 位 XOR 的裕量只有 ~0.01–0.06 dB（物理上脆弱）。
    "baseline": {"preset": "L90"},
    # 有信息量的场景（几何 / 功率探索）：吸波屏挡住馈源→探头视距（block_direct）+ ROI 拉近到 0.35 m。
    # M1 XOR 1-bit 裕量中位数 ~ +3.9 dB；3 位奇偶（full）只有连续相位 oracle 在 Rapp@+20/+25/+30 解出（3/3，裕量 ~4 dB），
    # M1 / M1b / M0 0/3；任何 1-bit 码本都没解出 3 位奇偶。
    "informative": {"preset": "L90", "roi_offset": (0.35, 0.0, 0.0), "block_direct": True},
    # 同上 + 探头方位 45° / 75°（第三探头 60°），探索中最好的探头几何旋钮。
    "informative_probe": {"preset": "L90", "roi_offset": (0.35, 0.0, 0.0), "block_direct": True,
                          "probe_az_deg": (45.0, 75.0, 60.0)},
}


def apply_scenario(cfg_or_kwargs, name: str):
    """把命名场景 SCENARIOS[name] 叠加到 SystemConfig（返回 dataclasses.replace 的新对象）或 kwargs dict（返回新 dict）。
    场景字段覆盖原值；"baseline" 也会把 roi_offset / block_direct / 探头覆盖复位为预设默认（场景之间可互换）。"""
    if name not in SCENARIOS:
        raise ValueError(f"未知场景 {name!r}，可选 {list(SCENARIOS)}")
    reset = {"roi_offset": None, "block_direct": False, "skip_gain_db": 0.0, "feed_az_deg": None, "feed_dist": None,
             "probe_az_deg": None, "probe_dist": None}
    over = dict(reset)
    over.update(SCENARIOS[name])
    if isinstance(cfg_or_kwargs, SystemConfig):
        return dataclasses.replace(cfg_or_kwargs, **over)
    out = dict(cfg_or_kwargs or {})
    out.update(over)
    return out
