"""门 G1 的仿真侧自检（INTEGRATION H，DESIGN §10；方案 §4.8 验证、§6.1 奇偶自检、§8 门 G1）。

用法：python -m ris_nlsim.selfcheck [--fast] [--seed S] [--json PATH]
依次打印 PASS/FAIL（--fast 跳过的项打印 SKIP，只报告的项打印 INFO）与关键数值；全部 PASS（SKIP/INFO 视为通过）时退出码 0，否则 1。

检查项（每项一个函数，返回 CheckResult）：
1. reciprocity      互易性：两天线 A↔B 经物体（两个球）+ 环境（各向同性有耗点 + PEC PO 板）+ 无源面（随机相位）的传输相等，
                     full / born（DWBA）/ intra 三种物体模式；另核对 scat_to_scat 对称、src_to_rcv 满足 Friis。
2. optical_theorem  光学定理：solver.solve_objects_planewave + sigma_ext / sigma_sca，无耗介质（εr=4、εr=10 三球耦合）与
                     scalar_rr 金属（软球极限）σ_ext = σ_sca；有耗水 σ_abs > 0。
3. born_vs_full     εr = 1.01 时 Born（DWBA，p = α·E_inc）与完全耦合的远场一致（相对误差 < 2%），且误差随 χ 一阶缩放
                     （εr = 1.1 时误差约 10 倍）。
4. voxel_convergence 体素收敛（--fast 跳过）：λ/3 球远场 vs 标量 Mie（solver.mie_scalar_far_field），εr=4 与 εr=10 在 λ/10、λ/20，
                     以及水在自动体素（tasks.resolve_voxel_d）下。
5. dwba_affinity    M0 + Born 3 位奇偶：多组随机码本下探头场 z 的 Walsh W≥2 与 y1−y2（线性功率）的 W≥3 < 1e-20；
                     对照 M0 + full 的 W≥3 > 1e-12（说明量尺能看到高阶项）。
5b. relay_dwba_affinity（审查 P-2）DWBA 不变量对中继条件的正确推广：M1/M3 + Born（L90、flat0）下方案 §2 的
                     u_in、d、C、K 各自对 x 仿射（W≥2 < 1e-20）；objects_see_relay_tx=False 时线性中继的 y1−y2 W≥3 < 1e-20；
                     非空对照：默认时 M1 闭环 u 的 W≥2 > 1e-12。
    m1_born_parity_info（INFO）M1 + Born 的 y1−y2 W≥3：默认 / 去掉环路 / 物体不在发射波束。
                     注意：中继条件下物体处在中继发射波束中，C(x)·u_in(x) 是双线性的、环路 K(x) 经过物体，所以
                     M1 + Born 的探头功率含 3 阶以上 Walsh 项（L90 ~1e-5、flat0 ~1e-1）。方案 §6.1 预测表里 M1 + Born
                     n=3 的"—"（理论不可解）只在发射→物体耦合与依赖物体的环路都不存在时成立（objects_see_relay_tx=False）；
                     实验 A/D 里 M1 + Born"解出 / 接近解出"3 位奇偶是物理，不是代码错误。门 G1 仍只看 M0 + Born（INTEGRATION H）。
6. m0_born_parity   M0 + Born 3 位奇偶不可解：1-bit oracle（模型损失）、1-bit oracle（直接最大化带符号裕量 / 判对数）、DSPSA
                     都不能 8 个全对（报最佳准确率与裕量，裕量必须 ≤ 0）；非 --fast 时再加连续相位 oracle。
                     默认几何与视距遮挡（block_direct）两种场景都测。正对照 m0_born_presence：同一管线在 1 位有/无检测（一次函数）
                     上能全对——说明"解不出"不是优化器坏了；2 位 XOR 只作 INFO（需压掉背景偏置，方案推论 3）。
7. torch_numpy      TorchSystem 与 numpy PhysicalSystem 的无噪声前向一致（中继 M3 + full，与 M0 + Born；1-bit 与连续相位），
                     探头功率相对误差 < 1e-8，损失一致。
8. calibration      源功率标定自洽（INTEGRATION B）：M2/M3/M4 下聚焦码本的开环 PA 输入 = P1dB·10^{backoff/10}（误差 < 1e-9 dB），
                     M0 与 M3 同一 P_feed；另报闭环工作点（不作判据）。

奇偶任务的物体默认用实验材料水（自动体素 ≈ λ/28，born → volume_rr，tasks.object_model），tiny=True（单元测试）改用 λ/10。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .constants import LAMBDA, K0, F0, wavelength, wavenumber
from .geometry import board_elements, voxelize_sphere
from .greens import scat_to_scat, src_to_rcv
from .materials import alpha_passive, polarizability, voxels_to_nodes
from .metrics import walsh_weight_ge
from .patterns import Pattern, element_pattern, horn_pattern, po_cell_pattern, probe_pattern
from .scene import NodeSet, Scene
from .solver import (far_field_amplitude, mie_scalar_far_field, sigma_ext, sigma_sca, solve_objects_planewave,
                     solve_scene)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    values: dict = field(default_factory=dict)
    skipped: bool = False
    time_s: float = 0.0
    info: bool = False                  # 只报告、不作判据（passed 恒为 True）

    @property
    def status(self) -> str:
        if self.info:
            return "INFO"
        return "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")

    def line(self) -> str:
        return f"[{self.status}] {self.name}: {self.detail} ({self.time_s:.1f} s)"


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _sphere(center, eps, d, diameter=LAMBDA / 3, group=0, model=None):
    pos, vol = voxelize_sphere(center, diameter, d)
    kw = {} if model is None else {"model": model}
    return voxels_to_nodes(pos, vol, eps, K0, group=group, **kw)


# ============================================================ 1. 互易性
def check_reciprocity(rng: np.random.Generator) -> CheckResult:
    """A→B 与 B→A 的传输相等（DESIGN §1.2 必测项的场景级版本）。"""
    k, lam = K0, LAMBDA
    pa = np.array([1.6, -0.7, 0.10])
    pb = np.array([1.3, 0.9, -0.05])
    c = np.zeros(3)
    O = NodeSet.concat([_sphere([0.45, 0.05, 0.02], "water", LAMBDA / 8, group=0),
                        _sphere([0.42, -0.10, -0.03], "eps4", LAMBDA / 8, group=1)])
    env_pts = NodeSet(rng.uniform(-0.3, 0.3, (6, 3)) + np.array([0.8, 0.0, 0.0]), None, Pattern(),
                      alpha=(rng.uniform(0.5, 2.0, 6) - 1j * rng.uniform(0.0, 1.0, 6)) * 1e-5)
    cell = lam / 4
    plate_pos = board_elements([0.9, -0.5, -0.2], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 4, 4, cell)
    plate = NodeSet(plate_pos, [-1.0, 0.0, 0.0], po_cell_pattern(cell**2, lam), alpha=alpha_passive(-1.0, k, lam))
    ppos = board_elements([0.0, -0.1, -0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 5, 5, lam / 2)
    P = NodeSet(ppos, [1.0, 0.0, 0.0], element_pattern(), alpha=alpha_passive(1.0, k, lam))
    phase = rng.uniform(0, 2 * np.pi, len(P))
    A = NodeSet(pa, c - pa, horn_pattern())
    B = NodeSet(pb, c - pb, probe_pattern())
    errs = {}
    for mode in ("full", "born", "intra"):
        s_ab = Scene(F0, A, probes=B, env=[env_pts, plate], passive=P)
        s_ba = Scene(F0, B, probes=A, env=[env_pts, plate], passive=P)
        h_ab = solve_scene(s_ab, O, phase, mode)[0, 0]
        h_ba = solve_scene(s_ba, O, phase, mode)[0, 0]
        errs[mode] = float(abs(h_ab - h_ba) / abs(h_ab))
    # 耦合对称 + Friis
    C = scat_to_scat(O, O, k, same=True)
    sym = _rel(C, C.T)
    R = np.linalg.norm(pa - pb)
    friis = abs(src_to_rcv(B, A, k)[0, 0]) ** 2
    da = horn_pattern().gain(A.normal, (pb - pa) / R)[0]
    db = probe_pattern().gain(B.normal, (pa - pb) / R)[0]
    friis_err = abs(friis / (da * db * (lam / (4 * np.pi * R)) ** 2) - 1.0)
    worst = max(max(errs.values()), sym, friis_err)
    ok = worst < 1e-10
    return CheckResult("reciprocity", ok,
                       f"max |H_AB−H_BA|/|H_AB| = {max(errs.values()):.2e} (full {errs['full']:.1e}, born "
                       f"{errs['born']:.1e}, intra {errs['intra']:.1e}); C 对称 {sym:.1e}; Friis {friis_err:.1e}",
                       {"rel_err": errs, "coupling_sym": sym, "friis_err": friis_err})


# ============================================================ 2. 光学定理
def check_optical_theorem() -> CheckResult:
    k = K0
    kv = np.array([0.0, 0.0, 1.0])
    vals = {}
    cases = {
        "eps4": _sphere([0, 0, 0], "eps4", LAMBDA / 10),
        "eps10_3spheres": NodeSet.concat([_sphere([0, 0, z], "eps10", LAMBDA / 10, group=g)
                                          for g, z in enumerate((-LAMBDA / 2, 0.0, LAMBDA / 2))]),
        "metal_scalar_rr": _sphere([0, 0, 0], "metal", LAMBDA / 15),
    }
    worst = 0.0
    for name, O in cases.items():
        p = solve_objects_planewave(O, k, kv, "full")
        se, ss = sigma_ext(p, O, k, kv), sigma_sca(p, O, k)
        e = abs(se - ss) / se
        vals[name] = {"sigma_ext": se, "sigma_sca": ss, "rel_diff": e}
        worst = max(worst, e)
    Ow = _sphere([0, 0, 0], "water", LAMBDA / 15)
    pw = solve_objects_planewave(Ow, k, kv, "full")
    se, ss = sigma_ext(pw, Ow, k, kv), sigma_sca(pw, Ow, k)
    vals["water"] = {"sigma_ext": se, "sigma_sca": ss, "sigma_abs_frac": (se - ss) / se}
    ok = worst < 1e-9 and (se - ss) / se > 1e-3
    return CheckResult("optical_theorem", ok,
                       f"无耗 max |σ_ext−σ_sca|/σ_ext = {worst:.2e}（εr=4、εr=10 三球、金属软球）；"
                       f"水 σ_abs/σ_ext = {(se - ss) / se:.3f} > 0", vals)


# ============================================================ 3. Born ≈ Full
def _ff_dirs(n: int = 37) -> np.ndarray:
    th = np.linspace(0.0, np.pi, n)
    return np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=1)


def _born_full_err(eps: float, d: float) -> float:
    k = K0
    kv = np.array([0.0, 0.0, 1.0])
    O = _sphere([0, 0, 0], eps, d)
    dirs = _ff_dirs()
    f_full = far_field_amplitude(solve_objects_planewave(O, k, kv, "full"), O, k, dirs)
    f_born = far_field_amplitude(solve_objects_planewave(O, k, kv, "born"), O, k, dirs)
    return float(np.linalg.norm(f_born - f_full) / np.linalg.norm(f_full))


def check_born_vs_full() -> CheckResult:
    e1 = _born_full_err(1.01, LAMBDA / 10)
    e2 = _born_full_err(1.1, LAMBDA / 10)
    ratio = e2 / e1
    ok = e1 < 0.02 and 5.0 < ratio < 20.0
    return CheckResult("born_vs_full", ok,
                       f"εr=1.01 远场相对误差 {e1:.2e} (< 2e-2)；εr=1.1 为 {e2:.2e}，比值 {ratio:.1f}（一阶 χ 缩放，期望 ≈10）",
                       {"err_eps1.01": e1, "err_eps1.1": e2, "ratio": ratio})


# ============================================================ 4. 体素收敛
def _mie_err(eps, d: float) -> float:
    k = K0
    O = _sphere([0, 0, 0], eps, d)
    dirs = _ff_dirs()
    f = far_field_amplitude(solve_objects_planewave(O, k, [0, 0, 1.0], "full"), O, k, dirs)
    ref = mie_scalar_far_field(eps, LAMBDA / 6, k, dirs[:, 2])
    return float(np.linalg.norm(f - ref) / np.linalg.norm(ref))


def check_voxel_convergence(fast: bool = False) -> CheckResult:
    if fast:
        return CheckResult("voxel_convergence", True, "跳过（--fast）", skipped=True)
    from .tasks import resolve_voxel_d

    vals = {}
    limits = {("eps4", 10): 0.02, ("eps4", 20): 0.02, ("eps10", 10): 0.05, ("eps10", 20): 0.05}
    ok = True
    for (mat, n), lim in limits.items():
        e = _mie_err(mat, LAMBDA / n)
        vals[f"{mat}_lambda/{n}"] = e
        ok &= e < lim
    k = K0
    dirs = _ff_dirs()
    O10, O20 = _sphere([0, 0, 0], "eps4", LAMBDA / 10), _sphere([0, 0, 0], "eps4", LAMBDA / 20)
    f10 = far_field_amplitude(solve_objects_planewave(O10, k, [0, 0, 1.0], "full"), O10, k, dirs)
    f20 = far_field_amplitude(solve_objects_planewave(O20, k, [0, 0, 1.0], "full"), O20, k, dirs)
    change = float(np.linalg.norm(f20 - f10) / np.linalg.norm(f20))
    vals["eps4_change_10_to_20"] = change
    ok &= change < 0.02
    dw = resolve_voxel_d("water")
    ew = _mie_err("water", dw)
    vals["water_auto"] = ew
    vals["water_auto_voxel_lambda_over"] = LAMBDA / dw
    ok &= ew < 0.10
    return CheckResult("voxel_convergence", bool(ok),
                       f"vs 标量 Mie：εr=4 λ/10 {vals['eps4_lambda/10']:.1%}、λ/20 {vals['eps4_lambda/20']:.1%}；"
                       f"εr=10 λ/10 {vals['eps10_lambda/10']:.1%}、λ/20 {vals['eps10_lambda/20']:.1%}；"
                       f"εr=4 λ/10→λ/20 变化 {change:.1%}；水 λ/{LAMBDA / dw:.0f} {ew:.1%}", vals)


# ============================================================ 系统构造（5–8 共用）
def _parity_system(cond: str, mode: str, n: int, tiny: bool, **kw):
    from .system import SystemConfig, make_system

    vox = LAMBDA / 10 if tiny else None
    return make_system(SystemConfig(condition=cond, object_mode=mode, material="water", voxel_d=vox, **kw),
                       "parity", n=n)


# ============================================================ 5. DWBA 仿射性
def check_dwba_affinity(rng: np.random.Generator, sys_m0_born=None, tiny: bool = False, n_codebooks: int = 8
                        ) -> CheckResult:
    s = sys_m0_born if sys_m0_born is not None else _parity_system("M0", "born", 3, tiny)
    w3, wz = [], []
    for _ in range(n_codebooks):
        m = s.forward(s.phases_from_params(s.random_params(rng)), noisy=False)
        w3.append(walsh_weight_ge(m.p_w[:, 0] - m.p_w[:, 1], 3, 3))
        wz.append(walsh_weight_ge(m.z, 3, 2))
    # 连续相位同样仿射
    m = s.forward({"phase_p": rng.uniform(0, 2 * np.pi, s.n_params)}, noisy=False)
    w3.append(walsh_weight_ge(m.p_w[:, 0] - m.p_w[:, 1], 3, 3))
    wz.append(walsh_weight_ge(m.z, 3, 2))
    sf = s.derive(object_mode="full")
    mf = sf.forward(sf.phases_from_params(sf.random_params(rng)), noisy=False)
    w3_full = walsh_weight_ge(mf.p_w[:, 0] - mf.p_w[:, 1], 3, 3)
    ok = max(w3) < 1e-20 and max(wz) < 1e-20 and w3_full > 1e-12
    return CheckResult("dwba_affinity", bool(ok),
                       f"M0+Born：max W≥3(y1−y2) = {max(w3):.1e}，max W≥2(z) = {max(wz):.1e}（< 1e-20，"
                       f"{len(w3)} 组码本）；对照 M0+full W≥3 = {w3_full:.1e}（> 1e-12）；体素 λ/{LAMBDA / s.task.voxel_d:.0f}",
                       {"max_W3_ydiff": max(w3), "max_W2_z": max(wz), "W3_full_control": w3_full,
                        "voxel_d": s.task.voxel_d, "object_model": s.object_model})


# ============================================================ 5b. 中继条件下的 DWBA 不变量（审查 P-2）
RELAY_DWBA_PRESETS = ("L90", "flat0")


def _relay_terms(m) -> dict:
    """方案 §2 的中继线性量（Measurement 里逐样本的 u_in、d、C、K）。"""
    return {"u_in": m.u_in, "d": m.d, "C": m.C, "K": m.K}


def _w3_ydiff(p_w) -> float:
    return walsh_weight_ge(p_w[:, 0] - p_w[:, 1], 3, 3)


def check_relay_dwba_affinity(rng: np.random.Generator, tiny: bool = False, n_codebooks: int = 3,
                              presets=RELAY_DWBA_PRESETS, systems: dict | None = None
                              ) -> tuple[CheckResult, CheckResult]:
    """DWBA 仿射不变量对中继条件的正确推广（审查 P-2），以及 M1 + Born 3 位奇偶"理论不可解"的适用范围（INFO）。

    PASS 判据（M1、M3 + Born，各预设、若干随机码本，2³ 个样本）：
      a) 方案 §2 的四个中继线性量 u_in(x)、d(x)、C(x)、K(x) 各自对 x 仿射：Walsh W≥2 < 1e-20
         （DWBA：物体只对无物体背景场一阶响应；馈源列与发射列各自仿射）。这一项挡住中继路径 DWBA 的回归。
      b) 理想化对照 objects_see_relay_tx=False（物体不被中继发射场照射 ⇒ K、C 与 x 无关）：线性中继 M1 的
         z W≥2 < 1e-20、y1−y2 W≥3 < 1e-20——方案 §6.1"无源或线性有源 + Born … 理论上解不出 3 位奇偶"在其前提下成立。
      c) 非空检查：默认（物体在发射波束中）时 M1 的闭环 u 的 W≥2 > 1e-12（量尺看得到环路带来的高阶项）。
    INFO（不作判据）：M1 + Born 的 y1−y2 W≥3——默认（再照射 C(x)·u_in(x) 双线性 + 经物体的环路 K(x)）、去掉环路
      （z_open = d + C·G_v·u_in）、objects_see_relay_tx=False。前两者 > 0：本仿真器里 M1 + Born 的"—"只在
      发射→物体耦合与依赖物体的环路都不存在时成立；实验 A/D 中 M1 + Born"解出 / 接近解出"3 位奇偶是物理，不是门 G1
      违例。门 G1 仍只看 M0 + Born（INTEGRATION H）。
    systems：{preset: M1+Born 系统}（单元测试可传入别的系统验证"检查不是空的"）。
    """
    vals, worst, ctrl, info = {}, 0.0, {}, {}
    for preset in presets:
        s1 = (systems or {}).get(preset)
        if s1 is None:
            s1 = _parity_system("M1", "born", 3, tiny, preset=preset)
        s3 = s1.derive(condition="M3")
        s_off = s1.derive(objects_see_relay_tx=False)
        G = s1.small_signal_gain()
        w2 = {}
        w_off, w3_def, w3_open, w3_off, w2_u = [], [], [], [], []
        for _ in range(n_codebooks):
            ph = s1.phases_from_params(s1.random_params(rng))
            for tag, s in (("M1", s1), ("M3", s3)):
                for k, v in _relay_terms(s.forward(ph, noisy=False)).items():
                    w2[f"{tag}/{k}"] = max(w2.get(f"{tag}/{k}", 0.0), walsh_weight_ge(v, 3, 2))
            m = s1.forward(ph, noisy=False)
            w3_def.append(_w3_ydiff(m.p_w))
            w2_u.append(walsh_weight_ge(m.u, 3, 2))
            z_open = m.d + G * np.einsum("bkr,br->bk", m.C, m.u_in)              # K → 0：只剩再照射
            w3_open.append(_w3_ydiff(np.abs(z_open) ** 2))
            mo = s_off.forward(ph, noisy=False)
            w_off += [walsh_weight_ge(mo.z, 3, 2), _w3_ydiff(mo.p_w)]
            w3_off.append(_w3_ydiff(mo.p_w))
        vals[preset] = {"max_W2_relay_terms": w2, "noTx_max_W2z_W3y": max(w_off), "M1_max_W2_u": max(w2_u),
                        "voxel_d": s1.task.voxel_d, "object_model": s1.object_model}
        worst = max(worst, max(w2.values()), max(w_off))
        ctrl[preset] = max(w2_u)
        info[preset] = {"W3_ydiff_default": max(w3_def), "W3_ydiff_no_loop": max(w3_open),
                        "W3_ydiff_objects_not_in_tx_beam": max(w3_off)}
    ok = worst < 1e-20 and max(ctrl.values()) > 1e-12
    r1 = CheckResult("relay_dwba_affinity", bool(ok),
                     f"M1/M3+Born（{'/'.join(presets)}，{n_codebooks} 组码本）：u_in、d、C、K 的 max W≥2 与 "
                     f"objects_see_relay_tx=False 时 M1 的 W≥2(z)/W≥3(y1−y2) 最大 {worst:.1e}（< 1e-20）；"
                     "非空对照 M1 闭环 u 的 W≥2 = "
                     + "，".join(f"{p} {v:.1e}" for p, v in ctrl.items()) + "（> 1e-12）",
                     {"per_preset": vals, "worst": worst, "control_W2_u": ctrl})
    r2 = CheckResult("m1_born_parity_info", True,
                     "INFO：M1+Born 的 y1−y2 W≥3（默认 / 去环路 / 物体不在发射波束）："
                     + "；".join(f"{p} {v['W3_ydiff_default']:.1e} / {v['W3_ydiff_no_loop']:.1e} / "
                                f"{v['W3_ydiff_objects_not_in_tx_beam']:.1e}" for p, v in info.items())
                     + "——方案 §6.1 的 M1+Born \"—\" 只在发射→物体耦合与经物体的环路都不存在时成立；门 G1 只看 M0+Born",
                     info, info=True)
    return r1, r2


# ============================================================ 6. M0 + Born 奇偶
def _margin_oracle(s, rng, n_restarts: int, max_sweeps: int = 30) -> dict:
    """直接最大化带符号裕量 min_x sgn(x)(y1−y2)（> 0 ⇔ 全对）的 1-bit oracle（= train.run_oracle_bits(objective="margin")，
    审查 S0-2 把它提升为共享实现）。"""
    from .train import run_oracle_bits

    r = run_oracle_bits(s, rng, n_restarts=n_restarts, max_sweeps=max_sweeps, n_noise_eval=0, objective="margin")
    ev = r["eval"]
    return {"acc_clean": ev["acc_clean"], "all_correct": ev["all_correct"], "margin_db": -float(r["loss"])}


def _accuracy_oracle(s, rng, n_restarts: int, max_sweeps: int = 30) -> dict:
    """最大化判对样本数的 1-bit oracle（train.run_oracle_bits(objective="accuracy")，审查 S0-2）：
    "这个结构最多判对几个"——M0 + Born 3 位奇偶的二次型最多 7/8（Minsky–Papert），绝不能 8/8。"""
    from .train import run_oracle_bits

    return _ev3(run_oracle_bits(s, rng, n_restarts=n_restarts, max_sweeps=max_sweeps, n_noise_eval=0,
                                objective="accuracy")["eval"])


def _ev3(ev: dict) -> dict:
    return {"acc_clean": ev["acc_clean"], "all_correct": bool(ev["all_correct"]),
            "margin_db": ev.get("margin_db_unflagged", ev.get("margin_db"))}


def check_m0_born_parity(rng: np.random.Generator, sys_m0_born=None, tiny: bool = False, fast: bool = False
                         ) -> tuple[CheckResult, CheckResult, CheckResult]:
    """门 G1（方案 §6.1 Minsky–Papert、§8）：M0 + Born 下 y1−y2 是 x 的二次多项式 ⇒ 3 位奇偶不可解。

    在默认几何（保留馈源→探头视距）与"吸波屏遮挡视距"（block_direct，对线性系统最有利：背景可被无源面抵消）两种
    场景下，1-bit oracle（模型损失 / 直接最大化带符号裕量）、DSPSA（及非 fast 时连续相位 oracle）都不能 8 个全对，
    且最佳裕量 ≤ 0。注意：二分类损失在不可解时近似 τlog2 − mean(s)/2，最小化它不等于最少错误，准确率可低于 0.5。
    正对照（门控）：同一管线在 1 位"有/无"检测（一次函数）上能全对；XOR（2 位，二次型原则上可表示，但需要压掉背景偏置，
    方案推论 3）只作 INFO。
    """
    from .train import run_dspsa, run_oracle_bits, run_oracle_continuous

    n_rs = 2 if tiny else (4 if fast else 8)
    n_it = 300 if tiny else (1000 if fast else 2000)
    t0 = time.perf_counter()
    res = {}
    for bd in (False, True):
        s = sys_m0_born if (sys_m0_born is not None and not bd) else _parity_system("M0", "born", 3, tiny,
                                                                                   block_direct=bd)
        tag = "los_blocked" if bd else "default"
        res[f"{tag}/oracle_loss"] = _ev3(run_oracle_bits(s, rng, n_restarts=n_rs, n_noise_eval=0)["eval"])
        res[f"{tag}/oracle_margin"] = _margin_oracle(s, rng, n_restarts=max(1, n_rs // 2))
        res[f"{tag}/oracle_accuracy"] = _accuracy_oracle(s, rng, n_restarts=max(1, n_rs // 2))
        res[f"{tag}/dspsa"] = _ev3(run_dspsa(s, n_it, rng, eval_every=max(n_it // 4, 1), n_noise_eval=0)["eval"])
        if not fast and not tiny:
            res[f"{tag}/oracle_continuous"] = _ev3(run_oracle_continuous(s, n_restarts=2, steps=400,
                                                                         n_noise_eval=0)["eval"])
    solved = [k for k, v in res.items() if v["all_correct"]]
    margins = [v["margin_db"] for v in res.values() if v["margin_db"] is not None]
    best_acc = max(v["acc_clean"] for v in res.values())
    best_margin = max(margins)
    ok = not solved and best_margin <= 0.0
    r1 = CheckResult("m0_born_parity3_unsolvable", bool(ok),
                     "全对：" + ("无" if not solved else ",".join(solved)) + f"；最佳准确率 {best_acc:.3f}，最佳裕量 "
                     f"{best_margin:+.3f} dB（必须 ≤ 0）；DSPSA {n_it} 迭代；"
                     + "，".join(f"{k} {v['acc_clean']:.3f}/{v['margin_db']:+.2f}dB" for k, v in res.items()),
                     {**res, "best_acc": best_acc, "best_margin_db": best_margin, "dspsa_iters": n_it},
                     time_s=time.perf_counter() - t0)
    # 正对照：1 位有/无检测（一次函数，必须能解）
    t1 = time.perf_counter()
    s1 = _parity_system("M0", "born", 1, tiny)
    o1 = _ev3(run_oracle_bits(s1, rng, n_restarts=n_rs, n_noise_eval=0)["eval"])
    d1 = _ev3(run_dspsa(s1, n_it, rng, eval_every=max(n_it // 4, 1), n_noise_eval=0)["eval"])
    r2 = CheckResult("m0_born_presence_solvable", bool(o1["all_correct"] and d1["all_correct"]),
                     f"正对照：1 位有/无（一次函数）oracle 裕量 {o1['margin_db']:+.2f} dB、DSPSA 裕量 {d1['margin_db']:+.2f} dB，"
                     "均全对 ⇒ 3 位\"解不出\"不是优化器失效", {"oracle": o1, "dspsa": d1},
                     time_s=time.perf_counter() - t1)
    # INFO：XOR（视距遮挡，裕量 oracle）
    t2 = time.perf_counter()
    s2 = _parity_system("M0", "born", 2, tiny, block_direct=True)
    o2 = _margin_oracle(s2, rng, n_restarts=n_rs)
    r3 = CheckResult("m0_born_xor_info", True,
                     f"INFO：2 位 XOR（视距遮挡）最佳裕量 {o2['margin_db']:+.3f} dB、准确率 {o2['acc_clean']:.2f}"
                     "（二次型可表示 XOR，但需 1-bit 面压掉背景偏置，方案推论 3；不作判据）", {"oracle_margin": o2},
                     info=True, time_s=time.perf_counter() - t2)
    return r1, r2, r3


# ============================================================ 7. torch / numpy
def check_torch_numpy(rng: np.random.Generator, systems: dict) -> CheckResult:
    import torch

    vals, worst, worst_loss = {}, 0.0, 0.0
    for name, s in systems.items():
        ts = s.to_torch()
        errs, lerrs = [], []
        for trial in range(4):
            if trial < 2:
                ph = s.phases_from_params(s.random_params(rng))
            else:                                   # 连续相位
                ph = {k: rng.uniform(0, 2 * np.pi, np.shape(v)) for k, v in
                      s.phases_from_params(s.random_params(rng)).items()}
            m = s.forward(ph, noisy=False)
            with torch.no_grad():
                out = ts.forward(ts.as_phases(ph))
                p_t = (out["z"].real ** 2 + out["z"].imag ** 2).numpy()
                l_t = float(ts.loss(ts.as_phases(ph)))
            errs.append(_rel(p_t, m.p_w))
            lerrs.append(abs(l_t - s.loss_from(m)) / max(abs(s.loss_from(m)), 1e-12))
        vals[name] = {"max_rel_power_err": max(errs), "max_rel_loss_err": max(lerrs)}
        worst, worst_loss = max(worst, max(errs)), max(worst_loss, max(lerrs))
    ok = worst < 1e-8 and worst_loss < 1e-8
    return CheckResult("torch_numpy", bool(ok),
                       f"探头功率 max 相对误差 {worst:.1e}、损失 {worst_loss:.1e}（< 1e-8；"
                       + ", ".join(vals) + "；1-bit 与连续相位）", vals)


# ============================================================ 8. 源功率标定
def check_calibration(sys_m3) -> CheckResult:
    from .relay import weights
    from .system import N_ELEM

    vals, worst = {}, 0.0
    for cond, s in (("M3", sys_m3), ("M2", sys_m3.derive(condition="M2")), ("M4", sys_m3.derive(condition="M4"))):
        bits = s.focus_bits_rx
        w, _ = weights(np.pi * bits, np.zeros_like(bits, dtype=float), s.params)
        u0 = s.s_feed * np.sum(w * s.H0[:s.n_rx, 0].reshape(s.n_relays, N_ELEM), axis=-1)
        rel = float(10 * np.log10(np.mean(np.abs(u0) ** 2) / s.p1db_in_w))
        err = abs(rel - s.backoff_db)
        worst = max(worst, err)
        m = s.forward(s.phases_from_params(s.focus_params()), noisy=False)
        vals[cond] = {"open_loop_rel_p1db_db": rel, "backoff_db": s.backoff_db, "err_db": err,
                      "p_feed_dbm": float(10 * np.log10(s.p_feed_w) + 30),
                      "closed_loop_focus_rel_p1db_db": float(10 * np.log10(np.abs(m.u[0, 0]) ** 2 / s.p1db_in_w)),
                      "flags_focus": bool(m.flags.any())}
    m0 = sys_m3.derive(condition="M0")
    same = abs(m0.p_feed_w / sys_m3.p_feed_w - 1.0)
    vals["M0_vs_M3_pfeed_rel"] = same
    ok = worst < 1e-9 and same < 1e-12
    return CheckResult("calibration", bool(ok),
                       f"聚焦码本开环 PA 输入 − P1dB = backoff（M2/M3/M4 误差 max {worst:.1e} dB）；M0 与 M3 同 P_feed "
                       f"（相对差 {same:.1e}）；P_feed(M3) = {vals['M3']['p_feed_dbm']:.2f} dBm，聚焦码本闭环工作点 "
                       f"{vals['M3']['closed_loop_focus_rel_p1db_db']:+.2f} dB rel P1dB", vals)


# ============================================================ 主流程
def _timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    dt = time.perf_counter() - t0
    for r in (out if isinstance(out, tuple) else (out,)):
        if r.time_s == 0.0:
            r.time_s = dt
    return out


def run_selfcheck(fast: bool = False, seed: int = 0, tiny: bool = False, verbose: bool = True) -> list[CheckResult]:
    """运行全部检查。tiny=True：奇偶系统用 λ/10 体素、少量 oracle 重启 / DSPSA 迭代（单元测试用）。"""
    rng = np.random.default_rng(seed)
    results: list[CheckResult] = []

    def emit(r):
        results.append(r)
        if verbose:
            print(r.line(), flush=True)

    def guarded(name, fn, *a, **kw):
        try:
            out = _timed(fn, *a, **kw)
        except Exception as e:  # noqa: BLE001 —— 任何异常都算该项 FAIL，不中断其余检查
            import traceback
            emit(CheckResult(name, False, f"异常：{e!r}", {"traceback": traceback.format_exc()}))
            return None
        for r in (out if isinstance(out, tuple) else (out,)):
            emit(r)
        return out

    guarded("reciprocity", check_reciprocity, rng)
    guarded("optical_theorem", check_optical_theorem)
    guarded("born_vs_full", check_born_vs_full)
    guarded("voxel_convergence", check_voxel_convergence, fast or tiny)
    t0 = time.perf_counter()
    try:
        m0b = _parity_system("M0", "born", 3, tiny)
        m3f = _parity_system("M3", "full", 3, True)          # torch / 标定检查只需小体素
    except Exception as e:  # noqa: BLE001
        emit(CheckResult("build_systems", False, f"异常：{e!r}"))
        return results
    if verbose:
        print(f"       （系统构造 {time.perf_counter() - t0:.1f} s：M0+Born 水 λ/{LAMBDA / m0b.task.voxel_d:.0f}，"
              f"{m0b.task.n_vox_per_object} 体素/物体）", flush=True)
    guarded("dwba_affinity", check_dwba_affinity, rng, m0b, tiny)
    # 独立的 Generator（同一 seed 派生）：新增检查不改变其余检查的随机流
    guarded("relay_dwba_affinity", check_relay_dwba_affinity, np.random.default_rng([seed, 52]), tiny)
    guarded("m0_born_parity", check_m0_born_parity, rng, m0b, tiny, fast)
    guarded("torch_numpy", check_torch_numpy, rng, {"M3_full": m3f, "M0_born": m0b})
    guarded("calibration", check_calibration, m3f)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="门 G1 仿真侧自检（INTEGRATION H）")
    ap.add_argument("--fast", action="store_true", help="跳过体素收敛、减少 oracle/DSPSA 预算、不跑连续 oracle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default=None, help="把结果写成 JSON（可选）")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = run_selfcheck(fast=args.fast, seed=args.seed)
    n_fail = sum(1 for r in res if not r.passed)
    print(f"selfcheck: {sum(r.status == 'PASS' for r in res)} PASS, {sum(r.skipped for r in res)} SKIP, "
          f"{sum(r.info for r in res)} INFO, {n_fail} FAIL（{time.perf_counter() - t0:.1f} s）")
    if args.json:
        from .experiments.common import save_json
        save_json([{"name": r.name, "status": r.status, "detail": r.detail, "values": r.values,
                    "time_s": r.time_s} for r in res], args.json)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
