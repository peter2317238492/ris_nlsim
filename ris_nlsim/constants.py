"""物理常数、单位换算与全局默认参数（见 DESIGN.md §1、§8）。

约定：时谐 e^{jωt}，格林函数 G(R)=exp(-jkR)/(4πR)；|E|² = 功率密度 (W/m²)。
"""
from __future__ import annotations

import numpy as np

C0 = 299_792_458.0          # 光速 (m/s)
F0 = 3.5e9                  # 默认工作频率 (Hz)
KB = 1.380649e-23           # 玻尔兹曼常数 (J/K)
T0 = 290.0                  # 参考噪声温度 (K)


def wavelength(freq_hz: float = F0) -> float:
    return C0 / freq_hz


def wavenumber(freq_hz: float = F0) -> float:
    return 2.0 * np.pi / wavelength(freq_hz)


LAMBDA = wavelength(F0)     # ≈ 0.085655 m
K0 = wavenumber(F0)         # ≈ 73.35 rad/m
PITCH = LAMBDA / 2.0        # 阵元间距 λ/2


def db2lin(db):
    """功率 dB → 线性。"""
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def lin2db(x, floor: float = 1e-300):
    """功率线性 → dB（对 0 做下限保护）。"""
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), floor))


def dbm2w(dbm):
    return 10.0 ** ((np.asarray(dbm, dtype=float) - 30.0) / 10.0)


def w2dbm(w, floor: float = 1e-300):
    return 10.0 * np.log10(np.maximum(np.asarray(w, dtype=float), floor)) + 30.0


# 默认参数：Phase 0 实测后替换（方案 §4.4、§8）
DEFAULTS: dict = {
    # PA（两级）
    "pa_gain_db": 30.0,
    "psat_out_dbm": 20.0,
    "rapp_p": 2.0,
    "ampm_max_deg": 12.0,
    "ampm_beta": 1.0,
    # 中继网络与噪声
    "loss_rx_db": 2.5,
    "loss_tx_db": 2.5,
    "isolation_db": -30.0,          # 端口参考面隔离；环路中再乘收发网络损耗（DESIGN §3.3）
    "kappa0_phase_deg": 90.0,       # 泄漏相位（未知硬件参数）；实验 D 扫描
    "nf_db": 5.0,
    "noise_bw_hz": 1e3,
    # 方向图
    "elem_q": 1.0,
    "elem_d0": float(np.pi),
    "elem_backlobe_db": -25.0,
    "horn_q": 10.0,
    "probe_q": 1.0,
    # 物体
    "voxel_d": LAMBDA / 10.0,
    "parity_diameter": LAMBDA / 3.0,
    "parity_spacing": LAMBDA / 2.0,
    # DSPSA
    "dspsa_A": 400.0,
    "dspsa_alpha": 0.501,
    # 几何预设（DESIGN.md §1.5）；角度单位为度，距离单位为米
    "presets": {
        "L90": {
            "fold_deg": 90.0,
            "feed_az_deg": -30.0, "feed_dist": 1.8,            # 相对接收板中心
            "probe_az_deg": [75.0, 105.0, 90.0], "probe_dist": 1.0,  # 相对发射板中心；前 n_probes 个
            "roi_offset": [0.5, 0.0, 0.0],                     # 相对接收板中心
            "roi_ref": "rx_center",
        },
        "flat0": {
            "fold_deg": 0.0,
            "feed_az_deg": -30.0, "feed_dist": 1.8,
            "probe_az_deg": [20.0, 40.0, 60.0], "probe_dist": 1.0,
            "roi_offset": [0.4, 0.0, 0.0],                     # 相对中继中心（合页）
            "roi_ref": "center",
        },
    },
}

MATERIALS: dict = {
    "eps1.5": 1.5,
    "eps4": 4.0,
    "eps10": 10.0,
    "water": 78.0 - 10.0j,
    "metal": "metal",
}
