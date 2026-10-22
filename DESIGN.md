# ris_nlsim 设计规格（实现契约）

本文件是《有源非线性超表面 × 散射非线性：虚拟实验方案 v0》（下称"方案"，原文未随仓库发布）的仿真器实现契约。
所有模块必须严格遵守这里的**约定、公式、接口**；有歧义时以本文件为准，并在代码注释里写明取舍。

- 语言：Python 3.12；依赖只允许 numpy / scipy / matplotlib / torch / pytest（另有 openpyxl 读实测表；见 requirements.txt）。
- 运行测试：在仓库根目录执行 `python -m pytest tests -q`。
- 不得 pip install 新依赖。
- 代码：标识符英文；docstring/注释用简洁中文，关键公式写出来并标注对应方案章节号（如 `# 方案 §4.4`）。
- 数值：一律 complex128 / float64。所有随机性通过显式传入的 `np.random.Generator` 控制（禁止全局 `np.random.*`）。
- Windows：多进程必须 `if __name__ == "__main__":` 保护；用 `concurrent.futures.ProcessPoolExecutor`。

---

## 0. 目录与文件归属

```
ris_nlsim/                      (仓库根目录)
  DESIGN.md  INTEGRATION.md  README.md
  ris_nlsim/
    __init__.py
    constants.py   patterns.py   geometry.py   materials.py   greens.py   scene.py   solver.py      ← 物理核心
    pa.py          relay.py      readout.py    metrics.py                                       ← 中继/PA/读出/指标
    dspsa.py       oracle_bits.py  oracle_torch.py                                              ← 训练
    tasks.py       system.py     selfcheck.py                                                   ← 集成
    experiments/__init__.py common.py exp_f_hardware.py exp_a_parity.py exp_b_snr.py
                exp_c_invariance.py exp_d_feedback.py exp_e_trainability.py
  tests/  test_<module>.py
  results/ (实验输出，git 忽略)
```

---

## 1. 物理约定（全局，不得更改）

| 项 | 约定 |
|---|---|
| 时谐 | e^{jωt}；格林函数 `G(R) = exp(-1j*k*R)/(4*pi*R)` |
| 频率 | 默认 f0 = 3.5e9 Hz，λ = c0/f0 ≈ 0.085655 m，k = 2π/λ。频率作为 `Scene.freq_hz` 传递，函数显式接收 `k` 或 `freq_hz` |
| 场 E | 标量，单位 √W/m，满足 **|E|² = 功率密度 (W/m²)** |
| 源幅度 s | √W，\|s\|² = 源的发射功率 (W) |
| 接收量 v | √W，\|v\|² = 天线端口接收功率 (W)（匹配） |
| 偶极矩 p | p = α·E_loc，α 为极化率 (m³) |
| 损耗材料 | εr = ε′ − jε″（工程约定，例如水 78 − 10j） |
| 1-bit 相位 | bit b∈{0,1} → 相位 φ = π·b → 权重 e^{jφ} = ±1 |
| 2-bit 相位 | level l∈{0,1,2,3} → φ = (π/2)·l |
| 方位角 az | 在 xy 平面内从 +x 转向 +y 计；仰角 el 从 xy 平面向 +z |

### 1.1 方向图（patterns.py）

```python
@dataclass(frozen=True)
class Pattern:
    q: float | None = None        # None → 各向同性，F ≡ 1
    d0: float | None = None       # 峰值方向性（线性）；None 时取 2(q+1)（cos^q 前半球归一化）
    backlobe_db: float = -30.0    # 后瓣/旁瓣地板（相对峰值，功率 dB）
    def amp(self, normals: np.ndarray, dirs: np.ndarray) -> np.ndarray: ...
```
幅度方向图：`F(θ) = sqrt( d0 * ( max(cosθ,0)**q + 10**(backlobe_db/10) ) )`，cosθ = n̂·d̂（d̂ 为**从该节点指向另一节点**的单位向量）。
各向同性节点 F ≡ 1。`normals` 与 `dirs` 支持广播：normals `(N,3)`，dirs `(N,M,3)` → 返回 `(N,M)`。

默认参数（写入 `constants.py` 的 `DEFAULTS` 字典，见 §8）：
- 中继收/发单元、无源面单元：`Pattern(q=1.0, d0=pi, backlobe_db=-25)`（(λ/2)² 口径受限方向性 4πA/λ² = π）。
- 馈源喇叭：`Pattern(q=10.0)`（d0 = 22，约 13.4 dBi），法向指向其瞄准点。
- 探头：`Pattern(q=1.0)`（d0=4），法向指向中继中心。
- 金属板 PO 单元（实验 F 参考）：`Pattern(q=1.0, d0=4*pi*A_cell/λ**2, backlobe_db=-60)`。
- 物体体素、环境散射点：各向同性。

### 1.2 节点间耦合（greens.py）

记 R = |r_a − r_b|，d̂_{a→b} = (r_b − r_a)/R。四种耦合（全部向量化，`a` 为行、`b` 为列）：

| 函数 | 含义 | 公式 |
|---|---|---|
| `src_to_scat` | 源 b 以单位幅度(1 √W)在散射体 a 处产生的（方向图加权）局部场 | `sqrt(4π) * F_b(d̂_{b→a}) * G(R) * F_a(d̂_{a→b})` |
| `scat_to_scat` | 散射体 b 的单位偶极矩在散射体 a 处的局部场 | `k² * G(R) * F_b(d̂_{b→a}) * F_a(d̂_{a→b})`，同一集合时对角置 0 |
| `scat_to_rcv` | 散射体 b 的单位偶极矩在接收端 a 产生的接收量 v | `(λ/sqrt(4π)) * F_a(d̂_{a→b}) * k² * G(R) * F_b(d̂_{b→a})` |
| `src_to_rcv` | 源 b 单位幅度在接收端 a 的接收量 v | `λ * F_a(d̂_{a→b}) * F_b(d̂_{b→a}) * G(R)` |

自检（必须有测试）：`src_to_rcv` 满足 Friis：\|v\|² = D_a D_b (λ/4πR)²；
`scat_to_scat` 对称（互易）；对两个天线 A、B，经任意散射体集合的 A→B 传输 = B→A 传输（互易）。
距离 R < 1e-9 m 的跨集合节点对直接 `raise ValueError`。

### 1.3 极化率（materials.py）— 方案 §4.1

```python
def polarizability(eps_r: complex | str, volume: float, k: float, model: str = "cm_rr") -> complex
```
- `"cm"`：α0 = 3V(εr−1)/(εr+2)
- `"cm_rr"`：α = α0 / (1 + 1j*k**3*α0/(4π))  ← **标量模型下保证光学定理的辐射修正**（e^{jωt} 约定）
- `"scalar_rr"`（**已裁定为默认**）：α = χV/(1 − χk²a²/2 + jχk³V/(4π))，a 为等体积球半径。审查发现矢量 CM 形式在标量 LS 方程里收敛到
  ε_eff = 1 + 3(εr−1)/(εr+2) ≤ 4 的连续极限（压缩 εr 扫描、金属退化成 εr=4 介质）；scalar_rr 是标量 LS 的一致离散（局域场 + 辐射修正），
  收敛到 εr 本身（有标量 Mie 回归测试）。方案 §4.1"CM + 辐射修正"的目的（局域场修正 + 光学定理）由它实现；cm_rr 仍可显式选用。
  体素：介质需 |√εr|·k·d ≲ 2，系统层按材料自动取 d = min(λ/10, λ/(π|√εr|))（水 ≈ λ/28），金属 λ/20。
- `"volume_rr"`：α0 = (εr−1)V，再做同样辐射修正
- `eps_r == "metal"`：α0 = 3V（CM 在 εr→∞ 的极限），再做辐射修正
- 预设材料表 `MATERIALS = {"eps1.5": 1.5, "eps4": 4.0, "eps10": 10.0, "water": 78 - 10j, "metal": "metal"}`

能量守恒推导（写进 docstring）：单偶极子 σ_ext = −k·Im(p·conj(E_inc))/|E0|²，σ_sca = k⁴|α|²/(4π)·|E_loc|²/|E0|²，
无耗 α0 实数时 Im(1/α) = k³/(4π) ⇒ σ_abs = 0。

无源/金属板单元的"极化率"：`alpha_passive(Gamma, k, lam) = lam * Gamma / k**2`（单元方向图另在耦合里乘；推导：天线接收→×Γ→再辐射，k²α_eff = λΓF_inF_out，与物理光学 4πA/λ 一致）。

### 1.4 几何（geometry.py）

```python
PITCH = λ/2
def board_elements(origin_corner, u_axis, v_axis, nu=5, nv=5, pitch=PITCH) -> pos (nu*nv,3)   # 单元在格心
@dataclass
class RelayGeom:  rx_pos (25,3), rx_normal (3,), tx_pos (25,3), tx_normal (3,), rx_center, tx_center, center, fold_deg
def make_relay(fold_deg=90.0, origin=(0,0,0), gap=0.01, n=5, pitch=PITCH) -> RelayGeom
def make_relays(n_relays, fold_deg, spacing_z=0.30) -> list[RelayGeom]       # 沿 z 等距排列、整体居中
def point_from(ref, az_deg, el_deg, dist) -> (3,)
def voxelize_sphere(center, diameter, d) -> (pos (N,3), vol_each float)       # 体积修正：N*vol_each = πD³/6；至少 1 个体素
def voxelize_box(center, size_xyz, d) -> (pos, vol_each)
LETTERS: dict[str, np.ndarray]  # 'A','B','C' 的 7 行 × 5 列 0/1 位图
def voxelize_letter(letter, center, plane_u, plane_v, pixel=λ/2, thickness=λ/2, d=λ/5) -> (pos, vol_each, pixel_ids)
def parity_slots(n, center, spacing, axis=(0,0,1)) -> (n,3)   # 沿 axis 等距、以 center 居中
```
中继几何（合页轴 = z 轴，过 origin）：
- 接收板：平面 x = origin.x，y ∈ [−gap/2 − W, −gap/2]，z ∈ [−W/2, W/2]，法向 +x；W = 5·pitch。
- 发射板：从合页沿 e_t = (−sinψ, cosψ, 0) 展开（起点偏移 gap/2），z 同上，法向 n_t = (cosψ, sinψ, 0)。
  ψ=0 共面（两板都朝 +x），ψ=90° L 形（发射板朝 +y、位于 x<0 侧），ψ=180° 背靠背。
  （实现：真实合页轴放在接收板平面后 gap/2 处，避免 ψ→180° 时两板重合；ψ=0 与上式完全一致，ψ=90° 平移 5 mm。）

### 1.5 几何预设（system.py 使用；数值在 `constants.DEFAULTS["presets"]`）

**`"L90"`（默认）**：ψ = 90°。馈源在接收板中心方位 az = −30°、距离 1.8 m，喇叭瞄准接收板中心。
两个探头在发射板中心方位 az = 75° 与 105°、距离 1.0 m（对 3 类任务再加 90°），探头法向指向发射板中心。
物体感兴趣区（ROI）中心 = 接收板中心 + (0.5, 0, 0) m；奇偶槽位沿 z 轴排列，默认间距 λ/2（可配）。

**`"flat0"`**：ψ = 0°（收发共面）。馈源在接收板中心 az = −30°、1.8 m；探头在发射板中心 az = +20°/+40°（3 类再加 +60°）、1.0 m；
ROI 中心 = 中继中心（合页）+ (0.4, 0, 0) m —— 物体同时处于收、发波束中，**物体在放大环路里**（实验 D 用）。

多中继 N_r>1：用 `make_relays` 沿 z 排列，馈源/探头/ROI 以所有中继的整体中心为参考。

---

## 2. 场景与线性求解（scene.py / solver.py）— 方案 §4.1–4.3

### 2.1 节点集合

```python
@dataclass
class NodeSet:
    pos: np.ndarray                 # (N,3)
    normal: np.ndarray              # (N,3)（各向同性节点可填 0）
    pattern: Pattern
    alpha: np.ndarray | None = None # (N,) complex，散射体用
    group: np.ndarray | None = None # (N,) int，物体槽位/物体编号（intra 模式按它分块），其余 -1
    @staticmethod
    def empty(pattern=Pattern()) -> "NodeSet"
    def __len__(self) -> int
    @staticmethod
    def concat(sets: list["NodeSet"]) -> "NodeSet"  # 要求 pattern 相同；不同 pattern 的集合请分开存放
```
由于方向图按集合共享，"多个不同方向图的集合"用 `list[NodeSet]` 表示（例如源 = [馈源集合, 发射单元集合]）。
greens 的四个耦合函数接收两个 NodeSet；scene 提供把 list 拼成大矩阵的辅助函数。

```python
@dataclass
class Scene:
    freq_hz: float
    feeds: NodeSet                  # 固定源；所有馈源合成为"第 0 列源"，幅度权重 feed_weights
    feed_weights: np.ndarray        # (Nf,) complex，默认全 1；实际功率在系统层乘 s=sqrt(P_feed)
    tx: list[NodeSet]               # 中继发射单元（源），按中继顺序拼接，每个中继 25 个；无中继时为空列表
    rx: list[NodeSet]               # 中继接收单元（接收端）
    probes: NodeSet                 # 探头（接收端）
    env: list[NodeSet]              # 固定散射体（环境点、金属板 PO 单元……），可空
    passive: NodeSet                # 可训练无源单元，alpha = λΓ/k²（不含相位），可空
    block_direct_feed_probe: bool = False   # 吸波屏：去掉 馈源→探头 的视距项（方案 §5 "直达路径 遮挡"）
    skip_gain_db: float = 0.0       # 馈源→探头视距项额外缩放（虚拟旋钮，0 dB = 物理值）
    include_direct_tx_rx: bool = False      # 同一中继发射单元→接收单元的自由空间直耦；默认 False，因板间泄漏由 κ0 统一表示
                                            # （不同中继之间的 tx→rx 耦合始终包含）
```
源列顺序：`[feed, tx_0..tx_{NT-1}]`（共 S = 1+N_T 列）；接收行顺序：`[rx_0..rx_{NRX-1}, probe_0..probe_{K-1}]`。
发射单元→探头、发射单元→物体/环境/无源面 的耦合**始终包含**（经物体的反馈环靠它）。

### 2.2 物体模式（方案 §4.1 三个开关）

一个"样本"= 一组物体节点 `objects: NodeSet`（体素，各向同性，alpha 已按材料算好，group = 物体编号）。
背景 B = 环境 env ∪ 可训练无源面 passive（无源面相位给定）。三种模式：

- `"full"`：物体、环境、无源面全部精确耦合求解：
  `(diag(1/α) − C) p = E0`，其中 C 为全部散射体两两 `scat_to_scat`，E0 为各源列 `src_to_scat`。
  接收：`E_R = E0_R + C_RS p`（E0_R = `src_to_rcv`，C_RS = `scat_to_rcv`）。
- `"born"`：**畸变波 Born 近似（DWBA）**——物体只对"无物体背景场"一阶响应，物体散射场经背景（含无源面、环境的多次反射）传到接收端，但**不再照射任何物体**：
  1. 背景解：`A_BB p_B^bg = E0_B`，A_BB = diag(1/α_B) − C_BB
  2. 物体处背景场：`E_O^bg = E0_O + C_OB p_B^bg`
  3. 物体响应：`p_O = α_O ⊙ E_O^bg`
  4. 背景对物体辐射的响应：`p_B = p_B^bg + A_BB^{-1} C_BO p_O`
  5. 接收：`E_R = E0_R + C_RB p_B + C_RO p_O`
  ⇒ 接收量对物体存在向量 x 是**仿射**的（奇偶自检的基础，必须有测试验证 Walsh 二阶以上权重 < 1e-20）。
- `"intra"`：同 born，但第 3 步换成每个物体内部在**自由空间**精确多次散射：`p_O = blockdiag_g[(diag(1/α_Og) − C_OgOg)^{-1}] E_O^bg`（按 group 分块）。物体之间、物体经背景回到自身的路径都不计。

```python
def solve_scene(scene: Scene, objects: NodeSet, passive_phase: np.ndarray | None, mode: str) -> np.ndarray
    # 直接（参考）求解，返回 H: (N_RCV, S) complex。passive_phase: (N_P,) 弧度，α_P = alpha_base * exp(1j*phase)
```

### 2.3 快速约化表示（训练无源面时用）

训练时只有无源面相位变化。对每个样本预先把与相位无关的部分算好：

```python
@dataclass
class ReducedSample:
    mode: str; n_rx: int; n_probe: int; n_tx: int; n_p: int
    H: np.ndarray            # (N_RCV, S)；n_p==0 时它就是完整响应
    alpha_p_base: np.ndarray | None   # (N_P,)
    Cpp: np.ndarray | None   # (N_P,N_P)  无源单元间等效耦合
    Ep: np.ndarray | None    # (N_P,S)    无源单元处等效激励
    Hrp: np.ndarray | None   # (N_RCV,N_P) 无源偶极矩→接收
    # 仅 born/intra 且 n_p>0 时：
    Eo: np.ndarray | None    # (N_O,S)    物体处背景场（环境已穿衣，不含无源面）
    Cop: np.ndarray | None   # (N_O,N_P)  无源→物体
    Cpo: np.ndarray | None   # (N_P,N_O)  物体→无源
    Lam: np.ndarray | None   # (N_O,N_O)  物体响应算子（born: diag(α_O)；intra: 分块逆）
    Hro: np.ndarray | None   # (N_RCV,N_O) 物体→接收
def reduce_sample(scene, objects, mode) -> ReducedSample
def eval_reduced(rs: ReducedSample, passive_phase: np.ndarray | None) -> np.ndarray   # (N_RCV,S)
```
所有量都已经被 env "穿衣"（Schur 补消去环境节点）。`full` 模式下物体也被 Schur 消去（H、Cpp、Ep、Hrp 含物体精确效应）。
`eval_reduced`：
- n_p == 0：返回 `H`
- full：`M = diag(1/(α_base·e^{jφ})) − Cpp`；`pP = M⁻¹Ep`；返回 `H + Hrp@pP`
- born/intra：`pP_bg = M⁻¹Ep`；`pO = Lam@(Eo + Cop@pP_bg)`；`pP = pP_bg + M⁻¹(Cpo@pO)`；返回 `H + Hrp@pP + Hro@pO`
  （此时 H、Cpp、Ep、Hrp 是"只含环境"的背景量）

**必测**：`eval_reduced(reduce_sample(...), φ) == solve_scene(..., φ)`（三种模式、有/无环境、有/无无源面，相对误差 < 1e-9）。

缓存：`system.py` 对每个样本缓存 ReducedSample；n_p==0 时直接缓存 H。

---

## 3. 中继、PA 与不动点（pa.py / relay.py）— 方案 §2、§4.3–4.5

### 3.1 PA 模型（pa.py）

```python
class PA:                       # 基类
    gain_v: float               # 小信号电压增益 G_v = 10**(gain_db/20)
    def g(self, u: np.ndarray) -> np.ndarray                  # 逐元素复传输
    def wirtinger(self, u) -> tuple[np.ndarray, np.ndarray]   # (a, b)：δg = a·δu + b·conj(δu)
    def p1db_in_w(self) -> float                              # 输入 1 dB 压缩点功率 (W)
class LinearPA(PA):  LinearPA(gain_db)           # g = G_v u；a = G_v, b = 0；p1db_in_w = inf
class RappPA(PA):    RappPA(gain_db=30, psat_out_dbm=20, p=2.0, ampm_max_deg=12.0, ampm_beta=1.0, ampm=True)
```
Rapp + Saleh（方案 §4.4，为量纲统一把 AM-PM 自变量归一化到输入饱和幅度）：
- A_s = sqrt(Psat_out)（√W），A_in = A_s / G_v，r = |u|/A_in
- |g| = G_v|u| / (1 + r^{2p})^{1/(2p)}
- ∠g = ∠u + Φ(r)，Φ(r) = α_φ r²/(1+β_φ r²)，α_φ = deg2rad(ampm_max_deg)·β_φ（深饱和极限 = ampm_max_deg）；`ampm=False` 时 Φ ≡ 0
- P1dB_in = A_in² · (10^{0.1p} − 1)^{1/p}
- Wirtinger：写 g(u) = h(|u|)·u/|u|，h(r) = |g|(r)·e^{jΦ(r)}（r 为 |u|），则 a = ½(h′(r) + h(r)/r)，b = ½·e^{2j∠u}·(h′(r) − h(r)/r)；u→0 时 a→G_v，b→0（数值上对 |u|<1e-30 特判）。
- 必测：数值差分验证 wirtinger；P1dB 处增益恰好压缩 1 dB；AM-PM 极限。

### 3.2 中继电参数与权重（relay.py）

```python
@dataclass
class RelayParams:
    n_relays: int = 1
    n_elem: int = 25
    loss_rx_db: float = 2.5          # 合路网络损耗
    loss_tx_db: float = 2.5          # 功分网络损耗
    isolation_db: float = -30.0      # 收发板端口隔离 |κ0|（方案 §4.3）
    kappa0_phase_deg: float = 0.0
    phase_err_rx: np.ndarray | None = None   # (R,25) 弧度，固定的通道相位误差
    phase_err_tx: np.ndarray | None = None
    nf_db: float = 5.0               # PA 噪声系数
    noise_bw_hz: float = 1e3         # 等效噪声带宽（VNA IF 带宽量级）
    temp_k: float = 290.0
def weights(phase_rx: (R,25), phase_tx: (R,25), params) -> (w: (R,25), ct: (R,25))
    # w = 10**(-loss_rx_db/20)/sqrt(N) * exp(1j*(phase_rx + err_rx)),  ct 同理（loss_tx_db）
def bits_to_phase(bits, levels=2) -> phase   # 1-bit: π·b；2-bit: (π/2)·l
```

### 3.3 中继线性量 —— 方案 §2 的 u、κ、d、c

给定样本响应 H（(N_RCV,S)，可批量 (B,N_RCV,S)）、源幅度 s = sqrt(P_feed)：
```python
@dataclass
class RelayLinear:  u_in (B,R), K (B,R,R), d (B,K), C (B,K,R)
def relay_linear(H, w, ct, s_feed, params) -> RelayLinear
```
- 行 rx 块 r = H[:, 25r:25(r+1), :]；探头行 = H[:, NRX:, :]；源列 0 = 馈源，列 1+25s'..1+25(s'+1) = 中继 s' 的发射单元
- `u_in[b,r] = s · Σ_n w[r,n] · H[b, rxrow(r,n), 0]`
- `K[b,r,s'] = Σ_{n,m} w[r,n] · H[b, rxrow(r,n), 1+txcol(s',m)] · ct[s',m]  + κ_eff·δ_{r s'}`，
  κ_eff = κ0·10^{−(loss_rx+loss_tx)/20}，κ0 = 10^{iso/20}·e^{jφ_κ} 是**收发板端口参考面**的隔离（实测量）；PA 输出经功分网络到发射端口、
  泄漏到接收端口后再经合路网络回到 PA 输入，所以环路里要乘两段网络损耗（**已裁定**，修正了初版漏乘损耗导致默认参数恰好落在谐振点的问题）。
  默认 30 dB PA、−30 dB 隔离、2.5+2.5 dB 损耗 ⇒ 小信号环路增益 −5 dB；φ_κ 默认 90°（未知硬件参数，实验 D 扫描）。
- `d[b,k] = s · H[b, NRX+k, 0]`
- `C[b,k,s'] = Σ_m H[b, NRX+k, 1+txcol(s',m)] · ct[s',m]`
- 偏置 u₀ := 空场景（无物体）的 u_in；零陷深度 = 10log10(|u₀(b)|²/|u₀,focus|²)

### 3.4 不动点（方案 §4.5）

```python
@dataclass
class FixedPointResult:
    u (B,R); g (B,R); converged (B,) bool; iters (B,) int
    loop_gain_ss (B,)   # 小信号环路增益 ρ(G_v·K)（复矩阵谱半径）
    loop_gain_op (B,)   # 工作点线性化环路增益：F 在 u* 处 2R×2R 实雅可比的谱半径
    possibly_oscillating (B,) bool   # = ~converged | (loop_gain_op >= 1)
def solve_fixed_point(u_in, K, pa: PA, noise=None, tol=1e-12, max_iter=3000, beta0=0.5, newton=True) -> FixedPointResult
```
- 方程：`u = u_in + n + K @ g(u)`（n 为 PA 输入噪声，(B,R)，可为 None）
- LinearPA：直接解 `(I − G_v K) u = u_in + n`，converged=True，loop_gain_op = loop_gain_ss。
- 其他：阻尼迭代 `u ← (1−β)u + βF(u)`，初值 u = u_in + n；逐样本自适应 β（残差上升 → β 减半，下限 1e-3；连续 10 步下降 → β = min(1, 1.5β)）；
  收敛判据 ‖F(u)−u‖ ≤ tol·(‖u_in+n‖ + 1e-30)。之后可选 Newton 精修（实 2R×2R 雅可比，由 K 与 wirtinger 组装：
  δF_r = Σ_s K_rs (a_s δu_s + b_s conj(δu_s))），Newton 不降残差则保留迭代结果。
- 实雅可比组装（对每个复分量 δz = a δu + b δū）：`[[Re(a+b), −Im(a−b)], [Im(a+b), Re(a−b)]]`。
- 注意：阻尼迭代可能收敛到无阻尼映射的不稳定不动点，所以必须用 loop_gain_op 判定"可能自激"（CW 稳态近似，方案 §4.5）。
- 必测：线性 PA 与闭式 u_in/(1−G_vκ) 一致；小 κ 时 Rapp 解与线性解在深回退下一致；Newton 与纯迭代一致；|G_vκ|≥1 的线性中继被标记。

```python
def probe_fields(d, C, g, probe_noise=None) -> z (B,K)     # z = d + C@g (+ 探头加性噪声)
def pa_noise_power_w(params) -> float                      # k_B·T·B·F（W）
```

---

## 4. 读出与噪声（readout.py）— 方案 §4.6

```python
@dataclass
class Detector:
    kind: str = "vna"                # "ideal" | "vna" | "logdet"
    sigma_db: float = 0.002          # vna 0.002；logdet 0.1~0.5
    probe_noise_dbm: float | None = None   # 探头加性复高斯噪声功率；None 关闭（实验 B 用它扫 SNR）
    range_dbm: tuple = (-40.0, -10.0)      # logdet 量程
    offset_db: float = 0.0           # 检波前的增益/衰减；logdet 可用 calibrate() 自动设
    def calibrate(self, p_w_ref: np.ndarray) -> None   # logdet：offset 使参考读数中位数 = 量程中点；vna：不变
    def read(self, p_w: np.ndarray, rng) -> np.ndarray  # dBm 读数
```
流水线：功率(W) → dBm → +offset → logdet 时 clip 到量程 → + N(0, σ_dB²)（ideal 不加）。
探头加性噪声在 `relay.probe_fields`/系统层加到复场 z 上（在 |·|² 之前）。

---

## 5. 指标（metrics.py）— 方案 §7

```python
def walsh_coeffs(values: np.ndarray, n: int) -> np.ndarray
    # values 形状 (2**n, ...)，样本索引 i ↔ x_j = (i >> j) & 1；f̂(S) = 2^{-n} Σ_x f(x)(−1)^{|S∧x|}；返回同形状，索引 S 同编码
def walsh_weight_ge(values, n, d) -> float      # W≥d = Σ_{|S|≥d}|f̂(S)|² / Σ_{S≠∅}|f̂(S)|²（向量值对分量求和；分母为 0 返回 0）
def walsh_degree_profile(values, n) -> np.ndarray   # 长度 n，第 j 项 = 阶数恰为 j+1 的归一化权重
def parity_labels(n) -> (2**n,) int
def binary_margin_db(readings_dbm (B,2), labels) -> float   # min_x  sgn(x)·(y1−y2)，sgn = +1 奇 / −1 偶
def accuracy(pred, labels) -> float
def required_snr_db(snr_db, acc, target=0.9) -> float       # 线性插值；达不到返回 nan
def null_depth_db(u0, u0_ref) -> float
```
奇偶任务判决：y1 > y2 判为奇（label=1）。

---

## 6. 训练（dspsa.py / oracle_bits.py / oracle_torch.py）— 方案 §4.7

### 6.1 DSPSA（dspsa.py）

```python
@dataclass
class DSPSAResult: theta (p,), bits (p,) int, history: dict[str, np.ndarray], n_measurements: int
def dspsa(loss_fn: Callable[[np.ndarray, np.random.Generator], float], levels: np.ndarray, n_iter: int,
          rng, a: float | str = "auto", A: float = 400.0, alpha: float = 0.501,
          variant: str = "faithful", subset_size: int | None = None, theta0=None,
          callback: Callable[[int, np.ndarray], None] | None = None, callback_every: int = 50) -> DSPSAResult
```
- `faithful`（Wang & Spall 2011 DSPSA，即"照搬 Song"）：θ ∈ [0, L−1]（逐分量 L=levels），π(θ) = floor(θ)+½（floor 截到 ≤ L−2），
  Δ ~ Rademacher，x± = π(θ) ± Δ/2，ĝ = (L(x+) − L(x−))·Δ，θ ← clip(θ − a_k ĝ, 0, L−1−1e-9)，a_k = a/(1+k+A)^α。输出 bits = clip(round(θ))。
  **注意**：L=2（1-bit）时 floor(θ)≡0，x± 永远是一对互补码本，与当前迭代无关 → 退化。必须保留该实现并写测试证明退化
  （对满足 L(x)=L(1−x) 的函数，θ 不动/梯度估计恒为 0）。
- `subset`（1-bit 实用变体，默认用于 1-bit）：每次随机选 m = subset_size（默认 max(2, p//8)）个分量，Δ_S ~ Rademacher，其余分量取当前判决 round(θ)；
  x±_S = (1 ± Δ_S)/2（1-bit）或 π(θ_S) ± Δ_S/2（多电平），ĝ_S = (L(x+) − L(x−))Δ_S，只更新 S。每次迭代仍 2 次测量、扰动 ±½。
- `a="auto"`：先用 10 次独立梯度估计的 mean|ĝ| 设 a = 0.3·(1+A)^α / mean|ĝ|（这 20 次测量计入 n_measurements）。
- history 至少记录：`loss_plus`, `loss_minus`（每迭代），以及 callback 返回值不强制。

### 6.2 1-bit 模型 oracle（oracle_bits.py）

```python
def oracle_bits(loss_fn_exact: Callable[[np.ndarray], float], levels, rng, n_restarts=8, max_sweeps=30,
                anneal_steps=0) -> tuple[np.ndarray, float]
```
随机重启 + 坐标下降（逐分量尝试所有电平，取最优，直到一整轮无改进），可选模拟退火预热。用于"这个 1-bit 结构最多能做到多好"。

### 6.3 连续相位 + 自动微分 oracle（oracle_torch.py）

方案写的是 JAX；本机已有 torch 2.8，**用 torch 实现**（complex128，CPU）。
```python
def eval_reduced_torch(rs: ReducedSample, passive_phase: torch.Tensor | None) -> torch.Tensor   # 与 solver.eval_reduced 完全同公式
class TorchSystem:  # 由 system.PhysicalSystem 构造，持有各样本的 ReducedSample、PA 参数、RelayParams、s_feed、读出、任务损失
    def readings_db(self, phases: dict[str, torch.Tensor]) -> torch.Tensor     # 无噪声读数 (B,K) dBm
    def loss(self, phases) -> torch.Tensor
def oracle_continuous(system, n_restarts=8, steps=1500, lr=0.05, seed=0) -> dict   # best_phases, best_loss, exact accuracy/margin（用 numpy 系统按连续相位复评）
```
- 相位参数：`phase_rx (R,25)`, `phase_tx (R,25)`, `phase_p (N_P,)`（按条件存在与否）。
- 不动点：可用"no_grad 求解 + 隐函数求导"（推荐：自定义 autograd.Function，反传解 (I − J_Fᵀ)λ = ḡ，J_F 为 2R×2R 实雅可比），
  或从解出发再展开若干步。**必须**有 torch 梯度 vs 有限差分的测试，以及 torch 前向 vs numpy 前向一致性测试（相对误差 < 1e-8）。
- 自激惩罚：loop_gain_op ≥ 0.95 时加 `100*(ρ−0.95)²`。

---

## 7. 任务、系统与条件（tasks.py / system.py）— 方案 §5–§6

### 7.1 条件（方案 §5）

| 代号 | surface | PA | 源功率设定 |
|---|---|---|---|
| M0 | 无源 1-bit 反射面：位置 = 收、发两块板的全部 50 个单元位置与法向，Γ=1，方向图同中继单元 | — | 与同实验中 M3 相同的 P_feed |
| M1 | 中继 | LinearPA(G=Rapp 小信号增益) | 由 `backoff_db` 指定（默认 0，即与 M3 同功率），用于与 M2/M3/M4 逐一配对 |
| M2/M3/M4 | 中继 | RappPA | backoff_db = −10 / 0 / +10 |

源功率标定：`P_feed` 使"参考 PA 输入功率" = P1dB_in · 10^{backoff/10}。参考 = 空场景下、接收 1-bit 码本取**对馈源的量化共轭相位**
（相位共轭后对 16 个全局相位偏移量化取 |u₀| 最大者）时的 |u₀|²/P_feed。所有结果都要额外记录**实际工作点**：每样本 PA 输入功率相对 P1dB 的 dB 值（均值/范围）。

### 7.2 SystemConfig 与 PhysicalSystem

```python
@dataclass
class SystemConfig:
    preset: str = "L90"; fold_deg: float | None = None; n_relays: int = 1
    condition: str = "M3"; backoff_db: float | None = None   # None → 按条件表
    object_mode: str = "full"; material: str = "water"; voxel_d: float = λ/10
    block_direct: bool = False; skip_gain_db: float = 0.0
    bg_cancel: str = "free"            # "free" | "null"：null 时损失加零陷惩罚（方案 §5 背景相消）
    null_target_db: float = -20.0; null_weight: float = 1.0
    isolation_db: float = -30.0; kappa0_phase_deg: float = 0.0
    ampm: bool = True; pa_gain_db: float = 30.0; psat_out_dbm: float = 20.0; rapp_p: float = 2.0
    detector: str = "vna"; detector_sigma_db: float | None = None; probe_noise_dbm: float | None = None
    passive_levels: int = 2            # M0 无源面电平数（2 = 1-bit，4 = 2-bit）
    freq_hz: float = 3.5e9
class PhysicalSystem:
    def __init__(self, cfg: SystemConfig, task: "Task")
    n_params: int; levels: np.ndarray      # 可训练参数（中继：先 rx 后 tx，按中继顺序；M0：无源单元）
    def phases_from_params(self, params_int) -> dict          # 'phase_rx','phase_tx','phase_p'
    def forward(self, phases: dict, rng=None, noisy=True, sample_idx=None) -> Measurement
    def measure(self, params_int, rng) -> Measurement         # = forward(phases_from_params, noisy=True)
    def loss(self, params_int, rng=None, noisy=True) -> float  # 任务损失 + 零陷惩罚 + 自激惩罚
    def evaluate(self, params_int, rng, n_noise=20) -> dict   # 无噪声准确率/裕量 + 有噪声平均准确率 + 自激标记 + 工作点
    def to_torch(self) -> TorchSystem
@dataclass
class Measurement: readings_dbm (B,K); p_w (B,K); u (B,R)|None; g (B,R)|None; u_in; flags (B,) bool; loop_gain_op (B,)
```
自激处理：被标记的样本在训练损失里按"错误 + osc_penalty(=1.0)"计；`evaluate` 输出 `oscillating_run`（最终码本下任一样本被标记）——
按方案 §4.5，这样的 run **单独统计、不参与精度均值**。

### 7.3 任务（tasks.py）

```python
@dataclass
class Sample: objects: NodeSet; label: int; meta: dict
class Task:  name; n_probes; samples: list[Sample]; def loss(readings_dbm, labels) -> float; def predict(readings_dbm) -> np.ndarray
class ParityTask(Task):  ParityTask(n, slots_center, spacing=λ/2, diameter=λ/3, material, voxel_d, axis)  # 2^n 个样本，x 编码见 metrics
class LetterTask(Task):  3 类 A/B/C，像素 λ/2，材料默认 metal；可选抖动 dist_jitter=λ/4、power_jitter_db=6（meta 里记录，系统层按 meta 缩放 P_feed）
class LocalizationTask(Task): 8 格阈值定位（Song 式）：ROI 分 2×4 格，单物体随机落在某格；每格训练一个码本做"在/不在"阈值判决（单探头）
class FocusTask(Task):  无物体，最大化指定探头功率（实验 F）
```
损失：二分类 = mean softplus(−s/τ)·τ，s = sgn(label)·(y1−y2)（dB），τ = 1 dB；多分类 = 读数/τ 的 softmax 交叉熵；阈值检测 = 两类读数的 logistic 判别（阈值取两类中位数中点）。

---

## 8. 默认参数（constants.DEFAULTS，Phase 0 实测后替换）

PA：gain 30 dB（两级）、Psat_out 20 dBm、Rapp p=2、AM-PM 极限 12°、β_φ=1；网络损耗收/发各 2.5 dB（净环路小信号增益 25 dB）；
NF 5 dB、噪声带宽 1 kHz；κ0 = −30 dB；单元方向图 q=1, d0=π, 后瓣 −25 dB；喇叭 q=10；体素 λ/10；奇偶散射体直径 λ/3；DSPSA A=400, α=0.501。

---

## 9. 实验脚本（experiments/）— 方案 §6

每个脚本：`python -m ris_nlsim.experiments.exp_x_... [--quick] [--workers N] [--seeds 5] [--out results/...]`；
结果写 JSONL（每行一个 run）+ 汇总 CSV + Markdown 表 + PNG 图；`--quick` 须在笔记本上 ≤ 10 分钟跑完。
- F 硬件对齐：几何（馈源 az 30° 1.8 m / az 50° 1.5 m，折叠角扫描）下中继相对同尺寸金属板（PO 单元）增益；反推与实测 20–27 dB 吻合所需的净增益；门 G1 判据 ±3 dB。实测表来自 `测试数据.xlsx`（实测数据，不随仓库发布）。
- A 奇偶阶梯（主实验）：n ∈ {2,3,4} × 条件 M0–M4 × 物体模式 × εr × N_r；DSPSA + 1-bit oracle + 连续 oracle；报"全对比例"、平均准确率、裕量、W≥3 分解（e、u_in、u、g、y1−y2）。
- B SNR：probe_noise_dbm 扫描；准确率–SNR 曲线；所需 SNR。
- C 归一化利弊：字母 + 8 格定位，功率 ±6 dB / 距离 ±λ/4 抖动；不变性损失。
- D 反馈环：flat0 预设，κ0 ∈ {−40,−25,−15} dB × 物体距离；类间距、自激比例（M1 vs M4）。
- E 可训练性：M2→M4，DSPSA 达到 oracle 90% 所需迭代数。

## 10. 自检（selfcheck.py，对应门 G1）

`python -m ris_nlsim.selfcheck` 依次打印 PASS/FAIL：互易性、光学定理、Born=Full（εr=1.01）、体素收敛（εr=4，λ/10→λ/20 场变化）、
DWBA 仿射性（M0+Born 的 y1−y2 Walsh W≥3 < 1e-20）、M0+Born 3 位奇偶 oracle 解不出（全对率 < 1）、torch/numpy 前向一致。
