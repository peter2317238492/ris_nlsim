# ris_nlsim：有源非线性超表面 × 散射非线性 虚拟实验仿真器

按《有源非线性超表面 × 散射非线性：虚拟实验方案 v0》实现的仿真器（方案原文未随仓库发布，文中"方案 §x"指其章节）。
实现契约见 [DESIGN.md](DESIGN.md)（物理约定、核心接口）与 [INTEGRATION.md](INTEGRATION.md)（系统、任务、实验）。

> 状态（2026-10-20）：v0.2。核心求解器、中继/PA、训练器、实验 A–F、训练器基准与自检全部可运行，600 个测试通过，自检门 G1（仿真侧）PASS。
> 所有实验目前只跑过 `--quick`，结果是**初步**的；PA、隔离度等参数是文献典型值，需在 Phase 0 实测后替换（见 §6）。

## 1. 快速开始

在仓库根目录下执行。需要 Python 3.12 及 numpy、scipy、matplotlib、torch、openpyxl、pytest（见 `requirements.txt`）：

```bash
python -m pip install -r requirements.txt
```

```bash
python -m pytest tests -q
```

```bash
python -m ris_nlsim.selfcheck --fast
```

```bash
python -m ris_nlsim.experiments.exp_a_parity --quick --workers 12
```

实验输出写在 `results/`、中间文件写在 `_scratch/`，两者都不进版本库。实验 F 需要实测数据表 `测试数据.xlsx`（不随仓库发布），用 `--xlsx` 指定路径。

其余实验：`exp_f_hardware`（硬件对齐，先跑）、`exp_b_snr`、`exp_c_invariance`、`exp_d_feedback`、`exp_e_trainability`、`exp_g_phase_bits`（相位精度阶梯），以及训练器基准 `bench_trainers`。通用参数 `--quick / --workers / --seeds / --out`；A–E 另有 `--scenario {baseline,informative,informative_probe}`（D 保持 flat0 几何，只沿用其中的直达路径开关），A/B 有 `--dspsa-variant`、E 有 `--variant`。各自 `--help` 有完整选项。
输出在 `results/<实验>/<时间戳>/`：`summary.md`（结论表）、`summary.csv`、`runs.jsonl`（逐 run 原始结果）、`config.json`、`*.png`。

在 Python 里直接用：

```python
from ris_nlsim.system import SystemConfig, PhysicalSystem, make_task
from ris_nlsim import train
import numpy as np

cfg = SystemConfig(preset="L90", condition="M3", object_mode="full", material="water")
sysm = PhysicalSystem(cfg, make_task(cfg, "parity", n=3))
rng = np.random.default_rng(0)
res = train.run_oracle_bits(sysm, rng, objective="accuracy")     # 1-bit 结构能力上界
print(res["eval"]["acc_clean"], res["eval"]["margin_db"])
```

## 2. 模型一览（对应方案 §2、§4）

| 层 | 实现 | 文件 |
|---|---|---|
| 场求解 | 3D 标量耦合偶极子，G = e^{−jkR}/(4πR)；物体体素、环境点、无源面单元同一个线性方程组 | `greens.py` `scene.py` `solver.py` |
| 物体三开关 | `born`（畸变波 Born：物体只响应无物体背景场，**保证 M0+Born 的探头场对存在向量 x 严格仿射**）/ `intra`（物体内部多次散射）/ `full`（完全耦合） | `solver.py` |
| 极化率 | 默认 `scalar_rr`（标量 LS 的一致离散 = 局域场 + 辐射修正，收敛到标量 Mie）；`cm_rr` 可选；Born 用 χV | `materials.py` |
| 中继 | 25 收 → 合路 → PA → 功分 → 25 发，1-bit；u、κ、d、c 从散射矩阵显式算出（方案 §2 四个推论） | `relay.py` |
| PA | Rapp AM-AM + Saleh AM-PM，Wirtinger 导数解析；M1 = 同增益线性，M1b = Bussgang 等效增益 | `pa.py` |
| 反馈环 | 阻尼不动点 + Newton；"可能自激" = 不收敛或工作点线性化环路增益 ≥ 1（另有相位稳健、小信号两种判据） | `relay.py` |
| 读出 | VNA（0.002 dB）/ 对数检波器（AD8317 类，量程裁剪）/ 探头加性噪声 | `readout.py` |
| 训练 | DSPSA：1-bit 默认 `sign_acc`（判对优先可测目标 + sign-SPSA，§4.7 骨架不变），另有 Wang & Spall 原版 `faithful` 与旧 `subset`；迭代局部搜索 ILS（RQ6 对照）；1-bit 模型 oracle；torch 连续相位 oracle（隐函数求导）及设计初值版 cont-plus | `dspsa.py` `ils.py` `oracle_bits.py` `oracle_torch.py` `train.py` |
| 指标 | Boolean Fourier W≥d、判决裕量、所需 SNR、零陷深度 | `metrics.py` |
| 任务 | n 位奇偶、字母 A/B/C（薄金属板）、Song 式 8 格定位（3 探头阈值 → 3-bit）、聚焦 | `tasks.py` |
| 系统 | 几何预设 L90 / flat0、条件 M0–M4、源功率标定、torch 镜像 | `system.py` |

## 3. 与方案不同或方案没写清、由仿真器裁定的地方

1. **"Born" = 畸变波 Born（DWBA）**：物体只对无物体背景场（含无源面、环境）一阶响应，散射场经背景传到接收端但不再照射任何物体。这样 M0+Born 的 y₁−y₂ 严格是 x 的二次型，3 位奇偶自检（门 G1）在数学上成立（实测 W≥3 ≈ 1e-29）。
2. **M1+Born 不是严格二次的**：中继的发射波会照到物体，经物体回到接收阵，于是环路 K(x)、发射→探头系数 C(x) 都依赖 x，线性 PA 下 u = u_in/(1−GK(x)) 也含高阶项。方案 §6.1 表中"M1+Born —"只在"物体不在发射波束里、且没有经物体的环路"时成立（实验 A 的 `M1@M3:noTx` 条件即此理想情况）。门 G1 只用 M0+Born。
3. **极化率**：方案写的 CM 形式在标量方程里会收敛到 ε_eff = 1+3(εr−1)/(εr+2) ≤ 4（压缩 εr 扫描、金属变成 εr=4 介质），所以默认改用标量一致的 `scalar_rr`；体素按材料自动取 d = min(λ/10, λ/(π|√εr|))（水约 λ/28）。
4. **κ₀ 定义在收发板端口**：环路里乘两段网络损耗。默认 30 dB PA、−30 dB 隔离、各 2.5 dB 损耗 ⇒ 小信号环路增益 −5 dB；κ₀ 相位未知，默认 90°（实验 D/F 扫描）。
5. **DSPSA 在 1-bit 参数上退化**：中点法则 π(θ)=floor(θ)+½ 在 1-bit 时恒为 ½，两次测量永远是一对互补码本，与当前迭代无关；中继读数又对整体取反对称，所以梯度估计只剩测量噪声（实验 E 有数值演示）。Song 2026 用 2-bit 没有这个问题。仿真器保留原版（`variant="faithful"`）。1-bit 默认 `variant="sign_acc"`：随机子集 ±½ 中点扰动（每迭代 2 次测量）+ sign-SPSA 增益（a_k = a/(1+k+A)^0.602，A=400，首步 0.1）+ 判对优先的可测目标 + pocket。旧的子集变体（`"subset"`，τ=1 dB softplus + a='auto'）会停在 y₁≈y₂ 的 τ·ln2 平台，详见 INTEGRATION §T。
6. **源功率标定是开环的**：M2/M3/M4 = 接收波束对准馈源（量化共轭码本）时开环 PA 输入比 P1dB 低 10 / 等于 / 高 10 dB。训练后的码本会压掉入射偏置，**实际闭环工作点往往比名义值低 5–20 dB**，所有结果表都单列实测工作点；另有 `Rapp@+20` 等更高功率条件。
7. **8 格定位**按 Song 2026 原文：一套码本、3 个探头各自阈值 → 3-bit 格号；留出集评估用训练集阈值。

## 4. 初步结果（A/B/C/E/G 与 D、F 均为 full；详见 [results/RESULTS_SUMMARY.md](results/RESULTS_SUMMARY.md)）

**两个场景**（`--scenario`，INTEGRATION §S）：`baseline` = L90，水球 λ/3 距接收板 0.5 m，馈源→探头直达保留；
`informative` = 同几何但 ROI 拉近到 0.35 m、吸波屏挡住馈源→探头直达（实验 A/B 的 quick 默认；下面 A/B/C/E 的 full 结果都用它）。

- **自检（门 G1 仿真侧）**：互易性 2e-15、光学定理 5e-16、M0+Born 仿射性 W≥3 ≈ 1e-26、M0+Born 与理想线性中继 3 位奇偶都解不出（最好 7/8）、1 位有/无正对照可解、torch/numpy 一致 1e-15。
- **baseline 场景是偏置主导的（方案推论 3）**：物体对 PA 输入的调制约 −30 dB，判决量 0.01–0.1 dB，1-bit 结构连 2 位 XOR 都只能勉强做到；各条件无差别。
- **实验 A（informative full，n ∈ {2, 3} × full/intra × 水/金属，主结论 RQ1、RQ2）**：
  - 2 位 XOR：M0、M1、M1b@+25、M3、M4、Rapp@+20/+25/+30 全部可解（best 5/5 全对、连续相位 5/5 robust；1-bit accuracy oracle 单独 2–5/5），符合"二次型能做 XOR"。
  - 3 位奇偶（完全耦合、水）：**连续相位**下 Rapp@+20/+25/+30 分别 5/5、5/5、3/5 全对（robust ≥ 0.5 dB：1/5、4/5、2/5）；同功率线性 M1、Bussgang 增益匹配 M1b@+25、无源 M0、名义 M3 全部 0/5（最好 7/8）。
    **非线性收益（Rapp@+25 − M1b@+25，连续 oracle）：Δrobust = +0.80 [0.24, 1.00]，Δsolved = +1.00**，95% CI 不含 0 ⇒ 在这个场景里 PA 饱和本身提供了线性中继给不了的高阶项。intra（水、金属）与 full 金属三格 Δrobust 都是 +1.00 [1.00, 1.00]。
  - **RQ2（full vs intra，含密排补充组）**：间距 λ/2 时物体多次散射很弱（W≥3(e) ≈ 2e-4），线性 / 无源条件 0/5，判决量的高阶项来自 PA。物体贴紧（间距 0.35λ）、εr = 10、4 块中继时线性中继也能勉强解出（M1b@+25 robust 4/5，裕量约 0.8 dB；饱和中继约 7 dB）；物体互不耦合（intra）时线性中继始终 0/5 ⇒ 强散射非线性可**部分替代** PA，PA 的额外收益随之变小（Δrobust 从 +1.0 降到 +0.2），未见互补。
  - **1-bit 码本在任何条件下都解不出 3 位奇偶**（含深度模拟退火搜索，最好 7/8）；连续解要求相位精度约 1–2°。⇒ 3 位奇偶检验需要多比特 / 模拟移相；1-bit 硬件上奇偶阶梯止于 XOR。
  - 名义 M3/M4（聚焦码本标定）在训练后码本下实际工作在线性区（训练会把入射偏置零陷 20–28 dB）；有判别力的窗口是聚焦标定之上 +20…+30 dB。
- **训练器（RQ6，bench_trainers）**：旧 DSPSA 子集变体在分类任务上停在 τ·ln2 平台（chance 归一化得分 −0.01），根因是 a='auto' 步长两头失配 + softplus 损失奖励 y₁≈y₂ 塌缩；新默认 `sign_acc` 0.90（35/50 达标），ILS 约 0.95（3000 次测量）。faithful DSPSA 在 1-bit 上退化（ĝ 只剩测量噪声），数值确认。
- **实验 E（RQ6，informative full）**："深饱和更难训"不成立——字母任务 5 个条件都能训到 oracle 90%（达到比例 0.8–1.0；ρ(回退, 首达) = −0.11、ρ(回退, acc 首达) = −0.21），聚焦任务饱和越深越快（ρ = −0.49），3 位奇偶的 acc 达到比例反而随饱和上升（M1@M3 0.20 → Rapp@+20 1.00，acc* = 7/8）。quick（baseline 场景）里字母上的弱支持（ρ = +0.41）没有复现。
- **实验 B（RQ3，informative full）**：2 位 XOR 达到 90% 准确率所需 SNR：连续码本约 2 dB（M1、M3、Rapp@+25 相近），1-bit oracle 码本约 18–21 dB（逐种子均值），噪声下训练的 DSPSA（4000 次）约 22–31 dB。3 位奇偶只有 Rapp@+25 的连续码本能达到（所需约 15 dB，5/5 种子）。同一 SNR 下 Rapp@+25 相对 M1 的准确率差：−20…0 dB 时约 0，10 dB 时 +0.14，20 dB 时 +0.35，30 dB 时 +0.48（相对 M1b@+25：+0.14/+0.34/+0.43）。字母任务（金属）上 Rapp@+25 − M1 在 10 dB 有 +0.20（1-bit oracle 码本），但相对 M1b@+25 只有 −0.03，是增益效应。即低 SNR 下增益主导、各条件无差别，非线性收益只在高 SNR、且只在需要高阶项的任务上出现，与 RQ3 假设一致。
- **实验 F（硬件对齐）**：开环预测下，"收00-发00"与实测增益提升（约 25 dB）吻合需 PA 小信号增益约 29 dB（净 24 dB）；默认 κ₀（−30 dB/90°）时 17/17 非共面配置落在 ±3 dB 内，但依赖未测的 κ₀ 相位（0° 时只有 2/17）。共面配置实测比仿真多掉 6–9 dB，模型解释不了（多半是共面时馈源泄漏到发射阵、隔离变差）。
- **实验 C（RQ4，informative full，5 个种子）**：**仍不足以下结论，只有定位一侧有弱支持**。32 项配对检验 4 项显著：Rapp@+20 定位无抖动 oracle −0.17 [−0.26, −0.09]、距离抖动 DSPSA −0.17 [−0.28, −0.07]（饱和压掉幅度 → 定位更差），功率抖动 oracle +0.13 [+0.04, +0.23]（AGC 帮助固定阈值定位）；字母 DSPSA Rapp@+20 双抖动 +0.07 [+0.01, +0.14] 与"饱和更不变"相反。字母接近天花板（acc 0.88–1.00），名义 M4 训练后基本不饱和（字母 sat ≤ 0.2），形状识别一侧的 AGC 假设没有被检验到。
- **实验 D（RQ5，full）**：环路小信号增益超过 1 时，线性中继全部自激；饱和中继的自激比例显著更低（M3 −0.47 [−0.66, −0.28]，M4 −0.80 [−0.95, −0.65]，相位稳健判据下仍显著）⇒ PA 饱和能把环路钳住。但"接近谐振时类间距变大"没有看到（Spearman ρ 在 −0.2 到 +0.1 之间）。
- **实验 G（相位精度阶梯，新增）**：informative 场景、3 位奇偶、5 个种子，把连续解量化到 L 电平再精修：饱和中继 Rapp@+25 在 1/2/3/4/5 bit 时分别解出 0/0/1/4/5 个（物体完全耦合），线性对照 M1、M1b@+25 在任何精度下都是 0/5。静态相位误差 1° 时仍有 65–88% 可用，2° 时 35–58%，5° 时不到 10%。⇒ 3 位奇偶检验需要 ≥ 4-bit 移相器（或矢量调制器）与约 1° 的相位稳定度。运行：`python -m ris_nlsim.experiments.exp_g_phase_bits`。

## 5. 相关工作与新颖性边界

- PA 饱和作激活函数已有先例：Hua、Bergel、Gündüz，arXiv 2607.18354（MIMO 中继级联，Rapp p=2，仅 AM-AM，反向传播，线性基线未做增益匹配）；同组 2604.04212、综述 2602.14094。
- 编码 / 结构非线性的分离：Hammami…del Hougne，arXiv 2603.13602。无源谐振腔递归散射做 2 位 XOR：Zhu…Chen，arXiv 2609.22745（2 位 XOR 在 Born+平方律下本来就可解，3 位奇偶才是严格检验）。
- 环路稳定性：Larsson & Bai，arXiv 2405.01074。最接近的硬件：崔铁军组 AF-RIS，arXiv 2501.09759。Bussgang 增益：Demir & Björnson，arXiv 2604.13745。
- 仍可主张的空白：1-bit 单神经元中继（收→合路→PA→分路→发）、物体作为输入的 3D 散射场景、n≥3 奇偶阶梯 + Minsky–Papert + Walsh W≥d、增益匹配（含 Bussgang）的线性对照、AGC 不变性（RQ4）、经场景与板间泄漏的反馈环 + PA 钳位（RQ5）、非线性对 DSPSA 的影响与 1-bit DSPSA 退化（RQ6）。
- 可作数值对照、只参考不拷代码（许可证限制）：PhysFad（2D 标量耦合偶极子）、torchGDM（3D 矢量 GDM，GPL-3）、ADDA、openEMS RIS 模板、HermesPy（Rapp/Saleh，AGPL-3）。

## 6. 局限与 Phase 0 之后要替换的参数

- 标量模型：忽略极化与单元间互耦；绝对数值只能靠实验 F 和实测方向图标定，结论只依赖 M0–M4 的相对比较。
- 反馈环是 CW 稳态近似：自激判据只是必要条件，真实时域动态不在范围内。
- 待实测替换（`constants.DEFAULTS`）：PA 增益 / P1dB / Psat / AM-PM、噪声系数、收发隔离及其相位（随折叠角）、单元方向图 q、网络损耗。
- 尚未做：openEMS/MEEP 全波对照（方案 §4.8）、Phase 4 硬件小验证、D/F 的 full 模式、A 的 N_r / 间距 0.35λ / 介质 εr 块（informative）。
- 1-bit 下 3 位奇偶"解不出"是大量搜索后的经验结论，没有证明是结构极限。
