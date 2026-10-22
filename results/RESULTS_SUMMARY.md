# 结果汇总（2026-10-20；A/B/C/E 为 full + informative，D/F 仍为 `--quick`）

参数为文献典型值（PA 30 dB / Psat 20 dBm / Rapp p=2 / AM-PM 12°，隔离 −30 dB、相位 90°），Phase 0 实测后需重跑。A/B/C/E 为 full 模式、informative 场景、每格 5 个种子（`--workers 16`，墙钟 A 1521 s、B 1289 s、C 559 s、E 299 s；`--seeds 1` 估计全部远低于 2 h，**未缩网格**）；D、F 仍是 quick（D 3 个种子），结论是"方向"。连续 oracle（cont-plus）的设定与种子在 quick / full 相同、计算确定，所以 A 的 full + 水格点、B 的 oracle 码本曲线与 quick 逐位一致；full 新增的是 intra / 金属格点、字母任务、4000 次 DSPSA 与 loss-oracle 诊断。

## 输出目录

| 实验 | 目录 | 场景 / 训练器 |
|---|---|---|
| 自检（门 G1 仿真侧） | `python -m ris_nlsim.selfcheck` | 9 PASS / 0 FAIL |
| F 硬件对齐 | `results/exp_f/20261019-201804/` | — |
| A 奇偶阶梯 | `results/exp_a/20261020-105703/` | informative full：n ∈ {2, 3} × {M0, M1, M1b@+25, Rapp@+20/+25/+30, M3, M4} × {full, intra} × {水, 金属} 全交叉（`--design factorial`，N_r = 1，间距 λ/2，无功率扫描块）+ 门 G1 born 格点；sign_acc（4000 次）+ 1-bit oracle + loss-oracle + 连续 oracle（cont-plus）。quick：`results/exp_a/20261020-044142/` |
| A（旧场景对照） | `results/exp_a/20261019-202857/` | baseline，旧 subset DSPSA |
| B SNR | `results/exp_b/20261020-112736/` | informative full：parity3 / parity2 / 字母（金属）× full × {M0, M1, M1b@+25, Rapp@+25, M3}，10 个名义 SNR 点，sign_acc 4000 次。quick：`results/exp_b/20261020-044621/` |
| C 不变性 | `results/exp_c_invariance/20261020-115347/` | informative full（默认网格：字母 / 定位 × 5 条件 × 4 抖动 × {oracle, DSPSA 3000 次}），sign_acc。quick（baseline，3 种子）：`results/exp_c_invariance/20261020-045207/` |
| D 反馈环 | `results/exp_d_feedback/20261019-201154/` | flat0，旧 subset DSPSA（结论只用 oracle 码本与随机码本相位扫描）；未重跑（D 现在也有 `--scenario`，只沿用直达路径开关） |
| E 可训练性 | `results/exp_e_trainability/20261020-120526/` | informative full（默认网格：parity3 / 字母 / 聚焦 × 5 条件，DSPSA 4000 次，oracle 8 次重启），sign_acc。quick（baseline）：`results/exp_e_trainability/20261020-045004/` |
| 训练器基准 | `results/bench_trainers/baseline_b3000_s5/`；新训练器确认 `_scratch/integration-T/bench_confirm_b1000/` | — |

## 各研究问题

| RQ | 结论（A/B/C/E full，D quick） | 依据 |
|---|---|---|
| RQ1 PA 饱和是否增加表达能力 | **支持（仅连续相位）**：3 位奇偶 Rapp@+25 连续 oracle 在 full/intra × 水/金属 四格都 5/5 全对，robust 4/5（full 水）、5/5（其余三格）；同功率 M1、Bussgang M1b@+25、M0、名义 M3 全部 0/5（最好 7/8）。Δrobust（Rapp@+25 − M1b@+25，连续 oracle）= +0.80 [0.24, 1.00]（full 水）、+1.00 [1.00, 1.00]（intra 水、full / intra 金属），Δsolved 四格都 +1.00。**1-bit 码本（1-bit oracle、DSPSA）任何条件都 0/5**。2 位 XOR 全部条件可解（best 5/5、连续 5/5 robust；1-bit accuracy oracle 单独 2–5/5），无差别，符合预测。 | A |
| RQ2 两种非线性互补还是替代 | **PA 单独足够，未见散射非线性的贡献**：intra（物体之间不耦合，物体 W≥3(e) ≈ 1e-31）下 Rapp@+25 仍 5/5 robust（水、金属）；full（W≥3(e) ≈ 2.0e-4 水、2.4e-4 金属）下线性 / 无源条件仍 0/5，Rapp 条件也不比 intra 好（Rapp@+20/+25/+30 robust：水 full 1/4/2 vs intra 2/5/1，金属 full 3/5/3 vs intra 1/5/2）。即在本几何（ROI 0.35 m、λ/3 球、间距 λ/2）下物体多次散射既替代不了 PA，也没有互补收益；间距 0.35λ 与介质 εr 块（A 的 blocks 设计）未跑。 | A |
| RQ3 增益还是非线性 | **支持**：低 SNR（≤ 0 dB）各条件无差别；3 位奇偶连续 oracle 码本 Rapp@+25 − M1 在 10/20/30 dB 为 +0.14/+0.35/+0.48，相对增益匹配的 M1b@+25 为 +0.14/+0.34/+0.43。字母任务（新增）：1-bit oracle 码本 Rapp@+25 − M1 在 10 dB 为 +0.20，但相对 M1b@+25 为 −0.03 ⇒ 字母上的差别来自增益不是非线性。 | B |
| RQ4 归一化利弊 | **仍不足以下结论；只有定位一侧有弱支持**：32 项配对检验 4 项 95% CI 不含 0——定位 Rapp@+20 无抖动 oracle d_acc = −0.17 [−0.26, −0.09]、距离抖动 DSPSA −0.17 [−0.28, −0.07]（饱和压掉幅度信息 → 定位更差，支持），功率抖动 oracle +0.13 [+0.04, +0.23]（AGC 帮助固定阈值定位，支持）；字母 DSPSA Rapp@+20 双抖动不变性损失 +0.07 [+0.01, +0.14]（与"饱和更不变"相反）。字母在 informative 下接近天花板（acc 0.88–1.00），名义 M4 训练后在字母上 sat ≤ 0.2（不检验 AGC），定位上 0–0.65。线性自检：oracle 的 acc(M1@M4) − acc(M1@M3) 全为 0，唯一偏离 0.094 来自 1 个 DSPSA run。 | C |
| RQ5 反馈环谐振区 | 方向性支持（quick）：|Gκ| → 1 时判决灵敏度随 κ₀ 相位 0.7–2.3 dB 变化；+1 dB 以上线性中继全自激，Rapp 中继部分相位下被饱和锁定为稳定态。 | D |
| RQ6 非线性是否更难训 | **informative full 下不支持**（quick baseline 的弱支持没有复现）：字母 5 个条件都能训到 oracle 90%（达到比例 0.8–1.0），ρ(回退, 首达) = −0.11、ρ(回退, acc 首达) = −0.21；聚焦更快（ρ = −0.49）；3 位奇偶损失判据不可检验（全部删失），acc 判据下达到比例反而随饱和上升（M1@M3 0.20 → M2 0.60 → M3 0.80 → M4 0.80 → Rapp@+20 1.00，acc* = 7/8）。另：faithful DSPSA 在 1-bit 上退化（max\|L(x) − L(1−x)\| = 4e-14）；旧 subset DSPSA 停在 τ·ln2 平台，新 sign_acc 修复（得分 −0.01 → 0.90）。 | E、bench |
| 门 G1 | 仿真侧 PASS（M0+Born 3 位奇偶不可解、W≥3 ≈ 1e-26；A full informative 内的 G1 对照也 PASS，W≥3 = 8.7e-26，最好 7/8）；实验 F 侧"有条件 PASS"：依赖未测的 κ₀ 相位，开环 17/17。 | selfcheck、A、F |

## 对实验设计的含义

1. 默认几何（物体 0.5 m、直达保留）被入射偏置主导，奇偶阶梯没有判别力；需要吸波屏挡住馈源→探头直达、物体拉近（~0.35 m）。
2. 有判别力的功率窗口是"接收波束对准馈源时 PA 输入比 P1dB 高 20–30 dB"：训练会把偏置零陷 20–28 dB，名义 M3/M4 实际在线性区。实验室源功率需约 30–40 dBm（1.8 m）。
3. 3 位奇偶的判别只在连续/多比特相位下出现，且要求相位精度 ~1–2°；1-bit 硬件上建议阶梯止于 XOR，非线性收益用裕量曲线呈现。物体间多次散射（full vs intra）与金属物体都不改变结论，硬件上不必追求紧密排布或强散射体。
