# ris_nlsim 集成层规格（DESIGN.md 的补充，§7 / §9 / §10 的细化）

本文件细化 `tasks.py`、`system.py`、`train.py`、`selfcheck.py` 与 `experiments/` 的实现。与 DESIGN.md 冲突时，以**已实现并通过测试的核心模块的实际接口**为准（先读源码），其次本文件，再次 DESIGN.md。

## A. 场景构建 `system.build_scene(cfg, n_probes) -> (Scene, info)`

1. `relays = geometry.make_relays(cfg.n_relays, fold_deg)`，fold_deg = cfg.fold_deg 或预设值。
2. 参考点：`rx_c` = 各中继接收板中心的平均，`tx_c` = 发射板中心平均，`c` = 中继中心平均。
3. 馈源：`point_from(rx_c, feed_az_deg, 0, feed_dist)`，法向 = 单位向量(rx_c − feed)，`Pattern(q=horn_q)`；feed_weights = [1]。
4. 探头：前 n_probes 个 `point_from(tx_c, probe_az_deg[i], 0, probe_dist)`，法向指向 tx_c，`Pattern(q=probe_q)`。
5. ROI 中心：预设 `roi_ref` 为 "rx_center" 用 rx_c，为 "center" 用 c，再加 `roi_offset`；实验可用 `cfg.roi_offset` 覆盖（实验 D 扫物体距离）。
6. 中继条件（M1–M4）：`tx = [NodeSet(relay.tx_pos, relay.tx_normal 广播, elem pattern)]`（每中继一个），rx 同理；passive 为空。
7. M0：tx/rx 为空；passive = 所有中继的收+发板全部单元（每中继 50 个，法向各自板法向），`alpha = alpha_passive(1.0, k, λ)`。
8. env：默认空（预留 `cfg.env_points: list[(pos, alpha)]`）。
9. `block_direct_feed_probe = cfg.block_direct`，`skip_gain_db = cfg.skip_gain_db`，`include_direct_tx_rx = False`。

`info` 至少含：relays、rx_c、tx_c、c、roi_center、feed_pos、probe_pos。

## B. 源功率标定 `reference_feed_power(cfg) -> (p_feed_w, p_ref_per_w)`

- 始终用**中继版本**的同一几何（即使 cfg 是 M0）建空场景（无物体），取 H0 的 rx 行、馈源列：v_n。
- 聚焦码本：对 16 个全局偏移 θ_j = 2πj/16，b_n = 0 若 cos(−∠v_n + θ_j) ≥ 0 否则 1；`u0 = Σ_n w_n(b) v_n`（含 loss_rx/√N）；取 |u0| 最大者。N_r>1 时对各中继取 |u0|² 的平均。
- `p_ref_per_w = |u0|²`（P_feed = 1 W 时 PA 输入功率），`P_feed = P1dB_in(Rapp 参数) · 10^{backoff/10} / p_ref_per_w`。
- backoff：M2 −10、M3 0、M4 +10、M1 用 `cfg.backoff_db`（默认 0）、M0 固定用 0（与 M3 同功率）。
- M1 的"等增益"定义（`cfg.m1_gain`）：`"small_signal"`（默认，= Rapp 小信号增益，方案 §4.4 原意）或 `"bussgang"`（更严格的对照：
  饱和 PA 在该源功率下的最佳线性近似 G_B = Σ_b g(u_b)·conj(u_b) / Σ_b |u_b|²，u_b 取"同一源功率、同一几何、随机 16 个码本 × 全部样本"下 Rapp 中继的不动点 PA 输入；
  参见 Demir & Björnson arXiv 2604.13745）。Bussgang 版本记为 "M1b"，实验 A/B 的非线性收益同时报相对 M1 与 M1b 的值。
- 系统对象保存 `p_feed_w`、`s_feed = sqrt(p_feed_w)`、`u0_focus_abs2`（P_feed 下的聚焦 |u0|²，用于零陷深度归一化）、`p1db_in_w`。

## B2. 已裁定的核心模块变更（Workflow 1 之后）

- κ0 为端口参考面隔离，环路中用 `RelayParams.kappa0`（= κ0_port × 收发网络损耗）；`RelayParams.kappa0_port` 是原始隔离。默认相位 90°。
- 材料默认 `scalar_rr`（见 DESIGN §1.3）。`SystemConfig.voxel_d=None` 表示按材料自动：介质 d = min(λ/10, λ/(π|√εr|))，金属 λ/20；
  字母任务默认薄板（厚度 = 1 个体素，d = λ/10），否则体素数爆炸。所有结果记录实际 voxel_d 与 `effective_eps`。
- 多中继：不同中继之间的 tx→rx 自由空间耦合始终计入 H；同一中继的板间泄漏只由 κ0 表示。
- `relay.solve_fixed_point` 对长时间不收敛的样本有停滞早停；线性 PA 在 σ_min(I−G_vK) 极小时标记谐振。
- `losses.py` 已有 binary / multiclass / focus / threshold / loc3bit 损失，均支持 `reduction="none"`，以及 `task_loss(kind, ...)`、`chance_loss`。
  tasks.py / system.py 必须复用它（numpy 与 torch 同源）。
- `oracle_torch.TorchSystem` 的构造签名见其源码（含 `task_kwargs`，例如 loc3bit 的 thresholds_db / loss_weight）；
  `oracle_continuous(system_or_torchsystem, ...)` 若对象有 `evaluate_phases` 会自动复评。

## B3. 实验构建者反馈后的共享模块增补（2026-10-19，审查 S0–S2）

- 自激判据 `SystemConfig.osc_criterion`（`relay.OSC_CRITERIA`）："cw"（默认，即 DESIGN §3.4 的 ~converged | ρ_op ≥ 1）、
  "robust"（再要求相位稳健工作点环路增益 ρ_rob = ‖K‖₂·max(|a|+|b|) < 1：任意泄漏相位 / 带内频率不起振的充分条件，
  R=1 时恰为最坏泄漏相位下的 ρ_op）、"small_signal"（再要求 ρ(G_v K) < 1：源关闭也不起振；此时 M1 与 M2–M4 同码本标记相同，
  RQ5 的"M4 自激比例低于 M1"按构造不会出现）。强注入可把 ρ_ss ≥ 1 的中继锁成稳定饱和态，这是 CW 判据的定义行为，不是 bug。
  `Measurement` 增加 `loop_gain_robust`、`ss_unstable`、`robust_unstable`、`d`、`C`；`evaluate` 总报 `n_ss_unstable`、
  `ss_unstable_run`、`n_robust_unstable`、`robust_unstable_run`、`loop_gain_robust_max`、`osc_criterion`，可事后换判据统计。
- `PhysicalSystem.direct_field()`：方案 §2 的 d_k（中继：s_b·H[探头, 馈源]；M0：去掉无源面后的探头场）。
- loc3bit 留出集评估：`calibrate_thresholds(params)`（训练集阈值）→ `evaluate(..., thresholds_db=t)`；"auto" 阈值总由
  被评估样本自身的标签求出，在测试集上会泄漏标签。
- `LocalizationTask` 增加 `dist_jitter`、`power_jitter_db`（与 LetterTask 同一约定，见 E）。
- `train.run_oracle_bits(objective=...)`："loss"（默认）| "margin"（最大化最小带符号裕量）| "accuracy"（最大化判对数）。
  softplus 损失在不可解任务上 ≈ τlog2 − mean(s)/2，其 oracle 的 acc_clean 可低于随机，不是"最好可达准确率"。
  `Task.sample_margins_db` 给逐样本带符号裕量（margin_db 为其最小值）。
- `train.run_dspsa(pocket=True)`：另返回训练中测到的最好码本（model-free）及其复评；`train.calibrate_dspsa_gain` /
  `dspsa.auto_gain`：在参考条件上标定一次 a，再把同一数值传给各条件（a="auto" 会把整体变平的损失地形归一化掉，RQ6）。
- `system.bare_loop_gain_db(cfg)` / `isolation_for_loop_gain_db(cfg, g)`：只含板间泄漏的小信号环路增益与其反函数。
- born（volume_rr）下 material="metal" 是 εr_eff = 4 介质（构造任务时 RuntimeWarning）；导体请用 full / intra。
- `experiments.common` 导入时设置中文字体回退（DejaVu Sans 优先，缺字回退 Microsoft YaHei 等），图中可直接用中文。

## C. PhysicalSystem 细节

- `levels`：中继条件 = 2 × (50·N_r)（参数顺序：中继 0 的 rx 25、tx 25，中继 1 的 rx 25、tx 25……）；M0 = `cfg.passive_levels` × (50·N_r)。
- `phases_from_params`：1-bit → π·b；多电平 → (π/2)·l（M0 2-bit）。
- 预计算：对每个样本 `reduce_sample(scene, sample.objects, cfg.object_mode)` 并缓存；中继条件（n_p=0）直接缓存 `H`（堆叠为 (B,N_RCV,S)）。
- 每样本功率缩放 `power_scale[b] = 10^{meta.get("power_db",0)/10}`；样本有效馈源幅度 `s_b = s_feed·sqrt(power_scale[b])`。
- `forward(phases, rng, noisy, sample_idx)`：
  - 中继：`relay_linear(H, w, ct, 1.0, params)` 后把 `u_in`、`d` 逐样本乘 `s_b`；noisy 时 PA 输入噪声 `n ~ CN(0, pa_noise_power_w)`；`solve_fixed_point`；`z = probe_fields(d, C, g)`；noisy 且 `probe_noise_dbm` 非空时 z 加 CN(0, P_n)；`p_w = |z|²`；`readings = detector.read(p_w, rng)`（noisy=False 时用 ideal 读数）。
  - M0：`H_b = eval_reduced(rs_b, phase_p)`；`z = s_b·H_b[N_RX + k, 0]`；其余同上。
- 检测器：`cfg.detector` ∈ {"ideal","vna","logdet"}，sigma 默认 vna 0.002 / logdet 0.3；logdet 在构造时用"随机码本 + 全部样本"的无噪声功率 `calibrate` 一次后固定。
- `loss(params, rng, noisy=True)`：逐样本任务损失（losses.py 增加 `reduction="none"` 支持，或在本模块逐样本计算，但数学必须与 losses.py 一致）；被标记 `possibly_oscillating` 的样本损失替换为 `tau·log(2) + osc_penalty`（osc_penalty = 1.0）；`bg_cancel == "null"` 时加 `null_weight·softplus(nd_db − null_target_db)`，`nd_db = 10log10(|u0(b)|²/u0_focus_abs2)`（空场景、当前 rx 相位、无噪声；N_r>1 取各中继最大值）。
- `evaluate(params, rng, n_noise=20)` 与 `evaluate_phases(phases, rng, n_noise=20)` 返回 dict：
  `acc_clean`、`all_correct`（无噪声全部判对）、`margin_db`（二分类；多分类用最小 top1−top2）、`acc_noisy`（n_noise 次有噪声读数平均）、
  `oscillating_run`（任一样本被标记）、`n_flagged`、`loop_gain_ss_max`、`loop_gain_op_max`、`pa_in_rel_p1db_db`（每样本 10log10(|u|²/P1dB_in) 的 min/mean/max，中继条件）、`null_depth_db`、`loss_clean`。
- `nonlinearity_profile(phases)`（仅当任务是 ParityTask）：在无噪声下对 x 的 2^n 个样本计算并返回 W≥2、W≥3：
  `e`（s·H[:, :N_RX, 0]，接收单元场向量；M0 用无源单元处的激励场或直接用探头场）、`u_in`、`u`、`g`、`z`（探头复场）、`y_diff_lin`（P1−P2，W）、`y_diff_db`。
- `to_torch()`：用 oracle_torch.TorchSystem 的构造签名打包（relay：reduced=缓存的 ReducedSample 列表，pa dict 来自 cfg，relay dict 来自 RelayParams，null_ref 在 bg_cancel="null" 时提供）。

## D. train.py

```python
def run_dspsa(system, n_iter, rng, variant="auto", subset_size=None, a="auto", eval_every=100, ..., noise_rng=None) -> dict
    # variant "auto"：所有 levels>2 → "faithful"，否则 "sign_acc"（T 节的新 1-bit 默认；旧默认 "subset" 仍可显式选择）；
    # callback 每 eval_every 次记录无噪声 loss/acc（不计入测量次数）
    # 返回 {"params","eval","history","n_measurements","variant","train_objective",...}
def run_ils(system, n_meas, rng, extras=False, ...) -> dict    # T 节：迭代局部搜索，与 run_dspsa 同形返回
def run_oracle_bits(system, rng, n_restarts=8, max_sweeps=30) -> dict      # 用 system.loss(noisy=False)
def run_oracle_continuous(system, n_restarts=4, steps=800, lr=0.05, seed=0) -> dict   # torch 优化后用 system.evaluate_phases 复评
```

## E. tasks.py

- `ParityTask(n, roi_center, spacing=λ/2, diameter=λ/3, material="water", voxel_d=λ/10, axis=(0,0,1), k)`：槽位 `parity_slots`；每槽体素化球（group=槽号，alpha 由 `polarizability(MATERIALS[material], vol, k)`）；样本 i 的 x_j=(i>>j)&1，label=popcount 奇偶；n_probes=2；predict：y1>y2 → 1。
- `LetterTask(roi_center, letters="ABC", pixel=λ/2, voxel_d=λ/5, material="metal", plane_u=(0,1,0), plane_v=(0,0,1), n_per_class=1, dist_jitter=0.0, power_jitter_db=0.0, rng=None)`：
  字母平面法向 = plane_u×plane_v；抖动：沿法向位移 U(−dist_jitter, +dist_jitter)，meta["power_db"] = U(−power_jitter_db, +power_jitter_db)；n_probes=3；argmax。
- `LocalizationTask(roi_center, cell=λ/2, grid=(2,4), obj_diameter=λ/3, material="water", n_per_cell=1, jitter=0.0, plane_u=(0,1,0), plane_v=(0,0,1), thresholds_db="auto", loss_weight=1.0, rng=None, ..., dist_jitter=0.0, power_jitter_db=0.0)`：
  （B3：dist_jitter / power_jitter_db 与 LetterTask 同一约定，沿平面法向位移、meta["power_db"]；每样本抽样顺序 格内偏移 → 法向 → 功率，未启用的不耗随机数）
  **按 Song 2026（Nat. Commun. 17:7807）已核实的做法**：一套码本、**3 个探头各自阈值判决 → 3-bit 标签编码 8 个格子**（格 c 的标签位 j = (c>>j)&1，探头 j 读数 > 阈值 t_j 判 1）。
  格子在 plane_u/plane_v 平面内 2×4 排列；样本 = 物体在每格（含格内抖动，可多次）；n_probes=3；kind="loc3bit"。
  阈值：`thresholds_db` 为长度 3 的数组时固定使用（Song 实测为 −50/−53/−41.5 dB，量级依赖其硬件，本仿真默认不用）；"auto" 时每个探头 t_j = (median(y_j|bit=1) + median(y_j|bit=0))/2（不回传梯度，训练中随读数更新）。
  损失 = 3 个探头上的阈值 logistic 之和：s_bj = sgn(bit)·(y_bj − t_j)，mean_b Σ_j τ·softplus(−s_bj/τ)，再乘 `loss_weight`（Song 文中定位损失有一个修正系数 2.6，含义未完全公开，作为可配权重保留，默认 1.0）。
  predict：3 个比特拼成格号；准确率 = 格号全对比例；同时报逐比特准确率。
- `FocusTask(n_probes=1)`：单个空样本，focus 损失（最大化探头 0 功率）。
- 所有 Task 暴露：`name`、`n_probes`、`samples`、`labels`（ndarray）、`kind` ∈ {"binary","multiclass","loc3bit","focus"}、`loss(readings_db, labels, tau)`、`predict(readings_db)`。

## F. experiments/common.py

- `run_jobs(fn, jobs: list[dict], workers) -> list[dict]`：ProcessPoolExecutor（spawn 安全，fn 为模块顶层函数），每个 job 内 `torch.set_num_threads(1)`、`os.environ["OMP_NUM_THREADS"]="1"`；异常捕获为 `{"error": traceback}` 不中断其余 job。
- `save_jsonl(rows, path)`、`aggregate(rows, keys, metrics) -> list[dict]`（均值、标准差、95% 区间、n）、`to_markdown(table)`、`savefig(fig, path)`（matplotlib Agg 后端）。
- 每个实验输出目录 `results/<exp>/<YYYYmmdd-HHMMSS>/`：`runs.jsonl`、`summary.csv`、`summary.md`、`*.png`、`config.json`。
- CLI 通用参数：`--quick`、`--workers`（默认 max(1, os.cpu_count()-2)）、`--seeds`、`--out`。

## G. 实验定义（--quick 须 ≤ 10 分钟）

- **F（exp_f_hardware.py）**：几何 = 实测设置（馈源 az 30°/1.8 m 与 az 50°/1.5 m；探头在发射板法向 1.8 m）；折叠角扫描；中继码本 = 量化共轭（接收对准馈源、发射对准探头）与 1-bit oracle（最大化探头功率）两种；
  金属板参考 = 与一块板等面积（可选两块）的 PEC 板，PO 单元 λ/4，Γ=−1，作为 env（带 Pattern(q=1, d0=4πA_cell/λ², backlobe −60)），板法向取馈源与探头方向的角平分线（镜面最佳取向）。
  输出：相对金属板增益 vs 净增益（G_PA − 网络损耗）曲线；反推使"收00-发00"类配置落在实测均值所需的 G_PA；读取 `测试数据.xlsx`（实测数据，不随仓库发布，`--xlsx` 指定） 的"增益相较金属板增益提升"两张表做对比；门 G1：预测与实测区间偏差 ≤ ±3 dB 判 PASS。
- **A（exp_a_parity.py）**：网格 n × 条件(M0..M4) × object_mode × material × N_r；每格 ≥5 seeds；每 run：DSPSA（quick 1500 迭代 / full 4000）、1-bit oracle、可选连续 oracle；记录 evaluate 结果 + nonlinearity_profile（随机码本上 5 个平均 + 训练后码本）。
  汇总表：全对比例、平均准确率、裕量、W≥3（e/g/y）；**非线性收益 = acc(M3或M4) − acc(M1 同功率)**。quick：n∈{2,3}，M0/M1/M3/M4，born/intra/full（审查 R2-INT-1 后改为
  `experiments.common.EXP_A_OBJECT_MODES`：intra 是 RQ2 的强度匹配对照，born 留给门 G1），water，N_r=1，5 seeds。
  "最好可达准确率"请用 `run_oracle_bits(objective="accuracy")`，不要用默认损失 oracle 的 acc_clean（B3）。
- **B（exp_b_snr.py）**：3 位奇偶（+ 可选字母），M0/M1/M3/M4；probe_noise_dbm 扫描（quick 5 点）；在该噪声下用 DSPSA 训练并评估；准确率–SNR 曲线（SNR 定义：训练后码本下探头信号功率中位数 / P_n）与所需 SNR。
- **C（exp_c_invariance.py）**：字母 A/B/C + 8 格定位；有/无抖动（±6 dB 功率、±λ/4 距离）；M1 vs M3 vs M4；不变性损失 = 无抖动准确率 − 有抖动准确率。
- **D（exp_d_feedback.py）**：flat0 预设，3 位奇偶；κ0 ∈ {−40,−25,−15} dB × ROI 距离 ∈ {0.25,0.4,0.8} m（quick 取子集）；M1 vs M4（+M3）；报类间距（margin）、自激比例（oscillating_run 比例，含 DSPSA 中间码本被标记的比例）、准确率（排除自激 run）。
  注意（审查 S2-3）：B2 裁定环路里乘网络损耗后，默认 30 dB PA、2.5+2.5 dB 损耗下 κ0 = −25 dB 的裸环路增益恰为 0 dB
  （`system.bare_loop_gain_db`），无物体样本 K = κ_eff 处在临界上，线性中继按谐振容差必被标记 → M1 自激 100% 是构造性的。
  建议另加偏离临界的点（如 −28 dB = −3 dB 环路增益、−22 dB = +3 dB），或直接按环路增益取点（`isolation_for_loop_gain_db`）；
  自激比例按三种判据（evaluate 的 n_flagged / n_robust_unstable / n_ss_unstable，B3）分别报告。
- **E（exp_e_trainability.py）**：3 位奇偶，M2/M3/M4（+M1）；DSPSA 学习曲线（每 50 迭代无噪声 loss）；先跑 1-bit oracle 得到 oracle loss L*；"达到 oracle 90%" 定义为 (L0 − L_k) ≥ 0.9 (L0 − L*)，L0 为初始 loss；报所需迭代数（未达到记为 >n_iter）。

## G2. 来自文献核实的补充（2026-10-19，Song 2026 原文 + GitHub 调研）

- DSPSA：Song 的 a 由早期梯度统计标定（文中示例 a=1），投影边界 τ = 1e-10（θ 截到 [0, L−1−τ]）；dspsa.py 的 clip 常数统一为 1e-10。
  Song 用 2-bit，中点法则无退化；1-bit 下退化是本仿真器的新发现，README 与结果中明确写出。
- 环路稳定性诊断（relay.py 可选增强）：除谱半径外，增加 Gershgorin 上界 `max_r Σ_s |G_v K_rs|`（≥ ρ，便宜的保守判据）与 `|det(I − G_v K)|`（接近 0 即谐振，参见 Larsson & Bai arXiv 2405.01074），一并写进 evaluate 输出。
- README 必须有"相关工作与新颖性边界"一节：PA 饱和作激活函数已有先例（Hua/Bergel/Gündüz arXiv 2607.18354，Rapp p=2、仅 AM-AM、反向传播、线性基线未做增益匹配）；
  编码/结构非线性分离（Hammami…del Hougne arXiv 2603.13602）；无源谐振腔递归散射做 2 位 XOR（Zhu…Chen arXiv 2609.22745，注意 2 位 XOR 在 Born+平方律下本就可解，3 位奇偶才是严格检验）；
  最接近的硬件为崔铁军组 AF-RIS（arXiv 2501.09759）。本仿真器可主张的空白：1-bit 单神经元中继、物体作为输入的 3D 散射场景、n≥3 奇偶阶梯 + Minsky–Papert + Walsh W≥d、增益匹配线性对照（含 Bussgang）、AGC 不变性、经场景与板间泄漏的反馈环 + PA 钳位、非线性对 DSPSA 的影响。
- 可选对照（不在本轮实现，README 列为后续）：PhysFad（2D 标量耦合偶极子，MATLAB，无许可证，只能对照结构）、torchGDM（3D 矢量 GDM，GPL-3，量化标量近似误差）、ADDA（物体散射对照）、openEMS 模板（全波收发板耦合）、HermesPy（AGPL-3，仅用于核对 Rapp/Saleh 公式）。所有这些只作数值对照，不拷代码。

## H. selfcheck.py（门 G1 的仿真侧）

依次打印 PASS/FAIL 与关键数值：互易性；光学定理；Born≈Full（εr=1.01）；体素收敛（可 --fast 跳过）；DWBA 仿射性（M0+Born 的 y1−y2 线性功率 W≥3 < 1e-20）；
M0+Born 3 位奇偶：1-bit oracle 与 DSPSA 均不能全对（报最佳准确率）；torch/numpy 前向一致（中继 M3 与 M0 各一个）；源功率标定自洽（M3 下聚焦码本的 PA 输入 = P1dB）。退出码：全部 PASS 为 0。

## S. 有信息量的场景（几何 / 功率探索之后，2026-10-20）

探索代码与原始结果：`_scratch/scenarios/geometry_power/`（`gp_common.py`、`scan_xor.py`、`scan_ladder.py`，表在 `results/table_*.md`）。

**几何覆盖**：`SystemConfig.feed_az_deg / feed_dist / probe_az_deg（列表，前 n_probes 个）/ probe_dist`，None = 预设值；
经 `system.preset_geometry → scene_geometry → build_scene`（A.3–A.4 同一构造规则），并计入 `_scene_key`（derive 不复用旧几何的样本响应）。

**命名场景** `system.SCENARIOS` + `apply_scenario(cfg_or_kwargs, name)`（只覆盖几何 / 场景字段，条件、物体模式、材料由实验网格决定；
"baseline" 会把 roi_offset / block_direct / 馈源探头覆盖复位成预设默认，场景之间可互换）：

| 名称 | 字段 | 说明 |
|---|---|---|
| baseline | L90 预设（ROI = 接收板前 0.5 m，馈源→探头视距保留） | 原默认奇偶场景 |
| informative | L90 + `roi_offset=(0.35,0,0)` + `block_direct=True` | 吸波屏挡住馈源→探头视距、ROI 拉近；水球 λ/3、间距 λ/2（任务默认） |
| informative_probe | informative + `probe_az_deg=(45,75,60)` | 探索中最好的探头几何旋钮 |

**结论（如实）**：
1. **baseline 是偏置主导的（方案推论 3）**：物体对 PA 输入的调制 ~ −30 dB，1-bit 结构对 2 位 XOR 的裕量只有 ~0.01–0.06 dB
   （物理上脆弱：≪ 0.1 dB 检波器级噪声）；intra / full 下 3 位奇偶对 RQ1 没有判别力。
2. **informative**（L90，ROI 0.35 m，遮挡直达，水 λ/3）：M1 的 2 位 XOR 1-bit 裕量中位数 ~ +3.9 dB。
3. **3 位奇偶（full）只在连续相位下有判别力**：Rapp @ 回退 +20 / +25 / +30 dB（focus 标定）时连续相位 oracle 各 3/3 种子解出
   （solved），最好裕量 3.9 / 3.9 / 2.0 dB；robust（≥ 0.5 dB）2/3、3/3、2/3（逐种子裕量 0.02–3.9 dB，不是一律 ~4 dB）；
   同功率 M1、M1b（Bussgang，@+20/+25/+30）与 M0 均 0/3（最好 7/8，裕量 ≤ 0）；名义 M3 / M4 也 0/3。
4. **没有任何 1-bit 码本在任何探索场景里解出 3 位奇偶**（含深度模拟退火）；连续解要求相位精度 ≤ 1–2°（`phase_err` / `phase_bits`）。
   所以 1-bit 硬件 + 3 位奇偶不是 RQ1 的可行检验；n = 3 的判别需要连续 / 多比特相位。
5. 本场景里判决量的 W≥3 几乎全部来自 PA（物体 W≥3(e) ~ 2e-4；+20 / +30 dB 时 PA 超额 W≥3(y) ≈ 0.74 / 0.45）。
6. 训练后的码本把偏置零陷 ~25–28 dB，所以名义 M3 / M4（focus 标定）实际工作在线性区；有判别力的窗口是 focus +20…+30 dB
   （`cal_ref="nulled"` 时 M2 / M3 / M4 约等于 focus +11 / +21 / +31）。
7. **搜索很难**：随机初值的连续 oracle（softplus）从未解出；解只能从"设计初值"（接收码本零陷馈源、相干接收物体场，推论 3）
   出发再做 soft-min 裕量精修得到。复现确认：`exp_a_parity.run_cont_plus` 在探索者的种子上与 `_scratch` 结果逐位一致，
   但单次设计初值重启的成功率只有 ~20–60%（随种子而变），实验 A/B 的 cont-plus 因此用 6 次设计初值重启并按 (判对数, 最小裕量)
   而不是 softplus 损失挑选（探索者原实现按损失挑选，会丢掉已全对的候选）。quick 的 5 个种子里解出的比例会低于探索者的 3/3。

**实验接入**：
- 实验 A：`--scenario {baseline,informative,informative_probe}`；**quick 默认 informative**（full 默认仍 baseline）：条件
  M0、M1、M1b@+25、Rapp@+20、Rapp@+25、Rapp@+30、M3、M4（`Rapp@<dB>` = Rapp PA + 显式回退，与实验 C/E 同一写法，内部条件 M4），
  只跑 full + 门 G1 的 born 格点（M0、M1@M3:noTx，n = 3），不再加功率扫描块；连续 oracle 开且默认 cont-plus（τ = 0.1 dB、
  1 × 600 步随机重启 + 6 × 600 步设计初值重启 + 300 步 soft-min 精修，40 dB 动态范围受限读数选择）。
  **非线性收益主表用 robust 判据**（全对且最小裕量 ≥ 0.5 dB），solved（> 0.006 dB）与 Δacc 并列；非 baseline 场景下没有同回退
  M1 的 Rapp 条件与 M1（0 dB）配对（线性中继的无噪声 y1 − y2 与源功率无关），M1b 只与同回退配对。
- 实验 B：同样的 `--scenario`；quick 默认 informative：parity3 / parity2 × full × {M0, M1, M1b@+25, Rapp@+25, M3}，
  DSPSA 1000 迭代（从 1500 减少以控制墙钟），oracle 曲线的连续 oracle 用 cont-plus；RQ3 条件对按同一配对规则动态生成。
- 实验 C：`--scenario`（默认 baseline，quick / full 都不改）；`make_c_cfg` 把 `system.apply_scenario` 叠加到 L90 + 条件上，字母 / 定位任务随 ROI 一起移到 0.35 m。
- 实验 D：`--scenario`（默认 baseline）；几何保持 flat0 + `--distances` 网格，只沿用场景里的直达路径开关 `D_SCENARIO_FIELDS = (block_direct, skip_gain_db)`（经 `apply_scenario` 取值）。L90 的 ROI 偏移与 informative_probe 的探头方位在 flat0 下没有对应，不沿用（informative_probe ≡ informative）。实测 flat0、d = 0.25 m：遮挡后馈源→探头视距项为 0，读数 / 随机码本灵敏度变 ~0.1–0.7 dB，环路诊断（ρ、自激比例、rel_du）不变——视距项不进环路。
- 实验 E：`--scenario`（默认 baseline）。
- 实验 A 的 `--materials` 另接受 `metal`（只能显式给出；intra / full 下 scalar_rr 软球极限）。
- `--scenario baseline` 复现改动前的默认网格（job 完全相同）。
- 墙钟（--workers 12）：实验 A quick（informative，90 个 run）：按 `--seeds 1` 小跑（18 个 run，84 s）在 12 进程并行负载下的逐 run 实测耗时模拟调度，估计 ≈ 6 min（CPU 合计 ~72 min；Rapp 条件每 run ~65–80 s，其中 cont-plus ~50 s）；实验 B quick（informative，300 个 job）：`--seeds 1` 小跑（60 个 job，71 s）估计 ≈ 5 min（CPU ~57 min）。实验 A 的 DSPSA 迭代保持 1500，实验 B 由 1500 降到 1000（summary 头里注明）。小跑结果（种子 0）与探索一致：n = 3 时只有连续 oracle 在 Rapp@+20 / +25 / +30 全对（裕量 0.23 / 3.98 / 0.005 dB，robust 只有 +25），M0 / M1 / M1b@+25 / M3 / M4 均不全对；门 G1 PASS。

## T. 训练器选择（训练器探索之后，2026-10-20）

探索代码与原始结果：`_scratch/trainers/dspsa_family.py`（DSPSA 家族）、`_scratch/trainers/alt_optimizers.py`（ILS 等）。
bench_trainers，5 个分类任务（letters、loc3bit、parity2_born、parity3_born、parity2_full）× 10 个种子（0–4 调参、5–9 确认），
预算 3000 次测量；分数 = chance 归一化 (acc − chance)/(ref − chance) 的均值，达标 = acc ≥ 0.9·ref（ref = 全部种子 oracle 池）：

| 训练器 | 分数 | 达标 run | 备注 |
|---|---|---|---|
| dspsa_subset（旧默认） | −0.01 | 0/50 | τ·ln2 平台 |
| dspsa_subset_pocket | 0.41 | 6/50 | |
| coord_descent | 0.64 | 17/50 | 测量版逐单元贪心 |
| **dspsa_sign_acc（新默认，= df_u_nores）** | **0.90** | **35/50** | 预算 1000 时领先 |
| ils（= ao_ils_acc） | ~0.95（预算 3000）/ 0.71（预算 1000） | 50/60、32/60（含 focus） | 大预算最好 |

消融（累加）：基线 −0.01 → + pocket 0.41 → + sign-SPSA（首步 0.1、α 0.602）0.57 → + 判对优先目标 0.81 → + 逐单元自适应 τ 0.865
→ 去掉重启 0.90（与带重启的 df_u_p300 统计上不可分，取更简单者）。

**根因**：(1) a = "auto" 在 θ0 处按 mean|ĝ| 归一化，两头都把步长定错——初始码本的偏置使 |ΔL| 偏大时（parity2_full：a = 2.6）
偏置拉平后步长 ~0.003、θ 冻结；初始 |ΔL| 很小时（loc3bit：a = 64）步长 ~0.17、每迭代翻 ~0.8 位的随机游走；
(2) softplus τ = 1 dB 的损失在判决量 ≪ τ 时 ≈ τ·ln2 − mean(s)/2，奖励把 y1 ≈ y2 塌缩（"拉平偏置"）而不是判对，停在 τ·ln2 平台。

**新默认 `train.run_dspsa(variant="auto")` → "sign_acc"（1-bit 参数；多电平仍 faithful，旧 "subset" 可显式选择）**：
- 目标（只用一次测量的读数，`train.AccuracyObjective`）：J = −(n_correct + ½·mean_{b,j} tanh(M_bj/τ_j))，M = `Task.decision_units`
  （二分类 sgn·(y1 − y2)；多分类 y_true − y_k，k ≠ true；loc3bit sgn(bit)·(y − t_auto)）；被标记样本判错、M = −1e3；
  τ_j = max(1e-3, 0.5·exp(EMA_0.05 log 稳健离散度_j))，每次测量先更新再求值。focus 任务退回系统可测损失（读数上的任务损失）。
- 增益：sign-SPSA ĝ = sgn(J+ − J−)·Δ，a_k = a/(1+k+A)^0.602，A = 400，a = 0.1·(1+A)^0.602（首步 0.1，无标定；k 只在实际更新时递增）。
- 扰动：子集 m = max(2, p//8) 的 ±½ 中点扰动（每迭代 2 次测量，方案 §4.7 骨架）。
- pocket：键 = 判对优先目标 @ τ = cfg.tau_db（带噪测量，逐单元平局项；二分类时即 `oracle_objective("accuracy")` 作用在测量读数上），
  严格改进才替换；**输出 = pocket 码本**（`params`/`eval`），最终 round(θ) 在 `decision_params`；无重启。
- 实现：`dspsa.dspsa(grad="sign", step0=…, gain_index="update")`、`train.acc_first_loss_fn`、`train._run_sign_acc`；
  旧变体（subset / faithful）行为逐位不变（与插件里独立重写的基线 df_base / df_base_pk 同一码本）。
- `train.run_ils(system, n_meas, rng, extras=False)`（`ils.ils_bits`）：同一判对优先目标（样本裕量、固定 τ），单比特首次改进爬山 +
  2–6 位踢出 + 中性漂移，返回 pocket；`extras=True`（= ao_ils4_k0_drift）另加拒绝时重测 x*（≤ 8 次）与末段 4×3 终选赛。
- 与探索插件逐位一致（`tests/test_trainers_t.py`：parity2_born / letters / focus 的码本与学习曲线；`run_dspsa` / `run_ils` 用
  `noise_rng` 分开算法与噪声随机流后与插件同一码本）。
- bench_trainers 内置 `dspsa_sign_acc`、`ils`、`ils_extras`；参照 ref 改为池化缓存目录里同一实例哈希的**全部**缓存种子
  （`n_pooled_seeds`），不再随本次 --seeds 变化。
- 实验：A / B / C / D / E 都经 `train.run_dspsa` 自动用新默认；A / B 的"训练到位"对 sign_acc 改用准确率判据
  （无噪声 acc ≥ 0.9·acc*，acc* = 同种子 1-bit accuracy oracle，同 bench），sign_acc 的输出就是 pocket，不再单列 "dspsa_pocket"；
  实验 E 默认 sign_acc（无需增益标定，符号梯度的步长对各条件天然相同），`--variant subset` 保留原 shared / auto 增益设计；
  `calibrate_dspsa_gain(variant="auto")` 在 1-bit 上按 subset 扰动标定。
- 确认（本次集成，bench 预算 1000，种子 0–2，参照池化 10 个缓存种子；acc 均值 · 达标种子数 · chance 归一化分数）：

| 训练器 | parity2_born | letters | loc3bit |
|---|---|---|---|
| dspsa_sign_acc（新默认） | 1.00 · 3/3 · 1.00 | 0.89 · 2/3 · 0.83 | 0.54 · 0/3 · 0.67 |
| ils | 1.00 · 3/3 · 1.00 | 0.67 · 0/3 · 0.50 | 0.58 · 0/3 · 0.73 |
| dspsa_subset（旧默认） | 0.25 · 0/3 · −0.50 | 0.44 · 0/3 · 0.17 | 0.25 · 0/3 · 0.20 |
| dspsa_subset_pocket | 0.83 · 1/3 · 0.67 | 0.56 · 0/3 · 0.33 | 0.29 · 0/3 · 0.27 |
