"""tasks.py（DESIGN §7.3，INTEGRATION E）：样本构造、标签编码、判决与损失委托 losses.py。"""
import numpy as np
import pytest

from ris_nlsim import losses
from ris_nlsim.constants import K0, LAMBDA
from ris_nlsim.geometry import LETTERS
from ris_nlsim.materials import polarizability
from ris_nlsim.metrics import parity_labels
from ris_nlsim.tasks import FocusTask, LetterTask, LocalizationTask, ParityTask, resolve_voxel_d

ROI = np.array([0.5, -0.11, 0.0])


def test_resolve_voxel_d_rules():
    # 介质：min(λ/10, λ/(π|√εr|))；水 ≈ λ/28；εr=4 → λ/10；金属 λ/20；显式值原样返回（INTEGRATION B2）
    assert resolve_voxel_d("eps4") == pytest.approx(LAMBDA / 10)
    assert LAMBDA / resolve_voxel_d("water") == pytest.approx(np.pi * abs(np.sqrt(78 - 10j)), rel=1e-12)
    assert 27 < LAMBDA / resolve_voxel_d("water") < 29
    assert resolve_voxel_d("metal") == pytest.approx(LAMBDA / 20)
    assert resolve_voxel_d("water", 0.01) == 0.01
    with pytest.raises(ValueError):
        resolve_voxel_d("water", -1.0)


@pytest.mark.parametrize("n", [2, 3])
def test_parity_samples_and_labels(n):
    t = ParityTask(n, ROI, material="eps10", voxel_d=LAMBDA / 10)
    assert t.kind == "binary" and t.n_probes == 2 and len(t.samples) == 2 ** n
    assert np.array_equal(t.labels, parity_labels(n))
    nv = t.n_vox_per_object
    alpha = polarizability("eps10", np.pi * (LAMBDA / 3) ** 3 / 6 / nv, K0)
    for i, s in enumerate(t.samples):
        x = [(i >> j) & 1 for j in range(n)]
        assert list(s.meta["x"]) == x
        assert len(s.objects) == nv * sum(x)
        if sum(x):
            # group = 槽号，存在的槽各有 nv 个体素；极化率与材料一致
            assert sorted(set(s.objects.group.tolist())) == [j for j in range(n) if x[j]]
            assert np.allclose(s.objects.alpha, alpha)
            for j in range(n):
                if x[j]:
                    c = s.objects.pos[s.objects.group == j].mean(axis=0)
                    assert np.allclose(c, t.slots[j], atol=1e-12)
    # 槽位沿 z、间距 λ/2、以 ROI 居中
    assert np.allclose(np.diff(t.slots[:, 2]), LAMBDA / 2)
    assert np.allclose(t.slots.mean(axis=0), ROI)


def test_parity_predict_margin_and_loss_delegation():
    t = ParityTask(2, ROI, voxel_d=LAMBDA / 10)
    y = np.array([[-20.0, -21.0], [-22.0, -20.0], [-19.0, -25.0], [-30.0, -29.0]])
    assert np.array_equal(t.predict(y), [1, 0, 1, 0])
    labels = t.labels                                   # [0,1,1,0]
    sgn = np.where(labels == 1, 1.0, -1.0)
    assert t.margin_db(y) == pytest.approx(np.min(sgn * (y[:, 0] - y[:, 1])))
    assert t.loss(y, labels, 1.0) == pytest.approx(losses.binary_loss(y, labels, 1.0), rel=1e-15)
    per = t.loss(y, labels, 0.5, reduction="none")
    assert per.shape == (4,) and np.allclose(per, losses.binary_loss(y, labels, 0.5, reduction="none"))


def test_parity_auto_voxel_records_effective_eps():
    t = ParityTask(1, ROI, material="water")
    d = t.describe()
    assert d["voxel_d"] == pytest.approx(resolve_voxel_d("water"))
    assert d["effective_eps"] == "78-10j" and d["n_vox_per_object"] > 300


def test_letter_task_thin_plate_and_jitter():
    t = LetterTask(ROI)
    assert t.kind == "multiclass" and t.n_probes == 3 and np.array_equal(t.labels, [0, 1, 2])
    assert t.voxel_d == pytest.approx(LAMBDA / 10) and t.thickness == pytest.approx(LAMBDA / 10)
    for s, letter in zip(t.samples, "ABC"):
        # 薄板：像素 λ/2 按 λ/10 细分 5×5，厚 1 层；平面法向 +x（u=+y, v=+z）
        assert len(s.objects) == 25 * int(LETTERS[letter].sum())
        assert np.ptp(s.objects.pos[:, 0]) < 1e-12
        assert s.meta["power_db"] == 0.0
    rng = np.random.default_rng(0)
    tj = LetterTask(ROI, n_per_class=3, dist_jitter=LAMBDA / 4, power_jitter_db=6.0, rng=rng)
    assert len(tj.samples) == 9 and np.array_equal(tj.labels, np.tile([0, 1, 2], 3))
    shifts = np.array([s.meta["dist_shift"] for s in tj.samples])
    pdb = tj.power_db()
    assert np.all(np.abs(shifts) <= LAMBDA / 4) and np.all(np.abs(pdb) <= 6.0) and np.std(pdb) > 0
    for s in tj.samples:                                 # 位移沿法向 +x
        assert np.allclose(s.objects.pos[:, 0], ROI[0] + s.meta["dist_shift"])
    with pytest.raises(ValueError):
        LetterTask(ROI, dist_jitter=0.01)                # 有抖动必须给 rng
    y = np.array([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0], [0.0, 0.5, 0.2]])
    assert np.array_equal(t.predict(y), [0, 1, 1])
    assert t.margin_db(y) == pytest.approx(-0.3)        # 第 3 个样本判错：0.2 − 0.5
    assert t.loss(y, t.labels) == pytest.approx(losses.multiclass_loss(y, t.labels, 1.0))


def test_localization_cells_bits_and_predict():
    t = LocalizationTask(ROI, voxel_d=LAMBDA / 10)
    assert t.kind == "loc3bit" and t.n_probes == 3 and len(t.samples) == 8
    assert np.array_equal(t.labels, np.arange(8))
    # 2×4 格、间距 λ/2，以 ROI 居中，位于 y-z 平面（plane_u=+y, plane_v=+z）
    cc = t.cell_centers
    assert np.allclose(cc.mean(axis=0), ROI)
    assert np.allclose(cc[:, 0], ROI[0])
    assert np.allclose(cc[1] - cc[0], [0, LAMBDA / 2, 0]) and np.allclose(cc[4] - cc[0], [0, 0, LAMBDA / 2])
    for s in t.samples:
        assert np.allclose(s.objects.pos.mean(axis=0), cc[s.label], atol=1e-12)
    # predict：3 个探头各自阈值判决 → 格号 Σ bit_j 2^j
    t_fix = np.array([-30.0, -31.0, -32.0])
    tf = LocalizationTask(ROI, voxel_d=LAMBDA / 10, thresholds_db=t_fix)
    bits = (np.arange(8)[:, None] >> np.arange(3)[None, :]) & 1
    y = t_fix[None, :] + np.where(bits == 1, 0.7, -0.4)
    assert np.array_equal(tf.predict(y), np.arange(8))
    assert np.array_equal(tf.predict(y[::-1]), np.arange(8)[::-1])
    assert tf.margin_db(y) == pytest.approx(0.4)
    assert np.allclose(tf.bit_accuracy(y), 1.0)
    # auto 阈值：两类中位数中点（与 losses.loc3bit_thresholds 一致），同样能还原格号
    y2 = np.where(bits == 1, -20.0, -26.0) + np.random.default_rng(1).normal(0, 0.3, (8, 3))
    assert np.array_equal(t.predict(y2), np.arange(8))
    assert np.allclose(t.thresholds(y2), losses.loc3bit_thresholds(y2, t.labels))
    assert t.loss(y2, t.labels) == pytest.approx(losses.loc3bit_loss(y2, t.labels, 1.0))
    tw = LocalizationTask(ROI, voxel_d=LAMBDA / 10, loss_weight=2.6)
    assert tw.task_kwargs == {"thresholds_db": "auto", "loss_weight": 2.6}
    assert tw.loss(y2, tw.labels) == pytest.approx(2.6 * losses.loc3bit_loss(y2, tw.labels, 1.0))


def test_localization_jitter_and_reps():
    rng = np.random.default_rng(2)
    t = LocalizationTask(ROI, voxel_d=LAMBDA / 10, n_per_cell=2, jitter=LAMBDA / 8, rng=rng)
    assert len(t.samples) == 16 and np.array_equal(t.labels, np.tile(np.arange(8), 2))
    for s in t.samples:
        off = s.objects.pos.mean(axis=0) - t.cell_centers[s.label]
        assert np.all(np.abs(off) <= LAMBDA / 8 + 1e-12) and abs(off[0]) < 1e-12
    with pytest.raises(ValueError):
        LocalizationTask(ROI, grid=(2, 3))               # 格子数须为 2 的幂


def test_focus_task():
    t = FocusTask()
    assert t.kind == "focus" and t.n_probes == 1 and len(t.samples) == 1 and len(t.samples[0].objects) == 0
    y = np.array([[-12.5]])
    assert t.loss(y, t.labels) == pytest.approx(12.5)
    assert np.array_equal(t.predict(y), [0])


# ============================================================ 审查 R2-INT-1：born 物体的极化率模型
def test_object_model_by_mode_and_shared_voxel_grid():
    from ris_nlsim.materials import DEFAULT_MODEL
    from ris_nlsim.tasks import BORN_MODEL, object_model

    assert BORN_MODEL == "volume_rr" and DEFAULT_MODEL == "scalar_rr"
    assert object_model("born") == "volume_rr"
    assert object_model("full") == object_model("intra") == object_model(None) == "scalar_rr"
    assert object_model("born", "scalar_rr") == "scalar_rr" and object_model("full", "cm_rr") == "cm_rr"   # 显式优先
    # born 替身不改体素网格：水 λ/28、金属 λ/20 与 scalar_rr 相同 → 三种物体模式同一离散
    for mat in ("water", "metal", "eps4"):
        tb = ParityTask(1, ROI, material=mat, model=BORN_MODEL)
        tf = ParityTask(1, ROI, material=mat)
        assert tb.voxel_d == tf.voxel_d == pytest.approx(resolve_voxel_d(mat))
        assert np.array_equal(tb.samples[1].objects.pos, tf.samples[1].objects.pos)
        vol = np.pi * (LAMBDA / 3) ** 3 / 6 / tb.n_vox_per_object
        assert np.allclose(tb.samples[1].objects.alpha, polarizability(mat, vol, K0, "volume_rr"))
    # 记录实际模型与对应 ε_eff（金属在 volume_rr 下是 α0 = 3V，即 ε_eff = 4）
    d = ParityTask(1, ROI, material="metal", model=BORN_MODEL).describe()
    assert d["model"] == "volume_rr" and d["effective_eps"] == "4+0j"
    assert ParityTask(1, ROI, material="metal").describe()["effective_eps"] == "metal"
    lt = LocalizationTask(ROI, material="water", model=BORN_MODEL)
    assert lt.voxel_d == pytest.approx(resolve_voxel_d("water")) and lt.describe()["model"] == "volume_rr"
    assert LetterTask(ROI, model=BORN_MODEL).describe()["effective_eps"] == "4+0j"


def test_born_voxel_polarizability_converges_scalar_rr_does_not():
    """一阶 Born 的体素偶极矩 χV·E：volume_rr 的 |α/(χV)| 随 d→0 单调趋于 1（λ/20 起 ≥ 0.96）；
    scalar_rr 在体素互不耦合时 1 − χk²a²/2 不被抵消，水在自动体素 λ/28 附近近谐振（|α/(χV)| ≈ 3.2，非单调）。"""
    chi = (78 - 10j) - 1.0
    r_vol, r_sca = [], []
    for div in (20, 28, 40, 56, 80):
        V = (LAMBDA / div) ** 3
        r_vol.append(abs(polarizability("water", V, K0, "volume_rr") / (chi * V)))
        r_sca.append(abs(polarizability("water", V, K0, "scalar_rr") / (chi * V)))
    assert np.all(np.diff(r_vol) > 0) and r_vol[0] > 0.95 and r_vol[-1] < 1.0
    assert r_sca[1] > 3.0 and np.argmax(r_sca) == 1                      # λ/28 处的伪谐振


# ============================================================ 审查 R2-INT-2：判决平局
def test_ties_and_decide_mark_undecided():
    t = ParityTask(2, ROI, voxel_d=LAMBDA / 10)                          # labels [0,1,1,0]
    y = np.array([[-40.0, -40.0], [-20.0, -21.0], [-40.0, -40.0], [-30.0, -29.0]])
    assert np.array_equal(t.predict(y), [0, 1, 0, 0])                    # 严格规则：平局落到"偶"
    assert np.array_equal(t.ties(y), [True, False, True, False])
    assert np.array_equal(t.decide(y), [-1, 1, -1, 0])                   # 平局未决 → 计分为错
    assert t.margin_db(y) == 0.0
    lt = LetterTask(ROI)
    yl = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.5, 0.5]])
    assert np.array_equal(lt.ties(yl), [True, False, True])
    assert np.array_equal(lt.decide(yl), [-1, 1, -1])
    loc = LocalizationTask(ROI, voxel_d=LAMBDA / 10, thresholds_db=np.array([-30.0, -31.0, -32.0]))
    bits = (np.arange(8)[:, None] >> np.arange(3)[None, :]) & 1
    yz = np.array([-30.0, -31.0, -32.0])[None, :] + np.where(bits == 1, 0.7, -0.4)
    yz[5, 1] = -31.0                                                     # 恰在阈值上
    assert np.array_equal(loc.ties(yz), np.arange(8) == 5)
    assert loc.decide(yz)[5] == -1 and np.array_equal(loc.decide(yz)[:5], np.arange(5))
    assert np.allclose(loc.bit_accuracy(yz), [1.0, 7 / 8, 1.0])          # 平局比特计错
    assert loc.margin_db(yz) == 0.0


# ============================================================ 审查 R2-INT-3：掩码裕量
def test_masked_margin_keeps_rule_from_all_samples():
    t = LocalizationTask(ROI, voxel_d=LAMBDA / 10)                       # thresholds "auto"
    bits = (np.arange(8)[:, None] >> np.arange(3)[None, :]) & 1
    y = np.where(bits == 1, -20.0, -26.0) + np.random.default_rng(3).normal(0, 0.5, (8, 3))
    mask = np.zeros(8, dtype=bool)
    mask[[2, 7]] = True                                                  # 只剩格 2、7：比特 1 只有一类
    with pytest.raises(ValueError):
        t.margin_db(y[mask], t.labels[mask])                             # 旧做法：子集上重算自动阈值
    tt = t.thresholds(y)
    sgn = 2.0 * bits - 1.0
    assert t.margin_db(y, mask=mask) == pytest.approx(np.min((sgn * (y - tt))[mask]))
    assert t.margin_db(y, mask=np.ones(8, bool)) == pytest.approx(t.margin_db(y))
    assert np.isnan(t.margin_db(y, mask=np.zeros(8, bool)))
    with pytest.raises(ValueError):
        t.margin_db(y, mask=mask[:5])
    p = ParityTask(2, ROI, voxel_d=LAMBDA / 10)
    yp = np.array([[-20.0, -21.0], [-22.0, -20.0], [-19.0, -25.0], [-30.0, -29.0]])
    m = np.array([True, False, True, True])
    assert p.margin_db(yp, mask=m) == pytest.approx(p.margin_db(yp[m], p.labels[m]))
    lt = LetterTask(ROI)
    yl = np.array([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0], [0.0, 0.5, 0.2]])
    assert lt.margin_db(yl, mask=[True, True, False]) == pytest.approx(1.0)
    assert np.isnan(FocusTask().margin_db(np.zeros((1, 1)), mask=[True]))


# ============================================================ 审查 S0-2 / S1-1 / S2-0 增补
def test_sample_margins_consistent_with_margin_and_decisions():
    """S0-2：逐样本带符号裕量 > 0 ⇔ 该样本判对；margin_db = 其最小值（含 mask）。"""
    rng = np.random.default_rng(0)
    tp = ParityTask(2, ROI, material="eps4", voxel_d=LAMBDA / 10)
    tl = LetterTask(ROI)
    tz = LocalizationTask(ROI, material="eps4", voxel_d=LAMBDA / 10)
    for t in (tp, tl, tz):
        y = rng.normal(-30.0, 3.0, size=(t.n_samples, t.n_probes))
        s = t.sample_margins_db(y)
        assert s.shape == (t.n_samples,)
        assert np.array_equal(s > 0, t.decide(y) == t.labels)
        assert t.margin_db(y) == pytest.approx(s.min())
        mask = np.arange(t.n_samples) % 2 == 0
        assert t.margin_db(y, mask=mask) == pytest.approx(s[mask].min())
    assert np.all(np.isnan(FocusTask().sample_margins_db(np.zeros((1, 1)))))


def test_metal_with_non_scalar_model_warns_eps4():
    """S1-1：born（volume_rr）下的金属是 εr_eff = 4 介质 → 构造时警告；scalar_rr（full / intra）不警告。"""
    import warnings

    from ris_nlsim.tasks import BORN_MODEL

    with pytest.warns(RuntimeWarning, match="S1-1"):
        t = LetterTask(ROI, model=BORN_MODEL)
    assert t.describe()["effective_eps"] == "4+0j"
    with pytest.warns(RuntimeWarning, match="S1-1"):
        ParityTask(1, ROI, material="metal", voxel_d=LAMBDA / 10, model=BORN_MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert LetterTask(ROI).describe()["effective_eps"] == "metal"
        ParityTask(1, ROI, material="water", voxel_d=LAMBDA / 10, model=BORN_MODEL)


def test_localization_dist_and_power_jitter():
    """S2-0：与 LetterTask 同一约定——沿 n̂ = û×v̂ 位移 U(±dist_jitter)、meta power_db = U(±power_jitter_db)；
    不启用的抖动不消耗随机数（只有格内抖动时样本与旧实现相同）；有抖动无 rng 报错。"""
    kw = dict(material="eps4", voxel_d=LAMBDA / 10)
    base = LocalizationTask(ROI, **kw)
    t = LocalizationTask(ROI, dist_jitter=LAMBDA / 4, power_jitter_db=6.0, rng=np.random.default_rng(1), **kw)
    nrm = np.cross([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    for s0, s1 in zip(base.samples, t.samples):
        sh = s1.meta["dist_shift"]
        assert abs(sh) <= LAMBDA / 4 and abs(s1.meta["power_db"]) <= 6.0
        assert np.allclose(s1.objects.pos, s0.objects.pos + sh * nrm)
    assert np.ptp(t.power_db()) > 1.0 and np.all(base.power_db() == 0.0)
    d = t.describe()
    assert d["dist_jitter"] == pytest.approx(LAMBDA / 4) and d["power_jitter_db"] == 6.0
    # 只有格内抖动：随机数消耗与旧实现一致（每样本 2 个均匀数）
    t_in = LocalizationTask(ROI, jitter=0.01, rng=np.random.default_rng(2), **kw)
    r = np.random.default_rng(2)
    for smp in t_in.samples:
        assert np.allclose(smp.meta["offset"], r.uniform(-0.01, 0.01, size=2))
        assert smp.meta["dist_shift"] == 0.0 and smp.meta["power_db"] == 0.0
    with pytest.raises(ValueError):
        LocalizationTask(ROI, power_jitter_db=3.0, **kw)
