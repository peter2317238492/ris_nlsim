"""materials 测试：极化率模型、光学定理（闭式 + 球面能流数值积分）、无源单元等效极化率（DESIGN §1.3）。"""
import numpy as np
import pytest

from ris_nlsim.constants import K0, LAMBDA, MATERIALS
from ris_nlsim.greens import green
from ris_nlsim.materials import (DEFAULT_MODEL, MODELS, alpha_passive, effective_eps, polarizability,
                                 radiative_correction, resolve_eps, single_dipole_cross_sections, voxels_to_nodes)
from ris_nlsim.patterns import Pattern
from ris_nlsim.scene import NodeSet

K = K0
LAM = LAMBDA
V_SMALL = (LAM / 10) ** 3
V_PARITY = np.pi * (LAM / 3) ** 3 / 6        # 奇偶散射体整体（λ/3 球）当一个偶极子


def net_outward_flux(p, k, E0=1.0, radius=1.7 * LAMBDA, n_mu=200):
    """原点偶极子 p 与 +z 平面波 E0·e^{-jkz} 的总场穿过球面的净外向功率（W）。

    标量场功率流（e^{jωt}，|E|²=功率密度）：S_r = −Im(conj(ψ)·∂_rψ)/k；
    方位对称，cosθ 用 Gauss–Legendre。返回 (P_net, P_sca)：P_net = P_sca − P_ext = −P_abs。
    """
    mu, w = np.polynomial.legendre.leggauss(n_mu)
    r = radius
    psi_i = E0 * np.exp(-1j * k * r * mu)
    dpsi_i = -1j * k * mu * psi_i
    psi_s = k**2 * green(r, k) * p * np.ones_like(mu)
    dpsi_s = (-1j * k - 1.0 / r) * psi_s
    psi, dpsi = psi_i + psi_s, dpsi_i + dpsi_s
    S = -np.imag(np.conj(psi) * dpsi) / k
    P_net = 2 * np.pi * r**2 * np.sum(w * S)
    P_sca = 2 * np.pi * r**2 * np.sum(w * np.abs(psi_s) ** 2)   # 纯散射场能流
    return P_net, P_sca


# ---------------------------------------------------------------- 模型公式

def test_cm_and_rr_closed_forms():
    eps = 4.0
    a0 = 3 * V_SMALL * (eps - 1) / (eps + 2)
    assert polarizability(eps, V_SMALL, K, "cm") == pytest.approx(a0, rel=1e-15)
    a_rr = a0 / (1 + 1j * K**3 * a0 / (4 * np.pi))
    assert polarizability(eps, V_SMALL, K, "cm_rr") == pytest.approx(a_rr, rel=1e-15)
    assert polarizability(eps, V_SMALL, K, "volume_rr") == pytest.approx(
        complex(radiative_correction((eps - 1) * V_SMALL, K)), rel=1e-15)
    # 1/α = 1/α0 + j k³/(4π)
    assert (1 / a_rr - 1 / a0) == pytest.approx(1j * K**3 / (4 * np.pi), rel=1e-12)


def test_default_model_is_scalar_rr():
    """回归（PHY-R2-1）：默认路径（tasks 按 INTEGRATION §E 不传 model）必须求解 εr 本身而非 CM 压缩后的 ε_eff。

    cm_rr 的连续极限 ε_eff = 1 + 3(εr−1)/(εr+2) ≤ 4：金属退化成 εr=4 介质、水（|εr|≈78）与 eps10 几乎一样强。
    """
    assert DEFAULT_MODEL == "scalar_rr"
    for mat in MATERIALS:
        for V in [V_SMALL, V_PARITY]:
            assert polarizability(mat, V, K) == polarizability(mat, V, K, "scalar_rr")
        assert effective_eps(mat) == effective_eps(mat, "scalar_rr")
    pos = np.zeros((3, 3)) + np.arange(3)[:, None] * 0.01
    np.testing.assert_array_equal(voxels_to_nodes(pos, V_SMALL, "metal", K).alpha,
                                  polarizability("metal", V_SMALL, K, "scalar_rr"))
    # 默认模型下 εr 扫描不被压缩：ε_eff 就是表里的 εr，金属是 χ→∞ 极限而不是 εr=4
    assert [effective_eps(m) for m in ["eps1.5", "eps4", "eps10", "water"]] == [1.5, 4.0, 10.0, 78 - 10j]
    assert effective_eps("metal") == "metal"
    # 小体素（|m|kd ≪ 1）时默认极化率 ≈ χV，随 εr 单调增大（cm_rr 被钳在 ≤ 3V）
    a = [abs(polarizability(e, (LAM / 40) ** 3, K)) / (LAM / 40) ** 3 for e in [1.5, 4.0, 10.0, 30.0]]
    assert a == sorted(a) and a[-1] > 20
    assert abs(polarizability(30.0, (LAM / 40) ** 3, K, "cm_rr")) < 3 * (LAM / 40) ** 3


def test_cm_tends_to_chi_v_for_eps_to_one():
    for delta in [1e-3, 1e-5, 1e-7]:
        a = polarizability(1 + delta, V_SMALL, K, "cm")
        assert a / (delta * V_SMALL) == pytest.approx(1.0, rel=2 * delta)
        # RR 修正因子 1/(1+j k³α0/4π)，k³V/(4π) ≈ 0.2 ⇒ 相对偏差 ≲ 0.2χ，同样随 χ→0 消失
        for model in ["cm_rr", "volume_rr", "scalar_rr"]:
            a_rr = polarizability(1 + delta, V_SMALL, K, model)
            assert a_rr / (delta * V_SMALL) == pytest.approx(1.0, rel=delta)


def test_metal_limit_is_three_v():
    for name in ["metal", MATERIALS["metal"]]:
        assert polarizability(name, V_SMALL, K, "cm") == pytest.approx(3 * V_SMALL, rel=1e-15)
        a = polarizability(name, V_SMALL, K, "cm_rr")
        assert a == pytest.approx(3 * V_SMALL / (1 + 1j * K**3 * 3 * V_SMALL / (4 * np.pi)), rel=1e-15)
    # CM 随 εr → ∞ 收敛到 3V
    assert polarizability(1e9, V_SMALL, K, "cm") == pytest.approx(3 * V_SMALL, rel=1e-8)


def test_material_table_lookup_and_errors():
    assert resolve_eps("water") == 78 - 10j
    assert resolve_eps("eps4") == 4.0
    assert polarizability("eps10", V_SMALL, K) == polarizability(10.0, V_SMALL, K)
    with pytest.raises(ValueError):
        polarizability("unobtainium", V_SMALL, K)
    with pytest.raises(ValueError):
        polarizability(4.0, V_SMALL, K, "dda")
    with pytest.raises(ValueError):
        polarizability(-2.0, V_SMALL, K, "cm")
    vols = np.array([V_SMALL, 2 * V_SMALL])
    arr = polarizability("water", vols, K)
    assert arr.shape == (2,) and arr.dtype == np.complex128
    assert isinstance(polarizability("water", V_SMALL, K), complex)


def a_sphere(V):
    """等体积球半径 a = (3V/4π)^{1/3}。"""
    return (3 * V / (4 * np.pi)) ** (1 / 3)


@pytest.mark.parametrize("eps", [1.5, 4.0, 10.0, 78 - 10j])
@pytest.mark.parametrize("V", [V_SMALL, V_PARITY])
def test_scalar_rr_closed_form(eps, V):
    """scalar_rr：α = χV/(1 − χk²a²/2 + jχk³V/4π) ⇔ 1/α = 1/(χV) − k²a²/(2V) + jk³/(4π)（materials docstring）。"""
    chi = eps - 1
    a = a_sphere(V)
    alpha = polarizability(eps, V, K, "scalar_rr")
    assert alpha == pytest.approx(chi * V / (1 - chi * K**2 * a**2 / 2 + 1j * chi * K**3 * V / (4 * np.pi)),
                                  rel=1e-14)
    assert 1 / alpha == pytest.approx(1 / (chi * V) - K**2 * a**2 / (2 * V) + 1j * K**3 / (4 * np.pi), rel=1e-12)


def test_scalar_self_term_matches_sphere_integral():
    """自作用 S = k²∫_{球} G dV（数值求积）的展开：Re S = k²a²/2 + O((ka)⁴)，−Im S = k³V/(4π) + O((ka)⁵)。"""
    from scipy.integrate import quad
    for V in [V_SMALL, (LAM / 20) ** 3]:
        a = a_sphere(V)
        # ∫_{|r|<a} e^{−jkr}/(4πr) dV = ∫_0^a r e^{−jkr} dr
        re = quad(lambda r: r * np.cos(K * r), 0, a, epsabs=0, epsrel=1e-13)[0]
        im = quad(lambda r: -r * np.sin(K * r), 0, a, epsabs=0, epsrel=1e-13)[0]
        S = K**2 * (re + 1j * im)
        ka = K * a
        assert abs(S.real - ka**2 / 2) <= 1.01 * ka**4 / 8          # 下一项 −(ka)⁴/8
        assert abs(-S.imag - K**3 * V / (4 * np.pi)) <= 1.01 * ka**5 / 30
        # 截断后的 scalar_rr 与"精确球积分自作用"版本只差高阶项
        chi = 3.0
        a_exact = chi * V / (1 - chi * S)
        assert abs(polarizability(1 + chi, V, K, "scalar_rr") / a_exact - 1) < 2 * chi * ka**4


def test_scalar_rr_metal_is_infinite_contrast_limit():
    for V in [V_SMALL, V_PARITY]:
        a = a_sphere(V)
        am = polarizability("metal", V, K, "scalar_rr")
        assert am == pytest.approx(V / (-K**2 * a**2 / 2 + 1j * K**3 * V / (4 * np.pi)), rel=1e-14)
        assert polarizability(1e12, V, K, "scalar_rr") == pytest.approx(am, rel=1e-9)
    np.testing.assert_array_equal(polarizability("metal", np.array([0.0, 0.0]), K, "scalar_rr"), 0)
    assert polarizability(1.0, V_SMALL, K, "scalar_rr") == 0


def test_scalar_rr_static_resonance_is_unitary_limit():
    """实 χ 恰好使 χk²a²/2 = 1 时 α = −j4π/k³（σ_ext = σ_sca = 4π/k²，单偶极子最大散射），不出现 1/0。"""
    a = a_sphere(V_SMALL)
    chi = 2 / (K * a) ** 2
    alpha = polarizability(1 + chi, V_SMALL, K, "scalar_rr")
    assert alpha == pytest.approx(-4j * np.pi / K**3, rel=1e-12)
    cs = single_dipole_cross_sections(alpha, K)
    assert cs["sigma_ext"] == pytest.approx(4 * np.pi / K**2, rel=1e-12)
    assert cs["sigma_sca"] == pytest.approx(4 * np.pi / K**2, rel=1e-12)


def test_effective_eps():
    """cm/cm_rr 的连续极限对应 ε_eff = 1 + 3(εr−1)/(εr+2)；scalar_rr / volume_rr 就是 εr。"""
    assert effective_eps("eps4", "cm_rr") == pytest.approx(2.5)
    assert effective_eps("eps10", "cm") == pytest.approx(3.25)
    assert effective_eps("water", "cm_rr") == pytest.approx(1 + 3 * (77 - 10j) / (80 - 10j))
    assert abs(effective_eps("water", "cm_rr")) < 4          # χ_eff ≤ 3：水与 eps10 几乎同样强
    assert effective_eps("metal", "cm_rr") == 4.0
    for m in ["scalar_rr", "volume_rr"]:
        assert effective_eps("water", m) == 78 - 10j
    assert effective_eps("metal", "scalar_rr") == "metal"
    # 一致性：CM 的 α0 恰是 (ε_eff−1)V
    for eps in [1.5, 4.0, 78 - 10j]:
        assert polarizability(eps, V_SMALL, K, "cm") == pytest.approx((effective_eps(eps, "cm") - 1) * V_SMALL)


def test_physics_sign_convention_rejected():
    """εr 的虚部 > 0（物理约定 ε′ + jε″）在 e^{jωt} 下是增益介质 ⇒ 报错（DESIGN §1）。"""
    for model in MODELS:
        with pytest.raises(ValueError, match="虚部"):
            polarizability(78 + 10j, V_SMALL, K, model)
    with pytest.raises(ValueError):
        resolve_eps(4 + 1e-3j)
    assert resolve_eps(4 + 1e-15j) == 4 + 1e-15j          # 舍入噪声级别容许
    assert resolve_eps(4 - 0.1j) == 4 - 0.1j


# ---------------------------------------------------------------- 光学定理

@pytest.mark.parametrize("model", ["cm_rr", "scalar_rr"])
@pytest.mark.parametrize("eps", [1.5, 4.0, 10.0, "eps4", "metal"])
@pytest.mark.parametrize("V", [V_SMALL, V_PARITY, (LAM / 2) ** 3])
def test_optical_theorem_lossless(eps, V, model):
    a = polarizability(eps, V, K, model)
    cs = single_dipole_cross_sections(a, K)
    assert cs["sigma_ext"] == pytest.approx(cs["sigma_sca"], rel=1e-12)
    assert abs(cs["sigma_abs"]) <= 1e-12 * cs["sigma_sca"]


@pytest.mark.parametrize("model", ["cm_rr", "volume_rr", "scalar_rr"])
def test_lossy_water_absorbs(model):
    for V in [V_SMALL, V_PARITY]:
        a = polarizability("water", V, K, model)
        cs = single_dipole_cross_sections(a, K)
        assert cs["sigma_abs"] > 0
        eps = 78 - 10j
        # α0 = 辐射修正前的极化率；scalar_rr 的实自作用项 k²a²/2 不影响 Im(1/α0) = Im(1/(χV))
        a0 = {"cm_rr": 3 * V * (eps - 1) / (eps + 2), "volume_rr": (eps - 1) * V,
              "scalar_rr": (eps - 1) * V}[model]
        # σ_abs = k|α|² Im(1/α0)
        assert cs["sigma_abs"] == pytest.approx(K * abs(a) ** 2 * np.imag(1 / a0), rel=1e-9)


def test_cm_without_rr_violates_energy_conservation():
    # 对照：不做辐射修正时，无耗 α 为实数 ⇒ σ_ext = 0 但 σ_sca > 0（能量不守恒）
    a = polarizability(4.0, V_PARITY, K, "cm")
    cs = single_dipole_cross_sections(a, K)
    assert cs["sigma_ext"] == 0 and cs["sigma_sca"] > 0


@pytest.mark.parametrize("model", ["cm_rr", "scalar_rr"])
@pytest.mark.parametrize("eps", [4.0, 10.0, "metal", "water"])
def test_flux_integral_confirms_extinction_formula(eps, model):
    # 独立验证：球面数值能流积分得到的 P_ext 与 σ_ext = −k·Im α 一致；无耗时净能流为 0
    a = polarizability(eps, V_PARITY, K, model)
    P_net, P_sca = net_outward_flux(a * 1.0, K)
    cs = single_dipole_cross_sections(a, K)
    assert P_sca == pytest.approx(cs["sigma_sca"], rel=1e-12)
    P_ext_num = P_sca - P_net
    assert P_ext_num == pytest.approx(cs["sigma_ext"], rel=1e-9)
    if eps == "water":
        assert -P_net == pytest.approx(cs["sigma_abs"], rel=1e-7)
        assert -P_net > 0
    else:
        assert abs(P_net) < 1e-9 * P_sca


def test_flux_integral_independent_of_sphere_radius():
    a = polarizability("water", V_PARITY, K)
    nets = [net_outward_flux(a, K, radius=r)[0] for r in [0.5 * LAM, 1.3 * LAM, 3.1 * LAM]]
    np.testing.assert_allclose(nets, nets[0], rtol=1e-9)


# ---------------------------------------------------------------- 无源单元

def test_alpha_passive_formula_and_po_rcs():
    assert alpha_passive(1.0, K, LAM) == pytest.approx(LAM / K**2, rel=1e-15)
    g = np.exp(1j * np.array([0.0, np.pi / 2]))
    np.testing.assert_allclose(alpha_passive(g, K, LAM), LAM * g / K**2, rtol=1e-15)
    # 金属板 PO 单元：d0 = 4πA/λ²，Γ=1，视轴入射/出射 ⇒ 单站 RCS = 4πA²/λ²（物理光学）
    A = (LAM / 2) ** 2
    pat = Pattern(q=1.0, d0=4 * np.pi * A / LAM**2, backlobe_db=-200.0)
    F2 = pat.peak_directivity
    a = alpha_passive(1.0, K, LAM)
    R = 10.0
    E_back = K**2 * green(R, K) * a * F2           # 单位入射场 → 背向散射场（方向图进出各乘一次）
    sigma = 4 * np.pi * R**2 * abs(E_back) ** 2
    assert sigma == pytest.approx(4 * np.pi * A**2 / LAM**2, rel=1e-12)


def test_voxels_to_nodes():
    pos = np.zeros((5, 3)) + np.arange(5)[:, None] * 0.01
    ns = voxels_to_nodes(pos, V_SMALL, "water", K, group=3)
    assert len(ns) == 5 and ns.pattern.isotropic
    assert np.all(ns.alpha == polarizability("water", V_SMALL, K))
    assert np.all(ns.group == 3)


@pytest.mark.parametrize("Gamma", [1.0, 1j, -1j, 0.3 - 0.4j])
def test_alpha_passive_equals_receive_gamma_retransmit(Gamma):
    """单元 = 天线链 "接收 v → ×Γ → 以 Γv 为源再辐射"（alpha_passive 文档的推导，PHY-8 的模型含义）：
    scat_to_rcv·α·src_to_scat == src_to_rcv(探头←单元)·Γ·src_to_rcv(单元←馈源)。"""
    from ris_nlsim.greens import scat_to_rcv, src_to_rcv, src_to_scat
    from ris_nlsim.patterns import element_pattern, horn_pattern, probe_pattern
    ep = element_pattern()
    cell = np.array([0.0, 0.0, 0.0])
    elem = NodeSet(cell, [1.0, 0.2, 0.0], ep, alpha=alpha_passive(Gamma, K, LAM))
    fpos, ppos = np.array([1.5, -0.8, 0.1]), np.array([1.1, 1.3, -0.2])
    feed = NodeSet(fpos, cell - fpos, horn_pattern())
    probe = NodeSet(ppos, cell - ppos, probe_pattern())
    dipole = scat_to_rcv(probe, elem, K)[0, 0] * elem.alpha[0] * src_to_scat(elem, feed, K)[0, 0]
    chain = src_to_rcv(probe, elem, K)[0, 0] * Gamma * src_to_rcv(elem, feed, K)[0, 0]
    assert abs(dipole - chain) <= 1e-13 * abs(chain)
