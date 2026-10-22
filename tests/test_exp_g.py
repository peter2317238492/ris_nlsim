"""实验 G（相位精度阶梯）冒烟测试：极小预算跑通一个 job 与汇总。"""
from ris_nlsim.experiments import exp_g_phase_bits as g


def test_exp_g_tiny_job_and_summary():
    job = {"scenario": "informative", "n": 3, "cond": "M1", "mode": "full", "material": "water", "spacing_lam": 0.5,
           "seed": 0, "levels": [2, 4], "sigmas": [1.0], "n_draw": 2, "polish_sweeps": 1, "cont_restarts": 1,
           "cont_steps": 20, "cont_init_restarts": 1, "cont_refine_steps": 10}
    r = g.run_job(job)
    assert set(r["results"]) == {"cont", "L2", "L4"}
    for L in ("L2", "L4"):
        assert r["results"][L]["polished"]["n_ok"] >= r["results"][L]["quantized"]["n_ok"]   # 精修不降判对数
    # 线性中继 + 3 位奇偶：二次阈值函数至多 7/8（Minsky–Papert），任何电平都不能全对
    assert not r["results"]["L4"]["polished"]["all_correct"]
    table, md = g.summarize([r], [2, 4], [1.0])
    assert table and "电平数阶梯" in md


def test_quantize_roundtrip():
    import numpy as np
    ph = {"phase_rx": np.array([[0.0, np.pi / 2, np.pi]]), "phase_tx": np.array([[3 * np.pi / 2, 2 * np.pi, 0.1]])}
    q = g.quantize(ph, 4)
    assert q.tolist() == [0, 1, 2, 3, 0, 0]
