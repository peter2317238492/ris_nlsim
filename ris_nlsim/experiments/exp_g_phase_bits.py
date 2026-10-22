"""实验 G：相位精度阶梯——中继移相器要几 bit 才能让"饱和 vs 线性"在 3 位奇偶上分出来（方案 §6.1 的延伸）。

背景（INTEGRATION §S、实验 A full）：informative 场景下 3 位奇偶只有连续相位的饱和中继（Rapp@+25）能稳健解出，
1-bit 码本在任何条件下都不行。本实验回答两个硬件问题：
  1. 电平数 L = 2, 4, 8, 16, 32（1–5 bit）时还能不能解：连续 oracle 解 φ* → 量化到 L 电平 → 字典序（判对数, 最小裕量）
     坐标下降精修（exp_a.polish_min_margin，全部电平都试）；
  2. 静态相位误差容忍度：在连续解与各 L 的精修解上加 σ = 0.5/1/2/5° 的随机通道相位误差（每档 n_draw 次），
     统计仍全对 / 仍 robust 的比例。
对照：同功率线性 M1、Bussgang 增益匹配 M1b@+25（理论上任何相位都解不出 3 位奇偶，用来确认"解出来"不是量化伪迹）。
所有评估无噪声（结构能力），判据同实验 A：solved = 全对且最小裕量 > 3σ_det，robust = 全对且最小裕量 ≥ 0.5 dB。
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import common
from .exp_a_parity import (DEFAULT_SPACING_LAM, ROBUST_MARGIN_DB, CONT_PLUS_INIT_RESTARTS, get_system, margin_key,
                           polish_min_margin, run_cont_plus, scenario_overrides, solved_threshold_db)

EXP = "exp_g"
LEVELS_FULL = (2, 4, 8, 16, 32)
LEVELS_QUICK = (2, 4, 8)
SIGMAS_DEG = (0.5, 1.0, 2.0, 5.0)
CONDITIONS = ("Rapp@+25", "M1b@+25", "M1")


def _job_base(job: dict, levels: int) -> dict:
    over = dict(scenario_overrides(job["scenario"]))
    over["relay_levels"] = int(levels)
    return {"preset": "L90", "n": int(job["n"]), "cond": job["cond"], "mode": job["mode"],
            "material": job["material"], "n_relays": 1, "spacing_lam": float(job["spacing_lam"]),
            "cfg_over": over}


def quantize(phases: dict, L: int) -> np.ndarray:
    """连续相位 → L 电平整数码本（就近量化，顺序同 PhysicalSystem 参数：中继 r 的 rx 25、tx 25）。"""
    step = 2.0 * np.pi / L
    rx = np.mod(np.rint(np.asarray(phases["phase_rx"]) / step), L).astype(np.int64)
    tx = np.mod(np.rint(np.asarray(phases["phase_tx"]) / step), L).astype(np.int64)
    return np.stack([rx, tx], axis=1).ravel()


def _key_dict(k: tuple, B: int, thr_solved: float) -> dict:
    n_ok, m = int(k[0]), float(k[1])
    allc = n_ok == B
    return {"n_ok": n_ok, "min_margin_db": m, "all_correct": allc,
            "solved": bool(allc and m > thr_solved), "robust": bool(allc and m >= ROBUST_MARGIN_DB)}


def tolerance(s, phases: dict, sigmas, n_draw: int, rng: np.random.Generator, thr_solved: float) -> dict:
    """静态通道相位误差（每通道独立 N(0, σ²)）下仍全对 / solved / robust 的比例（无噪声评估）。"""
    B = len(s.labels)
    out = {}
    for sg in sigmas:
        res = []
        for _ in range(int(n_draw)):
            ph = {k: np.asarray(v) + np.deg2rad(sg) * rng.standard_normal(np.shape(v)) for k, v in phases.items()}
            res.append(_key_dict(margin_key(s, ph, dr_db=None), B, thr_solved))
        out[f"{sg:g}"] = {"all_correct": float(np.mean([r["all_correct"] for r in res])),
                          "solved": float(np.mean([r["solved"] for r in res])),
                          "robust": float(np.mean([r["robust"] for r in res])),
                          "margin_median_db": float(np.median([r["min_margin_db"] for r in res]))}
    return out


def run_job(job: dict) -> dict:
    t0 = time.perf_counter()
    s2 = get_system(_job_base(job, 2))
    B = len(s2.labels)
    thr = solved_threshold_db(s2)
    cell = (job["cond"], job["mode"], job["material"], job["seed"])
    # 1. 连续相位 oracle（cont-plus，与实验 A informative 同设定）
    cseed = int(common.seed_rng(EXP, "cont", *cell).integers(0, 2 ** 31 - 1))
    c = run_cont_plus(s2, cseed, int(job["cont_restarts"]), int(job["cont_steps"]), 0,
                      common.seed_rng(EXP, "cont-eval", *cell), init_restarts=int(job["cont_init_restarts"]),
                      refine_steps=int(job["cont_refine_steps"]))
    phases = {k: np.asarray(v) for k, v in c["phases"].items()}
    cont = _key_dict(margin_key(s2, phases, dr_db=None), B, thr)
    cont["tolerance"] = tolerance(s2, phases, job["sigmas"], job["n_draw"], common.seed_rng(EXP, "tol-c", *cell), thr)
    rows = {"cont": cont}
    # 2. 各电平数：量化 → 精修 → 误差容忍度
    for L in job["levels"]:
        sL = get_system(_job_base(job, L))
        q = quantize(phases, L)
        kq = _key_dict(margin_key(sL, sL.phases_from_params(q), dr_db=None), B, thr)
        p = polish_min_margin(sL, q, common.seed_rng(EXP, "polish", L, *cell), max_sweeps=int(job["polish_sweeps"]))
        kp = _key_dict((p["n_ok"], p["min_margin_db"]), B, thr)
        kp["tolerance"] = tolerance(sL, sL.phases_from_params(p["params"]), job["sigmas"], job["n_draw"],
                                    common.seed_rng(EXP, "tol", L, *cell), thr)
        kp["polish_evals"] = int(p["n_evals"])
        rows[f"L{L}"] = {"quantized": kq, "polished": kp}
    return {"job": {k: v for k, v in job.items() if k not in ("sigmas",)}, "cond": job["cond"], "mode": job["mode"],
            "material": job["material"], "seed": int(job["seed"]), "B": B, "cont_source": c.get("source"),
            "p_feed_dbm": float(10 * np.log10(s2.p_feed_w) + 30), "results": rows,
            "time_s": time.perf_counter() - t0}


# ------------------------------------------------------------------ 汇总
def _frac(vals) -> str:
    vals = [bool(v) for v in vals]
    return f"{sum(vals)}/{len(vals)}" if vals else "–"


def summarize(rows: list[dict], levels, sigmas) -> tuple[list[dict], str]:
    ok = [r for r in rows if "error" not in r]
    conds = [c for c in CONDITIONS if any(r["cond"] == c for r in ok)]
    modes = sorted({r["mode"] for r in ok})
    table, md = [], []
    md.append("## 1. 电平数阶梯（3 位奇偶，无噪声；solved = 全对且裕量 > 3σ_det，robust = 全对且裕量 ≥ 0.5 dB）\n")
    md.append("每格：robust / solved（种子数），括号内为最小裕量中位数 dB。量化 = 连续解就近量化；精修 = 量化后坐标下降。\n")
    hdr = ["条件", "模式", "连续"] + [f"{int(math.log2(L))} bit（L={L}）量化 → 精修" for L in levels]
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "---|" * len(hdr))
    for cond in conds:
        for mode in modes:
            rr = [r for r in ok if r["cond"] == cond and r["mode"] == mode]
            if not rr:
                continue
            cont = [r["results"]["cont"] for r in rr]
            cells = [f"{_frac(x['robust'] for x in cont)} / {_frac(x['solved'] for x in cont)} "
                     f"({np.median([x['min_margin_db'] for x in cont]):+.2f})"]
            row = {"cond": cond, "mode": mode, "n": len(rr),
                   "cont_robust": float(np.mean([x["robust"] for x in cont])),
                   "cont_solved": float(np.mean([x["solved"] for x in cont]))}
            for L in levels:
                qs = [r["results"][f"L{L}"]["quantized"] for r in rr]
                ps = [r["results"][f"L{L}"]["polished"] for r in rr]
                cells.append(f"{_frac(x['robust'] for x in qs)} / {_frac(x['solved'] for x in qs)} → "
                             f"{_frac(x['robust'] for x in ps)} / {_frac(x['solved'] for x in ps)} "
                             f"({np.median([x['min_margin_db'] for x in ps]):+.2f})")
                row[f"L{L}_robust"] = float(np.mean([x["robust"] for x in ps]))
                row[f"L{L}_solved"] = float(np.mean([x["solved"] for x in ps]))
                row[f"L{L}_margin_med"] = float(np.median([x["min_margin_db"] for x in ps]))
            table.append(row)
            md.append(f"| {cond} | {mode} | " + " | ".join(cells) + " |")
    md.append("\n## 2. 静态相位误差容忍度（每档 σ 随机抽样；格内 = 仍 solved 的比例，种子平均；只统计无误差时 solved 的种子）\n")
    hdr = ["条件", "模式", "码本"] + [f"σ = {sg:g}°" for sg in sigmas]
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "---|" * len(hdr))
    for cond in conds:
        for mode in modes:
            rr = [r for r in ok if r["cond"] == cond and r["mode"] == mode]
            for name in ["cont"] + [f"L{L}" for L in levels]:
                res = [(r["results"][name] if name == "cont" else r["results"][name]["polished"]) for r in rr]
                res = [x for x in res if x["solved"]]
                if not res:
                    continue
                label = "连续" if name == "cont" else f"{int(math.log2(int(name[1:])))} bit"
                cells = [f"{np.mean([x['tolerance'][f'{sg:g}']['solved'] for x in res]):.2f}" for sg in sigmas]
                md.append(f"| {cond} | {mode} | {label}（{len(res)} 种子） | " + " | ".join(cells) + " |")
    return table, "\n".join(md)


def plot(table: list[dict], levels, outdir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bits = [int(math.log2(L)) for L in levels]
    for row in table:
        if row["mode"] != table[0]["mode"]:
            continue
        ax.plot(bits, [row[f"L{L}_robust"] for L in levels], marker="o", label=f"{row['cond']}（robust）")
        ax.axhline(row["cont_robust"], ls=":", lw=1, color=ax.lines[-1].get_color())
    ax.set_xlabel("中继移相器比特数")
    ax.set_ylabel("3 位奇偶稳健解出的种子比例")
    ax.set_xticks(bits)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"相位精度阶梯（{table[0]['mode']}；虚线 = 连续相位）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return [common.savefig(fig, f"{outdir}/fig_phase_bits.png")]


def main(argv=None) -> None:
    p = common.base_argparser("实验 G：中继移相器比特数与相位误差容忍度（3 位奇偶）")
    p.add_argument("--scenario", default="informative")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--modes", nargs="+", default=None, help="物体模式（默认 full；full 模式 full intra）")
    p.add_argument("--material", default="water")
    p.add_argument("--levels", nargs="+", type=int, default=None)
    p.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS_DEG))
    p.add_argument("--n-draw", type=int, default=None)
    p.add_argument("--polish-sweeps", type=int, default=10)
    a = p.parse_args(argv)
    q = a.quick
    levels = a.levels or list(LEVELS_QUICK if q else LEVELS_FULL)
    modes = a.modes or (["full"] if q else ["full", "intra"])
    seeds = 2 if (q and a.seeds == 5) else a.seeds
    n_draw = a.n_draw or (10 if q else 20)
    jobs = [{"scenario": a.scenario, "n": 3, "cond": cond, "mode": mode, "material": a.material,
             "spacing_lam": DEFAULT_SPACING_LAM, "seed": sd, "levels": levels, "sigmas": a.sigmas, "n_draw": n_draw,
             "polish_sweeps": a.polish_sweeps, "cont_restarts": 1, "cont_steps": 600,
             "cont_init_restarts": CONT_PLUS_INIT_RESTARTS, "cont_refine_steps": 300}
            for cond in a.conditions for mode in modes for sd in range(seeds)]
    outdir = common.make_outdir(EXP, a.out)
    t0 = time.perf_counter()
    rows = common.run_jobs(run_job, jobs, workers=a.workers, progress=True)
    table, md = summarize(rows, levels, a.sigmas)
    cfg = {"scenario": a.scenario, "conditions": a.conditions, "modes": modes, "material": a.material,
           "levels": levels, "sigmas": a.sigmas, "n_draw": n_draw, "seeds": seeds, "quick": q,
           "wall_s": time.perf_counter() - t0}
    common.write_outputs(outdir, rows, table, cfg, title="实验 G：相位精度阶梯")
    head = (f"# 实验 G：相位精度阶梯（{'quick' if q else 'full'}，场景 {a.scenario}，材料 {a.material}，"
            f"{seeds} 个种子，墙钟 {cfg['wall_s']:.0f} s）\n\n"
            "连续 oracle = 实验 A 的 cont-plus（1×600 步随机重启 + 6 次设计初值重启 + 300 步 soft-min 精修）。"
            "量化后的精修是全部电平的坐标下降（字典序：判对数、最小裕量）。全部为无噪声评估。\n\n")
    figs = plot(table, levels, outdir) if table else []
    with open(f"{outdir}/summary.md", "w", encoding="utf-8") as f:
        f.write(head + md + "\n\n" + "".join(f"![]({x.split('/')[-1]})\n" for x in figs))
    n_err = sum(1 for r in rows if "error" in r)
    print(f"完成：{cfg['wall_s']:.0f} s，{len(rows)} 个 job（{n_err} 个出错）；输出 {outdir}")


if __name__ == "__main__":
    main()
