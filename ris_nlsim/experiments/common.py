"""实验公共工具（INTEGRATION F）：多进程运行、JSONL/CSV/Markdown 输出、分组汇总（均值/标准差/95% 区间）、作图。

Windows 多进程：ProcessPoolExecutor 用 spawn，job 函数必须是模块顶层函数（可 pickle），调用方脚本要有
`if __name__ == "__main__":` 保护。子进程环境在启动前设好单线程（OMP/MKL/OpenBLAS），job 内再 torch.set_num_threads(1)。
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")                    # 无界面后端（DESIGN §9）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ============================================================ 中文字体（审查 S0-1）
CJK_FONT_CANDIDATES = ("Microsoft YaHei", "SimHei", "DengXian", "Noto Sans CJK SC", "Source Han Sans SC",
                       "WenQuanYi Micro Hei", "PingFang SC")


def setup_cjk_fonts() -> str | None:
    """matplotlib 默认字体（DejaVu Sans）没有中文字形，中文标题 / 刻度会画成方框（审查 S0-1）。

    在已安装字体中找第一个中文字体，把它放到 font.sans-serif 最前（DejaVu Sans 等原列表保留在后），并设
    axes.unicode_minus = False（中文字体常缺 U+2212 负号）。matplotlib ≥ 3.6 按 font.family 列表逐字形回退，
    所以另把 font.family 设为 ["DejaVu Sans", 中文字体, "sans-serif"]：拉丁 / 希腊字母与数学符号仍用 DejaVu
    （图的观感不变），DejaVu 没有的中文字形逐字回退到中文字体。
    幂等；本模块导入时自动调用一次（所有实验脚本都经 common 导入 matplotlib）。返回所用字体名，找不到时 None（保持原样）。
    """
    from matplotlib import font_manager

    try:
        names = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:          # noqa: BLE001 —— 字体缓存异常时不影响实验本身
        return None
    cjk = next((c for c in CJK_FONT_CANDIDATES if c in names), None)
    if cjk is None:
        return None
    rc = matplotlib.rcParams
    rc["font.sans-serif"] = [cjk] + [f for f in rc["font.sans-serif"] if f != cjk]
    rc["font.family"] = ["DejaVu Sans", cjk, "sans-serif"]
    rc["axes.unicode_minus"] = False
    return cjk


CJK_FONT = setup_cjk_fonts()

_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "RIS_NLSIM_BLAS_THREADS")
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")

# ============================================================ 实验 A 的物体模式（审查 R2-INT-1；方案 §4.1 三个开关、§6.1）
# born 物体已改用体素尺寸收敛的一阶 Born（volume_rr，tasks.object_model），但对水（χ = 77、λ/3 球）收敛后的 Born
# 仍比完全耦合强约 27 dB（单物体在接收单元处 −0.3 dB vs −27.2 dB 相对入射场）：born vs full 的差别同时混入了
# "物体多次散射"与"物体散射强度相差 27 dB → PA 工作点完全不同"两件事。intra（每个物体自身精确多次散射、物体之间
# 不耦合）单物体与 full 逐位相同，且 M0 下同样对 x 仿射（Walsh W≥3 ≈ 1e-28）→ 它才是 RQ2 "无物体间多次散射"的
# 强度匹配对照。born 保留给门 G1（"M0 + Born 解不出 3 位奇偶"）与线性理论参考。
# 实验 A 的网格（含 --quick）应包含全部三种模式：RQ2 的非线性收益比较取 intra vs full。
# （INTEGRATION G 原写 quick = born/full，已按本常量更新：审查 S1-2。）
EXP_A_OBJECT_MODES = ("born", "intra", "full")
RQ2_REFERENCE_MODE = "intra"            # RQ2：与 full 比较的"无物体间多次散射"对照（强度匹配）
G1_OBJECT_MODE = "born"                 # 门 G1：M0 + Born 3 位奇偶不可解（仿射性自检）


# ============================================================ 多进程
def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def _single_thread() -> None:
    """子进程内限制线程：BLAS 线程由父进程在 spawn 前用环境变量设好（对已加载的库无效）；torch 已导入时补 set_num_threads。"""
    for k in _THREAD_ENV:
        os.environ[k] = "1"
    import sys
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


def _run_one(fn: Callable, job: dict, index: int, in_child: bool = True) -> dict:
    """顶层包装：异常（含 SystemExit 等 BaseException，KeyboardInterrupt 除外）捕获为 {"error": traceback} 行，
    不中断其余 job（审查 INT-NUM-1）。"""
    if in_child:
        _single_thread()
    t0 = time.perf_counter()
    try:
        res = fn(job)
        row = dict(res) if isinstance(res, dict) else {"result": res}
    except KeyboardInterrupt:
        raise
    except BaseException:      # noqa: BLE001 —— 任何异常（含 sys.exit）都记录下来
        row = {"error": traceback.format_exc()}
    row.setdefault("job", job)
    row["job_index"] = index
    row["elapsed_s"] = time.perf_counter() - t0
    return row


def _max_workers(workers: int, n_jobs: int) -> int:
    """进程数上限：不超过 job 数；Windows 的 ProcessPoolExecutor 最多 61 个进程（WaitForMultipleObjects 限制）。"""
    cap = 61 if os.name == "nt" else max(1, int(workers))
    return max(1, min(int(workers), int(n_jobs), cap))


def _crash_row(job: dict, index: int, detail: str) -> dict:
    return {"error": f"worker crashed (BrokenProcessPool): {detail}", "job": job, "job_index": index, "crashed": True}


def _run_isolated(fn: Callable, job: dict, index: int, ctx) -> dict:
    """单 job 独占一个单进程池运行：它若使进程崩溃（原生访问冲突、OOM、os._exit），只影响自己这一行。"""
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
            return ex.submit(_run_one, fn, job, index).result()
    except BrokenProcessPool as e:
        return _crash_row(job, index, repr(e))
    except Exception:          # noqa: BLE001 —— pickle 失败等
        return {"error": traceback.format_exc(), "job": job, "job_index": index}


def run_jobs(fn: Callable[[dict], dict], jobs: list[dict], workers: int | None = None,
             progress: bool = False) -> list[dict]:
    """并行运行 fn(job) → dict（INTEGRATION F）。返回行按 jobs 顺序；异常行为 {"error": ..., "job": job}。

    workers ≤ 1 时在本进程顺序运行（便于调试；job 内 os._exit 会结束本进程）。子进程用 spawn 启动，启动前设置单线程
    环境变量（子进程继承）。健壮性（审查 INT-NUM-1）：
      - job 内的任何异常（含 SystemExit）都成为该 job 的 error 行；
      - 某个 job 使工作进程崩溃（BrokenProcessPool）时，共享池里被连带的 job（正在运行或排队）不记为错误，
        而是在第二轮里各自独占一个单进程池重跑（最多 workers 个并发）；只有让自己的单进程池崩溃的 job 得到
        {"error": "worker crashed (BrokenProcessPool): ...", "crashed": True} 行。
      - Windows 上进程数上限 61。
    """
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor

    jobs = list(jobs)
    workers = default_workers() if workers is None else int(workers)
    if workers <= 1 or len(jobs) <= 1:
        rows = []
        for i, j in enumerate(jobs):
            rows.append(_run_one(fn, j, i, in_child=False))
            if progress:
                print(f"[{i + 1}/{len(jobs)}] {'ERR' if 'error' in rows[-1] else 'ok'}", flush=True)
        return rows
    saved = {k: os.environ.get(k) for k in _THREAD_ENV}
    for k in _THREAD_ENV:
        os.environ[k] = "1"
    rows: list = [None] * len(jobs)
    broken: list[int] = []
    ctx = mp.get_context("spawn")
    nw = _max_workers(workers, len(jobs))
    done = 0

    def report(i):
        nonlocal done
        done += 1
        if progress:
            print(f"[{done}/{len(jobs)}] {'ERR' if 'error' in rows[i] else 'ok'}", flush=True)

    try:
        with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
            futs = {}
            for i, j in enumerate(jobs):
                try:
                    futs[ex.submit(_run_one, fn, j, i)] = i
                except BrokenProcessPool:          # 提交途中池已崩溃：余下的 job 进入第二轮
                    broken.append(i)
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    rows[i] = fut.result()
                except BrokenProcessPool:
                    broken.append(i)
                    continue
                except Exception:      # noqa: BLE001 —— pickle 失败等
                    rows[i] = {"error": traceback.format_exc(), "job": jobs[i], "job_index": i}
                report(i)
        if broken:                                  # 第二轮：逐 job 独占单进程池，隔离真正崩溃的 job
            broken.sort()
            with ThreadPoolExecutor(max_workers=min(nw, len(broken))) as tp:
                futs2 = {tp.submit(_run_isolated, fn, jobs[i], i, ctx): i for i in broken}
                for fut in as_completed(futs2):
                    i = futs2[fut]
                    rows[i] = fut.result()
                    report(i)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return rows


def demo_job(job: dict) -> dict:
    """测试用的顶层 job：返回 x² 与进程号。job["fail"] → 抛异常；job["exit"] → sys.exit(2)；
    job["crash"] → os._exit(3)（模拟工作进程硬崩溃）；job["sleep"] → 先睡眠若干秒。"""
    if job.get("sleep"):
        time.sleep(float(job["sleep"]))
    if job.get("fail"):
        raise RuntimeError(f"intentional failure for x={job.get('x')}")
    if job.get("exit"):
        import sys
        sys.exit(2)
    if job.get("crash"):
        os._exit(3)
    return {"x": job["x"], "y": job["x"] ** 2, "pid": os.getpid()}


# ============================================================ JSON
def to_jsonable(obj):
    """numpy / complex / dataclass 等 → JSON 可序列化对象（非有限浮点 → None）。"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (complex, np.complexfloating)):
        return {"re": to_jsonable(float(np.real(obj))), "im": to_jsonable(float(np.imag(obj)))}
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses
        return to_jsonable(dataclasses.asdict(obj))
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return repr(obj)


def save_jsonl(rows: Iterable[dict], path: str) -> str:
    """每行一个 run（JSON）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_jsonable(r), ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(obj, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)
    return path


# ============================================================ 汇总
def _get(row: dict, key: str):
    """支持 "a.b" 形式的嵌套键。"""
    cur = row
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _t975(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    from scipy.stats import t
    return float(t.ppf(0.975, dof))


def _group_key(v):
    """分组键归一化（审查 INT-NUM-4）：先转 JSON 友好对象，list/dict（含原 tuple、ndarray）再序列化成字符串以便哈希。"""
    v = to_jsonable(v)
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v


_PROPORTION_PREFIXES = ("acc", "all_correct")       # 末段以此开头的指标视为比例 → 区间截到 [0, 1]


def _metric_bounds(metric: str, raw_vals: list, bounds) -> tuple | None:
    """区间截断范围：bounds 显式给出时用之；"auto" 时布尔指标、或末段以 acc / all_correct 开头的指标为 (0, 1)。"""
    if isinstance(bounds, dict):
        return bounds.get(metric)
    if bounds != "auto":
        return None
    last = metric.split(".")[-1]
    if (raw_vals and all(isinstance(v, (bool, np.bool_)) for v in raw_vals)) or last.startswith(_PROPORTION_PREFIXES):
        return (0.0, 1.0)
    return None


def aggregate(rows: list[dict], keys: list[str], metrics: list[str], skip_errors: bool = True,
              exclude_if: str | None = None, bounds="auto") -> list[dict]:
    """按 keys 分组，对每个 metric 给出均值、标准差（ddof=1）、95% 区间（t 分布）与有效样本数 n。

    输出每组一行：keys 的取值 + n_total（该组全部行）+ n_errors（error 行）+ n_excluded（exclude_if 为真的行）
    + n_runs（参与统计的有效行）+ 对每个 metric m：m_mean、m_std、m_ci95（半宽）、m_lo、m_hi、m_n。
    - 布尔值按 0/1 平均（即比例）；None / 非有限值不计入（m_n 反映有效个数）。键支持 "a.b" 嵌套；
      tuple / ndarray / list / dict 键值先转 JSON 再序列化（审查 INT-NUM-4）。
    - n = 1 时 std / ci95 / lo / hi 为 nan（ddof=1 无定义，审查 INT-NUM-5）。
    - error 行（skip_errors=True）不参与统计，但仍归入其分组并计入 n_errors：全部出错的格点也会出现（n_runs = 0）。
      请用 job 字段（如 "job.condition"）分组，error 行才落在正确的格点。
    - exclude_if：嵌套键，取值为真的行不参与统计（计入 n_excluded），例如 "eval.oscillating_run"
      （方案 §4.5：自激 run 单独统计、不参与精度均值；也可直接汇总 evaluate 的 *_valid 字段）。
    - bounds："auto"（默认）把比例类指标（布尔，或末段以 acc / all_correct 开头）的 lo/hi 截到 [0, 1]；
      dict {metric: (lo, hi)} 显式指定；None 不截断。ci95 半宽本身不截断。
    """
    groups: dict = {}
    order = []
    for r in rows:
        g = tuple(_group_key(_get(r, k)) for k in keys)
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(r)
    table = []
    for g in order:
        allrows = groups[g]
        errs = [r for r in allrows if "error" in r] if skip_errors else []
        rs = [r for r in allrows if not (skip_errors and "error" in r)]
        excl = [r for r in rs if exclude_if is not None and bool(_get(r, exclude_if))]
        rs = [r for r in rs if not (exclude_if is not None and bool(_get(r, exclude_if)))]
        out = dict(zip(keys, g))
        out.update(n_total=len(allrows), n_errors=len(errs), n_excluded=len(excl), n_runs=len(rs))
        for m in metrics:
            vals, raw = [], []
            for r in rs:
                v = _get(r, m)
                if v is not None:
                    raw.append(v)
                if isinstance(v, (bool, np.bool_)):
                    v = float(v)
                if isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(float(v)):
                    vals.append(float(v))
            n = len(vals)
            nan = float("nan")
            if n:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if n > 1 else nan
                half = _t975(n - 1) * std / math.sqrt(n) if n > 1 else nan
            else:
                mean = std = half = nan
            lo, hi = (mean - half, mean + half) if n > 1 else (nan, nan)
            bd = _metric_bounds(m, raw, bounds)
            if bd is not None and n > 1:
                lo, hi = max(lo, bd[0]), min(hi, bd[1])
            out.update({f"{m}_mean": mean, f"{m}_std": std, f"{m}_ci95": half, f"{m}_lo": lo, f"{m}_hi": hi,
                        f"{m}_n": n})
        table.append(out)
    return table


def _fmt(v, floatfmt: str) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "nan" if not math.isfinite(float(v)) else format(float(v), floatfmt)
    return str(v)


def to_markdown(table: list[dict], columns: list[str] | None = None, floatfmt: str = ".4g") -> str:
    """表（dict 列表）→ Markdown 表格字符串。"""
    if not table:
        return "_(empty)_\n"
    cols = columns or list(dict.fromkeys(k for row in table for k in row))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in table:
        lines.append("| " + " | ".join(_fmt(row.get(c), floatfmt).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_csv(table: list[dict], path: str, columns: list[str] | None = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = columns or list(dict.fromkeys(k for row in table for k in row))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in table:
            w.writerow({c: to_jsonable(row.get(c)) for c in cols})
    return path


def savefig(fig, path: str, dpi: int = 150) -> str:
    """保存 matplotlib 图（Agg 后端）并关闭。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================ 输出目录与 CLI
def make_outdir(exp: str, out: str | None = None, stamp: str | None = None) -> str:
    """results/<exp>/<YYYYmmdd-HHMMSS>/（out 给定时直接用 out）。"""
    if out:
        path = out
    else:
        stamp = stamp or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(RESULTS_ROOT, exp, stamp)
    os.makedirs(path, exist_ok=True)
    return path


def write_outputs(outdir: str, rows: list[dict], table: list[dict], config: dict | None = None,
                  columns: list[str] | None = None, title: str | None = None) -> dict:
    """标准输出：runs.jsonl、summary.csv、summary.md、config.json。返回各文件路径。"""
    paths = {"runs": save_jsonl(rows, os.path.join(outdir, "runs.jsonl")),
             "csv": save_csv(table, os.path.join(outdir, "summary.csv"), columns)}
    md = (f"# {title}\n\n" if title else "") + to_markdown(table, columns)
    n_err = sum(1 for r in rows if "error" in r)
    if n_err:
        md += f"\n{n_err} 个 job 出错（见 runs.jsonl 的 error 字段）。\n"
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    paths["md"] = os.path.join(outdir, "summary.md")
    if config is not None:
        paths["config"] = save_json(config, os.path.join(outdir, "config.json"))
    return paths


def base_argparser(description: str = "") -> argparse.ArgumentParser:
    """实验脚本通用 CLI：--quick、--workers（默认 cpu−2）、--seeds、--out。"""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="快速模式（笔记本 ≤ 10 分钟）")
    p.add_argument("--workers", type=int, default=default_workers(), help="并行进程数")
    p.add_argument("--seeds", type=int, default=5, help="每格随机种子数")
    p.add_argument("--out", type=str, default=None, help="输出目录（默认 results/<exp>/<时间戳>/）")
    return p


def seed_rng(*parts) -> np.random.Generator:
    """由若干整数/字符串确定性地派生 Generator（每个 job 独立、可复现；字符串用 sha256，不用随进程变化的 hash()）。"""
    import hashlib

    ints = []
    for p in parts:
        if isinstance(p, (int, np.integer)):
            ints.append(int(p) & 0xFFFFFFFF)
        else:
            ints.append(int.from_bytes(hashlib.sha256(str(p).encode("utf-8")).digest()[:4], "little"))
    return np.random.default_rng(np.random.SeedSequence(ints))
