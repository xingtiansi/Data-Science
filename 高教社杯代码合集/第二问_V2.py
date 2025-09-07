"""
Q2 完整代码（可直接粘贴为 .py 或放进 Notebook 运行）
------------------------------------------------------
思路：
1) 以个体为单位重构(孕周, Y%)序列，做单调平滑，求“首次达标周”T（阈值=4%+缓冲delta）。
2) 以 T 为目标变量，保留 BMI（和可选批次/医院等FE）作为特征；此处用简单、稳健的“经验分布+监督分箱”方案，
   直接在每个候选 BMI 区间上，以“期望成本”EC(t)=c_early*P(T>t)+c_late*E[(t-T)_+ * 1{T>W_late}] 为目标，
   在离散周 t∈[min_week, max_week] 暴力搜索最优 t*，并用动态规划从所有切法中挑总成本最小的分组方案。
3) 提供灵敏度分析（delta、c_early/c_late、组数K、晚发现阈值W_late）。

依赖：pandas, numpy, scikit-learn（仅IsotonicRegression），matplotlib(可选)。
注意：为兼容不同数据列名，运行前请在 CONFIG 区域配置列名与文件路径。
"""

from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# =============================
# ======== CONFIG 区 ==========
# =============================
CONFIG = {
    "file_path": "/Users/shiyidianqianyaoshuijiao/Desktop/数模国赛/cleaned_data.xlsx",
    "id_col": "孕妇代码",      # 受试者ID
    "week_col": "检测孕周",    # 孕周
    "y_col": "Y染色体浓度",   # Y 浓度
    "bmi_col": "孕妇BMI",     # 已有 BMI 列
    "is_excel": True,
    "excel_sheet": 0,
}


# =============================
# ====== 数据加载与清洗 ========
# =============================

def load_and_prepare_data(cfg: Dict) -> pd.DataFrame:
    if cfg.get("is_excel", True):
        df = pd.read_excel(cfg["file_path"], sheet_name=cfg.get("excel_sheet", 0))
    else:
        df = pd.read_csv(cfg["file_path"])  # 如需 encoding/sep 可自行添加

    # 仅保留需要的列并统一命名
    cols_map = {
        cfg["id_col"]: "id",
        cfg["week_col"]: "week",
        cfg["y_col"]: "y",
        cfg["bmi_col"]: "bmi",
    }
    try:
        df = df[list(cols_map.keys())].rename(columns=cols_map)
    except KeyError as e:
        missing = [k for k in cols_map.keys() if k not in df.columns]
        raise KeyError(f"数据中缺少这些列，请在 CONFIG 里修正列名：{missing}") from e

    # 基本清洗
    df = df.dropna(subset=["id", "week", "y", "bmi"]).copy()

    # 数值化 & 合理范围
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df = df.dropna(subset=["week", "y", "bmi"]).copy()

    # Y% 单位自动判定（>1 视为百分数）
    if df["y"].median() > 1:
        df["y"] = df["y"] / 100.0

    # 裁剪到合理范围
    df = df[(df["week"] >= 0) & (df["week"] <= 45)].copy()
    df = df[(df["y"] >= 0) & (df["y"] <= 1)].copy()
    df = df[(df["bmi"] >= 10) & (df["bmi"] <= 60)].copy()

    # 同一人同一孕周若有重复，取均值（Y 取均值，BMI 取中位）
    df = (df.groupby(["id", "week"], as_index=False)
            .agg({"y": "mean", "bmi": "median"}))

    return df

# ==========================================
# ====== 单人序列：单调平滑 + 首次达标T ======
# ==========================================

def estimate_T_for_one(person_df: pd.DataFrame,
                       threshold: float = 0.04,
                       delta: float = 0.003,
                       enforce_monotonic: bool = True,
                       return_curve: bool = False) -> Tuple[Optional[float], bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    对同一受试者的序列（列：week, y），按孕周排序，做平滑并求“首次达标周”T。
    返回：
      T (float 或 None)  —— 若始终未达标，返回 None（视作右删失）
      censored (bool)    —— True 表示右删失（至末次观测仍未≥阈值）
      weeks_sorted (np.ndarray) —— （可选）排序后的周
      y_smooth (np.ndarray)     —— （可选）平滑/单调化后的 y
    """
    sdf = person_df.sort_values("week")
    w = sdf["week"].values
    y = sdf["y"].values

    if len(w) == 0:
        return None, True, None, None

    # 单调平滑：Y 应随孕周上升，用等距下的各向同性单调回归近似
    if enforce_monotonic and len(w) >= 2:
        # 为避免 x 的重复值问题，稍作抖动（或预先 groupby week 取均值，已做）
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        y_hat = iso.fit_transform(w, y)
    else:
        y_hat = y.copy()

    thr = threshold + delta

    # 若最后一点仍未达标，则视为右删失
    if y_hat[-1] < thr:
        return None, True, w, y_hat

    # 找最早交叉点：线性插值求精确周
    for i in range(1, len(w)):
        if y_hat[i-1] < thr <= y_hat[i]:
            # 线性插值 t = w[i-1] + (thr - y1)*(w2-w1)/(y2-y1)
            y1, y2 = y_hat[i-1], y_hat[i]
            x1, x2 = w[i-1], w[i]
            if y2 == y1:
                t_cross = x2
            else:
                t_cross = x1 + (thr - y1) * (x2 - x1) / (y2 - y1)
            return float(t_cross), False, w, y_hat

    # 理论上走不到这里；兜底
    return float(w[-1]), False, w, y_hat


def build_T_dataset(df: pd.DataFrame,
                    threshold: float = 0.04,
                    delta: float = 0.003) -> pd.DataFrame:
    """为每个受试者计算 (T, censored, bmi)。bmi 取个体中位数。"""
    records = []
    for pid, g in df.groupby("id"):
        T, cens, _, _ = estimate_T_for_one(g[["week", "y"]], threshold=threshold, delta=delta)
        bmi_val = float(g["bmi"].median()) if len(g) else np.nan
        records.append({"id": pid, "T": T, "censored": cens, "bmi": bmi_val})
    out = pd.DataFrame(records)
    out = out.dropna(subset=["bmi"])  # T 可能是 None（右删失），bmi 必须有
    return out

# ========================================
# ===== Kaplan-Meier 生存 + 期望成本 =====
# ========================================

def km_survival(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    简单 Kaplan-Meier：
    输入：times (事件或删失时间)，events(1=事件发生；0=右删失)
    输出：unique_event_times, S(t) 在这些事件时刻的取值（阶梯函数右极限）
    备注：用于估 P(T>t)。
    """
    # 仅在事件时刻更新 S
    df = pd.DataFrame({"t": times, "e": events}).sort_values(["t", "e"], ascending=[True, False])
    uniq_event_times = np.sort(df.loc[df["e"] == 1, "t"].unique())
    S = 1.0
    S_list = []
    at_risk_n = len(df)
    idx = 0
    # 迭代每个事件时刻，计算当时的在险人数和事件数
    for u in uniq_event_times:
        # 在时刻 u 之前离开的（删失/事件 <u）已在风险集中剔除；这里简化处理：逐时刻重算在险人数
        at_risk_n = ((df["t"] >= u)).sum()
        d_u = ((df["t"] == u) & (df["e"] == 1)).sum()
        if at_risk_n > 0:
            S *= (1.0 - d_u / at_risk_n)
        S_list.append(S)
    return uniq_event_times, np.array(S_list, dtype=float)


def S_of_t(t: float, event_times: np.ndarray, S_vals: np.ndarray) -> float:
    """返回任意 t 的 S(t)（右连续阶梯）：取 t 所在的最后一个事件时刻的 S。"""
    if len(event_times) == 0:
        return 1.0
    idx = np.searchsorted(event_times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(S_vals[idx])


def expected_late_term(t: float, W_late: float,
                       event_times: np.ndarray,
                       S_vals: np.ndarray) -> float:
    """
    计算 E[(t - T)_+ * 1{T>W_late}] ≈ sum_{u in (W_late, t]} (t - u) * dF(u)，
    其中 dF(u) ≈ -ΔS(u) 在事件时刻的跳跃量（来自 KM 曲线）。
    """
    if len(event_times) == 0 or t <= W_late:
        return 0.0
    # 计算跳跃量 dF(u) = -ΔS(u)
    S_prev = 1.0
    val = 0.0
    for u, S_u in zip(event_times, S_vals):
        if u <= W_late:
            S_prev = S_u
            continue
        if u > t:
            break
        dF = max(0.0, S_prev - S_u)
        val += (t - float(u)) * dF
        S_prev = S_u
    return float(val)


@dataclass
class CostParams:
    c_early: float = 1.0
    c_late: float = 1.0
    W_late: float = 12.0   # 晚发现开始计惩罚的阈值周（示例）


def compute_bin_cost(times: np.ndarray, events: np.ndarray,
                     t_grid: np.ndarray, params: CostParams) -> Tuple[float, float, Dict]:
    """
    给定一个 BMI 分组样本(的 T, event)，在 t_grid 上找使 EC(t) 最小的 t*。
    返回：(最小成本, t*, 附加信息dict)
    附加信息包含：P_early(t*), P_late(t*), S曲线节点等。
    """
    # KM 生存曲线
    ev_times, S_vals = km_survival(times, events)

    best_cost = math.inf
    best_t = None
    best_info = {}

    for t in t_grid:
        P_early = S_of_t(t, ev_times, S_vals)  # P(T>t)
        late_term = expected_late_term(t, params.W_late, ev_times, S_vals)
        cost = params.c_early * P_early + params.c_late * late_term
        if cost < best_cost:
            # 粗略估计晚发现概率：P(W_late < T ≤ t) ≈ F(t) - F(W_late)
            S_t = S_of_t(t, ev_times, S_vals)
            S_w = S_of_t(params.W_late, ev_times, S_vals)
            P_late = max(0.0, (1 - S_t) - (1 - S_w))
            best_cost = cost
            best_t = float(t)
            best_info = {
                "P_early": float(P_early),
                "P_late": float(P_late),
                "ev_times": ev_times,
                "S_vals": S_vals,
            }
    return best_cost, best_t, best_info

# ======================================
# ====== 监督分箱（动态规划切分） ======
# ======================================

def precompute_bin_costs(dfT: pd.DataFrame,
                         candidate_edges: np.ndarray,
                         t_grid: np.ndarray,
                         params: CostParams,
                         min_bin_size: int = 30) -> Tuple[np.ndarray, np.ndarray, List[List[Dict]]]:
    """
    对所有候选 BMI 区间 [edge[i], edge[j]) 预先计算：
      - cost[i,j]：该区间的最小期望成本
      - best_t[i,j]：对应的最优检测周
      - extras[i][j]：字典，含 P_early, P_late 等
    不满足最小样本量的区间记为 +inf。
    """
    M = len(candidate_edges)
    cost = np.full((M, M), np.inf, dtype=float)
    best_t = np.full((M, M), np.nan, dtype=float)
    extras: List[List[Dict]] = [[{} for _ in range(M)] for __ in range(M)]

    for i in range(M - 1):
        for j in range(i + 1, M):
            lo, hi = candidate_edges[i], candidate_edges[j]
            sub = dfT[(dfT["bmi"] >= lo) & (dfT["bmi"] < hi)]
            if len(sub) < min_bin_size:
                continue
            times = sub["T"].fillna(sub["T"].max() + 1e-6).values  # 对None已在构建阶段保留为空；这里用 fillna 仅防极端
            events = (~sub["censored"]).astype(int).values
            c, t_star, info = compute_bin_cost(times, events, t_grid, params)
            cost[i, j] = c
            best_t[i, j] = t_star
            extras[i][j] = info | {"n": int(len(sub)), "bmi_range": (float(lo), float(hi))}
    return cost, best_t, extras


def optimal_binning_dp(candidate_edges: np.ndarray,
                        cost_mat: np.ndarray,
                        K: int) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
    """
    动态规划：在候选边界上选 K 个分箱（K 个区间），使区间成本和最小。
    返回：
      bins_ranges: 每个箱的 (lo, hi)
      idx_pairs: 每个箱对应的 (i, j) 索引区间
    """
    M = len(candidate_edges)
    dp = np.full((K + 1, M), np.inf)
    prev = [[-1] * M for _ in range(K + 1)]

    dp[0, 0] = 0.0

    for k in range(1, K + 1):
        for j in range(1, M):
            # 枚举上一个断点 i
            best_val = np.inf
            best_i = -1
            for i in range(0, j):
                c = cost_mat[i, j]
                if not np.isfinite(c):
                    continue
                if np.isfinite(dp[k - 1, i]):
                    v = dp[k - 1, i] + c
                    if v < best_val:
                        best_val = v
                        best_i = i
            dp[k, j] = best_val
            prev[k][j] = best_i

    # 终点在 M-1（必须用到最右边界）
    if not np.isfinite(dp[K, M - 1]):
        raise RuntimeError("无法在给定候选边界与样本量约束下找到可行的 K 组切分；请降低 K 或放宽 min_bin_size。")

    # 回溯区间
    idx_pairs: List[Tuple[int, int]] = []
    j = M - 1
    for k in range(K, 0, -1):
        i = prev[k][j]
        if i < 0:
            raise RuntimeError("回溯失败，请检查 cost 矩阵是否连通。")
        idx_pairs.append((i, j))
        j = i
    idx_pairs.reverse()

    bins_ranges = [(float(candidate_edges[i]), float(candidate_edges[j])) for (i, j) in idx_pairs]
    return bins_ranges, idx_pairs

# ==================================
# ===== 主流程 + 灵敏度分析 ========
# ==================================

def run_pipeline(cfg: Dict,
                 threshold: float = 0.04,
                 delta: float = 0.003,
                 K: int = 4,
                 min_week: float = 9.0,
                 max_week: float = 16.0,
                 week_step: float = 0.25,
                 min_bin_size: int = 30,
                 cost_params: Optional[CostParams] = None,
                 candidate_mode: str = "quantile",
                 n_edges: int = 21,
                 verbose: bool = True) -> Dict:
    """
    运行完整管线：
     1) 读数-清洗；2) 估计 T；3) 预计算每个 BMI 区间成本；4) DP 找最优分箱；5) 汇报结果。
    参数：
      - K: BMI 分组数量
      - delta: 阈值缓冲（默认 0.003 = 0.3个百分点）
      - candidate_mode: 候选边界生成方式（'quantile' 或 'unique'）
      - n_edges: 若为 quantile，则在 [min,max] 均匀分位生成 n_edges 个边界点
    返回：包含关键中间结果与最终方案的字典。
    """
    if cost_params is None:
        cost_params = CostParams()

    if verbose:
        print("[1/5] 读取与清洗数据…")
    df = load_and_prepare_data(cfg)

    if verbose:
        print("[2/5] 估计各个体的首次达标周 T（带阈值缓冲）…")
    dfT = build_T_dataset(df, threshold=threshold, delta=delta)

    # 候选 BMI 边界
    if candidate_mode == "quantile":
        qs = np.linspace(0, 1, n_edges)
        edges = np.quantile(dfT["bmi"].values, qs)
        edges[0] = float(dfT["bmi"].min())
        edges[-1] = float(dfT["bmi"].max()) + 1e-9  # 右开区间保障覆盖
        candidate_edges = np.unique(edges)
    else:
        vals = np.sort(dfT["bmi"].values)
        candidate_edges = np.r_[vals[::max(1, len(vals)//(n_edges-1))], vals[-1] + 1e-9]

    # t 的离散搜索网格
    t_grid = np.arange(min_week, max_week + 1e-9, week_step)

    if verbose:
        print("[3/5] 预计算所有候选 BMI 区间的最优 t* 与成本…")
    cost_mat, best_t_mat, extras = precompute_bin_costs(
        dfT, candidate_edges, t_grid, cost_params, min_bin_size=min_bin_size
    )

    if verbose:
        print("[4/5] 动态规划搜索 K 组的全局最优切分…")
    bin_ranges, idx_pairs = optimal_binning_dp(candidate_edges, cost_mat, K)

    # 汇总每组信息
    groups = []
    total_cost = 0.0
    for (i, j), (lo, hi) in zip(idx_pairs, bin_ranges):
        c = cost_mat[i, j]
        t_star = best_t_mat[i, j]
        info = extras[i][j]
        groups.append({
            "BMI_range": (lo, hi),
            "t_star": float(t_star),
            "expected_cost": float(c),
            "P_early": float(info.get("P_early", np.nan)),
            "P_late": float(info.get("P_late", np.nan)),
            "n": int(info.get("n", 0)),
        })
        total_cost += float(c)

    result = {
        "config": cfg,
        "threshold": threshold,
        "delta": delta,
        "K": K,
        "min_week": min_week,
        "max_week": max_week,
        "week_step": week_step,
        "min_bin_size": min_bin_size,
        "cost_params": cost_params,
        "candidate_edges": candidate_edges,
        "t_grid": t_grid,
        "groups": groups,
        "total_expected_cost": total_cost,
        "dfT": dfT,
    }

    if verbose:
        print("[5/5] 完成。")
        print("—— 推荐方案 ——")
        for g in groups:
            lo, hi = g["BMI_range"]
            print(f"BMI∈[{lo:.2f}, {hi:.2f}) → 推荐检测周 t* = {g['t_star']:.2f}（样本 n={g['n']}）\n"
                  f"    早到概率≈P(T>t*)={g['P_early']:.3f}，晚发现概率≈{g['P_late']:.3f}，期望成本={g['expected_cost']:.4f}")
        print(f"总期望成本 = {total_cost:.4f}")

    return result

# ================================
# ======== 灵敏度分析工具 =========
# ================================

def sensitivity_grid(cfg: Dict,
                      K_list: List[int] = (3, 4, 5),
                      delta_list: List[float] = (0.002, 0.003, 0.005),
                      c_early_list: List[float] = (1.0, 1.0, 1.0),
                      c_late_list: List[float] = (0.5, 1.0, 2.0),
                      W_late_list: List[float] = (11.0, 12.0, 13.0),
                      **kwargs) -> pd.DataFrame:
    """网格扫描关键参数，比较总期望成本与方案稳定性。"""
    rows = []
    for K in K_list:
        for delta in delta_list:
            for ce in c_early_list:
                for cl in c_late_list:
                    for Wl in W_late_list:
                        params = CostParams(c_early=ce, c_late=cl, W_late=Wl)
                        try:
                            res = run_pipeline(cfg, K=K, delta=delta, cost_params=params, verbose=False, **kwargs)
                            rows.append({
                                "K": K,
                                "delta": delta,
                                "c_early": ce,
                                "c_late": cl,
                                "W_late": Wl,
                                "total_cost": res["total_expected_cost"],
                                "groups": res["groups"],
                            })
                        except Exception as e:
                            rows.append({
                                "K": K, "delta": delta, "c_early": ce, "c_late": cl, "W_late": Wl,
                                "total_cost": np.nan, "groups": str(e)
                            })
    return pd.DataFrame(rows)

# ================================
# =========== 示例运行 ============
# ================================
if __name__ == "__main__":
    # 1) 先把 CONFIG 里的列名改成你实际的数据列名，然后再运行。
    # 2) 设定关键参数：
    cfg = CONFIG.copy()

    params = CostParams(
        c_early=1.0,   # 过早（未达标/复检）的单位代价
        c_late=1.5,    # 过晚（超过 W_late 后）的单位代价
        W_late=12.0    # 晚发现起算周
    )

    # 运行主流程（K=4 组；t 在 9~16 周、步长 0.25 周 里选最优）
    result = run_pipeline(
        cfg,
        threshold=0.04,
        delta=0.003,
        K=4,
        min_week=9.0,
        max_week=16.0,
        week_step=0.25,
        min_bin_size=30,
        cost_params=params,
        candidate_mode="quantile",
        n_edges=21,
        verbose=True,
    )

    # 如需做灵敏度：
    # grid = sensitivity_grid(cfg, K_list=[3,4,5], delta_list=[0.002,0.003,0.005],
    #                         c_early_list=[1.0], c_late_list=[0.5,1.0,2.0], W_late_list=[11.0,12.0,13.0],
    #                         threshold=0.04, min_week=9.0, max_week=16.0, week_step=0.25, min_bin_size=30,
    #                         candidate_mode="quantile", n_edges=21)
    # print(grid.sort_values("total_cost").head(10))


#结果生成图：直观解释
# ====== 0) 解决中文乱码 ======
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 优先使用 macOS 的 PingFang SC；否则回退到常见中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC','Heiti SC','Heiti TC','Arial Unicode MS','SimHei','Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ====== 1) 准备数据 ======
USE_RESULT = False  # 如果你环境里有 run_pipeline 的返回值 result，就改成 True

if USE_RESULT:
    # A) 直接用 run_pipeline 的返回值
    # 假设你已经运行过：result = run_pipeline(...)
    groups = []
    for g in result["groups"]:
        lo, hi = g["BMI_range"]
        groups.append({
            "BMI_range": (lo, hi),
            "t_star": g["t_star"],
            "P_early": g["P_early"],
            "P_late": g["P_late"],
            "expected_cost": g["expected_cost"],
        })
else:
    # B) 用你贴出来的控制台结果手动构造
    groups = [
        {"BMI_range": (26.81, 31.22), "t_star": 14.75, "P_early": 0.880, "P_late": 0.120, "expected_cost": 0.9682},
        {"BMI_range": (31.22, 32.91), "t_star": 14.00, "P_early": 0.904, "P_late": 0.096, "expected_cost": 0.9343},
        {"BMI_range": (32.91, 34.16), "t_star": 12.50, "P_early": 0.949, "P_late": 0.026, "expected_cost": 0.9530},
        {"BMI_range": (34.16, 39.30), "t_star": 13.00, "P_early": 0.925, "P_late": 0.057, "expected_cost": 0.9524},
    ]

df_groups = pd.DataFrame(groups)
labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in df_groups["BMI_range"]]

# ====== 2) 图一：各 BMI 组的推荐检测周 t* ======
plt.figure(figsize=(8, 5))
plt.barh(labels, df_groups["t_star"])
plt.xlabel("推荐检测周 t*")
plt.ylabel("BMI 分组")
plt.title("各 BMI 组的推荐检测周")
for y, v in enumerate(df_groups["t_star"]):
    plt.text(v, y, f"  {v:.2f}", va="center")  # 条形末尾标注
plt.tight_layout()
# plt.savefig("fig_tstar_by_bmi.png", dpi=300)  # 如需保存，取消注释
plt.show()

# ====== 3) 图二：各组早到概率 & 晚发现概率 ======
x = np.arange(len(df_groups))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, df_groups["P_early"], width, label="早到概率 P(T>t*)")
rects2 = ax.bar(x + width/2, df_groups["P_late"],  width, label="晚发现概率")

ax.set_ylabel("概率")
ax.set_title("各 BMI 组的早到/晚发现概率")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 在柱顶标注数值
for r in rects1:
    h = r.get_height()
    ax.text(r.get_x() + r.get_width()/2, h, f"{h:.3f}", ha="center", va="bottom")
for r in rects2:
    h = r.get_height()
    ax.text(r.get_x() + r.get_width()/2, h, f"{h:.3f}", ha="center", va="bottom")

plt.tight_layout()
# plt.savefig("fig_probs_by_bmi.png", dpi=300)
plt.show()

# ====== 4) 图三：各组期望成本 ======
plt.figure(figsize=(8, 5))
plt.bar(labels, df_groups["expected_cost"])
plt.ylabel("期望成本")
plt.title("各 BMI 组的期望成本")
for i, v in enumerate(df_groups["expected_cost"]):
    plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.tight_layout()
# plt.savefig("fig_cost_by_bmi.png", dpi=300)
plt.show()