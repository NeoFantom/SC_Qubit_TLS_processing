import os
import re
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
from collections import Counter
from typing import Dict, Any, List

# -------------------------------------------------------------------------
# 字体配置：使用 SimHei 支持中文，避免 Glyph missing 警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# find_npy_info_from_pixel_fast：返回原始时间字符串及对应的 datetime 对象
NPY_PATH = r"C:\Users\zepin\Desktop\TLS\data\TLS_DATA_2024Nov_2025Apr.npy"
TLS_DATA = np.load(NPY_PATH, allow_pickle=True).item()

def find_npy_info_from_pixel_fast(qubit: int, pixel_x: int, pixel_y: int,
                                  figsize=(16, 8), dpi=360) -> Dict[str, Any]:
    data = TLS_DATA[qubit]
    xs_list, ys_list, times = data["xs"], data["ys"], data["times"]
    n = min(len(xs_list), len(ys_list), len(times))
    xs_list, ys_list, times = xs_list[:n], ys_list[:n], times[:n]

    coords, meta = [], []
    for i, (x_arr, y_arr) in enumerate(zip(xs_list, ys_list)):
        coords.append(np.column_stack([np.full_like(x_arr, i), x_arr]))
        meta.extend([(i, j) for j in range(len(x_arr))])
    coords = np.vstack(coords)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.5, len(xs_list) - 0.5)
    all_x = np.concatenate(xs_list)
    ax.set_ylim(all_x.min(), all_x.max())
    fig.canvas.draw()
    trans = ax.transData

    pix = trans.transform(coords)
    dist = np.hypot(pix[:, 0] - pixel_x, pix[:, 1] - pixel_y)
    idx = dist.argmin()
    i_sel, j_sel = meta[idx]

    t_str = times[i_sel]  # e.g. 'Wed Jan 22 05:16:31 2025'
    t_dt  = datetime.datetime.strptime(t_str, "%a %b %d %H:%M:%S %Y")

    plt.close(fig)
    return {
        'qubit': qubit,
        'i,j': (i_sel, j_sel),
        'x': float(xs_list[i_sel][j_sel]),
        'y': float(ys_list[i_sel][j_sel]),
        'time_str': t_str,
        'time_dt': t_dt
    }
# -------------------------------------------------------------------------

INPUT_FOLDER  = r"E:\TLS\u2net-pytorch-main\result4"
OUTPUT_FOLDER = r"E:\TLS\u2net-pytorch-main\result5"

def extract_qubit_from_image_name(image_name: str) -> int:
    m = re.search(r'_Q(\d+)_', image_name)
    if m:
        return int(m.group(1))
    raise ValueError(f"无法从文件名中提取 qubit: '{image_name}'")

def create_jump_plot(times_dt: List[datetime.datetime],
                     counts: List[int],
                     title: str,
                     xlabel: str,
                     ylabel: str,
                     color: str,
                     output_path: str,
                     figsize: tuple = (12, 6)) -> str:
    """
    创建跳跃图表：按索引均匀分布柱状，x 轴标签用两行显示时间，柱子上方标注次数
    figsize 参数用于指定图表宽度与高度，默认 (12,6)。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if times_dt and counts:
        xs = np.arange(len(times_dt))
        bars = ax.bar(xs, counts, width=0.8, alpha=0.7, color=color, label=ylabel)
        max_count = max(counts)
        for count, bar in zip(counts, bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count + max_count * 0.01,
                str(count),
                ha='center', va='bottom',
                fontsize=8, fontweight='bold'
            )
        labels = [dt.strftime('%Y-%m-%d\n%H:%M:%S') for dt in times_dt]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        fig.subplots_adjust(bottom=0.35)
    else:
        now = datetime.datetime.now()
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, '无跳跃数据', transform=ax.transAxes,
                ha='center', va='center', fontsize=16, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if times_dt and counts:
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def process_all():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 全局事件聚合
    qubit_events: Dict[int, Dict[str, List[Any]]] = {}
    all_qubits = set()

    # 遍历每个 connections.json
    for fname in os.listdir(INPUT_FOLDER):
        if not fname.endswith("_connections.json"):
            continue

        file_out_dt: List[datetime.datetime] = []
        file_in_dt:  List[datetime.datetime] = []

        path_in = os.path.join(INPUT_FOLDER, fname)
        with open(path_in, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_name = data["image_name"]
        qubit      = extract_qubit_from_image_name(image_name)
        all_qubits.add(qubit)

        qubit_events.setdefault(qubit, {
            "jump_out_str": [], "jump_in_str": [],
            "jump_out_dt":  [], "jump_in_dt":  []
        })

        new_conns = []
        for conn in data.get("connections", []):
            sx, sy = conn["start_pixel"]["x"], conn["start_pixel"]["y"]
            ex, ey = conn["end_pixel"]["x"],   conn["end_pixel"]["y"]

            out_info = find_npy_info_from_pixel_fast(qubit, sx, sy)
            in_info  = find_npy_info_from_pixel_fast(qubit, ex, ey)

            t_out_str, t_in_str = out_info["time_str"], in_info["time_str"]
            t_out_dt,  t_in_dt  = out_info["time_dt"],  in_info["time_dt"]

            # 更新全局数据
            qubit_events[qubit]["jump_out_str"].append(t_out_str)
            qubit_events[qubit]["jump_in_str"].append(t_in_str)
            qubit_events[qubit]["jump_out_dt"].append(t_out_dt)
            qubit_events[qubit]["jump_in_dt"].append(t_in_dt)

            # 收集当前文件数据
            file_out_dt.append(t_out_dt)
            file_in_dt.append(t_in_dt)

            new_conns.append({
                "connection_id": conn["connection_id"],
                "from": {
                    "rectangle": conn["source_rectangle"]["name"],
                    "pixel":     {"x": sx, "y": sy},
                    "time":      t_out_str
                },
                "to": {
                    "rectangle": conn["target_rectangle"]["name"],
                    "pixel":     {"x": ex, "y": ey},
                    "time":      t_in_str
                },
                "length": conn.get("length")
            })

        # 保存单文件 JSON
        out_name = fname.replace("_connections.json", "_time_connections.json")
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                "image_name":  image_name,
                "qubit":       qubit,
                "connections": new_conns
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ 已保存 JSON: {out_name}")

        # 单文件绘图 — 跳出
        cnts_out  = Counter(file_out_dt)
        times_out = sorted(cnts_out) if cnts_out else []
        vals_out  = [cnts_out[t] for t in times_out] if times_out else []
        fig_out   = out_path.replace(".json", "_jump_out_bar.png")
        create_jump_plot(times_out, vals_out,
                         f"{image_name} 跳出次数分布",
                         "真实时间", "跳出次数", 'red', fig_out)
        print(f"📈 已保存: {os.path.basename(fig_out)}")

        # 单文件绘图 — 跳入
        cnts_in   = Counter(file_in_dt)
        times_in  = sorted(cnts_in) if cnts_in else []
        vals_in   = [cnts_in[t] for t in times_in] if times_in else []
        fig_in    = out_path.replace(".json", "_jump_in_bar.png")
        create_jump_plot(times_in, vals_in,
                         f"{image_name} 跳入次数分布",
                         "真实时间", "跳入次数", 'blue', fig_in)
        print(f"📈 已保存: {os.path.basename(fig_in)}")

    # —— 全局汇总与统计 —— #
    summary = {
        q: {
            "jump_out": ev["jump_out_str"],
            "jump_in":  ev["jump_in_str"]
        }
        for q, ev in qubit_events.items()
    }
    summary_path = os.path.join(OUTPUT_FOLDER, "jump_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("✅ 已保存 全局 jump_summary.json")

    # 全局跳出柱状图：分为四张
    all_out_times = []
    for ev in qubit_events.values():
        all_out_times.extend(ev["jump_out_dt"])
    out_cnts     = Counter(all_out_times)
    sorted_out   = sorted(out_cnts) if out_cnts else []
    vals_all_out = [out_cnts[t] for t in sorted_out] if sorted_out else []

    n_out   = len(sorted_out)
    # 四等分大小，至少 1，否则 chunk 要为 0 会出错
    chunk_o = math.ceil(n_out / 4) if n_out else 1
    for part_idx in range(4):
        start = part_idx * chunk_o
        end   = min((part_idx + 1) * chunk_o, n_out)
        times_part = sorted_out[start:end]
        vals_part  = vals_all_out[start:end]
        suffix = f"_part{part_idx+1}"

        # 画图
        fig_name = f"all_qubits_jump_out_bar{suffix}.png"
        fig_out  = os.path.join(OUTPUT_FOLDER, fig_name)
        if not times_part:
            # 空数据也要生成一张空图
            create_jump_plot([], [], "所有 Qubit 跳出总次数分布",
                             "真实时间", "总跳出次数", 'red', fig_out)
            print(f"📊 已保存 全局跳出图: {fig_name}")
            continue

        # 宽度按数据量乘系数，至少 20
        width_out = max(20, len(times_part) * 0.4)
        create_jump_plot(times_part, vals_part,
                         "所有 Qubit 跳出总次数分布",
                         "真实时间", "总跳出次数", 'red', fig_out,
                         figsize=(width_out, 6))
        print(f"📊 已保存 全局跳出图: {fig_name}")

    # 全局跳入柱状图：分为四张
    all_in_times = []
    for ev in qubit_events.values():
        all_in_times.extend(ev["jump_in_dt"])
    in_cnts       = Counter(all_in_times)
    sorted_in     = sorted(in_cnts) if in_cnts else []
    vals_all_in   = [in_cnts[t] for t in sorted_in] if sorted_in else []

    n_in   = len(sorted_in)
    chunk_i = math.ceil(n_in / 4) if n_in else 1
    for part_idx in range(4):
        start = part_idx * chunk_i
        end   = min((part_idx + 1) * chunk_i, n_in)
        times_part = sorted_in[start:end]
        vals_part  = vals_all_in[start:end]
        suffix = f"_part{part_idx+1}"

        fig_name = f"all_qubits_jump_in_bar{suffix}.png"
        fig_in   = os.path.join(OUTPUT_FOLDER, fig_name)
        if not times_part:
            create_jump_plot([], [], "所有 Qubit 跳入总次数分布",
                             "真实时间", "总跳入次数", 'blue', fig_in)
            print(f"📊 已保存 全局跳入图: {fig_name}")
            continue

        width_in = max(20, len(times_part) * 0.4)
        create_jump_plot(times_part, vals_part,
                         "所有 Qubit 跳入总次数分布",
                         "真实时间", "总跳入次数", 'blue', fig_in,
                         figsize=(width_in, 6))
        print(f"📊 已保存 全局跳入图: {fig_name}")

    # 打印控制台统计
    total_out_events = len(all_out_times)
    total_in_events  = len(all_in_times)
    print("\n📈 全局统计信息:")
    print(f"总跳出事件数: {total_out_events}")
    print(f"总跳入事件数: {total_in_events}")
    print(f"总跳跃事件数: {total_out_events + total_in_events}")
    print(f"分析 Qubit 数量: {len(all_qubits)}")
    for q in sorted(all_qubits):
        o = len(qubit_events[q]["jump_out_dt"])
        i = len(qubit_events[q]["jump_in_dt"])
        status = "有跳跃" if (o + i) > 0 else "无跳跃"
        print(f"Qubit {q}: 跳出 {o} 次, 跳入 {i} 次 [{status}]")

if __name__ == "__main__":
    process_all()
