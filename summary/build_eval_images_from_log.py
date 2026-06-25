import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class EpochMetrics:
    epoch: int
    train_acc: Optional[float] = None
    val_sp: Optional[float] = None
    val_se: Optional[float] = None
    val_score: Optional[float] = None
    val_acc1: Optional[float] = None


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _default_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _nice_ticks(y_min: float, y_max: float, n: int = 6) -> List[float]:
    if n <= 1:
        return [y_min, y_max]
    step = (y_max - y_min) / (n - 1)
    return [y_min + i * step for i in range(n)]


def _draw_line_chart(
    out_path: str,
    title: str,
    epochs: List[int],
    series: Dict[str, List[Optional[float]]],
    y_min: float,
    y_max: float,
    y_label: str,
    x_label: str = "epoch",
    best_epochs: Optional[List[int]] = None,
    annotate_best: bool = False,
):
    width, height = 1400, 800
    margin_l, margin_r, margin_t, margin_b = 110, 320, 90, 110
    plot_l, plot_t = margin_l, margin_t
    plot_r, plot_b = width - margin_r, height - margin_b

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_title = _default_font(28)
    font_axis = _default_font(16)
    font_tick = _default_font(14)

    draw.text((margin_l, 25), title, fill=(0, 0, 0), font=font_title)

    draw.rectangle([plot_l, plot_t, plot_r, plot_b], outline=(0, 0, 0), width=2)

    ticks = _nice_ticks(y_min, y_max, n=6)
    for t in ticks:
        y = plot_b - (t - y_min) / (y_max - y_min + 1e-9) * (plot_b - plot_t)
        draw.line([plot_l, y, plot_r, y], fill=(230, 230, 230), width=1)
        draw.text((plot_l - 10, y - 7), f"{t:.1f}", fill=(0, 0, 0), font=font_tick, anchor="ra")

    if epochs:
        x_ticks = list(range(min(epochs), max(epochs) + 1))
        step = max(1, len(x_ticks) // 10)
        for e in x_ticks[::step]:
            x = plot_l + (e - x_ticks[0]) / (x_ticks[-1] - x_ticks[0] + 1e-9) * (plot_r - plot_l)
            draw.line([x, plot_t, x, plot_b], fill=(245, 245, 245), width=1)
            draw.text((x, plot_b + 10), str(e), fill=(0, 0, 0), font=font_tick, anchor="ma")

    draw.text(((plot_l + plot_r) / 2, height - 55), x_label, fill=(0, 0, 0), font=font_axis, anchor="ma")
    draw.text((25, (plot_t + plot_b) / 2), y_label, fill=(0, 0, 0), font=font_axis)

    colors = [
        (30, 136, 229),
        (67, 160, 71),
        (244, 67, 54),
        (156, 39, 176),
        (255, 152, 0),
    ]
    names = list(series.keys())
    for i, name in enumerate(names):
        vals = series[name]
        color = colors[i % len(colors)]
        last_xy = None
        for e, v in zip(epochs, vals):
            if v is None:
                last_xy = None
                continue
            x = plot_l + (e - epochs[0]) / (epochs[-1] - epochs[0] + 1e-9) * (plot_r - plot_l)
            y = plot_b - (v - y_min) / (y_max - y_min + 1e-9) * (plot_b - plot_t)
            if last_xy is not None:
                draw.line([last_xy[0], last_xy[1], x, y], fill=color, width=3)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color, outline=color)
            last_xy = (x, y)

    if best_epochs:
        for be in best_epochs:
            if not epochs or be < epochs[0] or be > epochs[-1]:
                continue
            x = plot_l + (be - epochs[0]) / (epochs[-1] - epochs[0] + 1e-9) * (plot_r - plot_l)
            draw.line([x, plot_t, x, plot_b], fill=(0, 0, 0), width=1)

    legend_x = plot_r + 30
    legend_y = plot_t + 20
    for i, name in enumerate(names):
        color = colors[i % len(colors)]
        y0 = legend_y + i * 34
        draw.line([legend_x, y0 + 10, legend_x + 40, y0 + 10], fill=color, width=4)
        draw.text((legend_x + 55, y0), name, fill=(0, 0, 0), font=font_axis)

    if annotate_best and "val_score" in series:
        best_e = None
        best_v = None
        for e, v in zip(epochs, series["val_score"]):
            if v is None:
                continue
            if best_v is None or v > best_v:
                best_v = v
                best_e = e
        if best_e is not None and best_v is not None:
            text = f"best: epoch={best_e}, score={best_v:.2f}"
            draw.text((legend_x, legend_y + len(names) * 34 + 25), text, fill=(0, 0, 0), font=font_axis)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def _draw_scatter(
    out_path: str,
    title: str,
    xs: List[float],
    ys: List[float],
    labels: Optional[List[str]] = None,
    x_min: float = 0,
    x_max: float = 100,
    y_min: float = 0,
    y_max: float = 100,
    x_label: str = "Sp (%)",
    y_label: str = "Se (%)",
):
    width, height = 1200, 900
    margin_l, margin_r, margin_t, margin_b = 110, 80, 90, 110
    plot_l, plot_t = margin_l, margin_t
    plot_r, plot_b = width - margin_r, height - margin_b

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_title = _default_font(28)
    font_axis = _default_font(16)
    font_tick = _default_font(14)

    draw.text((margin_l, 25), title, fill=(0, 0, 0), font=font_title)
    draw.rectangle([plot_l, plot_t, plot_r, plot_b], outline=(0, 0, 0), width=2)

    for t in _nice_ticks(x_min, x_max, n=6):
        x = plot_l + (t - x_min) / (x_max - x_min + 1e-9) * (plot_r - plot_l)
        draw.line([x, plot_t, x, plot_b], fill=(245, 245, 245), width=1)
        draw.text((x, plot_b + 10), f"{t:.0f}", fill=(0, 0, 0), font=font_tick, anchor="ma")
    for t in _nice_ticks(y_min, y_max, n=6):
        y = plot_b - (t - y_min) / (y_max - y_min + 1e-9) * (plot_b - plot_t)
        draw.line([plot_l, y, plot_r, y], fill=(245, 245, 245), width=1)
        draw.text((plot_l - 10, y - 7), f"{t:.0f}", fill=(0, 0, 0), font=font_tick, anchor="ra")

    draw.text(((plot_l + plot_r) / 2, height - 55), x_label, fill=(0, 0, 0), font=font_axis, anchor="ma")
    draw.text((25, (plot_t + plot_b) / 2), y_label, fill=(0, 0, 0), font=font_axis)

    for i, (xv, yv) in enumerate(zip(xs, ys)):
        x = plot_l + (xv - x_min) / (x_max - x_min + 1e-9) * (plot_r - plot_l)
        y = plot_b - (yv - y_min) / (y_max - y_min + 1e-9) * (plot_b - plot_t)
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(30, 136, 229), outline=(30, 136, 229))
        if labels and i < len(labels):
            draw.text((x + 8, y - 8), labels[i], fill=(0, 0, 0), font=font_tick)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def _draw_grouped_bars(
    out_path: str,
    title: str,
    categories: List[str],
    a_values: List[int],
    b_values: List[int],
    a_label: str,
    b_label: str,
    y_label: str = "count",
):
    width, height = 1400, 850
    margin_l, margin_r, margin_t, margin_b = 110, 200, 90, 120
    plot_l, plot_t = margin_l, margin_t
    plot_r, plot_b = width - margin_r, height - margin_b

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_title = _default_font(28)
    font_axis = _default_font(16)
    font_tick = _default_font(14)

    draw.text((margin_l, 25), title, fill=(0, 0, 0), font=font_title)
    draw.rectangle([plot_l, plot_t, plot_r, plot_b], outline=(0, 0, 0), width=2)

    max_v = max(a_values + b_values + [1])
    y_max = float(max_v) * 1.15

    for t in _nice_ticks(0, y_max, n=6):
        y = plot_b - (t / (y_max + 1e-9)) * (plot_b - plot_t)
        draw.line([plot_l, y, plot_r, y], fill=(230, 230, 230), width=1)
        draw.text((plot_l - 10, y - 7), f"{t:.0f}", fill=(0, 0, 0), font=font_tick, anchor="ra")

    n = len(categories)
    if n == 0:
        img.save(out_path)
        return

    group_w = (plot_r - plot_l) / n
    bar_w = group_w * 0.28
    gap = group_w * 0.12

    color_a = (67, 160, 71)
    color_b = (244, 67, 54)

    for i, cat in enumerate(categories):
        cx = plot_l + i * group_w + group_w / 2
        x1 = cx - bar_w - gap / 2
        x2 = cx + gap / 2

        for x_left, v, color in [(x1, a_values[i], color_a), (x2, b_values[i], color_b)]:
            y_top = plot_b - (v / (y_max + 1e-9)) * (plot_b - plot_t)
            draw.rectangle([x_left, y_top, x_left + bar_w, plot_b], fill=color, outline=color)
            draw.text((x_left + bar_w / 2, y_top - 6), str(v), fill=(0, 0, 0), font=font_tick, anchor="ms")

        draw.text((cx, plot_b + 10), cat, fill=(0, 0, 0), font=font_tick, anchor="ma")

    draw.text(((plot_l + plot_r) / 2, height - 55), "class", fill=(0, 0, 0), font=font_axis, anchor="ma")
    draw.text((25, (plot_t + plot_b) / 2), y_label, fill=(0, 0, 0), font=font_axis)

    legend_x = plot_r + 30
    legend_y = plot_t + 20
    draw.rectangle([legend_x, legend_y, legend_x + 30, legend_y + 20], fill=color_a, outline=color_a)
    draw.text((legend_x + 40, legend_y), a_label, fill=(0, 0, 0), font=font_axis)
    draw.rectangle([legend_x, legend_y + 35, legend_x + 30, legend_y + 55], fill=color_b, outline=color_b)
    draw.text((legend_x + 40, legend_y + 35), b_label, fill=(0, 0, 0), font=font_axis)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def parse_log(log_text: str) -> Tuple[List[EpochMetrics], List[int], Dict[str, Dict[int, Tuple[str, int, float]]]]:
    epoch_train_re = re.compile(r"Train epoch\s+(\d+),.*accuracy:([0-9.]+)")
    spse_re = re.compile(r"\*\s*S_p:\s*([0-9.]+),\s*S_e:\s*([0-9.]+),\s*Score:\s*([0-9.]+)")
    acc1_re = re.compile(r"\*\s*Acc@1\s*([0-9.]+)")
    best_re = re.compile(r"Best ckpt is modified.*Epoch\s*=\s*(\d+)")

    class_re = re.compile(r"Class\s+(\d+)\s+\(([^)]+)\)\s*:\s*(\d+)\s*\(([\d.]+)%\)")
    split_mode = None
    class_stats: Dict[str, Dict[int, Tuple[str, int, float]]] = {"train": {}, "test": {}}

    metrics_by_epoch: Dict[int, EpochMetrics] = {}
    best_epochs: List[int] = []
    current_epoch: Optional[int] = None

    for line in log_text.splitlines():
        if "[Preprocessed train dataset information]" in line:
            split_mode = "train"
        elif "[Preprocessed test dataset information]" in line:
            split_mode = "test"

        m = class_re.search(line)
        if m and split_mode in {"train", "test"}:
            cid = int(m.group(1))
            cname = m.group(2).strip()
            cnt = int(m.group(3))
            pct = float(m.group(4))
            class_stats[split_mode][cid] = (cname, cnt, pct)

        m = epoch_train_re.search(line)
        if m:
            current_epoch = int(m.group(1))
            train_acc = float(m.group(2))
            metrics_by_epoch.setdefault(current_epoch, EpochMetrics(epoch=current_epoch)).train_acc = train_acc
            continue

        m = spse_re.search(line)
        if m and current_epoch is not None:
            em = metrics_by_epoch.setdefault(current_epoch, EpochMetrics(epoch=current_epoch))
            em.val_sp = float(m.group(1))
            em.val_se = float(m.group(2))
            em.val_score = float(m.group(3))
            continue

        m = acc1_re.search(line)
        if m and current_epoch is not None:
            metrics_by_epoch.setdefault(current_epoch, EpochMetrics(epoch=current_epoch)).val_acc1 = float(m.group(1))
            continue

        m = best_re.search(line)
        if m:
            best_epochs.append(int(m.group(1)))

    epochs = sorted(metrics_by_epoch.keys())
    metrics_list = [metrics_by_epoch[e] for e in epochs]
    return metrics_list, sorted(set(best_epochs)), class_stats


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_path = os.path.join(repo_root, "python main.log")
    out_dir = os.path.dirname(__file__)

    log_text = _load_text(log_path)
    metrics, best_epochs, class_stats = parse_log(log_text)
    if not metrics:
        raise RuntimeError("未从日志中解析到 epoch 指标，请检查日志格式。")

    epochs = [m.epoch for m in metrics]

    def series_vals(attr: str) -> List[Optional[float]]:
        return [getattr(m, attr) for m in metrics]

    _draw_line_chart(
        out_path=os.path.join(out_dir, "train_val_acc.png"),
        title="Train Acc vs Val Acc@1 (from python main.log)",
        epochs=epochs,
        series={"train_acc": series_vals("train_acc"), "val_acc1": series_vals("val_acc1")},
        y_min=0,
        y_max=100,
        y_label="accuracy (%)",
    )

    _draw_line_chart(
        out_path=os.path.join(out_dir, "sp_se_score.png"),
        title="Val Sp / Se / Score (from python main.log)",
        epochs=epochs,
        series={"val_sp": series_vals("val_sp"), "val_se": series_vals("val_se"), "val_score": series_vals("val_score")},
        y_min=0,
        y_max=100,
        y_label="metric (%)",
        best_epochs=best_epochs,
        annotate_best=True,
    )

    _draw_line_chart(
        out_path=os.path.join(out_dir, "score_with_best_ckpt.png"),
        title="Val Score with Best-CKPT Epoch Markers",
        epochs=epochs,
        series={"val_score": series_vals("val_score")},
        y_min=0,
        y_max=100,
        y_label="score (%)",
        best_epochs=best_epochs,
        annotate_best=True,
    )

    scatter_x = []
    scatter_y = []
    scatter_labels = []
    for m in metrics:
        if m.val_sp is None or m.val_se is None:
            continue
        scatter_x.append(m.val_sp)
        scatter_y.append(m.val_se)
        scatter_labels.append(str(m.epoch))

    _draw_scatter(
        out_path=os.path.join(out_dir, "sp_vs_se_scatter.png"),
        title="Val Sp vs Se (label=epoch)",
        xs=scatter_x,
        ys=scatter_y,
        labels=scatter_labels,
        x_min=0,
        x_max=100,
        y_min=0,
        y_max=100,
        x_label="Sp (%)",
        y_label="Se (%)",
    )

    if class_stats["train"] and class_stats["test"]:
        classes = sorted(set(class_stats["train"].keys()) | set(class_stats["test"].keys()))
        labels = []
        train_counts = []
        test_counts = []
        for cid in classes:
            t = class_stats["train"].get(cid)
            v = class_stats["test"].get(cid)
            cname = t[0] if t else (v[0] if v else str(cid))
            labels.append(f"{cid}:{cname}")
            train_counts.append(t[1] if t else 0)
            test_counts.append(v[1] if v else 0)

        _draw_grouped_bars(
            out_path=os.path.join(out_dir, "class_distribution_train_test.png"),
            title="ICBHI class distribution (from log)",
            categories=labels,
            a_values=train_counts,
            b_values=test_counts,
            a_label="train cycles",
            b_label="test cycles",
            y_label="cycle count",
        )

    print("generated in:", out_dir)
    for fn in [
        "train_val_acc.png",
        "sp_se_score.png",
        "score_with_best_ckpt.png",
        "sp_vs_se_scatter.png",
        "class_distribution_train_test.png",
    ]:
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            print(p)


if __name__ == "__main__":
    main()

