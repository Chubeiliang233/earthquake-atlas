# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///

"""读取本地 USGS 地震数据，生成地图拼贴海报。

运行方法：uv run plot.py
生成文件：out/earthquake-atlas.png
"""

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = Path(__file__).parent
DATA = HERE / "data" / "usgs-earthquakes-2.5-month.geojson"
OUT = HERE / "out"
PICTURE = "earthquake-atlas.png"

BACKGROUND = "#0d0f0e"
PANEL = "#171a18"
GRID = "#343b36"
TEXT = "#e3e8df"
MUTED = "#8b958b"
SIGNAL = "#a6ff18"
DENSITY_COLOURS = LinearSegmentedColormap.from_list(
    "seismic_density", [PANEL, "#252b27", "#59615a", "#c3c8bf"]
)

# 三个放大区域使用相同的经纬度比例，避免把地图拉伸。
REGIONS = [
    ("NORTH PACIFIC", (-180, -119, 30, 80)),
    ("WEST PACIFIC", (115, 163, 8, 47)),
    ("SOUTH AMERICA", (-87, -39, -55, -16)),
]


def load_earthquakes(path):
    """从保存的 GeoJSON 中取出经纬度、深度、震级和日期。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    quakes = []
    for feature in raw["features"]:
        magnitude = feature["properties"]["mag"]
        coordinates = feature["geometry"]["coordinates"]
        if magnitude is None or not coordinates or len(coordinates) < 3:
            continue
        longitude, latitude, depth = coordinates[:3]
        occurred = dt.datetime.fromtimestamp(
            feature["properties"]["time"] / 1000, dt.timezone.utc
        )
        quakes.append({
            "longitude": longitude,
            "latitude": latitude,
            "depth": depth,
            "magnitude": magnitude,
            "date": occurred.date(),
        })
    return quakes


def dot_size(magnitude):
    """震级越高，地震点越大。"""
    return 3 * 2.2 ** (magnitude - 2.5)


def depth_gray(depth):
    """浅层地震较亮，深层地震较暗。"""
    shade = 0.78 - min(max(depth, 0), 400) / 400 * 0.5
    return (shade, shade, shade)


def seismic_density(quakes):
    """把地震放进一度宽的网格，再稍微平滑，供灰度纹理和细线使用。"""
    field = np.zeros((180, 360), dtype=float)
    for quake in quakes:
        x = min(359, max(0, int(quake["longitude"] + 180)))
        y = min(179, max(0, int(quake["latitude"] + 90)))
        field[y, x] += max(quake["magnitude"] - 2, 0.5) ** 2

    for _ in range(10):
        up = np.vstack((field[:1], field[:-1]))
        down = np.vstack((field[1:], field[-1:]))
        field = (
            4 * field
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
            + up + down
        ) / 8
    return np.log1p(field)


def draw_map(ax, quakes, density, bounds, show_coordinates=False):
    """用同一组真实数据绘制全球地图或局部放大图。"""
    west, east, south, north = bounds
    ax.set_facecolor(PANEL)
    upper = max(float(np.quantile(density, 0.995)), 0.01)
    ax.imshow(
        density, extent=(-180, 180, -90, 90), origin="lower",
        cmap=DENSITY_COLOURS, vmin=0, vmax=upper,
        interpolation="nearest", zorder=1,
    )

    # 这些细线表示地震活动密度相同的位置，不是地形等高线。
    values = density[density > 0.02]
    if len(values):
        levels = np.unique(np.quantile(values, [0.55, 0.72, 0.86, 0.95]))
        x = np.linspace(-179.5, 179.5, density.shape[1])
        y = np.linspace(-89.5, 89.5, density.shape[0])
        ax.contour(
            x, y, density, levels=levels,
            colors="#d0d7cc", linewidths=0.42, alpha=0.47, zorder=2,
        )

    visible = [
        q for q in quakes
        if west <= q["longitude"] <= east and south <= q["latitude"] <= north
    ]
    ax.scatter(
        [q["longitude"] for q in visible],
        [q["latitude"] for q in visible],
        s=[dot_size(q["magnitude"]) for q in visible],
        c=[depth_gray(q["depth"]) for q in visible],
        alpha=0.67, linewidths=0, zorder=3,
    )
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal")
    ax.grid(color=GRID, linewidth=0.5, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color("#777f77")
        spine.set_linewidth(0.65)

    if show_coordinates:
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.tick_params(colors=MUTED, labelsize=7, length=0)
        ax.set_xlabel("LONGITUDE  /  DEGREES", color=MUTED, fontsize=7, labelpad=9)
        ax.set_ylabel("LATITUDE  /  DEGREES", color=MUTED, fontsize=7, labelpad=9)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    return len(visible)


def main():
    quakes = load_earthquakes(DATA)
    if not quakes:
        raise SystemExit("No earthquakes found in the saved data file.")
    first = min(q["date"] for q in quakes)
    last = max(q["date"] for q in quakes)
    strongest = sorted(quakes, key=lambda q: q["magnitude"], reverse=True)[:8]
    density = seismic_density(quakes)
    print(f"{len(quakes)} earthquakes, {first} to {last} (UTC)")

    fig = plt.figure(figsize=(12, 16), facecolor=BACKGROUND)
    fig.text(0.08, 0.955, "FIELD RECORD  /  01", color=SIGNAL, fontsize=10, weight="bold")
    fig.text(0.08, 0.913, "EARTHQUAKE ATLAS", color=TEXT, fontsize=29, weight="bold")
    fig.text(
        0.08, 0.886,
        f"USGS  /  M2.5+  /  {first} — {last} UTC  /  {len(quakes):,} EVENTS",
        color=MUTED, fontsize=9,
    )
    fig.text(0.08, 0.835, "01 / GLOBAL SEISMIC FIELD", color=TEXT, fontsize=9)
    fig.text(0.92, 0.835, "180° W  —  180° E", color=MUTED, fontsize=8, ha="right")

    global_ax = fig.add_axes([0.08, 0.495, 0.84, 0.315])
    draw_map(global_ax, quakes, density, (-180, 180, -90, 90), show_coordinates=True)
    for number, (_, (west, east, south, north)) in enumerate(REGIONS, start=1):
        global_ax.add_patch(Rectangle(
            (west, south), east - west, north - south,
            fill=False, edgecolor=TEXT, linewidth=0.65,
            linestyle=(0, (3, 3)), alpha=0.6, zorder=3.5,
        ))
        global_ax.text(
            west + 2, north - 3, f"{number:02}", color=TEXT,
            fontsize=6, va="top", zorder=4,
        )
    for number, quake in enumerate(strongest, start=1):
        global_ax.scatter(
            quake["longitude"], quake["latitude"],
            s=dot_size(quake["magnitude"]) * 2,
            facecolors="none", edgecolors=SIGNAL,
            linewidths=1.1, zorder=4,
        )
        global_ax.annotate(
            f"{number:02}", (quake["longitude"], quake["latitude"]),
            xytext=(7, 7), textcoords="offset points",
            color=SIGNAL, fontsize=7, zorder=5,
        )

    fig.text(0.08, 0.469, "02 / REGIONAL DETAILS", color=TEXT, fontsize=9)
    positions = [0.08, 0.37, 0.66]
    for index, ((name, bounds), left) in enumerate(zip(REGIONS, positions), start=1):
        panel = fig.add_axes([left, 0.282, 0.26, 0.16])
        count = draw_map(panel, quakes, density, bounds)
        panel.text(
            0.03, 0.94, f"{index:02}  {name}",
            transform=panel.transAxes, va="top",
            color=TEXT, fontsize=8, weight="bold",
            bbox={"facecolor":BACKGROUND, "edgecolor":"none", "alpha":0.78, "pad":3},
        )
        panel.text(
            0.03, 0.06, f"{count:03} EVENTS  /  DETAIL",
            transform=panel.transAxes, va="bottom",
            color=SIGNAL, fontsize=7,
            bbox={"facecolor":BACKGROUND, "edgecolor":"none", "alpha":0.78, "pad":2},
        )

    fig.text(0.08, 0.251, "03 / EVENT FREQUENCY", color=TEXT, fontsize=9)
    fig.text(0.92, 0.251, "UTC DAYS  /  FIRST & LAST MAY BE PARTIAL",
             color=MUTED, fontsize=7, ha="right")
    dates = [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]
    counts = {date: 0 for date in dates}
    for quake in quakes:
        counts[quake["date"]] += 1
    time_ax = fig.add_axes([0.08, 0.113, 0.84, 0.115], facecolor=PANEL)
    time_ax.bar(
        range(len(dates)), [counts[date] for date in dates],
        color=["#59625a" if i in (0, len(dates) - 1) else "#aab2a7"
               for i in range(len(dates))],
        width=0.72,
    )
    time_ax.set_xlim(-0.8, len(dates) - 0.2)
    middle = len(dates) // 2
    time_ax.set_xticks([0, middle, len(dates) - 1])
    time_ax.set_xticklabels([str(dates[0]), str(dates[middle]), str(dates[-1])])
    time_ax.tick_params(colors=MUTED, labelsize=7, length=0)
    time_ax.set_ylabel("EVENTS / DAY", color=MUTED, fontsize=7, labelpad=9)
    time_ax.grid(axis="y", color=GRID, linewidth=0.5)
    time_ax.set_axisbelow(True)
    for spine in time_ax.spines.values():
        spine.set_color("#777f77")
        spine.set_linewidth(0.65)

    fig.text(
        0.08, 0.072,
        "POSITION / LONGITUDE + LATITUDE    SIZE / MAGNITUDE    GRAY / DEPTH (KM)    GREEN / 8 STRONGEST",
        color=TEXT, fontsize=7,
    )
    fig.text(
        0.08, 0.051,
        "THIN CONTOURS / SMOOTHED, MAGNITUDE-WEIGHTED EVENT DENSITY    •    SOURCE / USGS GEOJSON",
        color=MUTED, fontsize=7,
    )
    OUT.mkdir(exist_ok=True)
    target = OUT / PICTURE
    fig.savefig(target, dpi=150, facecolor=BACKGROUND)
    print(f"saved {target.relative_to(HERE)}")
    plt.show()


if __name__ == "__main__":
    main()
