# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///

"""从本地 GeoJSON 绘制地震地图拼贴。运行：uv run plot.py。"""

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

HERE = Path(__file__).parent
QUAKES = HERE / "data" / "usgs-earthquakes-2.5-month.geojson"
LAND = HERE / "data" / "ne_110m_land.geojson"
OUT = HERE / "out" / "earthquake-atlas.png"

BLACK = "#101210"
PAPER = "#e2e3dd"
SILVER = "#a3aaa3"
MID = "#555c56"
SIGNAL = "#b0ff2a"


def load_earthquakes():
    """每条记录保留经纬度、震级、深度和 UTC 日期。"""
    raw = json.loads(QUAKES.read_text(encoding="utf-8"))
    quakes = []
    for feature in raw["features"]:
        coordinates = feature["geometry"]["coordinates"]
        magnitude = feature["properties"]["mag"]
        if magnitude is None or len(coordinates) < 3:
            continue
        longitude, latitude, depth = coordinates[:3]
        date = dt.datetime.fromtimestamp(
            feature["properties"]["time"] / 1000, dt.timezone.utc
        ).date()
        quakes.append((longitude, latitude, depth, magnitude, date))
    return quakes


def load_land():
    """从保存的 Natural Earth 文件读取陆地多边形。"""
    raw = json.loads(LAND.read_text(encoding="utf-8"))
    rings = []
    for feature in raw["features"]:
        geometry = feature["geometry"]
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        for polygon in polygons:
            rings.append(polygon[0])  # 只绘制海岸线，不切出内陆湖泊。
    return rings


def density_field(quakes):
    """按经纬度汇总地震；线条表示平滑后的事件密度。"""
    field = np.zeros((180, 360), dtype=float)
    for longitude, latitude, _depth, magnitude, _date in quakes:
        x = min(359, max(0, int(longitude + 180)))
        y = min(179, max(0, int(latitude + 90)))
        field[y, x] += max(magnitude - 2.4, 0.1)
    for _ in range(7):
        above = np.vstack((field[:1], field[:-1]))
        below = np.vstack((field[1:], field[-1:]))
        field = (4 * field + np.roll(field, 1, 1)
                 + np.roll(field, -1, 1) + above + below) / 8
    return field


def draw_map(ax, quakes, land, density, bounds, palette, grid_step):
    """用同一份数据制作不同裁切和明暗的地图。"""
    west, east, south, north = bounds
    ocean, land_color, coast, points, contour = palette
    ax.set_facecolor(ocean)

    # 经纬网只作极细的定位辅助线，不与数据点争夺注意力。
    for x in range(-180, 181, grid_step):
        ax.plot([x, x], [south, north], color=coast, alpha=0.17,
                linewidth=0.35, zorder=1)
    for y in range(-90, 91, grid_step):
        ax.plot([west, east], [y, y], color=coast, alpha=0.17,
                linewidth=0.35, zorder=1)

    for ring in land:
        ax.add_patch(Polygon(
            ring, closed=True, facecolor=land_color, edgecolor=coast,
            linewidth=0.33, zorder=2,
        ))

    values = density[density > 0.03]
    if len(values):
        levels = np.unique(np.quantile(values, [0.52, 0.67, 0.80, 0.91]))
        ax.contour(
            np.linspace(-179.5, 179.5, 360),
            np.linspace(-89.5, 89.5, 180), density,
            levels=levels, colors=contour, linewidths=0.43,
            alpha=0.59, zorder=3,
        )

    visible = [q for q in quakes
               if west <= q[0] <= east and south <= q[1] <= north]
    ax.scatter(
        [q[0] for q in visible], [q[1] for q in visible],
        s=[1.5 + max(q[3] - 2.5, 0) ** 2 * 1.3 for q in visible],
        c=points, alpha=0.72, linewidths=0, zorder=4,
    )
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return len(visible)


def label(fig, x, y, message, size=8, color=SILVER, weight="normal", **kwargs):
    fig.text(x, y, message, fontsize=size, color=color, weight=weight, **kwargs)


def main():
    quakes = load_earthquakes()
    land = load_land()
    density = density_field(quakes)
    first = min(q[4] for q in quakes)
    last = max(q[4] for q in quakes)
    strongest = sorted(quakes, key=lambda q: q[3], reverse=True)[:5]

    fig = plt.figure(figsize=(12, 15), facecolor=BLACK)
    fig.patches.append(Rectangle(
        (0.055, 0.864), 0.575, 0.112, transform=fig.transFigure,
        facecolor=PAPER, edgecolor="none", zorder=1,
    ))
    fig.patches.append(Rectangle(
        (0.055, 0.864), 0.008, 0.112, transform=fig.transFigure,
        facecolor=SIGNAL, edgecolor="none", zorder=2,
    ))
    label(fig, 0.078, 0.952, "SEISMIC / FIELD RECORD 01", 8, BLACK, "bold")
    label(fig, 0.078, 0.898, "EARTHQUAKE ATLAS", 27, BLACK, "bold")
    label(fig, 0.672, 0.945, "USGS  /  M 2.5+", 10, PAPER, "bold")
    label(fig, 0.672, 0.916, f"{first} — {last}", 8, SILVER)
    label(fig, 0.672, 0.891, "PAST-MONTH GLOBAL RECORD  /  UTC", 7, SILVER)
    fig.add_artist(Line2D([0.055, 0.945], [0.846, 0.846],
                          transform=fig.transFigure, color=MID, linewidth=0.6))

    label(fig, 0.055, 0.829, "01   /   GLOBAL DISTRIBUTION", 8, PAPER, "bold")
    label(fig, 0.945, 0.829, "LONGITUDE  −180° TO +180°", 7, SILVER,
          ha="right")
    global_ax = fig.add_axes([0.055, 0.508, 0.89, 0.317])
    dark = ("#171a18", "#373d38", "#8b948b", PAPER, "#d8e0d6")
    draw_map(global_ax, quakes, land, density,
             (-180, 180, -80, 80), dark, 20)
    for quake in strongest:
        global_ax.scatter(quake[0], quake[1], s=51, facecolors="none",
                          edgecolors=SIGNAL, linewidths=0.9, zorder=6)

    # 三个裁切区在总图上留下极细的定位框。
    windows = [
        ("02", "WEST PACIFIC", (90, 177, -10, 44),
         [0.055, 0.188, 0.575, 0.285],
         (PAPER, "#929c92", "#464f48", BLACK, BLACK), 10),
        ("03", "ANDEAN EDGE", (-89, -28, -49, -15),
         [0.650, 0.340, 0.295, 0.133],
         ("#444a45", "#bfc5bc", "#222721", BLACK, BLACK), 10),
        ("04", "NORTH PACIFIC", (-180, -105, 25, 67),
         [0.650, 0.188, 0.295, 0.133],
         ("#1b201c", "#4b544c", "#9da89b", PAPER, PAPER), 10),
    ]
    for number, name, bounds, position, palette, grid_step in windows:
        west, east, south, north = bounds
        global_ax.add_patch(Rectangle(
            (west, south), east - west, north - south,
            fill=False, edgecolor=PAPER, linestyle=(0, (2, 3)),
            linewidth=0.55, alpha=0.58, zorder=5,
        ))
        panel = fig.add_axes(position)
        count = draw_map(panel, quakes, land, density, bounds,
                         palette, grid_step)
        panel.text(0.025, 0.955, f"{number}  /  {name}",
                   transform=panel.transAxes, va="top", fontsize=8,
                   weight="bold", color=palette[3],
                   bbox={"facecolor":palette[0], "edgecolor":"none",
                         "alpha":0.88, "pad":4})
        panel.text(0.025, 0.055, f"{count} RECORDED EVENTS",
                   transform=panel.transAxes, fontsize=7,
                   color=palette[3],
                   bbox={"facecolor":palette[0], "edgecolor":"none",
                         "alpha":0.88, "pad":3})

    label(fig, 0.055, 0.487, "A MONTH OF EARTH MOVEMENT, SEEN AT FOUR SCALES.",
          8, SILVER)
    label(fig, 0.055, 0.162, "USGS / EARTHQUAKE CATALOGUE", 8, PAPER, "bold")
    fig.add_artist(Line2D([0.055, 0.945], [0.147, 0.147],
                          transform=fig.transFigure, color=MID, linewidth=0.6))

    label(fig, 0.055, 0.071, f"{len(quakes):,}", 31, PAPER, "bold")
    label(fig, 0.230, 0.076, "EARTHQUAKES\nIN THE SAVED DATA", 8, SILVER,
          linespacing=1.5)
    label(fig, 0.515, 0.105, "POSITION  /  LONGITUDE + LATITUDE", 7, PAPER)
    label(fig, 0.515, 0.083, "DOT SIZE  /  MAGNITUDE", 7, PAPER)
    label(fig, 0.515, 0.061, "FINE LINES  /  SMOOTHED EVENT DENSITY", 7, PAPER)
    label(fig, 0.515, 0.039, "LIME RINGS  /  FIVE STRONGEST EVENTS", 7, SIGNAL)
    label(fig, 0.055, 0.020, "MAP OUTLINES / NATURAL EARTH 1:110M", 6, SILVER)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=BLACK)
    print(f"saved {OUT.relative_to(HERE)} from {len(quakes)} earthquakes")
    plt.show()


if __name__ == "__main__":
    main()
