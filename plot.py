# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///

"""读取保存在 data/ 中的地震数据，绘制第一版海报。

运行方法：uv run plot.py
图片将保存到 out/earthquake-atlas.png。
"""

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt

FILE = "usgs-earthquakes-2.5-month.geojson"
PICTURE = "earthquake-atlas.png"

HERE = Path(__file__).parent
DATA = HERE / "data" / FILE
OUT = HERE / "out"

BACKGROUND = "#101210"
PANEL = "#171a17"
GRID = "#343b35"
TEXT = "#e3e8df"
MUTED = "#8b958b"
SIGNAL = "#a6ff18"


def load_earthquakes(path):
    """从 GeoJSON 中取出绘图需要的数字。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    quakes = []
    for feature in raw["features"]:
        magnitude = feature["properties"]["mag"]
        coordinates = feature["geometry"]["coordinates"]
        if magnitude is None or not coordinates:
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
    """震级越高，点越大。"""
    return 5 * 2.2 ** (magnitude - 2.5)


def depth_gray(depth):
    """较浅的地震画得较亮；超过 400 km 的都使用最暗的灰。"""
    shade = 0.78 - min(max(depth, 0), 400) / 400 * 0.5
    return (shade, shade, shade)


def main():
    quakes = load_earthquakes(DATA)
    if not quakes:
        raise SystemExit("No earthquakes found in the saved data file.")
    first = min(q["date"] for q in quakes)
    last = max(q["date"] for q in quakes)
    strongest = sorted(quakes, key=lambda q: q["magnitude"], reverse=True)[:8]
    print(f"{len(quakes)} earthquakes, {first} to {last} (UTC)")

    fig, (map_ax, time_ax) = plt.subplots(
        2, 1, figsize=(12, 9), facecolor=BACKGROUND,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    fig.subplots_adjust(left=0.08, right=0.95, top=0.81, bottom=0.12, hspace=0.29)
    fig.text(0.08, 0.94, "SEISMIC FIELD / 01", color=SIGNAL, fontsize=12, weight="bold")
    fig.text(0.08, 0.89, "EARTHQUAKE ATLAS", color=TEXT, fontsize=27, weight="bold")
    fig.text(
        0.08, 0.85,
        f"USGS  /  M2.5+  /  {first} — {last} UTC  /  {len(quakes):,} EVENTS",
        color=MUTED, fontsize=10,
    )

    map_ax.set_facecolor(PANEL)
    map_ax.scatter(
        [q["longitude"] for q in quakes],
        [q["latitude"] for q in quakes],
        s=[dot_size(q["magnitude"]) for q in quakes],
        c=[depth_gray(q["depth"]) for q in quakes],
        alpha=0.68, linewidths=0, zorder=3,
    )
    # 只用荧光绿强调震级最高的八次地震。
    for number, quake in enumerate(strongest, start=1):
        map_ax.scatter(
            quake["longitude"], quake["latitude"],
            s=dot_size(quake["magnitude"]) * 1.8,
            facecolors="none", edgecolors=SIGNAL, linewidths=1.2, zorder=4,
        )
        map_ax.annotate(
            f"{number:02}", (quake["longitude"], quake["latitude"]),
            xytext=(7, 7), textcoords="offset points",
            color=SIGNAL, fontsize=8, zorder=5,
        )
    map_ax.set_xlim(-180, 180)
    map_ax.set_ylim(-90, 90)
    map_ax.set_aspect("equal")
    map_ax.set_xticks(range(-180, 181, 60))
    map_ax.set_yticks(range(-90, 91, 30))
    map_ax.grid(color=GRID, linewidth=0.65)
    map_ax.tick_params(colors=MUTED, labelsize=8, length=0)
    map_ax.set_xlabel("LONGITUDE  /  DEGREES", color=MUTED, fontsize=8, labelpad=10)
    map_ax.set_ylabel("LATITUDE  /  DEGREES", color=MUTED, fontsize=8, labelpad=10)
    for spine in map_ax.spines.values():
        spine.set_color(GRID)

    # 把同一天的地震数量累加起来，绘制下方的时间图。
    dates = [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]
    counts = {date: 0 for date in dates}
    for quake in quakes:
        counts[quake["date"]] += 1
    time_ax.set_facecolor(PANEL)
    time_ax.bar(range(len(dates)), [counts[date] for date in dates], color=MUTED, width=0.72)
    time_ax.set_xlim(-0.8, len(dates) - 0.2)
    time_ax.set_xticks([0, len(dates) // 2, len(dates) - 1])
    time_ax.set_xticklabels([str(dates[0]), str(dates[len(dates) // 2]), str(dates[-1])])
    time_ax.tick_params(colors=MUTED, labelsize=8, length=0)
    time_ax.set_ylabel("EVENTS / DAY", color=MUTED, fontsize=8, labelpad=10)
    time_ax.grid(axis="y", color=GRID, linewidth=0.65)
    time_ax.set_axisbelow(True)
    for spine in time_ax.spines.values():
        spine.set_color(GRID)

    fig.text(
        0.08, 0.04,
        "POSITION = LONGITUDE / LATITUDE     SIZE = MAGNITUDE     GRAY = DEPTH     GREEN = 8 STRONGEST",
        color=MUTED, fontsize=8,
    )
    OUT.mkdir(exist_ok=True)
    target = OUT / PICTURE
    fig.savefig(target, dpi=150, facecolor=BACKGROUND)
    print(f"saved {target.relative_to(HERE)}")
    plt.show()


if __name__ == "__main__":
    main()
